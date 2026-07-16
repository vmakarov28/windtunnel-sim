"""D3Q19 BGK solver — the readable PyTorch reference implementation (3D).

This is the 3D sibling of lbm/solver.py (the finished, validated 2D
program). It shares NO solver code with 2D — only lbm.units, the
dimension-blind Reynolds triangle — but it inherits every lesson the 2D
program paid for (see notes/NOTES.md, the four autopsies):

  * anechoic forcing sponge, not a viscosity ramp (absorbs wakes AND the
    acoustics a ramped inlet excites; pins the outlet pressure so the box
    cannot pressurize — the original 3D run settled at rho~1.06, ran 12%
    slow, and never shed because of this);
  * equilibrium velocity inlet (carries no non-equilibrium mode, so it is
    robust at low viscosity; Zou-He + regularization is the planned
    refinement for the quantitative 3D gates);
  * interior-only guards (a boundary condition must never grade itself);
  * Smagorinsky from the LOCAL Pi_neq (Hou et al. 1996) with the same
    closed form as 2D — the formula is dimension-independent;
  * local-Mach discipline: separated flow amplifies velocity ~3x, so
    high-Re scenes run u_lat = 0.05, resolution up via the triangle.

One time step (Kruger et al. 2017, ch. 3-6):

  1. macroscopics    rho = sum_q f_q,  u = (sum_q e_q f_q + F/2)/rho
  2. BGK collision   f_q += -omega (f_q - f_q^eq) + Guo source   (eq. 3.9,
       6.25); omega is a field when the Smagorinsky model is on
  2b. anechoic sponge over the last sponge_fraction of the domain
  3. streaming       f_q(x + e_q, t+1) = f_q(x, t)      -> torch.roll
  4. halfway bounce-back at solids (eq. 5.26); moving lid adds
       +6 w_q rho0 (e_q . u_wall)                       (eq. 5.27)
  4b. momentum-exchange force on the obstacle           (eq. 5.51)
  5. equilibrium velocity inlet at x = 0 (ramped)

Memory layout: f has shape (19, nx, ny, nz), fp32. All ops are per-
direction loops over q so peak temporaries stay ~a few scalar fields.

Unit discipline: tau and u_char always come from a resolved Scene
(lbm.units). The low-level constructor taking raw values exists ONLY for
tests and validation scripts that need engineered analytic cases.
"""

from __future__ import annotations

import math
import shutil
from datetime import datetime
from pathlib import Path

import torch

from .lattice import CS2, E, OPP, Q, W
from .config import Scene


class SimulationBlowup(RuntimeError):
    """NaN or runaway velocity detected. This is footage, not just an error."""


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Solver:
    def __init__(
        self,
        nx: int, ny: int, nz: int,
        tau: float,
        u_char: float,
        *,
        seed: int = 0,
        device: str = "auto",
        obstacle_mask: torch.Tensor | None = None,   # bool (nx,ny,nz), True=solid
        inlet_outlet: bool = True,      # False -> periodic/closed box
        ramp_steps: int = 2000,         # inlet ramp; 0 = impulsive start
        sponge_fraction: float = 0.08,  # of nx, before the outlet
        sponge_strength: float = 0.15,  # peak anechoic blend rate
        wall_y: bool = False,           # bounce-back walls at y=0, y=ny-1
        wall_x: bool = False,           # bounce-back walls at x edges (cavity)
        lid_velocity: float = 0.0,      # top wall moves in +x (needs wall_y)
        body_force: tuple[float, float, float] = (0.0, 0.0, 0.0),  # Guo
        sgs: bool = False,              # Smagorinsky subgrid model
        cs_smag: float = 0.14,          # Smagorinsky constant, ~0.1-0.17
        init_noise: float = 1e-3,       # *u_char, seeds the shedding asymmetry
        scene_name: str = "adhoc",
    ):
        self.device = pick_device(device)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.tau, self.u_char = float(tau), float(u_char)
        self.seed = seed
        self.step_count = 0
        self.inlet_outlet = inlet_outlet
        self.ramp_steps = ramp_steps
        self.lid_velocity = float(lid_velocity)
        self.force = tuple(float(c) for c in body_force)
        self.has_force = any(c != 0.0 for c in self.force)
        self.sgs = bool(sgs)
        self.cs_smag = float(cs_smag)
        self.last_tau_eff_max = self.tau
        self.scene_name = scene_name
        self.omega = 1.0 / self.tau

        # -- anechoic sponge profile (see module docstring) ----------------
        self._n_sp = 0
        if inlet_outlet and sponge_fraction > 0:
            self._n_sp = max(2, round(nx * sponge_fraction))
            s = torch.linspace(0.0, 1.0, self._n_sp)
            self._sponge_sigma = (
                sponge_strength * s * s              # gentle entry, C1
            ).to(self.device).view(self._n_sp, 1, 1)

        # -- solid mask (+ moving lid cells) --------------------------------
        mask = torch.zeros((nx, ny, nz), dtype=torch.bool)
        lid = torch.zeros((nx, ny, nz), dtype=torch.bool)
        if obstacle_mask is not None:
            mask |= obstacle_mask.cpu()
        if wall_y:
            mask[:, 0, :] = True
            if self.lid_velocity != 0.0:
                lid[:, -1, :] = True   # moving lid = the top wall
            mask[:, -1, :] = True
        if wall_x:
            mask[0, :, :] = True
            mask[-1, :, :] = True
            lid[0, :, :] = lid[-1, :, :] = False  # side walls never move
        self.mask = mask.to(self.device)
        self._lid_mask = lid.to(self.device)
        self.fluid_cells = int((~mask).sum())

        # Precompute bounce-back link indices (flat, per direction): fluid
        # cells whose pull-neighbor x - e_q is solid. After streaming,
        # f_q(x) would have come from inside the solid; halfway bounce-back
        # replaces it with the reflected post-collision population:
        # f_q(x,t+1) = f*_{opp(q)}(x,t)  [+ lid momentum, eq. 5.27].
        self._bounce_idx: list[torch.Tensor | None] = [None] * Q
        self._lid_idx: list[torch.Tensor | None] = [None] * Q
        self._lid_corr: list[float] = [0.0] * Q
        for q in range(1, Q):
            came_from_solid = torch.roll(
                self.mask, shifts=E[q], dims=(0, 1, 2)
            ) & ~self.mask
            idx = came_from_solid.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            self._bounce_idx[q] = idx if idx.numel() else None
            if self.lid_velocity != 0.0:
                up_lid = torch.roll(self._lid_mask, shifts=E[q],
                                    dims=(0, 1, 2)) & ~self.mask
                lidx = up_lid.reshape(-1).nonzero(as_tuple=False).squeeze(1)
                self._lid_idx[q] = lidx if lidx.numel() else None
                # +6 w_q (e_q . u_wall), u_wall = (lid_velocity, 0, 0)
                self._lid_corr[q] = 6.0 * W[q] * E[q][0] * self.lid_velocity
        self._solid_idx = self.mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)

        # Obstacle-only boundary links for the momentum-exchange force
        # (Kruger eq. 5.51). Walls/lid excluded — we weigh the model, not
        # the tunnel. Enabled on demand: set measure_force = True.
        self.measure_force = False
        self.last_force: torch.Tensor | None = None   # (3,) on device
        self._force_idx: list[torch.Tensor | None] = [None] * Q
        if obstacle_mask is not None:
            obs = obstacle_mask.cpu().to(self.device)
            for q in range(1, Q):
                up_obs = torch.roll(obs, shifts=E[q], dims=(0, 1, 2)) \
                    & ~self.mask
                idx = up_obs.reshape(-1).nonzero(as_tuple=False).squeeze(1)
                self._force_idx[q] = idx if idx.numel() else None

        # -- initial condition: equilibrium + seeded noise ------------------
        # Open scenes start at the (ramped) inlet velocity; closed boxes at
        # rest. Tiny seeded noise breaks the wake's metastable symmetry so
        # vortex shedding onsets reproducibly (seed -> same flow).
        gen = torch.Generator().manual_seed(seed)  # CPU gen: device-portable
        u = torch.zeros((3, nx, ny, nz), dtype=torch.float32)
        if inlet_outlet:
            u[0] = self.u_char * self._ramp_factor(0)
        u += (init_noise * self.u_char
              * (2.0 * torch.rand((3, nx, ny, nz), generator=gen) - 1.0))
        u[:, mask] = 0.0
        rho = torch.ones((nx, ny, nz), dtype=torch.float32)

        self.f = torch.empty((Q, nx, ny, nz), dtype=torch.float32,
                             device=self.device)
        self._write_equilibrium(self.f, rho.to(self.device), u.to(self.device))
        self.f[:, self.mask] = 0.0
        self.f2 = torch.empty_like(self.f)   # B buffer (A-B, like the kernel)
        self.initial_mass = float(self.f.sum())

    # ------------------------------------------------------------------
    @classmethod
    def from_scene(cls, scene: Scene, seed: int, device: str = "auto",
                   ramp: bool = True) -> "Solver":
        b = scene.raw.get("boundaries", {})
        obstacle = scene.raw.get("obstacle")
        mask = build_obstacle_mask(scene, obstacle) if obstacle else None
        top_bottom = b.get("top_bottom", "periodic")
        if top_bottom not in ("periodic", "bounce_back"):
            raise NotImplementedError(f"top_bottom = {top_bottom!r}")
        is_cavity = "lid" in b
        return cls(
            scene.nx, scene.ny, scene.nz,
            tau=scene.units.tau, u_char=scene.units.u_lat,
            seed=seed, device=device, obstacle_mask=mask,
            inlet_outlet="inlet" in b,
            ramp_steps=int(b.get("ramp_steps", 2000)) if ramp else 0,
            sponge_fraction=float(b.get("sponge_fraction", 0.08)),
            wall_y=(top_bottom == "bounce_back") or is_cavity,
            wall_x=is_cavity,
            lid_velocity=scene.units.u_lat if is_cavity else 0.0,
            sgs=scene.units.sgs,
            cs_smag=float(scene.raw.get("smagorinsky_cs", 0.14)),
            scene_name=scene.name,
        )

    # ------------------------------------------------------------------
    def _ramp_factor(self, step: int) -> float:
        """Smooth inlet ramp 0 -> 1 over ramp_steps (0 = impulsive start)."""
        if self.ramp_steps <= 0 or step >= self.ramp_steps:
            return 1.0
        return 0.5 - 0.5 * math.cos(math.pi * step / self.ramp_steps)

    def macroscopics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """rho (nx,ny,nz) and u (3,nx,ny,nz). With Guo forcing, u carries
        the half-step force shift u = (sum e f + F/2)/rho (eq. 6.27)."""
        rho = self.f.sum(dim=0)
        u = torch.zeros((3, self.nx, self.ny, self.nz),
                        dtype=torch.float32, device=self.device)
        for q in range(1, Q):
            ex, ey, ez = E[q]
            if ex: u[0] += ex * self.f[q]
            if ey: u[1] += ey * self.f[q]
            if ez: u[2] += ez * self.f[q]
        if self.has_force:
            u[0] += 0.5 * self.force[0]
            u[1] += 0.5 * self.force[1]
            u[2] += 0.5 * self.force[2]
        rho_safe = torch.where(rho > 1e-12, rho, torch.ones_like(rho))
        u /= rho_safe
        u[:, self.mask] = 0.0
        return rho, u

    def _write_equilibrium(self, out: torch.Tensor, rho: torch.Tensor,
                           u: torch.Tensor) -> None:
        """out[q] = f_q^eq(rho, u) for all q (eq. 3.54), per-q to save memory."""
        usq = (u * u).sum(dim=0)
        for q in range(Q):
            ex, ey, ez = E[q]
            cu = torch.zeros_like(usq)
            if ex: cu += ex * u[0]
            if ey: cu += ey * u[1]
            if ez: cu += ez * u[2]
            cu *= 3.0  # = (e.u)/c_s^2
            out[q] = W[q] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)

    def step(self) -> None:
        f, f2 = self.f, self.f2
        u_in = self.u_char * self._ramp_factor(self.step_count)
        fx, fy, fz = self.force

        # 1. macroscopics (with the Guo half-force shift).
        rho, u = self.macroscopics()
        usq = (u * u).sum(dim=0)

        # 1b. Smagorinsky effective relaxation from the LOCAL
        # non-equilibrium momentum flux (Hou et al. 1996; Kruger 14.5.2):
        #   Pi_ab = sum_q e_qa e_qb (f_q - f_q^eq)      (6 components in 3D)
        #   Qbar  = sqrt(2 Pi_ab Pi_ab)
        #   tau_eff = (tau0 + sqrt(tau0^2 + 18 Cs^2 Qbar / rho)) / 2
        # The closed form is dimension-independent (c_s^2 = 1/3 in both);
        # at resolved scales Qbar ~ 0 and tau_eff -> tau0: inert where the
        # grid already sees the flow.
        omega = self.omega
        if self.sgs:
            pxx = torch.zeros_like(usq); pyy = torch.zeros_like(usq)
            pzz = torch.zeros_like(usq); pxy = torch.zeros_like(usq)
            pxz = torch.zeros_like(usq); pyz = torch.zeros_like(usq)
            for q in range(1, Q):
                ex, ey, ez = E[q]
                cu = torch.zeros_like(usq)
                if ex: cu += ex * u[0]
                if ey: cu += ey * u[1]
                if ez: cu += ez * u[2]
                cu *= 3.0
                fneq = f[q] - W[q] * rho * (1.0 + cu + 0.5 * cu * cu
                                            - 1.5 * usq)
                if ex: pxx += (ex * ex) * fneq
                if ey: pyy += (ey * ey) * fneq
                if ez: pzz += (ez * ez) * fneq
                if ex and ey: pxy += (ex * ey) * fneq
                if ex and ez: pxz += (ex * ez) * fneq
                if ey and ez: pyz += (ey * ez) * fneq
            qbar = (2.0 * (pxx * pxx + pyy * pyy + pzz * pzz
                           + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz))).sqrt()
            rho_safe = torch.where(rho > 1e-12, rho, torch.ones_like(rho))
            tau_eff = 0.5 * (self.tau + (
                self.tau * self.tau
                + 18.0 * self.cs_smag ** 2 * qbar / rho_safe).sqrt())
            omega = 1.0 / tau_eff
            self.last_tau_eff_max = float(tau_eff.max())

        # 2. BGK collision (+ Guo source), in place, per direction.
        if self.has_force:
            uF = u[0] * fx + u[1] * fy + u[2] * fz      # u.F, reused per q
        for q in range(Q):
            ex, ey, ez = E[q]
            cu = torch.zeros_like(usq)
            if ex: cu += ex * u[0]
            if ey: cu += ey * u[1]
            if ez: cu += ez * u[2]
            cu *= 3.0
            feq = W[q] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
            f[q] -= omega * (f[q] - feq)   # omega is a field with SGS on
            if self.has_force:
                # Guo source (eq. 6.25): (1 - omega/2) w_q *
                #   [3(e-u).F + 9(e.u)(e.F)]   with (e.u) = cu/3
                eF = ex * fx + ey * fy + ez * fz
                s = 3.0 * (eF - uF) + 3.0 * cu * eF
                f[q] += (1.0 - 0.5 * omega) * W[q] * s

        # 2b. anechoic sponge: blend the last n_sp x-planes toward
        # feq(rho=1, (u_in, 0, 0)) — absorbs wakes and acoustics, pins the
        # outlet pressure (module docstring).
        if self._n_sp:
            n_sp = self._n_sp
            for q in range(Q):
                ex = E[q][0]
                cu = 3.0 * ex * u_in
                target = W[q] * (1.0 + cu + 0.5 * cu * cu
                                 - 1.5 * u_in * u_in)
                f[q, -n_sp:, :, :] += self._sponge_sigma \
                    * (target - f[q, -n_sp:, :, :])

        # 3. streaming: pull — f2_q(x) = f_q(x - e_q). torch.roll is
        # periodic in all dims; the inlet plane is overwritten below, the
        # sponge absorbs at the outlet, walls (if any) live in the mask.
        for q in range(Q):
            f2[q] = torch.roll(f[q], shifts=E[q], dims=(0, 1, 2))

        # 4. halfway bounce-back (eq. 5.26): where the pull came from
        # inside a solid, take the reflected post-collision population;
        # links from the moving lid add wall momentum (eq. 5.27).
        for q in range(1, Q):
            idx = self._bounce_idx[q]
            if idx is not None:
                f2[q].view(-1)[idx] = f[OPP[q]].view(-1)[idx]
            lidx = self._lid_idx[q]
            if lidx is not None:
                f2[q].view(-1)[lidx] += self._lid_corr[q]
        if self._solid_idx.numel():
            f2.view(Q, -1)[:, self._solid_idx] = 0.0  # solids hold no fluid

        # 4b. momentum-exchange force on the obstacle (eq. 5.51): each
        # boundary link transfers e_qbar * (f_incoming + f_bounced) to the
        # body, qbar pointing into the solid. 3-vector: drag, lift, and
        # spanwise force.
        if self.measure_force:
            force = torch.zeros(3, dtype=torch.float32, device=self.device)
            for q in range(1, Q):
                idx = self._force_idx[q]
                if idx is None:
                    continue
                qb = OPP[q]
                transfer = (f[qb].view(-1)[idx] + f2[q].view(-1)[idx]).sum()
                force[0] += E[qb][0] * transfer
                force[1] += E[qb][1] * transfer
                force[2] += E[qb][2] * transfer
            self.last_force = force

        # 5. open boundary: equilibrium velocity inlet at x = 0. No copy
        # outlet — the sponge IS the outlet (module docstring).
        if self.inlet_outlet:
            for q in range(Q):
                ex = E[q][0]
                cu = 3.0 * ex * u_in
                f2[q, 0, :, :] = W[q] * (1.0 + cu + 0.5 * cu * cu
                                         - 1.5 * u_in * u_in)

        # swap A/B buffers
        self.f, self.f2 = f2, f
        self.step_count += 1

    # ------------------------------------------------------------------
    def guards(self) -> dict:
        """Cheap health check: NaN, runaway velocity, mass drift.

        u_max is measured over the INTERIOR (clear of the inlet/outlet
        planes) — the original 3D bug hid because the imposed inlet plane
        WAS the global max. A BC must never grade itself."""
        rho, u = self.macroscopics()
        speed2 = (u * u).sum(dim=0)
        if self.inlet_outlet and self.nx > 8:
            speed2 = speed2[2:-2, :, :]
        u_max = float(speed2.max().sqrt())
        mass = float(rho.sum())
        has_nan = bool(torch.isnan(rho).any() or torch.isinf(rho).any()
                       or math.isnan(u_max))
        return {
            "step": self.step_count,
            "has_nan": has_nan,
            "u_max": u_max,
            "mass": mass,
            "mass_drift": mass / self.initial_mass - 1.0,
        }

    def check_guards(self) -> dict:
        """Run guards; raise SimulationBlowup on NaN or runaway velocity."""
        g = self.guards()
        if g["has_nan"]:
            raise SimulationBlowup(f"NaN at step {g['step']}")
        if g["u_max"] > 0.45:  # Ma > 0.78: unphysical, NaN is seconds away
            raise SimulationBlowup(
                f"runaway velocity u_max = {g['u_max']:.3f} at step {g['step']}"
            )
        return g

    # ------------------------------------------------------------------
    def checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "f": self.f.cpu(),
            "step": self.step_count,
            "seed": self.seed,
            "scene": self.scene_name,
            "tau": self.tau,
            "u_char": self.u_char,
        }, path)

    def restore(self, path: str | Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=True)
        if ck["f"].shape != self.f.shape:
            raise ValueError(
                f"checkpoint grid {tuple(ck['f'].shape)} != solver "
                f"{tuple(self.f.shape)}"
            )
        self.f = ck["f"].to(self.device)
        self.f2 = torch.empty_like(self.f)
        self.step_count = int(ck["step"])
        self.seed = int(ck["seed"])


# ----------------------------------------------------------------------
def build_obstacle_mask(scene: Scene, obstacle: dict) -> torch.Tensor:
    """Rasterize the scene's obstacle to a boolean (nx,ny,nz) mask."""
    kind = obstacle.get("type")
    if kind == "cylinder":
        n = scene.units.cells                    # diameter in cells
        cx = obstacle["center_x_chars"] * n
        cy = obstacle["center_y_chars"] * n
        r = n / 2.0
        x = torch.arange(scene.nx, dtype=torch.float32) + 0.5
        y = torch.arange(scene.ny, dtype=torch.float32) + 0.5
        circle = ((x[:, None] - cx) ** 2 + (y[None, :] - cy) ** 2) <= r * r
        return circle[:, :, None].expand(scene.nx, scene.ny, scene.nz).clone()
    raise NotImplementedError(
        f"3D obstacle type {kind!r} (the spanwise-periodic airfoil section "
        "arrives with the 3D airfoil phase)"
    )


def capture_failure(
    solver: Solver,
    reason: str,
    frames_dir: str | Path | None = None,
    failures_root: str | Path = "failures",
) -> Path:
    """Instability is footage: save checkpoint + seed + last 120 frames."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = Path(failures_root) / stamp
    dest.mkdir(parents=True, exist_ok=True)
    solver.checkpoint(dest / "checkpoint.pt")
    (dest / "meta.yaml").write_text(
        f"scene: {solver.scene_name}\nseed: {solver.seed}\n"
        f"step: {solver.step_count}\nreason: {reason}\n",
        encoding="utf-8",
    )
    if frames_dir is not None and Path(frames_dir).is_dir():
        frames = sorted(Path(frames_dir).glob("frame_*.png"))[-120:]
        for p in frames:
            shutil.copy2(p, dest / p.name)
    return dest
