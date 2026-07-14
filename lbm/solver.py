"""D2Q9 BGK solver — the readable PyTorch reference implementation.

This is the file that gets explained on camera; it stays readable even
after the fused-kernel port (Phase 4), which must match it within fp32
tolerance. (The 3D D3Q19 ancestor lives on the `3d-d3q19` branch; design
of the modular collision/streaming/BC split follows the spirit of
Flatscher's LB-t solver — consulted for ideas, no code copied.)

One time step (Kruger et al. 2017, ch. 3-6):

  1. macroscopics    rho = sum_q f_q,  u = (sum_q e_q f_q + F/2) / rho
  2. BGK collision   f_q += -(1/tau)(f_q - f_q^eq) + S_q       (eq. 3.9)
       f_q^eq = w_q rho (1 + 3(e.u) + 4.5(e.u)^2 - 1.5 u^2)    (eq. 3.54)
       S_q = (1 - 1/(2tau)) w_q [3(e-u) + 9(e.u)e] . F         (eq. 6.25,
                                                    Guo body force)
  3. streaming       f_q(x + e_q, t+1) = f_q(x, t)      -> torch.roll
  4. halfway bounce-back at solids (eq. 5.26); moving walls add
       +6 w_q rho0 (e_q . u_wall)                              (eq. 5.27)
  5. open boundaries (Zou & He 1997):
       inlet  x=0    : velocity imposed; density solved LOCALLY from the
                       populations already at the boundary column
       outlet x=nx-1 : pressure imposed (rho = 1); velocity solved locally

     Only the 3 unknown (inward-pointing) populations are reconstructed;
     everything the interior sent to the boundary is kept. (Two earlier
     schemes failed — NOTES.md 2026-07-13: equilibrium inlet at fixed
     rho=1 + copy outlet pressurized the domain 6% and ran 12% slow;
     extrapolating the inlet density created a positive feedback instead.)

  6. anechoic sponge (last sponge_fraction of the domain): f blends
     toward feq(rho=1, u_inlet) with a smoothly rising strength. A
     velocity inlet + pressure outlet is an acoustically CLOSED resonator:
     ramping the inlet from rest is a piston stroke of amplitude
     u/c_s ~ 14%, and that standing wave never decays on its own (bulk
     damping time >> any run; measured, NOTES.md). A viscosity-ramp
     sponge does nothing for acoustics. Forcing the state itself — the
     numerical analog of the foam wedges in a real wind tunnel's
     termination — absorbs vortices AND sound in one or two passes.

Memory layout: f has shape (9, nx, ny), fp32.

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
        nx: int, ny: int,
        tau: float,
        u_char: float,
        *,
        seed: int = 0,
        device: str = "auto",
        obstacle_mask: torch.Tensor | None = None,   # bool (nx,ny), True=solid
        inlet_outlet: bool = True,      # False -> periodic/closed box
        ramp_steps: int = 2000,         # inlet ramp; 0 = impulsive start
        sponge_fraction: float = 0.08,  # of nx, before the outlet
        sponge_strength: float = 0.15,  # peak blend rate in the sponge
        wall_y: bool = False,           # bounce-back walls at y=0, y=ny-1
        wall_x: bool = False,           # bounce-back walls at x edges (cavity)
        lid_velocity: float = 0.0,      # top wall moves in +x (needs wall_y)
        body_force: tuple[float, float] = (0.0, 0.0),  # Guo forcing (Phase 2)
        init_noise: float = 1e-3,       # *u_char, seeds the shedding asymmetry
        scene_name: str = "adhoc",
    ):
        self.device = pick_device(device)
        self.nx, self.ny = nx, ny
        self.tau, self.u_char = float(tau), float(u_char)
        self.seed = seed
        self.step_count = 0
        self.inlet_outlet = inlet_outlet
        self.ramp_steps = ramp_steps
        self.lid_velocity = float(lid_velocity)
        self.force = (float(body_force[0]), float(body_force[1]))
        self.has_force = any(c != 0.0 for c in self.force)
        self.scene_name = scene_name

        self.omega = 1.0 / self.tau

        # -- anechoic sponge profile (see module docs, point 6) -----------
        # sigma rises smoothly 0 -> sponge_strength over the last
        # sponge_fraction of the domain; zero elsewhere.
        self._n_sp = 0
        if inlet_outlet and sponge_fraction > 0:
            self._n_sp = max(2, round(nx * sponge_fraction))
            s = torch.linspace(0.0, 1.0, self._n_sp)
            self._sponge_sigma = (
                sponge_strength * s * s              # gentle entry, C1
            ).to(self.device).view(self._n_sp, 1)

        # -- solid mask ----------------------------------------------------
        mask = torch.zeros((nx, ny), dtype=torch.bool)
        lid = torch.zeros((nx, ny), dtype=torch.bool)
        if obstacle_mask is not None:
            mask |= obstacle_mask.cpu()
        if wall_y:
            mask[:, 0] = True
            if self.lid_velocity != 0.0:
                lid[:, -1] = True   # moving lid = the top wall
            mask[:, -1] = True
        if wall_x:
            mask[0, :] = True
            mask[-1, :] = True
            lid[0, :] = lid[-1, :] = False  # side walls never move
        self.mask = mask.to(self.device)
        self.fluid_cells = int((~mask).sum())

        # Precompute bounce-back link indices (flat, per direction):
        # fluid cells x whose pull-neighbor x - e_q is solid. After
        # streaming, f_q(x) would have come from inside the solid; halfway
        # bounce-back replaces it with the reflected post-collision
        # population: f_q(x,t+1) = f*_{opp(q)}(x,t)  [+ lid correction].
        self._bounce_idx: list[torch.Tensor | None] = [None] * Q
        self._lid_idx: list[torch.Tensor | None] = [None] * Q
        self._lid_corr: list[float] = [0.0] * Q
        lid_dev = lid.to(self.device)
        for q in range(1, Q):
            upstream_solid = torch.roll(self.mask, shifts=E[q], dims=(0, 1)) \
                & ~self.mask
            idx = upstream_solid.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            self._bounce_idx[q] = idx if idx.numel() else None
            if self.lid_velocity != 0.0:
                up_lid = torch.roll(lid_dev, shifts=E[q], dims=(0, 1)) \
                    & ~self.mask
                lidx = up_lid.reshape(-1).nonzero(as_tuple=False).squeeze(1)
                self._lid_idx[q] = lidx if lidx.numel() else None
                # Moving-wall momentum (eq. 5.27), rho_wall ~ 1:
                # +6 w_q (e_q . u_wall), u_wall = (lid_velocity, 0)
                self._lid_corr[q] = 6.0 * W[q] * E[q][0] * self.lid_velocity
        self._solid_idx = self.mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)

        # -- initial condition: equilibrium + seeded noise -----------------
        # Open scenes start at the (ramped) inlet velocity; closed boxes at
        # rest. Tiny seeded noise breaks the wake's metastable symmetry so
        # vortex shedding onsets reproducibly (seed -> same flow).
        gen = torch.Generator().manual_seed(seed)  # CPU gen: device-portable
        u = torch.zeros((2, nx, ny), dtype=torch.float32)
        if inlet_outlet:
            u[0] = self.u_char * self._ramp_factor(0)
        u += (init_noise * self.u_char
              * (2.0 * torch.rand((2, nx, ny), generator=gen) - 1.0))
        u[:, mask] = 0.0
        rho = torch.ones((nx, ny), dtype=torch.float32)

        self.f = torch.empty((Q, nx, ny), dtype=torch.float32,
                             device=self.device)
        self._write_equilibrium(self.f, rho.to(self.device), u.to(self.device))
        self.f[:, self.mask] = 0.0
        self.f2 = torch.empty_like(self.f)   # B buffer (A-B, like Phase 4)
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
        if scene.units.sgs:
            raise NotImplementedError(
                "this scene needs the Smagorinsky model (Phase 5)"
            )
        is_cavity = "lid" in b
        return cls(
            scene.nx, scene.ny,
            tau=scene.units.tau, u_char=scene.units.u_lat,
            seed=seed, device=device, obstacle_mask=mask,
            inlet_outlet="inlet" in b,
            ramp_steps=int(b.get("ramp_steps", 2000)) if ramp else 0,
            sponge_fraction=float(b.get("sponge_fraction", 0.08)),
            wall_y=(top_bottom == "bounce_back") or is_cavity,
            wall_x=is_cavity,
            lid_velocity=scene.units.u_lat if is_cavity else 0.0,
            scene_name=scene.name,
        )

    # ------------------------------------------------------------------
    def _ramp_factor(self, step: int) -> float:
        """Smooth inlet ramp 0 -> 1 over ramp_steps (0 = impulsive start)."""
        if self.ramp_steps <= 0 or step >= self.ramp_steps:
            return 1.0
        return 0.5 - 0.5 * math.cos(math.pi * step / self.ramp_steps)

    def macroscopics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """rho (nx,ny) and u (2,nx,ny). With Guo forcing, u includes the
        half-step force shift (Kruger eq. 6.27): u = (sum e f + F/2)/rho."""
        rho = self.f.sum(dim=0)
        u = torch.zeros((2, self.nx, self.ny),
                        dtype=torch.float32, device=self.device)
        for q in range(1, Q):
            ex, ey = E[q]
            if ex: u[0] += ex * self.f[q]
            if ey: u[1] += ey * self.f[q]
        if self.has_force:
            u[0] += 0.5 * self.force[0]
            u[1] += 0.5 * self.force[1]
        rho_safe = torch.where(rho > 1e-12, rho, torch.ones_like(rho))
        u /= rho_safe
        u[:, self.mask] = 0.0
        return rho, u

    def _write_equilibrium(self, out: torch.Tensor, rho: torch.Tensor,
                           u: torch.Tensor) -> None:
        """out[q] = f_q^eq(rho, u) for all q (eq. 3.54)."""
        usq = (u * u).sum(dim=0)
        for q in range(Q):
            ex, ey = E[q]
            cu = torch.zeros_like(usq)
            if ex: cu += ex * u[0]
            if ey: cu += ey * u[1]
            cu *= 3.0  # = (e.u)/c_s^2
            out[q] = W[q] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)

    def step(self) -> None:
        f, f2 = self.f, self.f2
        fx, fy = self.force
        u_in = self.u_char * self._ramp_factor(self.step_count)

        # 1+2. collision: relax toward equilibrium, in place, per direction.
        rho, u = self.macroscopics()
        usq = (u * u).sum(dim=0)
        if self.has_force:
            uF = u[0] * fx + u[1] * fy                 # u.F, reused per q
        for q in range(Q):
            ex, ey = E[q]
            cu = torch.zeros_like(usq)
            if ex: cu += ex * u[0]
            if ey: cu += ey * u[1]
            cu *= 3.0
            feq = W[q] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
            f[q] -= self.omega * (f[q] - feq)          # BGK, omega = 1/tau(x)
            if self.has_force:
                # Guo source (eq. 6.25): (1 - omega/2) w_q *
                #   [3(e-u).F + 9(e.u)(e.F)]   with (e.u) = cu/3
                eF = ex * fx + ey * fy
                s = 3.0 * (eF - uF) + 3.0 * cu * eF
                f[q] += (1.0 - 0.5 * self.omega) * W[q] * s

        # 2b. anechoic sponge: blend toward feq(rho=1, u_in) in the last
        # sponge_fraction of the domain (module docs, point 6).
        if self._n_sp:
            n_sp = self._n_sp
            for q in range(Q):
                ex = E[q][0]
                cu = 3.0 * ex * u_in
                target = W[q] * (1.0 + cu + 0.5 * cu * cu
                                 - 1.5 * u_in * u_in)
                f[q, -n_sp:, :] += self._sponge_sigma \
                    * (target - f[q, -n_sp:, :])

        # 3. streaming: pull — f2_q(x) = f_q(x - e_q). torch.roll wraps
        # periodically; x-plane wraparound is overwritten by inlet/outlet
        # below, y-walls (if any) live in the mask.
        for q in range(Q):
            f2[q] = torch.roll(f[q], shifts=E[q], dims=(0, 1))

        # 4. halfway bounce-back (eq. 5.26): where the pull came from
        # inside a solid, take the reflected post-collision population.
        for q in range(1, Q):
            idx = self._bounce_idx[q]
            if idx is not None:
                f2[q].view(-1)[idx] = f[OPP[q]].view(-1)[idx]
            lidx = self._lid_idx[q]
            if lidx is not None:  # moving wall adds momentum (eq. 5.27)
                f2[q].view(-1)[lidx] += self._lid_corr[q]
        if self._solid_idx.numel():
            f2.view(Q, -1)[:, self._solid_idx] = 0.0  # solids hold no fluid

        # 5. open boundaries, Zou-He (see module docs). Direction indices:
        #    1:(1,0) 2:(-1,0) 3:(0,1) 4:(0,-1) 5:(1,1) 6:(-1,-1)
        #    7:(1,-1) 8:(-1,1). Unknowns at the inlet are the +x triple
        #    (1,5,7); at the outlet the -x triple (2,6,8).
        if self.inlet_outlet:
            c = f2[:, 0, :]        # inlet column, post-streaming
            rho_in = (c[0] + c[3] + c[4]
                      + 2.0 * (c[2] + c[6] + c[8])) / (1.0 - u_in)
            t = 0.5 * (c[3] - c[4])            # transverse imbalance
            f2[1, 0, :] = c[2] + (2.0 / 3.0) * rho_in * u_in
            f2[5, 0, :] = c[6] - t + (1.0 / 6.0) * rho_in * u_in
            f2[7, 0, :] = c[8] + t + (1.0 / 6.0) * rho_in * u_in

            c = f2[:, -1, :]       # outlet column: rho = 1 imposed
            u_out = (c[0] + c[3] + c[4]
                     + 2.0 * (c[1] + c[5] + c[7])) - 1.0
            t = 0.5 * (c[3] - c[4])
            f2[2, -1, :] = c[1] - (2.0 / 3.0) * u_out
            f2[6, -1, :] = c[5] + t - (1.0 / 6.0) * u_out
            f2[8, -1, :] = c[7] - t - (1.0 / 6.0) * u_out

        # swap A/B buffers
        self.f, self.f2 = f2, f
        self.step_count += 1

    # ------------------------------------------------------------------
    def guards(self) -> dict:
        """Cheap health check: NaN, runaway velocity, mass drift.

        u_max is measured over the INTERIOR (2 columns clear of the open
        boundaries) — the 3D bug hid because the inlet plane itself was
        the global max. Never let a boundary condition grade itself."""
        rho, u = self.macroscopics()
        speed2 = (u * u).sum(dim=0)
        if self.inlet_outlet and self.nx > 8:
            speed2 = speed2[2:-2, :]
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
    """Rasterize the scene's obstacle to a boolean (nx,ny) mask."""
    kind = obstacle.get("type")
    if kind == "cylinder":
        n = scene.units.cells                    # diameter in cells
        cx = obstacle["center_x_chars"] * n
        cy = obstacle["center_y_chars"] * n
        r = n / 2.0
        x = torch.arange(scene.nx, dtype=torch.float32)[:, None] + 0.5
        y = torch.arange(scene.ny, dtype=torch.float32)[None, :] + 0.5
        return ((x - cx) ** 2 + (y - cy) ** 2) <= r * r
    raise NotImplementedError(
        f"obstacle type {kind!r} (airfoil_dat arrives in Phase 6)"
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
