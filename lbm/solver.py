"""D3Q19 BGK solver — the readable PyTorch reference implementation.

This is the file that gets explained on camera; it stays readable even
after the fused-kernel port (Phase 4), which must match it bit-for-bit
within fp32 tolerance.

One time step (Kruger et al. 2017, ch. 3-5):

  1. macroscopics    rho = sum_q f_q,   u = sum_q e_q f_q / rho   (eq. 3.57)
  2. BGK collision   f_q += -(1/tau) (f_q - f_q^eq)               (eq. 3.9)
       f_q^eq = w_q rho (1 + 3(e.u) + 4.5(e.u)^2 - 1.5 u^2)       (eq. 3.54)
  3. streaming       f_q(x + e_q, t+1) = f_q(x, t)      -> torch.roll
  4. halfway bounce-back at solids                                (eq. 5.26)
  5. equilibrium velocity inlet (ramped) + an ANECHOIC forcing sponge over
     the last 8% of the domain that blends toward the clean freestream
     (absorbs wakes AND acoustics, pins the outlet pressure). This
     replaced the original viscosity-sponge + copy-outlet, which
     pressurized the box and left the wake on the non-shedding symmetric
     branch — see the 2D program's open-boundary autopsy in NOTES.

Memory layout: f has shape (19, nx, ny, nz), fp32. All ops are per-
direction loops over q so peak temporaries stay ~2 scalar fields, not 19.

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
        inlet_outlet: bool = True,      # False -> fully periodic box
        ramp_steps: int = 2000,         # inlet ramp; 0 = impulsive start
        sponge_fraction: float = 0.08,  # of nx, before the outlet
        sponge_strength: float = 0.15,  # peak anechoic blend rate
        wall_y: bool = False,           # bounce-back walls at y=0, y=ny-1
        body_force: tuple[float, float, float] = (0.0, 0.0, 0.0),  # Guo
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
        self.force = tuple(float(c) for c in body_force)
        self.has_force = any(c != 0.0 for c in self.force)
        self.scene_name = scene_name

        self.omega = 1.0 / self.tau

        # -- anechoic sponge (ported from the 2D program, NOTES 2026-07-13/14)
        # A velocity inlet + open outlet is an acoustically closed
        # resonator; the OLD 3D code used a viscosity-ramp sponge, which
        # does nothing for sound and (with a fixed-rho inlet + copy outlet)
        # let the domain pressurize to rho~1.06 and slowed the flow ~12% —
        # the run settled on the unstable symmetric branch and never shed.
        # Fix: force the last sponge_fraction of the tunnel toward the clean
        # freestream feq(1, u_in) with a smoothly rising strength (the
        # numerical foam wedge). Absorbs vortices AND sound, and pins the
        # outlet pressure so nothing pressurizes.
        self._n_sp = 0
        if inlet_outlet and sponge_fraction > 0:
            self._n_sp = max(2, round(nx * sponge_fraction))
            s = torch.linspace(0.0, 1.0, self._n_sp)
            self._sponge_sigma = (
                sponge_strength * s * s              # gentle entry, C1
            ).to(self.device).view(self._n_sp, 1, 1)

        # -- solid mask ---------------------------------------------------
        mask = torch.zeros((nx, ny, nz), dtype=torch.bool)
        if obstacle_mask is not None:
            mask |= obstacle_mask.cpu()
        if wall_y:
            mask[:, 0, :] = True
            mask[:, -1, :] = True
        self.mask = mask.to(self.device)
        self.fluid_cells = int((~mask).sum())

        # Precompute bounce-back link indices (flat, per direction):
        # cells x that are fluid whose pull-neighbor x - e_q is solid.
        # After streaming, f_q(x) would have come from inside the solid;
        # halfway bounce-back replaces it with the population that left x
        # toward the wall and reflected: f_q(x,t+1) = f*_{opp(q)}(x,t).
        self._bounce_idx: list[torch.Tensor | None] = [None] * Q
        for q in range(1, Q):
            came_from_solid = torch.roll(
                self.mask, shifts=E[q], dims=(0, 1, 2)
            ) & ~self.mask
            idx = came_from_solid.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            self._bounce_idx[q] = idx if idx.numel() else None
        self._solid_idx = self.mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)

        # -- initial condition: equilibrium + seeded noise ----------------
        # Tiny seeded velocity noise breaks the wake's metastable symmetry
        # so vortex shedding onsets reproducibly (seed -> same flow).
        gen = torch.Generator().manual_seed(seed)  # CPU gen: same on any device
        u0 = self.u_char * self._ramp_factor(0)
        u = torch.zeros((3, nx, ny, nz), dtype=torch.float32)
        u[0] = u0
        u += (init_noise * self.u_char
              * (2.0 * torch.rand((3, nx, ny, nz), generator=gen) - 1.0))
        u[:, mask] = 0.0
        rho = torch.ones((nx, ny, nz), dtype=torch.float32)

        self.f = torch.empty((Q, nx, ny, nz), dtype=torch.float32,
                             device=self.device)
        self._write_equilibrium(self.f, rho.to(self.device), u.to(self.device))
        self.f[:, self.mask] = 0.0
        self.f2 = torch.empty_like(self.f)   # B buffer (A-B, like Phase 4 will)
        self.initial_mass = float(self.f.sum())

    # ------------------------------------------------------------------
    @classmethod
    def from_scene(cls, scene: Scene, seed: int, device: str = "auto",
                   ramp: bool = True) -> "Solver":
        b = scene.raw.get("boundaries", {})
        obstacle = scene.raw.get("obstacle")
        mask = None
        if obstacle is not None:
            mask = build_obstacle_mask(scene, obstacle)
        top_bottom = b.get("top_bottom", "periodic")
        if top_bottom not in ("periodic", "bounce_back"):
            raise NotImplementedError(f"top_bottom = {top_bottom!r}")
        if scene.units.sgs:
            raise NotImplementedError(
                "this scene needs the Smagorinsky model (Phase 5)"
            )
        return cls(
            scene.nx, scene.ny, scene.nz,
            tau=scene.units.tau, u_char=scene.units.u_lat,
            seed=seed, device=device, obstacle_mask=mask,
            inlet_outlet="inlet" in b,
            ramp_steps=int(b.get("ramp_steps", 2000)) if ramp else 0,
            sponge_fraction=float(b.get("sponge_fraction", 0.08)),
            wall_y=(top_bottom == "bounce_back"),
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
        the half-step force shift u = (sum e f + F/2)/rho (Kruger eq. 6.27)."""
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

        # 1+2. collision: relax toward equilibrium, in place, per direction.
        rho, u = self.macroscopics()
        usq = (u * u).sum(dim=0)
        fx, fy, fz = self.force
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
            f[q] -= self.omega * (f[q] - feq)   # BGK, omega = 1/tau
            if self.has_force:
                # Guo source (Kruger eq. 6.25): (1 - omega/2) w_q *
                #   [3(e-u).F + 9(e.u)(e.F)]   with (e.u) = cu/3
                eF = ex * fx + ey * fy + ez * fz
                s = 3.0 * (eF - uF) + 3.0 * cu * eF
                f[q] += (1.0 - 0.5 * self.omega) * W[q] * s

        # 2b. anechoic sponge: blend the last n_sp x-planes toward
        # feq(rho=1, (u_in, 0, 0)) so wakes and acoustic waves are absorbed
        # before the outlet (see constructor + NOTES).
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
        # sponge absorbs at the outlet, y-walls (if any) live in the mask.
        for q in range(Q):
            f2[q] = torch.roll(f[q], shifts=E[q], dims=(0, 1, 2))

        # 4. halfway bounce-back (eq. 5.26): where the pull came from
        # inside a solid, take the reflected post-collision population.
        for q in range(1, Q):
            idx = self._bounce_idx[q]
            if idx is not None:
                f2[q].view(-1)[idx] = f[OPP[q]].view(-1)[idx]
        if self._solid_idx.numel():
            f2.view(Q, -1)[:, self._solid_idx] = 0.0  # solids hold no fluid

        # 5. open boundaries: equilibrium velocity inlet at x=0. No special
        # outlet — the anechoic sponge above absorbs structures and pins the
        # outlet pressure, so a fixed-rho inlet no longer pressurizes the
        # box (the failure autopsied in NOTES). An equilibrium inlet carries
        # no non-equilibrium mode, so it is robust at low viscosity without
        # the Zou-He regularization the 2D program needed (a future refine-
        # ment for quantitative 3D gates).
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
        WAS the global max, so u_max read exactly u_in while the interior
        flow was actually slower. A boundary condition must never grade
        itself (NOTES 2026-07-14)."""
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
