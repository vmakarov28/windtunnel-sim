"""Phase 3 — the cinematography layer.

Everything here is about footage, not physics: passive tracers rendered
as fading streaklines, an advected dye field for smoke-plume shots,
colormap presets, and a camera (zoom region + upscale). None of it feeds
back into the solver.

Tracers: velocity sampled by bilinear interpolation, midpoint RK2
advection, continuous re-seeding at the inlet. Streaklines come from an
accumulation buffer that decays each frame — particles paint bright
heads, the decay leaves their history as a fading tail (additive
blending, so crossing paths glow).

Dye: semi-Lagrangian advection (unconditionally stable): the dye at x is
whatever dye was at x - u dt last step, sampled bilinearly. A source
region re-injects concentration 1 every step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.image import imsave

from .solver import Solver


def _bilinear(field_x: torch.Tensor, field_y: torch.Tensor,
              pos: torch.Tensor) -> torch.Tensor:
    """Sample two (nx,ny) fields at pos (2,N); returns (2,N)."""
    nx, ny = field_x.shape
    x = pos[0].clamp(0.0, nx - 1.001)
    y = pos[1].clamp(0.0, ny - 1.001)
    x0, y0 = x.floor().long(), y.floor().long()
    fx, fy = x - x0, y - y0
    x1, y1 = x0 + 1, y0 + 1
    out = torch.empty_like(pos)
    for i, f in enumerate((field_x, field_y)):
        f00 = f[x0, y0]; f10 = f[x1, y0]; f01 = f[x0, y1]; f11 = f[x1, y1]
        out[i] = (f00 * (1 - fx) * (1 - fy) + f10 * fx * (1 - fy)
                  + f01 * (1 - fx) * fy + f11 * fx * fy)
    return out


class Tracers:
    def __init__(self, solver: Solver, n: int = 300_000, seed: int = 0,
                 max_age: int | None = None):
        # Lifetime must cover the domain transit (nx / u_char steps), or
        # inlet-respawned particles die before ever reaching the camera —
        # the first streakline render was pure black because of this.
        if max_age is None:
            max_age = int(2.0 * solver.nx / max(solver.u_char, 1e-6))
        self.n, self.max_age = n, max_age
        dev = solver.device
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed)
        r = torch.rand((2, n), generator=self.gen, device=dev)
        self.pos = torch.empty((2, n), device=dev)
        self.pos[0] = r[0] * (solver.nx - 2) + 1
        self.pos[1] = r[1] * (solver.ny - 2) + 1
        self.age = (torch.rand(n, generator=self.gen, device=dev)
                    * max_age).long()

    def step(self, solver: Solver) -> None:
        _, u = solver.macroscopics()
        v1 = _bilinear(u[0], u[1], self.pos)          # RK2 midpoint
        v2 = _bilinear(u[0], u[1], self.pos + 0.5 * v1)
        self.pos += v2
        self.age += 1

        # respawn: left the domain, entered a solid, or expired
        nx, ny = solver.nx, solver.ny
        xi = self.pos[0].clamp(0, nx - 1).long()
        yi = self.pos[1].clamp(0, ny - 1).long()
        dead = (
            (self.pos[0] < 1) | (self.pos[0] > nx - 2)
            | (self.pos[1] < 1) | (self.pos[1] > ny - 2)
            | solver.mask[xi, yi] | (self.age >= self.max_age)
        )
        k = int(dead.sum())
        if k:
            r = torch.rand((2, k), generator=self.gen, device=self.pos.device)
            self.pos[0][dead] = r[0] * 3.0 + 1.0      # inlet band
            self.pos[1][dead] = r[1] * (ny - 2) + 1.0
            self.age[dead] = 0


class StreaklineBuffer:
    """Persistent additive accumulation buffer -> glowing streaklines.

    Trail length ~ u_char / (1 - decay) cells: 0.996 at u ~ 0.06 leaves
    ~15-cell tails (0.94 left 1-cell dots — that was a lesson)."""

    def __init__(self, solver: Solver, decay: float = 0.996):
        self.decay = decay
        self.buf = torch.zeros((solver.nx, solver.ny), device=solver.device)

    def splat(self, tracers: Tracers, brightness: float = 0.01) -> None:
        self.buf *= self.decay
        xi = tracers.pos[0].round().long().clamp(0, self.buf.shape[0] - 1)
        yi = tracers.pos[1].round().long().clamp(0, self.buf.shape[1] - 1)
        fade = 1.0 - tracers.age.float() / tracers.max_age
        flat = xi * self.buf.shape[1] + yi
        self.buf.view(-1).index_put_(
            (flat,), brightness * fade, accumulate=True)


class Dye:
    def __init__(self, solver: Solver,
                 source: tuple[float, float, float, float] | None = None,
                 decay: float = 0.99995):
        # decay 0.999 e-folds in 1000 steps ~ 60 cells of travel: the
        # first plume render died 1/4 of the way to the cylinder. Keep
        # decay negligible over a domain transit.
        self.decay = decay
        nx, ny = solver.nx, solver.ny
        self.field = torch.zeros((nx, ny), device=solver.device)
        # source rect in cells (x0, y0, x1, y1); default: thin inlet band
        # centered on the obstacle's vertical position
        self.source = source or (2, ny * 0.45, 6, ny * 0.55)
        gx = torch.arange(nx, dtype=torch.float32, device=solver.device)
        gy = torch.arange(ny, dtype=torch.float32, device=solver.device)
        self.gx = gx[:, None].expand(nx, ny)
        self.gy = gy[None, :].expand(nx, ny)

    def step(self, solver: Solver) -> None:
        _, u = solver.macroscopics()
        pos = torch.stack([
            (self.gx - u[0]).reshape(-1), (self.gy - u[1]).reshape(-1)])
        adv = _bilinear(self.field, self.field, pos)[0]
        self.field = adv.reshape(self.field.shape) * self.decay
        x0, y0, x1, y1 = self.source
        self.field[int(x0):int(x1), int(y0):int(y1)] = 1.0
        self.field[solver.mask] = 0.0


# ----------------------------------------------------------------------
# presets: name -> callable(solver, extras) -> rgba (ny, nx, 4) ndarray
def _vorticity_rgba(solver, extras, state):
    from .render import vorticity
    omega = vorticity(solver)
    flat = omega.abs().flatten()
    if flat.numel() > 4_000_000:      # CUDA quantile caps at ~16M elements
        flat = flat[:: flat.numel() // 4_000_000 + 1]
    p = float(torch.quantile(flat, torch.tensor(0.995, device=omega.device)))
    state["scale"] = max(state.get("scale", 0.0), p, 1e-9)
    img = (omega / (2 * state["scale"]) + 0.5).clamp(0, 1)
    return colormaps["RdBu_r"](img.cpu().numpy())


def _speed_rgba(solver, extras, state):
    _, u = solver.macroscopics()
    speed = (u * u).sum(0).sqrt().T / (1.8 * solver.u_char)
    return colormaps["magma"](speed.clamp(0, 1).cpu().numpy())


def _dye_rgba(solver, extras, state):
    dye = extras["dye"].field.T.clamp(0, 1)
    return colormaps["bone"](dye.cpu().numpy())


def _streaks_rgba(solver, extras, state):
    buf = extras["streaks"].buf.T.clamp(0, 1)
    return colormaps["afmhot"](buf.cpu().numpy())


PRESETS = {
    "vorticity": _vorticity_rgba,
    "speed": _speed_rgba,
    "dye": _dye_rgba,
    "streaklines": _streaks_rgba,
}


class CinemaWriter:
    """Preset + camera + obstacle compositing -> numbered PNG frames."""

    def __init__(self, out_dir: str | Path, preset: str = "vorticity",
                 zoom: tuple[float, float, float, float] | None = None,
                 upscale: int = 1):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.dir.glob("frame_*.png"))
        self.count = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0
        self.preset = preset
        self.fn = PRESETS[preset]
        self.zoom = zoom          # (x0, y0, x1, y1) in CELLS, or None
        self.upscale = upscale
        self.state: dict = {}

    def write(self, solver: Solver, extras: dict | None = None,
              overlay: str | None = None) -> Path:
        rgba = self.fn(solver, extras or {}, self.state)   # (ny, nx, 4)
        if self.preset in ("vorticity", "speed"):
            solid = solver.mask.T.cpu().numpy()
            rgba[solid] = (0.42, 0.42, 0.42, 1.0)
        else:  # dark presets get a lighter obstacle
            solid = solver.mask.T.cpu().numpy()
            rgba[solid] = (0.65, 0.65, 0.68, 1.0)
        if self.zoom is not None:
            x0, y0, x1, y1 = (int(v) for v in self.zoom)
            rgba = rgba[y0:y1, x0:x1]
        if self.upscale > 1:
            rgba = np.repeat(np.repeat(rgba, self.upscale, 0),
                             self.upscale, 1)
        path = self.dir / f"frame_{self.count:06d}.png"
        if overlay:
            from PIL import Image, ImageDraw
            img = Image.fromarray(
                (np.flipud(rgba) * 255).astype(np.uint8))
            draw = ImageDraw.Draw(img)
            light = self.preset in ("vorticity", "speed")
            draw.text((10, 8), overlay,
                      fill=(20, 20, 20) if light else (235, 235, 235))
            img.save(path)
        else:
            imsave(path, np.flipud(rgba))
        self.count += 1
        return path
