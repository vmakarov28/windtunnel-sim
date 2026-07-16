"""Headless 3D rendering: slices + Q-criterion volume projection -> PNG.

How 3D is visualized in this project (all pure tensor ops, no mesh or GL
dependencies — headless-first survives):

* "slice"      omega_z on the mid-span plane. The 2D-comparable read; at
               Re = 100 it should look exactly like the 2D tunnel, which
               is itself a statement worth filming.
* "three_pane" the slice PLUS an x-z pane of the SPANWISE velocity u_z at
               mid-height. u_z is identically zero for two-dimensional
               flow, so this pane is an honest 3D-ness meter — and to
               STAY honest it uses an ABSOLUTE color scale (full color at
               |u_z| = 0.15 u_char, the amplitude scale of real mode-A
               structure). A self-normalizing scale amplified 1e-6-level
               fp32 roundoff into loud fake structure on the first try;
               a meter needs a fixed reference, not a rolling one.
* "qcrit"      emission-absorption volume projection (along the span) of
               the Q-criterion, Q = (||Omega||^2 - ||S||^2)/2 > 0 in
               vortex cores (Hunt et al. 1988), colored by streamwise
               vorticity omega_x — counter-rotating streamwise vortex
               pairs render red/blue. This is the genuinely-3D shot.

Frame numbering continues across --resume (a 2D lesson), and percentile
auto-scaling subsamples above 4M elements (CUDA quantile caps at ~16M —
the crash that killed the first Re=50k 2D render).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.image import imsave

from .solver import Solver

_RDBU = colormaps["RdBu_r"]
_GRAY = (0.42, 0.42, 0.42, 1.0)


def _robust_p995(t: torch.Tensor) -> float:
    flat = t.abs().flatten().float()
    if flat.numel() > 4_000_000:
        flat = flat[:: flat.numel() // 4_000_000 + 1]
    return float(torch.quantile(flat, torch.tensor(0.995,
                                                   device=flat.device)))


def _grad(field: torch.Tensor, axis: int) -> torch.Tensor:
    """Central difference via roll (periodic wrap; the wrong planes at the
    open x edges sit inside the inlet plane / sponge zone)."""
    return 0.5 * (torch.roll(field, -1, axis) - torch.roll(field, 1, axis))


def vorticity_z_midplane(solver: Solver) -> torch.Tensor:
    """omega_z = dv/dx - du/dy at k = nz//2, (ny, nx) for display."""
    _, u = solver.macroscopics()
    k = solver.nz // 2
    ux, uy = u[0, :, :, k], u[1, :, :, k]
    dvdx = _grad(uy, 0)
    dudy = _grad(ux, 1)
    return (dvdx - dudy).T


def spanwise_velocity_midheight(solver: Solver) -> torch.Tensor:
    """u_z on the x-z plane at j = ny//2, (nz, nx) for display.

    Zero for 2D flow — this field IS the measurement of three-
    dimensionality."""
    _, u = solver.macroscopics()
    return u[2, :, solver.ny // 2, :].T


def q_criterion(u: torch.Tensor) -> torch.Tensor:
    """Q = (||Omega||^2 - ||S||^2)/2 from the velocity-gradient tensor
    A_ij = du_i/dx_j (central differences). Q > 0 where rotation beats
    strain: the vortex cores."""
    a = [[_grad(u[i], j) for j in range(3)] for i in range(3)]
    norm_s2 = torch.zeros_like(a[0][0])
    norm_o2 = torch.zeros_like(a[0][0])
    for i in range(3):
        for j in range(3):
            s = 0.5 * (a[i][j] + a[j][i])
            o = 0.5 * (a[i][j] - a[j][i])
            norm_s2 += s * s
            norm_o2 += o * o
    return 0.5 * (norm_o2 - norm_s2)


def streamwise_vorticity(u: torch.Tensor) -> torch.Tensor:
    """omega_x = dw/dy - dv/dz — the signature of mode-A/B 3D structure."""
    return _grad(u[2], 1) - _grad(u[1], 2)


def qcrit_projection(solver: Solver, q_scale: float,
                     wx_scale: float) -> np.ndarray:
    """Emission-absorption compositing of Q along the span -> (ny, nx, 4).

    Opacity per cell rises with Q (vortex cores glow); color is the local
    streamwise vorticity through a diverging map; solids are opaque gray,
    so occlusion falls out of the compositing for free."""
    _, u = solver.macroscopics()
    q = q_criterion(u).clamp(min=0.0) / max(q_scale, 1e-12)
    alpha = 1.0 - torch.exp(-3.0 * q)                     # (nx, ny, nz)
    wx = (streamwise_vorticity(u) / (2.0 * max(wx_scale, 1e-12))
          + 0.5).clamp(0.0, 1.0)

    dev = u.device
    color_lut = torch.tensor(_RDBU(np.linspace(0, 1, 256))[:, :3],
                             dtype=torch.float32, device=dev)
    solid_rgb = torch.tensor(_GRAY[:3], dtype=torch.float32, device=dev)

    nx, ny, nz = solver.nx, solver.ny, solver.nz
    acc = torch.zeros((nx, ny, 3), dtype=torch.float32, device=dev)
    trans = torch.ones((nx, ny, 1), dtype=torch.float32, device=dev)
    for k in range(nz):                                    # front-to-back
        a_k = alpha[:, :, k].unsqueeze(-1)
        idx = (wx[:, :, k] * 255).long().clamp(0, 255)
        c_k = color_lut[idx]                               # (nx, ny, 3)
        solid_k = solver.mask[:, :, k].unsqueeze(-1)
        a_k = torch.where(solid_k, torch.ones_like(a_k), a_k)
        c_k = torch.where(solid_k, solid_rgb.expand_as(c_k), c_k)
        acc += trans * a_k * c_k
        trans *= (1.0 - a_k)
    acc += trans                                           # white background
    rgba = torch.cat([acc, torch.ones((nx, ny, 1), device=dev)], dim=-1)
    return rgba.permute(1, 0, 2).cpu().numpy()             # (ny, nx, 4)


class FrameWriter:
    """Writes frame_%06d.png for one preset; scales are rolling 99.5th
    percentiles that only grow (stable colors, reproducible runs), and
    numbering continues across --resume instead of overwriting."""

    PRESETS = ("slice", "three_pane", "qcrit")

    def __init__(self, out_dir: str | Path, preset: str = "slice",
                 scale: float | None = None):
        if preset not in self.PRESETS:
            raise ValueError(f"preset {preset!r} not in {self.PRESETS}")
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.dir.glob("frame_*.png"))
        self.count = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0
        self.preset = preset
        self.scale = scale or 0.0          # omega_z (slice) scale
        self._fixed = scale is not None
        self.q_scale = 0.0
        self.wx_scale = 0.0
        # ABSOLUTE full-scale for the spanwise pane, as a fraction of
        # u_char (see module docstring: a meter needs a fixed reference).
        self.uz_fullscale_frac = 0.15

    # -- panes -----------------------------------------------------------
    def _slice_rgba(self, solver: Solver) -> np.ndarray:
        omega = vorticity_z_midplane(solver)
        if not self._fixed:
            self.scale = max(self.scale, _robust_p995(omega), 1e-9)
        img = (omega / (2.0 * self.scale) + 0.5).clamp(0.0, 1.0)
        rgba = _RDBU(img.cpu().numpy())
        solid = solver.mask[:, :, solver.nz // 2].T.cpu().numpy()
        rgba[solid] = _GRAY
        return rgba

    def _spanwise_rgba(self, solver: Solver) -> np.ndarray:
        uz = spanwise_velocity_midheight(solver)           # (nz, nx)
        full = self.uz_fullscale_frac * max(solver.u_char, 1e-9)
        img = (uz / (2.0 * full) + 0.5).clamp(0.0, 1.0)
        rgba = _RDBU(img.cpu().numpy())
        solid = solver.mask[:, solver.ny // 2, :].T.cpu().numpy()
        rgba[solid] = _GRAY
        # a thin span is hard to read at 1 px/cell; stretch it 4x
        return np.repeat(rgba, 4, axis=0)

    # -- write -------------------------------------------------------------
    def write(self, solver: Solver) -> Path:
        if self.preset == "slice":
            rgba = self._slice_rgba(solver)
            out = np.flipud(rgba)
        elif self.preset == "three_pane":
            top = np.flipud(self._slice_rgba(solver))
            bottom = np.flipud(self._spanwise_rgba(solver))
            sep = np.full((6, top.shape[1], 4), (1, 1, 1, 1), dtype=top.dtype)
            sep[2:4, :, :3] = 0.75
            out = np.vstack([top, sep, bottom])
        else:  # qcrit
            _, u = solver.macroscopics()
            self.q_scale = max(
                self.q_scale, _robust_p995(q_criterion(u).clamp(min=0)), 1e-12)
            self.wx_scale = max(
                self.wx_scale, _robust_p995(streamwise_vorticity(u)), 1e-12)
            out = np.flipud(qcrit_projection(solver, self.q_scale,
                                             self.wx_scale))
        path = self.dir / f"frame_{self.count:06d}.png"
        imsave(path, out)
        self.count += 1
        return path
