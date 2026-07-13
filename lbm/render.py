"""Headless vorticity rendering: mid-span slice -> PNG frames.

Phase 1 keeps the read identical to a 2D tunnel: the z-component of
vorticity, omega_z = dv/dx - du/dy, on the mid-span plane, in a diverging
colormap (blue = clockwise, red = counter-clockwise), obstacle composited
in flat neutral gray. Genuinely-3D shots (Q-criterion isosurfaces) are a
Phase 3 deliverable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.image import imsave

from .solver import Solver


def vorticity_z_midplane(solver: Solver) -> torch.Tensor:
    """omega_z = dv/dx - du/dy at k = nz//2, central differences (ny, nx)."""
    _, u = solver.macroscopics()
    k = solver.nz // 2
    ux, uy = u[0, :, :, k], u[1, :, :, k]
    # central differences; roll wraps at edges, matching periodic y (the
    # one wrong column at x edges sits inside inlet/outlet planes anyway)
    dvdx = (torch.roll(uy, -1, 0) - torch.roll(uy, 1, 0)) * 0.5
    dudy = (torch.roll(ux, -1, 1) - torch.roll(ux, 1, 1)) * 0.5
    return (dvdx - dudy).T  # (ny, nx) so the image reads left-to-right


class FrameWriter:
    """Writes frame_%06d.png; vorticity scale is a rolling 99.5th
    percentile that only ever grows, so colors are stable once the flow
    develops and a run is reproducible frame-for-frame."""

    def __init__(self, out_dir: str | Path, scale: float | None = None):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.scale = scale or 0.0
        self._fixed = scale is not None
        self.count = 0
        self.cmap = colormaps["RdBu_r"]

    def write(self, solver: Solver) -> Path:
        omega = vorticity_z_midplane(solver)
        if not self._fixed:
            p = float(torch.quantile(
                omega.abs().flatten().float(),
                torch.tensor(0.995, device=omega.device),
            ))
            self.scale = max(self.scale, p, 1e-9)
        img = (omega / (2.0 * self.scale) + 0.5).clamp(0.0, 1.0)
        rgba = self.cmap(img.cpu().numpy())          # (ny, nx, 4) float
        solid = solver.mask[:, :, solver.nz // 2].T.cpu().numpy()
        rgba[solid] = (0.42, 0.42, 0.42, 1.0)        # flat neutral obstacle
        path = self.dir / f"frame_{self.count:06d}.png"
        imsave(path, np.flipud(rgba))                # y up
        self.count += 1
        return path
