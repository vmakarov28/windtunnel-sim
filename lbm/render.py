"""Headless vorticity rendering: full 2D field -> PNG frames.

omega = dv/dx - du/dy in a diverging colormap (blue = clockwise, red =
counter-clockwise), obstacle composited in flat neutral gray. Colormap
presets, tracers, and dye arrive in Phase 3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.image import imsave

from .solver import Solver


def vorticity(solver: Solver) -> torch.Tensor:
    """omega = dv/dx - du/dy, central differences, (ny, nx) for display."""
    _, u = solver.macroscopics()
    ux, uy = u[0], u[1]
    # central differences; roll wraps at edges, matching periodic y (the
    # wrong column at x edges sits inside the inlet/outlet planes anyway)
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
        omega = vorticity(solver)
        if not self._fixed:
            p = float(torch.quantile(
                omega.abs().flatten().float(),
                torch.tensor(0.995, device=omega.device),
            ))
            self.scale = max(self.scale, p, 1e-9)
        img = (omega / (2.0 * self.scale) + 0.5).clamp(0.0, 1.0)
        rgba = self.cmap(img.cpu().numpy())          # (ny, nx, 4) float
        solid = solver.mask.T.cpu().numpy()
        rgba[solid] = (0.42, 0.42, 0.42, 1.0)        # flat neutral obstacle
        path = self.dir / f"frame_{self.count:06d}.png"
        imsave(path, np.flipud(rgba))                # y up
        self.count += 1
        return path
