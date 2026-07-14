#!/usr/bin/env python3
"""Phase 6: the honest staircase figure.

Zoomed views of the rasterized MH45 mask at the leading and trailing
edges with the true outline overlaid — the jagged edge IS the honest
explanation for part of the drag error, so it gets its own figure (and
video beat) instead of being hidden.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lbm.airfoil import load_selig, rasterize

ROOT = Path(__file__).resolve().parent.parent
ALPHA = 5.0


def main() -> int:
    name, poly = load_selig(ROOT / "assets" / "mh45.dat")
    chord, cx, cy = 400.0, 800.0, 1000.0
    nx, ny = 3200, 2000
    mask = rasterize(poly, nx, ny, chord, cx, cy, alpha_deg=ALPHA)

    a = math.radians(-ALPHA)
    ca, sa = math.cos(a), math.sin(a)
    p = poly - torch.tensor([0.25, 0.0])
    verts = torch.stack([p[:, 0] * ca - p[:, 1] * sa,
                         p[:, 0] * sa + p[:, 1] * ca], 1) * chord \
        + torch.tensor([cx, cy])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    views = [("leading edge", verts[:, 0].min() - 4, 40),
             ("trailing edge", verts[:, 0].max() - 36, 40)]
    for ax, (title, x0, w) in zip(axes, views):
        x0 = float(x0)
        near = verts[(verts[:, 0] > x0 - 5) & (verts[:, 0] < x0 + w + 5)]
        yc = float(near[:, 1].mean())
        y0, h = yc - w / 2, w
        ax.imshow(mask[int(x0):int(x0 + w), int(y0):int(y0 + h)].T.numpy(),
                  origin="lower", cmap="Greys", vmin=0, vmax=1.4,
                  extent=(x0, x0 + w, y0, y0 + h), interpolation="nearest")
        ax.plot(verts[:, 0], verts[:, 1], "r-", lw=1.2,
                label="true surface")
        ax.set_xlim(x0, x0 + w); ax.set_ylim(y0, y0 + h)
        ax.set_title(f"{title} — the staircase, {int(w)} cells wide")
        ax.set_xlabel("x [cells]")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        f"{name}: chord = 400 cells, alpha = {ALPHA} deg. The mask is a "
        "staircase — one honest reason Cd will run high.", fontsize=10)
    fig.tight_layout()
    out = ROOT / "validation" / "mh45_staircase.png"
    fig.savefig(out, dpi=150)
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
