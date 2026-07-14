#!/usr/bin/env python3
"""Phase 5: kinetic-energy spectrum of a wake window from a checkpoint.

E(k) by radial binning of the 2D FFT of the fluctuating velocity in a
square window behind the cylinder. Plotted with k^-3 and k^-5/3 guide
lines. Honesty note (printed on the figure): this is a 2D flow — the
k^-3 enstrophy-cascade slope is the physically expected one; we are NOT
claiming 3D Kolmogorov turbulence.

usage: spectrum.py <scene> <checkpoint.pt> [device]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lbm.config import load_scene
from lbm.solver import Solver


def main(scene_name: str, ckpt: str, device: str = "auto") -> int:
    scene = load_scene(scene_name)
    s = Solver.from_scene(scene, seed=0, device=device)
    s.restore(ckpt)
    _, u = s.macroscopics()

    d = int(scene.units.cells)
    cx = int(scene.raw["obstacle"]["center_x_chars"] * d)
    side = min(s.ny - 2, s.nx - (cx + 2 * d) - int(0.1 * s.nx)) // 2 * 2
    x0 = cx + 2 * d
    y0 = (s.ny - side) // 2
    win = u[:, x0:x0 + side, y0:y0 + side].cpu().numpy()
    win = win - win.mean(axis=(1, 2), keepdims=True)
    hann = np.hanning(side)
    w2d = hann[:, None] * hann[None, :]

    ek = np.zeros(side // 2)
    for comp in range(2):
        fh = np.fft.fft2(win[comp] * w2d)
        psd = (np.abs(fh) ** 2) / side**2
        kx = np.fft.fftfreq(side) * side
        kk = np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)
        for i in range(1, side // 2):
            shell = (kk >= i - 0.5) & (kk < i + 0.5)
            ek[i] += 0.5 * psd[shell].sum()

    k = np.arange(1, side // 2)
    e = ek[1:]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(k, e, lw=1.2, label="E(k), wake window")
    kref = np.array([3.0, side / 6])
    anchor_i = max(3, side // 60)
    a3 = e[anchor_i] * (k[anchor_i] ** 3)
    a53 = e[anchor_i] * (k[anchor_i] ** (5 / 3))
    ax.loglog(kref, a3 * kref ** -3.0, "k--", lw=1,
              label=r"$k^{-3}$ (2D enstrophy cascade)")
    ax.loglog(kref, a53 * kref ** (-5 / 3), "k:", lw=1,
              label=r"$k^{-5/3}$ (3D reference, NOT expected here)")
    ax.set_xlabel("k  [window modes]")
    ax.set_ylabel("E(k)")
    ax.set_title(f"{scene_name} — step {s.step_count}, "
                 f"window {side}$^2$ at x={x0}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "validation" \
        / f"spectrum_{scene_name}.png"
    fig.savefig(out, dpi=150)
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2],
                  sys.argv[3] if len(sys.argv) > 3 else "auto"))
