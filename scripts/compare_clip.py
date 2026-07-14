#!/usr/bin/env python3
"""Phase 5 deliverable: Re=100 vs Re=50k side-by-side comparison clip.

Runs both scenes forward from their final checkpoints in lockstep,
renders each to a normalized full-domain vorticity frame, stacks them
vertically (Re=100 on top, Re=50k below) with a label bar, and writes
numbered PNGs. Same convective-time cadence on both so the shedding
rhythms line up on screen despite wildly different step counts.

usage: compare_clip.py [n_frames] [device]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.image import imsave
from PIL import Image, ImageDraw

from lbm.config import load_scene
from lbm.fused import FusedSolver
from lbm.render import vorticity

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "clip_compare"
CMAP = colormaps["RdBu_r"]
WIDTH = 1200          # output width per panel (downsampled from the grid)


def load(scene_name: str, ckpt: Path, device: str):
    scene = load_scene(scene_name)
    s = FusedSolver.from_scene(scene, seed=0, device=device)
    s.restore(ckpt)
    return scene, s


def panel(s, scale: float) -> np.ndarray:
    omega = vorticity(s)                         # (ny, nx) on device
    img = (omega / (2.0 * scale) + 0.5).clamp(0.0, 1.0)
    # downsample to WIDTH by striding (cheap, adequate for a comparison)
    ny, nx = img.shape
    stride = max(1, nx // WIDTH)
    small = img[::stride, ::stride]
    rgba = CMAP(small.cpu().numpy())
    solid = s.mask.T[::stride, ::stride].cpu().numpy()
    rgba[solid] = (0.42, 0.42, 0.42, 1.0)
    return np.flipud(rgba)


def main() -> int:
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    OUT.mkdir(parents=True, exist_ok=True)

    sc_lo, s_lo = load("cylinder_re100",
                       ROOT / "out" / "cylinder_re100-seed0" / "final.pt",
                       device)
    sc_hi, s_hi = load("cylinder_re50k",
                       ROOT / "out" / "re50k" / "final.pt", device)

    # steps per convective time for each -> advance the same *physical*
    # time between frames (frame stride in convective units)
    conv_lo = sc_lo.units.cells / sc_lo.units.u_lat
    conv_hi = sc_hi.units.cells / sc_hi.units.u_lat
    # convective times advanced per frame. Small on purpose: the Re50k
    # panel steps dt_conv * 6800 steps/frame on 52M cells, so 0.15 would
    # cost ~1.5 h (more compute than the original run). 0.02 gives smooth
    # motion and a ~10-min render.
    dt_conv = 0.02
    steps_lo = max(1, int(dt_conv * conv_lo))
    steps_hi = max(1, int(dt_conv * conv_hi))

    # fixed color scales (99.5th pct of the first frame, per scene)
    def scale_of(s):
        o = vorticity(s).abs().flatten()
        o = o[:: max(1, o.numel() // 2_000_000)]
        return float(torch.quantile(o, torch.tensor(0.995, device=o.device)))
    scale_lo, scale_hi = scale_of(s_lo), scale_of(s_hi)

    for i in range(n_frames):
        p_lo = panel(s_lo, scale_lo)
        p_hi = panel(s_hi, scale_hi)
        w = min(p_lo.shape[1], p_hi.shape[1])
        stack = np.vstack([p_lo[:, :w], p_hi[:, :w]])
        img = Image.fromarray((stack * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        h_lo = p_lo.shape[0]
        draw.text((12, 10), "Re = 100  (laminar vortex street)",
                  fill=(20, 20, 20))
        draw.text((12, h_lo + 10),
                  "Re = 50,000  (Smagorinsky LES, 52M cells)",
                  fill=(20, 20, 20))
        img.save(OUT / f"frame_{i:06d}.png")
        for _ in range(steps_lo):
            s_lo.step()
        for _ in range(steps_hi):
            s_hi.step()
        if (i + 1) % 50 == 0:
            print(f"frame {i + 1}/{n_frames}", flush=True)

    print(f"wrote {n_frames} frames to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
