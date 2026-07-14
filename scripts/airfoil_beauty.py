#!/usr/bin/env python3
"""Phase 6 beauty clip: MH45 vorticity + dye at a chosen angle of attack.

Renders a zoomed vorticity view of the airfoil and its near wake — the
video beat that pays off the whole airfoil phase. Separate from the sweep
so it can use any alpha and a cinematic camera without touching the
measurement protocol.

usage: airfoil_beauty.py [alpha_deg] [preset] [device]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from lbm.cinema import CinemaWriter, Dye
from lbm.config import load_scene
from lbm.fused import FusedSolver

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    preset = sys.argv[2] if len(sys.argv) > 2 else "vorticity"
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda"

    scene = load_scene("airfoil_mh45_re20k")
    scene.raw["obstacle"]["alpha_deg"] = alpha
    s = FusedSolver.from_scene(scene, seed=0, device=device)
    c = scene.units.cells
    cx = scene.raw["obstacle"]["center_x_chars"] * c
    cy = scene.raw["obstacle"]["center_y_chars"] * c
    # camera: 0.7c ahead of the LE to 4c behind, +/-1.5c vertically
    zoom = (int(cx - 0.7 * c), int(cy - 1.5 * c),
            int(cx + 4.0 * c), int(cy + 1.5 * c))

    out = ROOT / "out" / f"airfoil_a{alpha:.0f}_{preset}"
    frames = CinemaWriter(out / "frames", preset=preset, zoom=zoom)
    extras = {"dye": Dye(s, source=(int(cx - 0.6 * c), int(cy - 0.4 * c),
                                    int(cx - 0.55 * c), int(cy + 0.4 * c)))} \
        if preset == "dye" else {}

    conv = int(c / scene.units.u_lat)          # steps per convective time
    warmup = 8 * conv                          # establish the flow
    print(f"MH45 alpha={alpha}, preset={preset}, warmup {warmup} steps")
    t0 = time.perf_counter()
    for n in range(warmup):
        s.step()
        if "dye" in extras:
            extras["dye"].step(s)
    for i in range(360):                       # ~6 s at 60 fps
        for _ in range(conv // 8):             # 1/8 conv time per frame
            s.step()
            if "dye" in extras:
                extras["dye"].step(s)
        frames.write(s, extras)
        if (i + 1) % 60 == 0:
            print(f"frame {i+1}/360  ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
    g = s.check_guards()
    print(f"done, u_max={g['u_max']:.3f}, frames in {frames.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
