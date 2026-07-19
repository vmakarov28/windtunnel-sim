#!/usr/bin/env python3
"""Phase 8 study 2: what the staircase costs — grid convergence of Cd.

Re = 100 cylinder at three resolutions (D = 24, 32, 44 cells), same
tau = 0.572 for every run (u_lat = 2.4/D keeps Re and tau fixed while
ONLY resolution changes — single-variable, per the rules), same 30D x
15D domain, same 0.2D shedding trigger. Two wall treatments:

  staircase — halfway bounce-back on the rasterized mask (the default)
  bouzidi   — interpolated bounce-back against the analytic circle

Mean Cd from the momentum-exchange force over the last 25 shedding
periods (first ~15 discarded). The claim under test: Bouzidi's Cd
varies LESS across resolution — the staircase is a resolution-dependent
error and the curved boundary deletes most of it.

Output: validation/cylinder_convergence.png + printed table.
GPU strongly recommended (~5-10 min); CPU works but is slow.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lbm.solver import Solver

RE = 100.0
TAU = 0.572                      # nu = 0.024 for every run
DS = [24, 32, 44]
ST_EST = 0.166                   # only used to size the step budget


def run_case(d: int, curved: bool, device: str) -> float:
    u = RE * (TAU - 0.5) / 3.0 / d          # keeps Re and tau fixed
    nx, ny = 30 * d, 15 * d
    cx, cy, r = 8.0 * d, 7.3 * d, d / 2.0   # same geometry as the gate
    x = torch.arange(nx, dtype=torch.float32)[:, None] + 0.5
    y = torch.arange(ny, dtype=torch.float32)[None, :] + 0.5
    mask = ((x - cx) ** 2 + (y - cy) ** 2) <= r * r
    s = Solver(nx, ny, tau=TAU, u_char=u, seed=0, device=device,
               obstacle_mask=mask, ramp_steps=2000,
               curved_geom=("circle", cx, cy, r) if curved else None)
    s.measure_force = True
    period = d / (ST_EST * u)               # steps per shedding period
    warm = int(15 * period)
    meas = int(25 * period)
    for _ in range(warm):
        s.step()
    fx_sum, n = 0.0, 0
    for i in range(meas):
        s.step()
        if i % 5 == 0:                      # sample every 5 steps
            fx_sum += float(s.last_force[0])
            n += 1
    cd = (fx_sum / n) / (0.5 * u * u * d)
    tag = "bouzidi  " if curved else "staircase"
    print(f"  D={d:3d} {tag} u={u:.4f} grid {nx}x{ny}: Cd = {cd:.4f}"
          + (f"   ({s.bouzidi_links} links, {s.bouzidi_fallback} fallback)"
             if curved else ""))
    return cd


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    results = {"staircase": [], "bouzidi": []}
    for d in DS:
        results["staircase"].append(run_case(d, False, device))
        results["bouzidi"].append(run_case(d, True, device))

    stair = np.array(results["staircase"])
    bouz = np.array(results["bouzidi"])
    # spread across resolutions: how much does Cd depend on the grid?
    spread_s = stair.max() - stair.min()
    spread_b = bouz.max() - bouz.min()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(DS, stair, "o-", color="#d62728",
            label=f"staircase (spread {spread_s:.3f})")
    ax.plot(DS, bouz, "s-", color="#2ca02c",
            label=f"Bouzidi (spread {spread_b:.3f})")
    ax.axhspan(1.25, 1.45, color="#2ca02c", alpha=0.08,
               label="Phase 2 gate band")
    ax.set_xlabel("cylinder diameter D [cells]")
    ax.set_ylabel("mean Cd (25 shedding periods)")
    ax.set_title(f"Re = {RE:.0f} cylinder: Cd vs resolution, "
                 "wall treatment compared")
    ax.legend()
    ax.grid(alpha=0.3)
    out = Path(__file__).parent / "cylinder_convergence.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"Cd spread across D: staircase {spread_s:.4f}, "
          f"bouzidi {spread_b:.4f}")
    ok = spread_b < spread_s
    print("PASS: curved boundary reduces the resolution dependence of Cd"
          if ok else "FAIL: no reduction — investigate before claiming")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
