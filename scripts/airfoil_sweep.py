#!/usr/bin/env python3
"""Phase 6: MH45 alpha sweep at Re = 20,000 (Smagorinsky LES, fused).

For each alpha in 0..10 deg: establish the flow past the transient,
then average Cl/Cd (momentum exchange) over a fixed measurement window,
reporting the std-dev as error bars and the dominant Cl frequency.

Writes benchmarks/../validation/mh45_sweep.csv incrementally (a crash
loses nothing) and the polar plots. If data/xfoil_mh45_re20k.csv exists
(columns: alpha,cl,cd — user-provided), it is overlaid; otherwise the
plots ship without the overlay and say so.

Honest expectations, stated up front (see CLAUDE.md Phase 6):
- lift-curve slope within ~15% of XFOIL pre-stall = win;
- Cd expected HIGH (staircase surface + BL thickness ~ c/sqrt(Re) ~ 3
  cells at chord 400: the boundary layer is barely resolved);
- stall angle explicitly untrusted (different transition physics).

usage: airfoil_sweep.py [alpha_start alpha_end] [device]
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lbm.config import load_scene
from lbm.fused import FusedSolver

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "validation" / "mh45_sweep.csv"

TRANSIENT_CONV = 10.0     # convective times c/U before measuring
MEASURE_CONV = 24.0       # convective times; >= 8 shedding periods (checked
                          # per-point from the Cl FFT and printed)


def run_alpha(alpha: float, device: str) -> dict:
    scene = load_scene("airfoil_mh45_re20k")
    scene.raw["obstacle"]["alpha_deg"] = alpha
    s = FusedSolver.from_scene(scene, seed=0, device=device)
    c = scene.units.cells
    u = scene.units.u_lat
    q_dyn = 0.5 * u * u * c
    conv = int(c / u)                       # steps per convective time

    t0 = time.perf_counter()
    for _ in range(int(TRANSIENT_CONV * conv)):
        s.step()
    s.measure_force = True
    n_meas = int(MEASURE_CONV * conv)
    hist = torch.empty((n_meas, 2), dtype=torch.float32, device=s.device)
    for i in range(n_meas):
        s.step()
        hist[i] = s.last_force
    g = s.check_guards()
    forces = hist.cpu().numpy().astype(np.float64) / q_dyn
    cd, cl = forces[:, 0], forces[:, 1]

    # dominant shedding frequency and how many periods we averaged
    win = np.hanning(len(cl))
    spec = np.abs(np.fft.rfft((cl - cl.mean()) * win))
    freqs = np.fft.rfftfreq(len(cl), d=1.0)
    pk = spec[1:].argmax() + 1
    st = freqs[pk] * c / u
    periods = len(cl) * freqs[pk]
    wall = time.perf_counter() - t0
    print(f"alpha={alpha:4.1f}: Cl={cl.mean():+.4f}±{cl.std():.4f}  "
          f"Cd={cd.mean():.4f}±{cd.std():.4f}  St={st:.3f} "
          f"({periods:.1f} periods)  u_max={g['u_max']:.3f}  {wall:.0f}s")
    return dict(alpha=alpha, cl=cl.mean(), cl_std=cl.std(),
                cd=cd.mean(), cd_std=cd.std(), st=st, periods=periods)


def plot(rows: list[dict]) -> None:
    a = [r["alpha"] for r in rows]
    xf = None
    xf_path = ROOT / "data" / "xfoil_mh45_re20k.csv"
    if xf_path.exists():
        xf = np.genfromtxt(xf_path, delimiter=",", names=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    a1.errorbar(a, [r["cl"] for r in rows],
                yerr=[r["cl_std"] for r in rows],
                fmt="o-", ms=4, capsize=3, label="LBM-LES (this work)")
    a2.errorbar(a, [r["cd"] for r in rows],
                yerr=[r["cd_std"] for r in rows],
                fmt="o-", ms=4, capsize=3, label="LBM-LES (this work)")
    if xf is not None:
        a1.plot(xf["alpha"], xf["cl"], "ks--", ms=4, mfc="none",
                label="XFOIL (user-provided)")
        a2.plot(xf["alpha"], xf["cd"], "ks--", ms=4, mfc="none",
                label="XFOIL (user-provided)")
    else:
        a1.set_title("(XFOIL overlay pending: drop polars at "
                     "data/xfoil_mh45_re20k.csv)", fontsize=8)
    a1.set_xlabel(r"$\alpha$ [deg]"); a1.set_ylabel(r"$C_l$")
    a2.set_xlabel(r"$\alpha$ [deg]"); a2.set_ylabel(r"$C_d$")
    a1.legend(frameon=False, fontsize=9)
    a2.legend(frameon=False, fontsize=9)
    fig.suptitle("MH45, Re = 20,000 — chord 400 cells, Smagorinsky LES")
    fig.tight_layout()
    fig.savefig(ROOT / "validation" / "mh45_polar.png", dpi=150)


def main() -> int:
    args = sys.argv[1:]
    a0, a1 = (float(args[0]), float(args[1])) if len(args) >= 2 else (0.0, 10.0)
    device = args[2] if len(args) > 2 else "cuda"

    rows = []
    if OUT_CSV.exists():   # resume a partial sweep
        with open(OUT_CSV) as fh:
            rows = [dict((k, float(v)) for k, v in r.items())
                    for r in csv.DictReader(fh)]
        print(f"resuming: {len(rows)} alphas already done")
    done = {round(r["alpha"], 3) for r in rows}

    alpha = a0
    while alpha <= a1 + 1e-9:
        if round(alpha, 3) not in done:
            rows.append(run_alpha(alpha, device))
            rows.sort(key=lambda r: r["alpha"])
            with open(OUT_CSV, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        alpha += 1.0
    plot(rows)
    print(f"sweep complete: {OUT_CSV}, validation/mh45_polar.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
