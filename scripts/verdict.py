#!/usr/bin/env python3
"""Phase 6: the verdict table — agreement expected? achieved? why/why not.

Reads validation/mh45_sweep.csv (+ data/xfoil_mh45_re20k.csv if the user
has provided it) and writes notes/phase6_verdict.md. The success criteria
are the HONEST ones from the project brief: lift-curve slope within ~15%
of XFOIL pre-stall = win; Cd expected high (staircase + BL ~ 3 cells);
stall angle explicitly untrusted. Disagreement-with-explanation beats
fake agreement.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "validation" / "mh45_sweep.csv"
XFOIL = ROOT / "data" / "xfoil_mh45_re20k.csv"
OUT = ROOT / "notes" / "phase6_verdict.md"

PRESTALL_MAX_ALPHA = 6.0     # fit the slope only where flow is attached


def slope_per_rad(alpha_deg: np.ndarray, cl: np.ndarray) -> float:
    m = alpha_deg <= PRESTALL_MAX_ALPHA
    a, b = np.polyfit(np.radians(alpha_deg[m]), cl[m], 1)
    return a


def main() -> int:
    rows = list(csv.DictReader(open(SWEEP)))
    alpha = np.array([float(r["alpha"]) for r in rows])
    cl = np.array([float(r["cl"]) for r in rows])
    cl_std = np.array([float(r["cl_std"]) for r in rows])
    cd = np.array([float(r["cd"]) for r in rows])

    ours = slope_per_rad(alpha, cl)
    lines = [
        "# Phase 6 verdict — MH45 at Re = 20,000",
        "",
        f"LBM-LES sweep: alpha 0-{alpha.max():.0f} deg, chord 400 cells, "
        "Smagorinsky Cs = 0.14, momentum-exchange forces, error bars = "
        "std-dev over the measurement window.",
        "",
        "| quantity | expected agreement? | achieved | verdict |",
        "|---|---|---|---|",
    ]

    if XFOIL.exists():
        xf = np.genfromtxt(XFOIL, delimiter=",", names=True)
        xf_slope = slope_per_rad(np.asarray(xf["alpha"]),
                                 np.asarray(xf["cl"]))
        rel = (ours - xf_slope) / xf_slope
        win = abs(rel) <= 0.15
        lines.append(
            f"| lift-curve slope (pre-stall) | yes, within ~15% "
            f"| ours {ours:.2f}/rad vs XFOIL {xf_slope:.2f}/rad "
            f"({rel * 100:+.1f}%) | {'WIN' if win else 'MISS — investigate'} |")
        # Cd comparison at matching alphas
        cd_ratio = np.interp(alpha, xf["alpha"], xf["cd"])
        mean_ratio = float(np.mean(cd / cd_ratio))
        lines.append(
            f"| drag level | NO — expected high (staircase surface; BL "
            f"~ c/sqrt(Re) ~ 3 cells barely resolved) | ours averages "
            f"{mean_ratio:.1f}x XFOIL | "
            f"{'as predicted' if mean_ratio > 1.1 else 'unexpectedly close'} |")
        lines.append(
            "| stall angle | NO — different transition physics (LES with "
            "no transition model vs XFOIL's e^N) | see Cl(alpha) plot | "
            "explicitly untrusted, by design |")
    else:
        lines.append(
            f"| lift-curve slope (pre-stall) | yes, within ~15% of XFOIL "
            f"| ours: {ours:.2f}/rad (thin-airfoil ref: 6.28/rad) "
            f"| PENDING — needs user XFOIL polars at "
            f"data/xfoil_mh45_re20k.csv |")
        lines.append(
            "| drag level | NO — expected high (staircase + ~3-cell BL) "
            f"| Cd(0) = {cd[alpha.argmin()]:.4f} | PENDING XFOIL overlay |")
        lines.append(
            "| stall angle | NO — different transition physics | — | "
            "explicitly untrusted, by design |")

    lines += [
        "",
        f"Mean Cl oscillation (shedding) amplitude: "
        f"{cl_std.mean():.3f}; per-alpha Strouhal in the CSV.",
        "",
        "Figures: validation/mh45_polar.png (Cl, Cd vs alpha), "
        "validation/mh45_staircase.png (the honest mask).",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
