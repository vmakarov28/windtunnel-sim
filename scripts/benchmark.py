#!/usr/bin/env python3
"""Phase 4 benchmark: MLUPS vs grid size, reference vs fused kernel.

The physics of the ceiling: D2Q9 fp32 A-B is 9 populations read + 9
written = 72 B/cell/step of compulsory traffic. At the RTX 5080's ~960
GB/s that is ~13.3 GLUPS. Score against it honestly — achieved fraction
of ceiling IS the result, whatever it is.

Writes benchmarks/results.csv and benchmarks/mlups.png.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lbm.fused import FusedSolver
from lbm.solver import Solver

GRIDS = [(512, 512), (1024, 512), (1024, 1024), (2048, 1024), (4096, 2048)]
CEILING_MLUPS = 960e9 / 72.0 / 1e6           # ~13,333 MLUPS


def rate(solver, warmup, steps) -> float:
    for _ in range(warmup):
        solver.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        solver.step()
    torch.cuda.synchronize()
    return solver.nx * solver.ny * steps / (time.perf_counter() - t0) / 1e6


def main() -> int:
    out = Path(__file__).resolve().parent.parent / "benchmarks"
    out.mkdir(exist_ok=True)
    rows = []
    for nx, ny in GRIDS:
        ref = Solver(nx, ny, tau=0.6, u_char=0.05, device="cuda",
                     inlet_outlet=True, ramp_steps=100)
        m_ref = rate(ref, 10, 30)
        del ref
        fus = FusedSolver(nx, ny, tau=0.6, u_char=0.05, device="cuda",
                          inlet_outlet=True, ramp_steps=100)
        m_fus = rate(fus, 50, 300)
        del fus
        torch.cuda.empty_cache()
        rows.append((nx, ny, m_ref, m_fus))
        print(f"{nx:5d}x{ny:<5d}  reference {m_ref:8.1f} MLUPS   "
              f"fused {m_fus:9.1f} MLUPS   ({m_fus / m_ref:5.1f}x, "
              f"{100 * m_fus / CEILING_MLUPS:4.1f}% of ceiling)")

    # kernel-only rows (periodic box, no Zou-He columns): what the fused
    # kernel does when the fixed per-step boundary cost is absent.
    krows = []
    for nx, ny in [(2048, 1024), (4096, 2048)]:
        fus = FusedSolver(nx, ny, tau=0.6, u_char=0.05, device="cuda",
                          inlet_outlet=False, init_noise=0.0)
        m = rate(fus, 50, 300)
        del fus
        torch.cuda.empty_cache()
        print(f"{nx:5d}x{ny:<5d}  fused, periodic (kernel only) "
              f"{m:9.1f} MLUPS   ({100 * m / CEILING_MLUPS:4.1f}% of ceiling)")
        krows.append((nx, ny, m))

    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["nx", "ny", "reference_mlups", "fused_mlups", "mode"])
        for r in rows:
            w.writerow([*r, "open_boundaries"])
        for nx, ny, m in krows:
            w.writerow([nx, ny, "", m, "periodic_kernel_only"])

    cells = [nx * ny / 1e6 for nx, ny, *_ in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cells, [r[2] for r in rows], "o-", label="PyTorch reference")
    ax.plot(cells, [r[3] for r in rows], "s-", label="fused Triton kernel")
    ax.plot([nx * ny / 1e6 for nx, ny, _ in krows], [m for *_, m in krows],
            "d--", label="fused, periodic (kernel only)")
    ax.axhline(CEILING_MLUPS, color="k", ls="--", lw=1,
               label=f"bandwidth ceiling ~{CEILING_MLUPS / 1e3:.1f} GLUPS "
                     "(72 B/cell @ 960 GB/s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grid size [Mcells]")
    ax.set_ylabel("MLUPS")
    ax.set_title("D2Q9 fp32, RTX 5080 (sm_120)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "mlups.png", dpi=150)
    print(f"written: {out / 'results.csv'}, {out / 'mlups.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
