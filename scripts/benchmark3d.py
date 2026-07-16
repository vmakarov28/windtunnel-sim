#!/usr/bin/env python3
"""3D benchmark: MLUPS, reference vs fused D3Q19, vs the DRAM ceiling.

D3Q19 fp32 A-B is 19 populations read + 19 written = 152 B/cell/step of
compulsory traffic; at the RTX 5080's ~960 GB/s that is ~6.3 GLUPS.
Writes benchmarks/results3d.csv and benchmarks/mlups3d.png.
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

from lbm3d.fused import FusedSolver
from lbm3d.solver import Solver

# (nx, ny, nz, open_bc): the dev grid, the full cylinder grid, and a
# mid-size cube-ish grid
GRIDS = [
    (576, 288, 16, True),
    (900, 450, 90, True),
    (512, 256, 64, True),
]
CEILING_MLUPS = 960e9 / 152.0 / 1e6      # ~6316


def rate(solver, warmup, steps) -> float:
    for _ in range(warmup):
        solver.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        solver.step()
    torch.cuda.synchronize()
    n = solver.nx * solver.ny * solver.nz
    return n * steps / (time.perf_counter() - t0) / 1e6


def main() -> int:
    out = Path(__file__).resolve().parent.parent / "benchmarks"
    out.mkdir(exist_ok=True)
    rows = []
    for nx, ny, nz, open_bc in GRIDS:
        ref = Solver(nx, ny, nz, tau=0.6, u_char=0.05, device="cuda",
                     inlet_outlet=open_bc, ramp_steps=100)
        m_ref = rate(ref, 5, 15)
        del ref
        torch.cuda.empty_cache()
        fus = FusedSolver(nx, ny, nz, tau=0.6, u_char=0.05, device="cuda",
                          inlet_outlet=open_bc, ramp_steps=100)
        m_fus = rate(fus, 30, 200)
        del fus
        torch.cuda.empty_cache()
        rows.append((nx, ny, nz, m_ref, m_fus))
        print(f"{nx}x{ny}x{nz}  reference {m_ref:7.1f} MLUPS   "
              f"fused {m_fus:8.1f} MLUPS   ({m_fus / m_ref:5.1f}x, "
              f"{100 * m_fus / CEILING_MLUPS:4.1f}% of ceiling)")

    # SGS-on cost on the big grid (the extra in-kernel pass)
    fus = FusedSolver(900, 450, 90, tau=0.52, u_char=0.05, device="cuda",
                      inlet_outlet=True, ramp_steps=100, sgs=True)
    m_sgs = rate(fus, 30, 200)
    del fus
    torch.cuda.empty_cache()
    print(f"900x450x90  fused + Smagorinsky {m_sgs:8.1f} MLUPS   "
          f"({100 * m_sgs / CEILING_MLUPS:4.1f}% of ceiling)")

    with open(out / "results3d.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["nx", "ny", "nz", "reference_mlups", "fused_mlups"])
        w.writerows(rows)
        w.writerow([900, 450, 90, "", m_sgs, ])

    cells = [nx * ny * nz / 1e6 for nx, ny, nz, *_ in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cells, [r[3] for r in rows], "o", label="PyTorch reference")
    ax.plot(cells, [r[4] for r in rows], "s", label="fused Triton kernel")
    ax.plot([900 * 450 * 90 / 1e6], [m_sgs], "d",
            label="fused + Smagorinsky")
    ax.axhline(CEILING_MLUPS, color="k", ls="--", lw=1,
               label=f"DRAM ceiling ~{CEILING_MLUPS / 1e3:.1f} GLUPS "
                     "(152 B/cell @ 960 GB/s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grid size [Mcells]")
    ax.set_ylabel("MLUPS")
    ax.set_title("D3Q19 fp32, RTX 5080 (sm_120)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "mlups3d.png", dpi=150)
    print(f"written: {out / 'results3d.csv'}, {out / 'mlups3d.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
