#!/usr/bin/env python3
"""Phase 4 correctness gate: fused Triton kernel vs PyTorch reference.

Runs the cylinder scene with both solvers side by side and compares the
macroscopic fields (rho, u) — which are collision invariants, so the two
storage conventions (post-BC vs post-collision) must agree step-for-step
up to fp32 op-ordering. Reports the drift; PASS if the fields match to
fp32 tolerance after 10k steps.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from lbm.config import load_scene
from lbm.fused import FusedSolver
from lbm.solver import Solver

STEPS = 10_000
CHECK_EVERY = 2_000
TOL_U = 5e-4          # in lattice-velocity units (u_char = 0.06)
TOL_RHO = 5e-4


def main() -> int:
    scene = load_scene("cylinder_re100")
    ref = Solver.from_scene(scene, seed=0, device="cuda")
    fus = FusedSolver.from_scene(scene, seed=0, device="cuda")

    worst_u = worst_rho = 0.0
    for n in range(1, STEPS + 1):
        ref.step()
        fus.step()
        if n % CHECK_EVERY == 0:
            rho_r, u_r = ref.macroscopics()
            rho_f, u_f = fus.macroscopics()
            du = float((u_r - u_f).abs().max())
            drho = float((rho_r - rho_f).abs().max())
            worst_u, worst_rho = max(worst_u, du), max(worst_rho, drho)
            print(f"step {n:6d}: max|du| = {du:.3e}   max|drho| = {drho:.3e}")

    ok = worst_u < TOL_U and worst_rho < TOL_RHO
    print(f"\nfused vs reference over {STEPS} steps: "
          f"max|du| = {worst_u:.3e} (tol {TOL_U}), "
          f"max|drho| = {worst_rho:.3e} (tol {TOL_RHO})  "
          f"->  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
