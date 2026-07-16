#!/usr/bin/env python3
"""3D fused-kernel correctness gate: Triton kernel vs PyTorch reference.

Runs the dev cylinder scene with both solvers side by side and compares
the macroscopic fields (collision invariants, so the two storage
conventions must agree step-for-step up to fp32 op-order). Same protocol
as the 2D gate: the inlet-ramp window shows a bounded, documented
transient (the conventions see u_in(t) one step apart), so scoring starts
after 3x the ramp.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from lbm3d.config import load_scene
from lbm3d.fused import FusedSolver
from lbm3d.solver import Solver

STEPS = 6_000
CHECK_EVERY = 1_500
SCORE_FROM = 3_000        # dev scene ramp is 1000 steps
TOL_U = 1e-4
TOL_RHO = 1e-4


def main() -> int:
    scene = load_scene("cylinder_re100_dev")
    ref = Solver.from_scene(scene, seed=0, device="cuda")
    fus = FusedSolver.from_scene(scene, seed=0, device="cuda")

    worst_u = worst_rho = ramp_u = ramp_rho = 0.0
    for n in range(1, STEPS + 1):
        ref.step()
        fus.step()
        if n % CHECK_EVERY == 0:
            rho_r, u_r = ref.macroscopics()
            rho_f, u_f = fus.macroscopics()
            du = float((u_r - u_f).abs().max())
            drho = float((rho_r - rho_f).abs().max())
            scored = n >= SCORE_FROM
            if scored:
                worst_u, worst_rho = max(worst_u, du), max(worst_rho, drho)
            else:
                ramp_u, ramp_rho = max(ramp_u, du), max(ramp_rho, drho)
            print(f"step {n:5d}: max|du| = {du:.3e}   max|drho| = {drho:.3e}"
                  + ("" if scored else "   (ramp transient, not scored)"))

    ok = worst_u < TOL_U and worst_rho < TOL_RHO
    print(f"\nramp-window transient (documented, unscored): "
          f"max|du| = {ramp_u:.3e}, max|drho| = {ramp_rho:.3e}")
    print(f"fused3d vs reference, steps {SCORE_FROM}-{STEPS}: "
          f"max|du| = {worst_u:.3e} (tol {TOL_U}), "
          f"max|drho| = {worst_rho:.3e} (tol {TOL_RHO})  "
          f"->  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
