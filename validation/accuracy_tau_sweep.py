#!/usr/bin/env python3
"""Phase 8 study 1: where is the wall, really? BGK vs TRT across tau.

Forced Poiseuille between halfway bounce-back walls. The no-slip plane
should sit exactly halfway between the last fluid and first solid cell
centers, independent of viscosity. With BGK it does not — the effective
wall position drifts with tau (a numerical-slip error). TRT with the
magic parameter LAMBDA = 3/16 pins it.

Measured, not asserted: fit a parabola to the converged profile and
find its roots — those ARE the effective wall positions. Also reports
profile L2 error against the analytic solution through the true walls.

Output: validation/accuracy_tau_sweep.png + a printed table.
Runs on CPU in a few minutes (the channels are tiny).
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

NY = 34                     # 32 fluid cells across the channel
TAUS = [0.55, 0.65, 0.8, 1.0, 1.5, 2.0, 3.0]
F = 1e-6


def run(collision: str, tau: float) -> tuple[float, float]:
    """-> (wall drift [cells], profile L2 rel. error)."""
    # transient ~ (H/pi)^2 / nu; give it 8x that, minimum 20k
    nu = (tau - 0.5) / 3.0
    steps = max(20_000, int(8 * (NY - 2) ** 2 / (nu * np.pi ** 2)))
    s = Solver(4, NY, tau=tau, u_char=0.0, seed=0, device="cpu",
               inlet_outlet=False, wall_y=True, body_force=(F, 0.0),
               init_noise=0.0, collision=collision)
    for _ in range(steps):
        s.step()
    _, u = s.macroscopics()
    ux = u[0, 2, 1:-1].numpy().astype(np.float64)
    yc = np.arange(1, NY - 1) + 0.5
    lo, hi = sorted(np.roots(np.polyfit(yc, ux, 2)).real)
    drift = 0.5 * (abs(lo - 1.0) + abs(hi - (NY - 1.0)))
    ua = F / (2 * nu) * (yc - 1.0) * ((NY - 1.0) - yc)
    l2 = float(np.sqrt(((ux - ua) ** 2).mean()) / ua.max())
    return drift, l2


def main() -> int:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rows = []
    print(f"{'tau':>5} | {'BGK wall drift':>15} {'TRT wall drift':>15} | "
          f"{'BGK L2':>9} {'TRT L2':>9}")
    for tau in TAUS:
        db, lb = run("bgk", tau)
        dt, lt = run("trt", tau)
        rows.append((tau, db, dt, lb, lt))
        print(f"{tau:5.2f} | {db:13.4f} c {dt:13.4f} c | "
              f"{lb * 100:8.3f}% {lt * 100:8.3f}%")

    tau_a = np.array([r[0] for r in rows])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.semilogy(tau_a, [max(r[1], 1e-5) for r in rows], "o-",
                 label="BGK", color="#d62728")
    ax1.semilogy(tau_a, [max(r[2], 1e-5) for r in rows], "s-",
                 label="TRT ($\\Lambda=3/16$)", color="#2ca02c")
    ax1.set_xlabel(r"$\tau$")
    ax1.set_ylabel("effective wall drift [cells]")
    ax1.set_title("Where the no-slip plane actually sits")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.semilogy(tau_a, [max(r[3], 1e-6) for r in rows], "o-",
                 label="BGK", color="#d62728")
    ax2.semilogy(tau_a, [max(r[4], 1e-6) for r in rows], "s-",
                 label="TRT", color="#2ca02c")
    ax2.set_xlabel(r"$\tau$")
    ax2.set_ylabel("Poiseuille profile L2 error")
    ax2.set_title("Profile error (32-cell channel)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.suptitle("BGK's wall moves with viscosity; TRT's does not",
                 y=1.02)
    fig.tight_layout()
    out = Path(__file__).parent / "accuracy_tau_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")

    # the claim this study exists to check:
    hi_tau = rows[-1]
    ok = hi_tau[2] < 0.1 * hi_tau[1]   # TRT drift < 10% of BGK's at tau=3
    print("PASS: TRT wall drift at tau=3 is "
          f"{hi_tau[2] / hi_tau[1] * 100:.1f}% of BGK's"
          if ok else "FAIL: TRT did not pin the wall")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
