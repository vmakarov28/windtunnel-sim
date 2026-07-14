#!/usr/bin/env python3
"""Phase 2 gate #1: body-force-driven plane Poiseuille flow.

Converged velocity profile vs the analytic parabola
    u(y) = 4 u_max yhat (1 - yhat),   u_max = F H^2 / (8 rho nu)
PASS: relative L2 error < 1%.

tau and u_max come from scenes/channel_poiseuille.yaml through the units
discipline; the grid adds the two wall rows on top of H so the channel is
exactly H = cells_per_char cells wide (halfway bounce-back puts the
no-slip planes half a cell inside the wall rows).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lbm.config import load_scene
from lbm.solver import Solver

OUT = Path(__file__).resolve().parent


def main(device: str = "auto") -> int:
    scene = load_scene("channel_poiseuille")
    tau, u_max = scene.units.tau, scene.units.u_lat
    h = int(scene.units.cells)             # channel height H in cells
    nu = scene.units.nu_lat
    fx = 8.0 * nu * u_max / h**2           # force that yields u_max
    nx, ny = 64, h + 2                     # periodic x; +2 wall rows

    s = Solver(nx, ny, tau=tau, u_char=u_max, device=device,
               inlet_outlet=False, wall_y=True, body_force=(fx, 0.0),
               init_noise=0.0, scene_name="poiseuille_validation")
    print(f"Poiseuille: H={h}, tau={tau}, nu={nu:.4g}, fx={fx:.4g}, "
          f"target u_max={u_max}, device={s.device}")

    prev = None
    for chunk in range(200):
        for _ in range(2000):
            s.step()
        _, u = s.macroscopics()
        prof = u[0, nx // 2, 1:-1].clone()
        if prev is not None:
            delta = float((prof - prev).abs().max()) / u_max
            if delta < 1e-9:
                break
        prev = prof
    print(f"converged after {s.step_count} steps (delta={delta:.2e})")

    y = torch.arange(1, ny - 1, dtype=torch.float32) - 0.5   # wall at y=0.5
    y_hat = y / h
    analytic = 4.0 * u_max * y_hat * (1.0 - y_hat)
    prof = prof.cpu()
    l2 = float(((prof - analytic) ** 2).mean().sqrt()
               / (analytic ** 2).mean().sqrt())
    ok = l2 < 0.01

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [2, 1]})
    ax.plot(analytic, y_hat, "k-", lw=1.5, label="analytic parabola")
    ax.plot(prof, y_hat, "o", ms=3.5, mfc="none", color="#d62728",
            label="LBM (halfway BB + Guo forcing)")
    ax.set_xlabel("u / (lattice units)")
    ax.set_ylabel("y / H")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"Poiseuille, H={h}, tau={tau}")
    err = (prof - analytic) / u_max * 100
    ax2.plot(err, y_hat, "-", color="#d62728")
    ax2.axvline(0, color="k", lw=0.5)
    ax2.set_xlabel("error [% of u_max]")
    ax2.set_title(f"L2 = {l2 * 100:.3f}%  ({'PASS' if ok else 'FAIL'} < 1%)")
    fig.tight_layout()
    fig.savefig(OUT / "poiseuille.png", dpi=150)

    print(f"L2 error = {l2 * 100:.4f}%  ->  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "auto"
    sys.exit(main(dev))
