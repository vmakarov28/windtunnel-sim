#!/usr/bin/env python3
"""3D gate #1: body-force-driven plane Poiseuille flow (D3Q19).

Spanwise-periodic channel: the analytic parabola remains the exact
reference, and the profile must also be spanwise-INVARIANT (a 3D code
must reproduce 2D physics where the physics is 2D).
PASS: relative L2 error < 1% and spanwise std < 1e-6.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lbm3d.config import load_scene
from lbm3d.solver import Solver

OUT = Path(__file__).resolve().parent


def main(device: str = "auto") -> int:
    scene = load_scene("channel_poiseuille")
    tau, u_max = scene.units.tau, scene.units.u_lat
    h = int(scene.units.cells)
    nu = scene.units.nu_lat
    fx = 8.0 * nu * u_max / h**2
    nx, ny, nz = 32, h + 2, scene.nz

    s = Solver(nx, ny, nz, tau=tau, u_char=u_max, device=device,
               inlet_outlet=False, wall_y=True,
               body_force=(fx, 0.0, 0.0), init_noise=0.0,
               scene_name="poiseuille3d_validation")
    print(f"Poiseuille3D: H={h}, nz={nz}, tau={tau}, fx={fx:.4g}, "
          f"device={s.device}")

    prev = None
    delta = float("inf")
    for _ in range(200):
        for _ in range(2000):
            s.step()
        _, u = s.macroscopics()
        prof = u[0, nx // 2, 1:-1, nz // 2].clone()
        if prev is not None:
            delta = float((prof - prev).abs().max()) / u_max
            if delta < 1e-9:
                break
        prev = prof
    print(f"converged after {s.step_count} steps (delta={delta:.2e})")

    y = torch.arange(1, ny - 1, dtype=torch.float32) - 0.5
    y_hat = y / h
    analytic = 4.0 * u_max * y_hat * (1.0 - y_hat)
    prof = prof.cpu()
    l2 = float(((prof - analytic) ** 2).mean().sqrt()
               / (analytic ** 2).mean().sqrt())
    _, u = s.macroscopics()
    span_std = float(u[0, nx // 2, ny // 2, :].std())
    ok = l2 < 0.01 and span_std < 1e-6

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [2, 1]})
    ax.plot(analytic, y_hat, "k-", lw=1.5, label="analytic parabola")
    ax.plot(prof, y_hat, "o", ms=3.5, mfc="none", color="#d62728",
            label="D3Q19 LBM (halfway BB + Guo)")
    ax.set_xlabel("u (lattice units)"); ax.set_ylabel("y / H")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"3D Poiseuille, H={h}, span nz={nz} (periodic)")
    err = (prof - analytic) / u_max * 100
    ax2.plot(err, y_hat, "-", color="#d62728")
    ax2.axvline(0, color="k", lw=0.5)
    ax2.set_xlabel("error [% of u_max]")
    ax2.set_title(f"L2 = {l2 * 100:.3f}%  span-std = {span_std:.1e}\n"
                  f"({'PASS' if ok else 'FAIL'})")
    fig.tight_layout()
    fig.savefig(OUT / "poiseuille3d.png", dpi=150)

    print(f"L2 = {l2 * 100:.4f}%  spanwise std = {span_std:.2e}  ->  "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "auto"))
