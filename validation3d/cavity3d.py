#!/usr/bin/env python3
"""3D gate #2: spanwise-periodic lid-driven cavity, Re = 100 (D3Q19).

At Re = 100 the cavity has no spanwise instability (Taylor-Goertler
onsets far above), so the converged MID-SPAN plane must reproduce Ghia,
Ghia & Shin (1982) — and the solution must be spanwise-invariant.
PASS: max centerline deviation < 3% of lid speed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lbm3d.config import load_scene

OUT = Path(__file__).resolve().parent

# Ghia, Ghia & Shin (1982), Re = 100 — same verified transcription as the
# 2D gate (validation/cavity.py); duplicated, not imported: the versions
# stay independent.
GHIA_Y = np.array([
    1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
    0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547,
    0.0000])
GHIA_U = np.array([
    1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332,
    -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434,
    -0.04775, -0.04192, -0.03717, 0.00000])
GHIA_X = np.array([
    1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
    0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625,
    0.0000])
GHIA_V = np.array([
    0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
    -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317,
    0.10890, 0.10091, 0.09233, 0.00000])


def main(device: str = "auto", solver: str = "fused") -> int:
    scene = load_scene("cavity_re100")
    if solver == "fused":
        from lbm3d.fused import FusedSolver as cls
    else:
        from lbm3d.solver import Solver as cls
    s = cls.from_scene(scene, seed=0, device=device)
    u_lid = scene.units.u_lat
    k = s.nz // 2
    print(f"Cavity3D: {scene.nx}x{scene.ny}x{scene.nz}, "
          f"tau={scene.units.tau}, solver={solver}, device={s.device}")

    prev = None
    delta = float("inf")
    for _ in range(300):
        for _ in range(2000):
            s.step()
        _, u = s.macroscopics()
        if prev is not None:
            delta = float((u - prev).abs().max()) / u_lid
            if delta < 1e-8:
                break
        prev = u.clone()
    print(f"converged after {s.step_count} steps (delta={delta:.2e})")

    _, u = s.macroscopics()
    nx, ny = s.nx, s.ny
    span = nx - 2
    cells_y = (torch.arange(1, ny - 1, dtype=torch.float32) - 0.5) / span
    cells_x = (torch.arange(1, nx - 1, dtype=torch.float32) - 0.5) / span
    u_line = 0.5 * (u[0, nx // 2 - 1, 1:-1, k] + u[0, nx // 2, 1:-1, k])
    v_line = 0.5 * (u[1, 1:-1, ny // 2 - 1, k] + u[1, 1:-1, ny // 2, k])
    u_sim = np.interp(GHIA_Y, cells_y.numpy(), u_line.cpu().numpy()) / u_lid
    v_sim = np.interp(GHIA_X, cells_x.numpy(), v_line.cpu().numpy()) / u_lid
    du = np.abs(u_sim[1:-1] - GHIA_U[1:-1]).max()
    dv = np.abs(v_sim[1:-1] - GHIA_V[1:-1]).max()
    # spanwise invariance of the full field (3D code, 2D physics)
    span_dev = float((u - u[:, :, :, k:k + 1]).abs().max()) / u_lid
    dev = max(du, dv)
    ok = dev < 0.03 and span_dev < 1e-3

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    yy = np.linspace(0, 1, 200)
    a1.plot(np.interp(yy, cells_y.numpy(), u_line.cpu().numpy()) / u_lid,
            yy, "-", color="#1f77b4", lw=1.5,
            label=f"D3Q19 mid-span (nz={s.nz})")
    a1.plot(GHIA_U, GHIA_Y, "ks", ms=4, mfc="none", label="Ghia et al. 1982")
    a1.set_xlabel("u / U_lid"); a1.set_ylabel("y / L")
    a1.set_title("vertical centerline"); a1.legend(frameon=False, fontsize=9)
    a2.plot(yy, np.interp(yy, cells_x.numpy(), v_line.cpu().numpy()) / u_lid,
            "-", color="#1f77b4", lw=1.5)
    a2.plot(GHIA_X, GHIA_V, "ks", ms=4, mfc="none")
    a2.set_xlabel("x / L"); a2.set_ylabel("v / U_lid")
    a2.set_title("horizontal centerline")
    fig.suptitle(
        f"3D lid-driven cavity Re=100 (mid-span) — max dev = {dev * 100:.2f}%"
        f", spanwise dev = {span_dev:.1e} ({'PASS' if ok else 'FAIL'})")
    fig.tight_layout()
    fig.savefig(OUT / "cavity3d.png", dpi=150)

    print(f"max deviation: u {du * 100:.2f}%, v {dv * 100:.2f}%; "
          f"spanwise dev {span_dev:.2e}  ->  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "auto"
    slv = sys.argv[2] if len(sys.argv) > 2 else "fused"
    sys.exit(main(dev, slv))
