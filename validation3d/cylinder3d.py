#!/usr/bin/env python3
"""3D gate #3: cylinder at Re = 100 — Strouhal number and drag (D3Q19).

Full-resolution scene (900x450x90 = 36.5M cells, spanwise-periodic).
At Re = 100 the wake is physically two-dimensional (mode-A onsets near
Re ~ 190), so the 2D bands remain the honest reference:
PASS: St in [0.155, 0.175] and mean Cd in [1.25, 1.45].
The spanwise force must also stay negligible — a 3D solver earning the
right to use 2D reference data has to PROVE the flow stayed 2D.

Only tractable on the fused kernel (~36.5M cells x ~100k steps).
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

TRANSIENT_STEPS = 60_000     # t* ~ 140: past shedding saturation (the 2D
                             # gate FAILED once for measuring pre-saturation)
MEASURE_STEPS = 40_000       # ~15 shedding cycles


def main(device: str = "auto", solver: str = "fused") -> int:
    scene = load_scene("cylinder_re100")
    if solver == "fused":
        from lbm3d.fused import FusedSolver as cls
    else:
        from lbm3d.solver import Solver as cls
    s = cls.from_scene(scene, seed=0, device=device)
    d = scene.units.cells
    u = scene.units.u_lat
    span = s.nz
    q_dyn = 0.5 * u * u * d * span        # 0.5 rho0 U^2 * frontal area
    print(f"Cylinder3D: {scene.nx}x{scene.ny}x{scene.nz}, D={d:g}, "
          f"tau={scene.units.tau:.4g}, u={u}, solver={solver}, "
          f"device={s.device}")

    for n in range(TRANSIENT_STEPS):
        s.step()
        if (n + 1) % 10_000 == 0:
            g = s.check_guards()
            print(f"  transient {n + 1}: u_max={g['u_max']:.3f} "
                  f"drift={g['mass_drift']:.1e}", flush=True)
    print(f"transient done ({TRANSIENT_STEPS} steps)")

    s.measure_force = True
    hist = torch.empty((MEASURE_STEPS, 3), dtype=torch.float32,
                       device=s.device)
    for i in range(MEASURE_STEPS):
        s.step()
        hist[i] = s.last_force
    g = s.check_guards()
    forces = hist.cpu().numpy().astype(np.float64) / q_dyn
    cd, cl, cz = forces[:, 0], forces[:, 1], forces[:, 2]
    print(f"measurement done (u_max={g['u_max']:.3f})")

    win = np.hanning(len(cl))
    spec = np.abs(np.fft.rfft((cl - cl.mean()) * win))
    freqs = np.fft.rfftfreq(len(cl), d=1.0)
    peak = spec[1:].argmax() + 1
    st = freqs[peak] * d / u
    cd_mean, cd_std = cd.mean(), cd.std()
    cl_amp = np.sqrt(2.0) * (cl - cl.mean()).std()
    cz_frac = np.abs(cz).max() / max(abs(cd_mean), 1e-12)

    ok_st = 0.155 <= st <= 0.175
    ok_cd = 1.25 <= cd_mean <= 1.45
    ok_2d = cz_frac < 0.01              # flow must have stayed 2D
    ok = ok_st and ok_cd and ok_2d

    t = np.arange(len(cl)) * u / d
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4),
                                 gridspec_kw={"width_ratios": [3, 2]})
    a1.plot(t, cd, lw=0.8, label=f"Cd (mean {cd_mean:.3f})")
    a1.plot(t, cl, lw=0.8, label=f"Cl (amp {cl_amp:.3f})")
    a1.plot(t, cz, lw=0.8, label=f"Cz (max {np.abs(cz).max():.4f})")
    a1.set_xlabel("t U / D"); a1.legend(frameon=False, fontsize=9)
    a1.set_title("3D force coefficients, momentum exchange")
    sts = freqs * d / u
    m = (sts > 0.05) & (sts < 0.5)
    a2.semilogy(sts[m], spec[m], lw=1.0)
    a2.axvline(st, color="#d62728", lw=1, ls="--")
    a2.axvspan(0.155, 0.175, color="green", alpha=0.12)
    a2.set_xlabel("St = f D / U"); a2.set_title(f"Cl spectrum: St = {st:.4f}")
    fig.suptitle(
        f"3D cylinder Re=100 (span {span / d:.0f}D periodic) — "
        f"St={st:.4f} [{'PASS' if ok_st else 'FAIL'}], "
        f"Cd={cd_mean:.3f}±{cd_std:.3f} [{'PASS' if ok_cd else 'FAIL'}], "
        f"|Cz|/Cd={cz_frac:.1e} [{'PASS' if ok_2d else 'FAIL'} 2D-ness]")
    fig.tight_layout()
    fig.savefig(OUT / "cylinder3d.png", dpi=150)

    print(f"St = {st:.4f}  ({'PASS' if ok_st else 'FAIL'})")
    print(f"Cd = {cd_mean:.4f} ± {cd_std:.4f}  ({'PASS' if ok_cd else 'FAIL'})")
    print(f"Cl amplitude = {cl_amp:.4f}")
    print(f"|Cz|/Cd = {cz_frac:.2e}  ({'PASS' if ok_2d else 'FAIL'}: "
          "spanwise force must vanish at Re=100)")
    return 0 if ok else 1


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "auto"
    slv = sys.argv[2] if len(sys.argv) > 2 else "fused"
    sys.exit(main(dev, slv))
