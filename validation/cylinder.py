#!/usr/bin/env python3
"""Phase 2 gate #3: cylinder at Re = 100 — Strouhal number and drag.

Forces via the momentum-exchange method over boundary links (born here,
reused for the airfoil in Phase 6). Cl(t) from >= 10 shedding cycles
after the transient; St from the Cl FFT peak, Cd averaged.

PASS: St in [0.155, 0.175] and mean Cd in [1.25, 1.45]
(2D cylinder at Re=100: Williamson 1996 puts St ~ 0.164; 2D simulations
cluster Cd ~ 1.33-1.40 at ~5-7% blockage).
If out of band: investigate blockage ratio and diameter resolution FIRST.
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

from lbm.config import load_scene
from lbm.solver import Solver

OUT = Path(__file__).resolve().parent

# First attempt used a 20k-step transient and FAILED: St = 0.148 with Cl
# amplitude 0.116 (literature: ~0.33) — the street was still GROWING and
# the growth phase oscillates below the saturated frequency. Phase 1's
# 80k-step run shows saturation near t* ~ 90-100 (~65k steps). Diagnosis
# in NOTES.md 2026-07-13; the fix is measurement protocol, not constants.
TRANSIENT_STEPS = 80_000     # past saturation (~t* = 120)
MEASURE_STEPS = 40_000       # ~10 shedding cycles of measurement


def main(device: str = "auto") -> int:
    scene = load_scene("cylinder_re100")
    s = Solver.from_scene(scene, seed=0, device=device)
    d = scene.units.cells                 # diameter [cells]
    u = scene.units.u_lat
    q_dyn = 0.5 * u * u * d               # 0.5 rho0 U^2 D, rho0 = 1
    print(f"Cylinder: {scene.nx}x{scene.ny}, D={d:g}, tau={scene.units.tau},"
          f" u={u}, device={s.device}")

    for _ in range(TRANSIENT_STEPS):
        s.step()
    print(f"transient done ({TRANSIENT_STEPS} steps)")

    s.measure_force = True
    hist = torch.empty((MEASURE_STEPS, 2), dtype=torch.float32,
                       device=s.device)
    for i in range(MEASURE_STEPS):
        s.step()
        hist[i] = s.last_force        # device-side copy: no per-step sync
    forces = hist.cpu().numpy().astype(np.float64) / q_dyn
    cd, cl = forces[:, 0], forces[:, 1]
    g = s.check_guards()
    print(f"measurement done (u_max={g['u_max']:.3f})")

    # Strouhal from the Cl spectrum: St = f D / U.
    win = np.hanning(len(cl))
    spec = np.abs(np.fft.rfft((cl - cl.mean()) * win))
    freqs = np.fft.rfftfreq(len(cl), d=1.0)
    peak = spec[1:].argmax() + 1
    st = freqs[peak] * d / u
    cd_mean, cd_std = cd.mean(), cd.std()
    cl_amp = np.sqrt(2.0) * (cl - cl.mean()).std()

    ok_st = 0.155 <= st <= 0.175
    ok_cd = 1.25 <= cd_mean <= 1.45
    ok = ok_st and ok_cd

    t = np.arange(len(cl)) * u / d        # time in convective units
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4),
                                 gridspec_kw={"width_ratios": [3, 2]})
    a1.plot(t, cd, lw=0.8, label=f"Cd (mean {cd_mean:.3f})")
    a1.plot(t, cl, lw=0.8, label=f"Cl (amp {cl_amp:.3f})")
    a1.set_xlabel("t U / D"); a1.legend(frameon=False, fontsize=9)
    a1.set_title("force coefficients, momentum exchange")
    sts = freqs * d / u
    m = (sts > 0.05) & (sts < 0.5)
    a2.semilogy(sts[m], spec[m], lw=1.0)
    a2.axvline(st, color="#d62728", lw=1, ls="--")
    a2.axvspan(0.155, 0.175, color="green", alpha=0.12)
    a2.set_xlabel("St = f D / U"); a2.set_title(f"Cl spectrum: St = {st:.4f}")
    fig.suptitle(
        f"Cylinder Re=100 — St={st:.4f} "
        f"[{'PASS' if ok_st else 'FAIL'} 0.155-0.175], "
        f"Cd={cd_mean:.3f}±{cd_std:.3f} "
        f"[{'PASS' if ok_cd else 'FAIL'} 1.25-1.45]")
    fig.tight_layout()
    fig.savefig(OUT / "cylinder.png", dpi=150)

    print(f"St = {st:.4f}  ({'PASS' if ok_st else 'FAIL'})")
    print(f"Cd = {cd_mean:.4f} ± {cd_std:.4f}  ({'PASS' if ok_cd else 'FAIL'})")
    print(f"Cl amplitude = {cl_amp:.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "auto"
    sys.exit(main(dev))
