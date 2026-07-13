# Project conventions (read before touching anything)

2D GPU lattice-Boltzmann wind tunnel for a YouTube dev-log episode.
Built in phases (0–7); STOP for the user's approval after each phase.
Target machine: RTX 5080 16 GB (Blackwell, sm_120), WSL2 Ubuntu on Win 11.

## Non-negotiable rules

- **Units discipline is absolute.** All scenes defined in physical terms in
  `scenes/*.yaml` and resolved through `lbm/units.py` (refuses tau < 0.55
  pre-SGS, u_lat > 0.1). Raw tau / lattice velocity never set by hand.
  Match Reynolds numbers, never raw speeds.
- **No constant-fudging to match any reference, ever.** Failed validation =
  investigate cause (grid, tau, BC, blockage), document in `notes/NOTES.md`.
- Every run: `python run.py --scene <name> --seed <n>`; reproducible from
  config + seed alone.
- Git tag at every milestone (`v0.1-first-flow`, ...); every tag must stay
  runnable (used for b-roll re-runs).
- `notes/NOTES.md` is the dev diary: every bug, instability, wrong result,
  and its diagnosis, dated. It becomes the video script.
- NaN/instability = FOOTAGE: on detection save state checkpoint, seed, and
  last 120 rendered frames to `failures/<timestamp>/`, then halt. Never
  silently clamp or reset.
- Rendering is headless-first: PNG frames + ffmpeg. No interactive-window
  dependencies in the core path.
- Performance changes never merge until the Phase 2 validation gates
  (Poiseuille < 1% L2, Ghia cavity < 3%, cylinder St/Cd bands) re-pass.
- Core kernels stay readable and explainable on camera: physics comments
  with equation references. Keep a readable reference implementation next
  to any clever optimization.
- Work only in this repo; commit everything to the public GitHub repo
  `windtunnel-sim`.

## Phase status

- Phase 0 (scaffold + units): done — tag `v0.0-scaffold`.
- Phase 1 (D2Q9 PyTorch core): next, pending user approval.
- Phases 2–7: validation gauntlet → cinematography → fused-kernel port →
  Smagorinsky SGS → MH45 airfoil sweep vs XFOIL → (stretch) WebGPU toy.

## Practicalities

- Tests: `python -m pytest` (Windows Python 3.13 works for pure-Python
  phases; GPU phases run under WSL2 — the repo is reachable there at
  `/mnt/c/Users/aipla/Desktop/windtunnel`).
- fp32 everywhere on the GPU.
