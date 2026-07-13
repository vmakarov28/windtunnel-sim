# Project conventions (read before touching anything)

3D GPU lattice-Boltzmann wind tunnel (D3Q19) for a YouTube dev-log episode.
Built in phases (0–7); STOP for the user's approval after each phase.
Target machine: RTX 5080 16 GB (Blackwell, sm_120), WSL2 Ubuntu on Win 11.
(Was 2D/D2Q9 through Phase 0; user pivoted to 3D — see NOTES.md 2026-07-13.
Phase 7 browser toy, if reached, stays 2D.)

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
  (Poiseuille < 1% L2, Ghia cavity < 3% on the mid-plane of the
  spanwise-periodic cavity, cylinder St/Cd bands) re-pass.
- VRAM discipline: scenes must keep the fp32 A-B population working set
  (152 B/cell) <= 12 GB (tested in tests/test_scenes.py). Phase 4 ceiling
  on this card is ~6.3 GLUPS; target >= 2 GLUPS on the cylinder grid.
- Core kernels stay readable and explainable on camera: physics comments
  with equation references. Keep a readable reference implementation next
  to any clever optimization.
- Work only in this repo; commit everything to the public GitHub repo
  `windtunnel-sim`.

## Phase status

- Phase 0 (scaffold + units): done — tag `v0.0-scaffold` (2D at the time).
- Phase 0.5 (3D pivot, user-directed): done — scenes/budgets re-derived.
- Phase 1 (D3Q19 PyTorch core): approved, in progress.
- Phases 2–7: validation gauntlet → cinematography → fused-kernel port →
  Smagorinsky SGS → MH45 spanwise-periodic section sweep vs XFOIL →
  (stretch) WebGPU toy (2D).

## Practicalities

- Tests: `python -m pytest` (Windows Python 3.13 works for pure-Python
  phases; GPU phases run under WSL2 — the repo is reachable there at
  `/mnt/c/Users/aipla/Desktop/windtunnel`).
- fp32 everywhere on the GPU.
