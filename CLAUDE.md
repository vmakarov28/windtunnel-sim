# Project conventions (read before touching anything)

2D GPU lattice-Boltzmann wind tunnel (D2Q9) for a YouTube dev-log episode.
Built in phases (0–7); the user directed: finish ALL phases in 2D with
real rendered output at every phase. The 3D (D3Q19) implementation is
preserved, runnable, on the `3d-d3q19` branch — do not delete it; 3D
resumes after the 2D program completes.
Target machine: RTX 5080 16 GB (Blackwell, sm_120), WSL2 Ubuntu on Win 11.
Reference (ideas only, NO code copying): Flatscher's LB-t solver
(github.com/2b-t/LB-t, MIT) — modular collision/streaming/BC design,
A-A vs A-B streaming patterns for Phase 4.

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
- Rendering is headless-first: PNG frames + ffmpeg (bundled via
  imageio-ffmpeg; `scripts/make_video.py`). No interactive-window
  dependencies in the core path.
- Performance changes never merge until the Phase 2 validation gates
  (Poiseuille < 1% L2, Ghia cavity < 3%, cylinder St in [0.155, 0.175] and
  Cd in [1.25, 1.45]) re-pass.
- An empty wind tunnel must be boring: open-boundary changes must keep
  `test_open_boundaries_hold_freestream` passing (see NOTES.md 2026-07-13
  for the three-bug autopsy that bought this rule).
- Core kernels stay readable and explainable on camera: physics comments
  with equation references. Keep a readable reference implementation next
  to any clever optimization.
- Work only in this repo; commit everything to the public GitHub repo
  `windtunnel-sim`.

## Phase status

- Phase 0 (scaffold + units): done — tag `v0.0-scaffold`.
- Phase 0.5 (3D pivot) + 0.75 (back to 2D): done — 3D lives on `3d-d3q19`.
- Phase 1 (D2Q9 core): in progress — solver + Zou-He open BCs + anechoic
  sponge + Guo forcing + moving lid done, 48 tests; cylinder video next.
- Phases 2–7: validation gauntlet → cinematography → fused-kernel port
  (Phase 4 ceiling: 72 B/cell/step ≈ 13 GLUPS on this card; target
  >= 3 GLUPS at 2048x1024) → Smagorinsky SGS → MH45 sweep vs XFOIL →
  (stretch, separate approval) WebGPU browser toy.

## Practicalities

- Tests: `python -m pytest` (Windows Python 3.13 = CPU torch; GPU runs in
  WSL2: `wsl -e bash -lc "cd /mnt/c/Users/aipla/Desktop/windtunnel &&
  ~/venvs/windtunnel/bin/python ..."` — torch 2.11+cu128 sees sm_120).
- fp32 everywhere on the GPU.
