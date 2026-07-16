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

## Layout: 2D and 3D live side by side, SEPARATE by design

The finished 2D program and the growing 3D program are independent
packages that run and test separately, each with its own scenes and its
own renderer ("custom designs"). Never fuse them into a dimension-generic
solver; never make one import the other's solver code.

| | 2D (D2Q9, complete) | 3D (D3Q19, active) |
|---|---|---|
| package | `lbm/` | `lbm3d/` |
| scenes | `scenes/` | `scenes3d/` (require `span_chars`) |
| entry | `run.py` | `run3d.py` |
| tests | `tests/` | `tests3d/` |
| output | `out/` | `out3d/` |
| renderer | cinema.py presets | slice / three_pane / qcrit |

The ONE shared module is `lbm/units.py` — the Reynolds triangle and its
guard rails are dimension-blind by construction, and a single source of
truth for the physics rails beats two copies that can drift. `pytest`
runs both suites; `pytest tests` or `pytest tests3d` runs one.
The `3d-d3q19` branch is HISTORICAL (pre-restructure) — 3D development
happens here on main in `lbm3d/`.

3D visualization design (lbm3d/render.py, all headless tensor ops):
- `slice`: omega_z mid-span — the 2D-comparable read.
- `three_pane`: slice + spanwise-velocity pane on an ABSOLUTE scale
  (full color = 0.15 u_char) — an honest 3D-ness meter: blank while the
  flow is 2D, alive with mode-A/B structure at higher Re.
- `qcrit`: Q-criterion emission-absorption volume projection along the
  span, colored by streamwise vorticity — the genuinely-3D shot.

## Phase status

- Phase 0 (scaffold + units): done — tag `v0.0-scaffold`.
- Phase 0.5 (3D pivot) + 0.75 (back to 2D): done — 3D lives on `3d-d3q19`.
- Phase 1 (D2Q9 core): done — tag `v0.1-first-flow`, vortex street video.
- Phase 2 (validation gauntlet): done — tag `v0.2-validated`. Poiseuille
  0.082%, Ghia 0.67%, cylinder St 0.1667 / Cd 1.436. Re-run these three
  (validation/*.py) after ANY change to the core or BCs — the sacred gate.
- Phase 3 (cinematography): done — tag `v0.3-cinema`. Tracers/streaklines/
  dye/presets/camera in lbm/cinema.py; three clips.
- Phase 4 (fused kernel): done — tag `v0.4-fused`. Triton, 10.8 GLUPS at
  2048x1024 (81% of the 13 GLUPS ceiling), gates re-passed. Correctness
  gate: scripts/check_fused.py.
- Phase 5 (Smagorinsky SGS): done — Re=10k/50k stable over 200k steps,
  k^-3 spectra, Re=100 gate still passes with SGS on. Two instabilities
  diagnosed + fixed (regularized BCs; local-Mach via units) — see NOTES.
- Phase 6 (MH45 airfoil sweep): done — tag `v0.6-airfoil`. Sweep 0-10 deg
  at Re=20k; lift slope 6.76/rad (within 8% of 2*pi), Cd high as predicted
  (staircase + ~3-cell BL), stall untrusted. Polars + verdict table +
  staircase figure + beauty clips. XFOIL overlay auto-appears when the
  user drops data/xfoil_mh45_re20k.csv (alpha,cl,cd) — review-time step.
- Phase 7 (WebGPU browser toy): done — tag `v0.7-webgpu`. Single WGSL
  compute shader (D2Q9 BGK+SGS) in `web/`, mouse-drawn obstacles,
  vorticity + tracers, GitHub Pages deploy workflow. Verified headless via
  wgpu-py (shaders compile; step kernel holds freestream + sheds a wake)
  AND runs live in a real WebGPU browser (all pipelines built, loop
  running). Only a pixel screenshot is unavailable (WebGPU canvas can't be
  captured by the in-app browser).

## 3D status (lbm3d/, tag v0.8-3d-core)

- D3Q19 core with every 2D lesson ported: anechoic sponge + equilibrium
  velocity inlet (GPU-confirmed shed street; the old BCs never shed),
  interior-only guards, Guo forcing (3D Poiseuille < 1% L2), moving lid
  (3D Couette < 1%), Smagorinsky (inert on resolved flow, activates in
  shear), momentum-exchange 3-vector force, offset-cylinder trigger,
  local-Mach discipline in the scenes. 35 tests in tests3d/.
- Next, in order: 3D validation gauntlet scripts + full-res GPU runs
  (spanwise-periodic Poiseuille/Ghia-midplane/cylinder St+Cd); Zou-He +
  regularized BCs for quantitative gates; fused D3Q19 Triton kernel
  (~6.3 GLUPS ceiling at 152 B/cell); SGS high-Re demos (mode-A/B in the
  qcrit view is the money shot); spanwise-periodic MH45 section.

## Practicalities

- Tests: `python -m pytest` (Windows Python 3.13 = CPU torch; GPU runs in
  WSL2: `wsl -e bash -lc "cd /mnt/c/Users/aipla/Desktop/windtunnel &&
  ~/venvs/windtunnel/bin/python ..."` — torch 2.11+cu128 sees sm_120).
- fp32 everywhere on the GPU.
