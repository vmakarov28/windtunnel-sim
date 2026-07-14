# windtunnel-sim

A 2D GPU lattice-Boltzmann (D2Q9) wind tunnel, built from scratch for a
dev-log video. Two goals, both mandatory: physically credible results
(validated against Poiseuille/Ghia/cylinder benchmarks and XFOIL/OpenFOAM
data) and constant visual output.

(A 3D D3Q19 version lives on the `3d-d3q19` branch — runnable, tested;
the 2D program ships first. Architecture ideas referenced from
[LB-t](https://github.com/2b-t/LB-t) by Tobit Flatscher, MIT license —
no code copied.)

## Running

Every experiment is a scene config plus a seed — nothing else:

```
python run.py --scene cylinder_re100 --seed 0
python run.py --list-scenes
```

Scenes are defined in **physical units** (meters, m/s, viscosity of air)
in `scenes/*.yaml` and converted to lattice units by `lbm/units.py`, which
refuses unstable (`tau < 0.55` without a turbulence model) or too-
compressible (`u_lat > 0.1`) configurations. Raw lattice parameters are
never set by hand.

## Layout

- `lbm/` — core: `units.py` (Reynolds triangle), `solver.py` (readable
  D2Q9 BGK reference), `fused.py` (Triton kernel), `cinema.py` (tracers/
  dye/camera), `airfoil.py` (Selig loader + rasterizer)
- `scenes/` — experiment configs, physical-units-first
- `scripts/` — run/benchmark/render/sweep tooling
- `validation/` — benchmark gauntlet (Poiseuille, lid-driven cavity,
  cylinder) + airfoil polars, all producing publication-style figures
- `notes/NOTES.md` — dev diary: every bug, instability, and diagnosis,
  dated (it doubles as the video script)
- `tests/` — `python -m pytest` (61 tests; CPU on Windows, GPU in WSL2)

## Milestones

Each phase ends in a git tag; every tag stays runnable (they get re-run
for b-roll).

| tag | phase | headline result |
|---|---|---|
| v0.0-scaffold | 0 — scaffold + units discipline | tau/u_lat guard rails refuse unstable scenes |
| v0.1-first-flow | 1 — D2Q9 core | Karman vortex street; Zou-He open BCs + anechoic sponge |
| v0.2-validated | 2 — validation gauntlet | Poiseuille 0.08%, Ghia 0.67%, cylinder St 0.167 / Cd 1.44 |
| v0.3-cinema | 3 — cinematography | tracers, streaklines, dye, camera |
| v0.4-fused | 4 — fused Triton kernel | 10.8 GLUPS, 81% of the bandwidth ceiling |
| v0.5-sgs | 5 — Smagorinsky LES | stable to Re=50k (52M cells), k⁻³ wake spectra |
| v0.6-airfoil | 6 — MH45 airfoil sweep | lift slope 6.76/rad (within 8% of 2π) |

Phase 7 (WebGPU browser toy) is a stretch goal, gated on separate
approval. A 3D D3Q19 implementation lives on the `3d-d3q19` branch and
resumes after the 2D program.
