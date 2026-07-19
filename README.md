# windtunnel-sim

A GPU lattice-Boltzmann wind tunnel built from scratch for a dev-log
video, in two deliberately separate versions that run and test
independently:

- **2D (D2Q9)** — `lbm/` + `run.py` + `scenes/` + `tests/`. Complete:
  validated benchmarks, cinematography, a fused Triton kernel at 81% of
  the bandwidth ceiling, Smagorinsky LES to Re=50k, the MH45 airfoil
  sweep, and a WebGPU browser toy.
- **3D (D3Q19)** — `lbm3d/` + `run3d.py` + `scenes3d/` + `tests3d/`.
  Active: the core carries every lesson the 2D program paid for (see the
  autopsies in `notes/NOTES.md`), with its own visualization design.

Two goals, both mandatory: physically credible results (validated against
Poiseuille/Ghia/cylinder benchmarks and XFOIL/OpenFOAM data) and constant
visual output. Architecture ideas referenced from
[LB-t](https://github.com/2b-t/LB-t) by Tobit Flatscher (MIT) — no code
copied.

## Running

**New here? [USAGE.md](USAGE.md) is the practical guide** — setup, your
first run, custom shapes (including a worked NACA 4412 example with any
`.dat` airfoil or the `scripts/make_naca.py` generator), rendering
presets, the 3D tunnel, the browser toy, and troubleshooting.

Every experiment is a scene config plus a seed — nothing else:

```
python run.py   --scene cylinder_re100     --seed 0        # 2D
python run3d.py --scene cylinder_re100_dev --seed 0 \
                --steps 26000 --preset qcrit                # 3D
python run.py --list-scenes ; python run3d.py --list-scenes
```

Scenes are defined in **physical units** (meters, m/s, viscosity of air)
and converted to lattice units by `lbm/units.py` — the ONE module both
versions share, because the Reynolds triangle is dimension-blind. It
refuses unstable (`tau < 0.55` without a turbulence model) or too-
compressible (`u_lat > 0.1`) configurations; raw lattice parameters are
never set by hand.

3D rendering is headless tensor ops, no mesh/GL dependencies: `slice`
(mid-span vorticity), `three_pane` (slice + a spanwise-velocity pane on
an absolute scale — an honest "3D-ness meter" that reads blank while the
flow is two-dimensional), and `qcrit` (Q-criterion volume projection
colored by streamwise vorticity — the vortex-core shot).

## Layout

- `lbm/` — 2D core: `units.py` (shared Reynolds triangle), `solver.py`
  (readable D2Q9 BGK reference), `fused.py` (Triton kernel), `cinema.py`
  (tracers/dye/camera), `airfoil.py` (Selig loader + rasterizer)
- `lbm3d/` — 3D core: D3Q19 `solver.py` (BGK + Guo + Smagorinsky + moving
  lid + momentum exchange), `render.py` (slice / three_pane / qcrit)
- `scenes/`, `scenes3d/` — experiment configs, physical-units-first
- `scripts/` — run/benchmark/render/sweep tooling (2D and 3D)
- `validation/` — benchmark gauntlet + airfoil polars, publication figures
- `notes/NOTES.md` — dev diary: every bug, instability, and diagnosis,
  dated (it doubles as the video script)
- `tests/`, `tests3d/` — `python -m pytest` runs both (112 tests); each
  suite also runs alone

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
| v0.7-webgpu | 7 — WebGPU browser toy | the same kernel in one WGSL compute shader ([`web/`](web/)) |
| v0.10-accuracy | 8 — TRT + curved boundaries | wall drift ÷17 at τ=3 (Λ=3/16, derived); Bouzidi walls: s=½ ≡ the old rule, bit-tested |

The [`web/`](web/) toy runs the same D2Q9 BGK+Smagorinsky kernel in the
browser — draw obstacles with the mouse. A 3D D3Q19 implementation lives
on the `3d-d3q19` branch and resumes next.
