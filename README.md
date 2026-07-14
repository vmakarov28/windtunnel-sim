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

- `lbm/` — core solver (units, config; D2Q9 arrives in Phase 1)
- `scenes/` — experiment configs, physical-units-first
- `scripts/` — run/benchmark/render tooling
- `validation/` — benchmark gauntlet (Poiseuille, lid-driven cavity, cylinder)
- `notes/NOTES.md` — dev diary: every bug, instability, and diagnosis, dated
- `tests/` — `python -m pytest`

## Milestones

Each phase ends in a git tag; every tag stays runnable.

| tag | phase |
|---|---|
| v0.0-scaffold | Phase 0 — scaffold + units discipline |
