# Dev diary — 2D GPU lattice-Boltzmann wind tunnel

Chronological log of decisions, bugs, wrong results, and their diagnoses.
This file is the video script's raw material — nothing gets deleted,
failures get documented with dates.

---

## 2026-07-13 — Phase 0: scaffold + units discipline

**What exists now:** repo skeleton (`lbm/`, `scenes/`, `scripts/`,
`validation/`, `notes/`, `tests/`), `run.py` entry point, and
`lbm/units.py` — the unit converter that everything else must pass through.

**Why units first (the on-camera explanation):** LBM works in lattice units
where dx = dt = 1, and the #1 way these simulations go silently wrong is a
bad conversion between "air at 3 m/s over a 10 cm chord" and "u = 0.1 over
400 cells". The two systems are linked by one dimensionless number — the
Reynolds number — so given the physical side (which fixes Re) you may pick
exactly two of {resolution N, lattice speed u_lat, relaxation time tau} and
the third is *solved*, never chosen. `units.resolve()` does that and
refuses anything unsafe:

- `tau < 0.55` refused (plain BGK becomes inaccurate, then unstable, as
  tau → 1/2). With the Phase 5 Smagorinsky model the floor drops to 0.501,
  because the model adds local eddy viscosity on top of the molecular tau.
- `u_lat > 0.1` refused (Ma ≈ 0.17; LBM is weakly compressible and the
  error grows like Ma², ≈ 3% at the ceiling).

**Decisions made:**

- Scene configs are YAML, physical-units-first; `lbm/config.py` is the only
  path from file → lattice parameters, and it goes through the guard rails.
  Grid sizes are derived (`domain in characteristic lengths` × `cells per
  characteristic length`), never written directly.
- Cylinder scene: D = 40 cells (spec minimum is 30) → grid 1200×600,
  blockage 6.7% (< 8%), tau = 0.572, u_lat = 0.06. Expected shedding period
  at St ≈ 0.166 is ≈ 4000 steps, so a 10-cycle Cd/St measurement will need
  ~50–100k steps after transient — noted for Phase 2 planning.
- Cavity: the canonical 256×256 / Re 100 setup for the Ghia et al. (1982)
  comparison; u_lat = 0.1 sits exactly at the compressibility cap, which is
  fine for a steady benchmark, and gives a comfortable tau = 1.268.
- Channel: H = 64 cells, tau = 0.8 (accuracy sweet spot), Re = 32; the body
  force will be derived from the target centerline velocity in Phase 2.

**First "failure" captured (by design):** the airfoil scene at Re = 20,000
(chord = 400 cells, u_lat = 0.1) resolves to tau = 0.506 and is **refused**
by the plain-BGK rail. The refusal message tells you the three ways out,
and the only physical one is the turbulence model. The scene file carries
`sgs: true` and loads — but this is the units discipline proving it earns
its keep before a single particle has moved: Re = 20k is simply not
reachable at this resolution without Phase 5.

**Footage from this phase:** the refusal itself. Terminal shot: resolve the
cylinder (green numbers appear — tau 0.572, dx 1.25 mm, dt 2.5 ms), then
ask for the airfoil *without* the turbulence model and get the red
`REFUSED: tau = 0.506 is below the plain-BGK floor…` with its suggested
fixes. Also the scene table (`notes/scene_table.md`) as a full-screen
graphic: four planned experiments, all defined in meters and m/s, lattice
numbers all derived.

