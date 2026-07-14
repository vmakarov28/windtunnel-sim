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

---

## 2026-07-13 — Phase 0.5: the 3D pivot

Decision (user): the tunnel goes 3D. Lattice becomes **D3Q19** (19
velocities; D3Q27's better rotational isotropy isn't worth 216 B/cell vs
152 B/cell on a 16 GB card at our Reynolds numbers).

**What survives untouched:** `lbm/units.py`, all 25 of its tests, and both
guard rails — the Reynolds triangle is dimension-blind, and c_s^2 = 1/3
for D3Q19 exactly as for D2Q9. The units-first bet paid off on day one.

**What the third dimension costs (the honest arithmetic):**

- Memory per cell: 72 -> 152 B (fp32 A-B double buffer), and every scene
  gains a span factor. The 16 GB card, not taste, now sizes every grid.
- Cylinder: D = 40 -> 30 cells (the spec floor), domain 30D x 15D with a
  3D periodic span -> 900 x 450 x 90 = 36.5M cells, 5.5 GB of populations.
  Physics justification for the short span: at Re = 100 the wake is
  two-dimensional (mode-A instability onsets near Re ~ 190), so the 2D
  St/Cd validation bands remain the honest reference.
- Airfoil: chord 400 -> 200 cells, spanwise-periodic section with span
  0.2c -> 1600 x 1000 x 40 = 64M cells, 9.7 GB. Consequence to state up
  front: BL thickness ~ c/sqrt(Re) ~ 1.4 cells (was ~2.8 in the 2D plan),
  so expect Cd overprediction to worsen; the Cl(alpha) slope comparison vs
  XFOIL stays the primary metric. Upside: a spanwise-periodic 3D section
  at Re = 20k is *more* physical than a strict 2D simulation, which
  over-organizes the separated shear layer.
- Cavity and channel become spanwise-periodic 3D (nz = 16). At Re = 100
  neither has a spanwise instability, so the Ghia tables and the analytic
  parabola remain valid references — and each doubles as a "3D code must
  reproduce 2D physics where physics IS 2D" test.
- Phase 4 ceiling moves: 152 B/cell/step at ~960 GB/s is a ~6.3 GLUPS
  roof (was ~13 for D2Q9). Target restated: >= 2 GLUPS sustained on the
  cylinder grid.
- Phase 7 browser toy, if reached, stays 2D — WebGPU + laptop GPUs.

**Renderer consequence:** vorticity becomes a tensor; Phase 1 renders the
omega_z mid-span slice (identical read to the 2D plan), and Phase 3 gains
the genuinely-3D shots (Q-criterion isosurfaces are now on the table).

---

## 2026-07-13 — Phase 1: D3Q19 core in PyTorch

Solver is live: BGK collision, pull streaming via `torch.roll`, halfway
bounce-back (Kruger eq. 5.26) over precomputed boundary-link indices,
ramped equilibrium inlet, zero-gradient outlet behind a cosine viscosity
sponge, guards (NaN -> failure capture, u_max, mass drift logged), bitwise
checkpoint/restore, and the omega_z mid-plane renderer.

**Physics gates that passed before any GPU touched this code:**

- Taylor-Green vortex decay matches the exact Navier-Stokes solution
  exp(-2 nu k^2 t) within 2% — this single test exercises collision,
  streaming, AND nu = (tau - 1/2)/3, i.e. the whole units story.
- Uniform flow is a fixed point of collide+stream to fp32 roundoff
  (the equilibrium moments are exact, so any drift = a bug).
- Halfway bounce-back conserves mass to < 1e-5 relative over 200 steps.
- Checkpoint -> restore -> continue is BITWISE identical to never
  stopping (CPU determinism; the b-roll re-run guarantee).

**Design choice worth explaining on camera:** bounce-back is not a field
operation but a gather over precomputed flat indices of boundary links
(fluid cells whose upstream neighbor is solid). For the cylinder that's
~10^4 links out of 36.5M cells — the boundary is a 2D skin on a 3D
volume, and treating it as such is both faster and closer to how the
Phase 4 kernel will think.

**Seeding:** the only stochastic element is a 1e-3 * u_lat velocity noise
at t=0 (breaks the wake's metastable symmetry so shedding onsets
reproducibly). Same seed -> same flow, bit for bit.

