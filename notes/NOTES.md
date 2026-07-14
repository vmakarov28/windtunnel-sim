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

---

## 2026-07-13 — Phase 0.75: back to 2D (user-directed); the 3D run's autopsy

Direction change: finish ALL phases in 2D first, with real footage at
every phase; the 3D D3Q19 work is preserved on the `3d-d3q19` branch
(fully runnable, tests passing, 700 frames of its one 35k-step run on
disk). Consulted Flatscher's LB-t repo (MIT) for architecture ideas —
composable collision/streaming/BC modules, A-A vs A-B streaming patterns
worth benchmarking in Phase 4. Ideas only; no code copied.

**The autopsy (best footage so far, three bugs deep):**

1. The 3D cylinder run "finished" 35k steps and produced a beautiful,
   PERFECTLY SYMMETRIC wake — the unstable steady branch (Fornberg's
   steady solutions), no vortex street. Worse, the guard log showed
   u_max = 0.0700 = exactly the inlet speed. Flow over a cylinder
   shoulder must reach ~1.4x freestream. Autopsy of the checkpoint: the
   shoulder peaked at 0.064 — BELOW freestream — and interior density
   sat at 1.059. Diagnosis: equilibrium inlet at FIXED rho=1 + copy
   outlet let the tunnel pressurize; it ran ~12% slow (effective
   Re ~ 90) with weakened perturbations. The guard hid it because the
   global u_max WAS the inlet plane. Guards now measure the interior
   only — a boundary condition must never grade itself.
2. First fix attempt (inlet density extrapolated from inside, outlet
   anchored at rho=1) failed the new regression test in the opposite
   direction: +55% acceleration down the tunnel. Nothing pinned the
   inlet density level — positive feedback. Lesson: extrapolation at an
   inflow is a feedback loop, not a boundary condition.
3. Proper Zou-He (1997) boundaries (velocity in, pressure out, both
   solved LOCALLY from the boundary column's own populations) — and the
   test STILL failed with the same profile. Profiling rho(x, t) revealed
   why: a standing acoustic wave, rho swinging 0.87 <-> 1.15 with a
   ~2500-step period. Ramping the inlet from rest is a piston stroke of
   amplitude u/c_s ~ 14%, and velocity-in/pressure-out is an acoustically
   closed resonator; bulk viscous damping of the fundamental takes tens
   of thousands of steps. The viscosity-ramp sponge did NOTHING for it —
   acoustics barely feel shear viscosity.

**The fix that finally holds:** an anechoic termination — the last 8% of
the tunnel blends f toward feq(rho=1, u_inlet(t)) with smoothly rising
strength (the numerical foam wedge). Empty-tunnel test: u = 0.0800,
rho = 1.0000 to four decimals by t = 2000, flat forever after. Locked in
as `test_open_boundaries_hold_freestream` — an empty wind tunnel must be
BORING, and now it provably is.

Also: the cylinder now sits 0.2 D below the centerline (Schaefer-Turek
style deliberate asymmetry) so shedding onsets deterministically instead
of waiting on roundoff to break a perfect mirror symmetry — the other
half of why the 3D run never shed.

**Footage:** the symmetric-wake 3D frame next to a (coming) 2D shedding
frame; the rho(x,t) slosh plot; the "empty tunnel is boring now" test.

---

## 2026-07-13 — Phase 1 complete: first flow (v0.1-first-flow)

The Karman vortex street exists: cylinder Re=100, 1200x600, 80k steps
(t* = 120 convective times), 2000 frames -> 33 s of 60 fps footage
(`out/cylinder_re100-seed0/vortex_street.mp4`). Timeline of the run:
symmetric twin shear layers first, sinuous waviness visible by t* ~ 60,
fully rolled-up street by t* ~ 90 — the instability's whole life story
in one clip. Vortices advect out through the anechoic sponge without
visible reflection (a faint seam at the sponge entrance is the one
honest artifact; it lives in the last non-physical 8%, crop it for
final footage).

PyTorch D2Q9 reference performance: **172 MLUPS sustained** on the RTX
5080 (720k cells, fp32, ~40 kernel launches/step). Phase 4's fused
kernel gets to chase the ~13 GLUPS bandwidth ceiling from here — a 75x
gap, which is the montage.

Deliverables: the video; checkpoint/restore round-trip test (bitwise,
`test_checkpoint_roundtrip_bitwise`); impulsive-start failure-reel run
(`--no-ramp`, out/cylinder_impulsive). Momentum-exchange force
measurement is implemented and unlocks the Phase 2 cylinder gate.

**Would-be footage from this phase:** the vortex street video itself;
the MLUPS counter climbing in the run log; side-by-side of the 3D
symmetric wake vs the 2D street (same physics, one missing trigger).

---

## 2026-07-13 — Phase 2: the validation gauntlet

**Gate 1, Poiseuille: PASS.** L2 = 0.082% vs the analytic parabola
(gate < 1%), converged to a bitwise-steady state. Halfway bounce-back +
Guo forcing behaving like the second-order methods they are.
Figure: validation/poiseuille.png.

**Gate 2, Ghia cavity: PASS.** Max deviation 0.44% (u-centerline),
0.67% (v-centerline) against Ghia, Ghia & Shin (1982), gate < 3%.
Two honest notes: (a) my from-memory transcription of Ghia Table II had
three phantom rows and drifted digits — caught by checking against a
published transcription before trusting the gate; reference data needs
provenance like everything else. (b) The 1e-8 convergence criterion was
never reached (stopped at the 600k-step cap, delta ~ 1e-5) — at tau =
1.268 the cavity's corner eddies creep for a long time; the centerline
profiles were long settled. Figure: validation/cavity.png.

**Gate 3, cylinder: FAILED FIRST, then diagnosed.** First run: St =
0.1481 (band 0.155-0.175), Cd = 1.251, Cl amplitude 0.116. That Cl
amplitude is the tell — literature says ~0.33 at Re = 100. A 20k-step
transient put the 45k-step measurement window inside the street's GROWTH
phase: a still-growing wake oscillates below the saturated limit-cycle
frequency and drags both St and mean Cd down. Phase 1's own footage
shows saturation near t* ~ 90-100. Fix: measure after an 80k-step
transient (t* = 120). No constants touched — the physics was fine; the
stopwatch was early.

**Gate 3, second run: PASS.** St = 0.1667 (Williamson's curve fit gives
~0.164 at Re = 100 — we land 1.6% over, consistent with confinement),
Cd = 1.436 ± 0.008 (high side of the band, as expected at 6.7% blockage:
confinement adds a few percent drag), Cl amplitude 0.361 — the saturated
limit cycle, confirming the first run's diagnosis. Momentum-exchange
forces are officially born and validated. Figure: validation/cylinder.png.

**Phase 2 verdict: all three gates pass. The tunnel is calibrated.**
Nothing downstream (performance, turbulence model, airfoil) starts from
an unvalidated core, and any future change must re-pass these three
scripts before it merges.

Bonus unblocked: MH45 coordinates (67-point Selig loop, "MH 45 9.85%")
transcribed into assets/mh45.dat from the UIUC Airfoil Data Site with
provenance in the header — the Phase 6 loader already passes its tests
against an analytic NACA section, welded-trailing-edge rasterization
included.

---

## 2026-07-13 — Phase 3: cinematography (v0.3-cinema)

Three clips delivered, all from the SAME physics checkpoint (the 80k-step
cylinder state — reproducibility doubles as a b-roll factory):

1. `out/cylinder_re100-seed0/vortex_street.mp4` — 33 s full-domain
   vorticity (from Phase 1).
2. `out/clip_streaklines/streaklines_closeup.mp4` — 8 s, 300k tracers,
   RK2 advection, 3x-upscaled zoom on the cylinder.
3. `out/clip_dye/dye_plume.mp4` — 10 s smoke-plume shot: semi-Lagrangian
   dye ribbon rolling up into the vortex cores.

Two cinematography bugs, both "the camera saw nothing" class:

- First streakline render was PURE BLACK. Tracer lifetime (1200 steps)
  was shorter than the transit from the inlet respawn band to the zoom
  region (u = 0.06 -> 72 cells of travel; camera at x = 200+). Particles
  died en route, forever. Lifetime now auto-scales to 2x domain transit.
  Locked in as a test: the buffer must light up in every quarter of the
  domain.
- First dye plume died 60 cells from the source: a 0.999/step decay
  e-folds in 1000 steps. Decay budget is now < 5% per domain transit
  (also a test).

One accidental aesthetic, kept: the streakline buffer renders
ink-on-paper rather than glow-on-dark — uniform freestream saturates the
accumulation buffer while the diverging wake thins it. It reads like a
schlieren etching and it is staying.

Also in this commit (Phase 4 groundwork, gates still to run): the fused
Triton collide+stream kernel (lbm/fused.py), correctness-gate and
benchmark scripts, --solver fused and --overlay-mlups in run.py.
Toolchain verified: Triton 3.6.0 compiles and runs on sm_120.

---

## 2026-07-14 — Phase 4: the fused kernel (v0.4-fused)

**The number: 10.8 GLUPS at 2048x1024 — 81% of the bandwidth ceiling,
12x the PyTorch reference on the same grid, 3.6x past the >= 3 GLUPS
target.** The ceiling math, stated up front and scored against: D2Q9
fp32 A-B is 72 B/cell/step of compulsory DRAM traffic; at ~960 GB/s
that is ~13.3 GLUPS. One kernel does pull-streaming + halfway
bounce-back + Zou-He + BGK/Smagorinsky collision + Guo forcing + the
anechoic sponge.

Honest wrinkle in the small-grid numbers: 512^2 to 1024^2 report
110-240% "of ceiling" — not a measurement error. Their whole working
set (18-72 MB) fits in the RTX 5080's 64 MB L2 cache, so DRAM stops
being the wall. The ceiling is a DRAM ceiling; score against it only
when the grid actually spills to DRAM (2048x1024 up: 76-81%).

Design story worth telling (three benchmark rounds):
1. Round 1: fused SLOWER than the reference at small grids (0.5x), 11%
   of ceiling at 4M cells. Cause: the two Zou-He boundary columns lived
   in PyTorch — ~50 kernel launches per step, plus `float(tensor)`
   sponge lookups forcing TWO GPU->CPU syncs per step (~2 ms on WSL2).
   The kernel itself took 0.19 ms; the step took 5.6 ms.
2. Round 2 (syncs removed, columns batched): 0.83 GLUPS. Better, still
   launch-bound.
3. Round 3: Zou-He moved INTO the kernel — the edge programs
   reconstruct their three unknown populations in-register; the whole
   step is ONE kernel launch. 10.8 GLUPS. Lesson for the episode:
   on WSL2 the enemy was never arithmetic, it was the CPU-GPU chatter.

Correctness held through all three rounds: max field drift vs the
readable reference 1.5e-5 (fp32, 10k steps; the ramp-window transient
of 8e-4 is the documented one-step phase offset between the storage
conventions and decays once the inlet ramp ends). All three Phase 2
gates re-passed on the fused solver — cylinder St = 0.1667 and
Cd = 1.4357, matching the reference to four digits.

Storage-convention note (the algebra that saved a refactor): the
reference stores post-boundary state and steps collide->stream->BC;
the fused kernel stores post-collision state and steps
stream->BC->collide. Since rho and u are collision invariants, the two
report IDENTICAL macroscopic fields step-for-step — no reordering of
the readable reference was ever needed, and the correctness gate
compares exactly the quantities physics cares about.

Also in this phase: Smagorinsky implemented in BOTH solvers (Hou et al.
1996 closed form from the local Pi_neq; Cs = 0 reduces exactly to BGK,
so the kernel has no turbulence branch), with tests: near-inert on
resolved laminar flow, activates in shear, stabilizes tau = 0.502.
Phase 5 inherits it ready-made.

**Footage:** the benchmark plot with the ceiling line; the three-round
overhead hunt as a whiteboard beat; the reference-vs-fused MLUPS
overlay clips (out/clip_overlay_ref, out/clip_overlay_fused).

