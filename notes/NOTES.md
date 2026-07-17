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

---

## 2026-07-14 — Phase 5, first blood: Zou-He eats itself at Re = 10k

The very first turbulence run (cylinder Re = 10,000, tau = 0.5018) died
at **step 200** — runaway velocity 0.488 while the ramped inlet was
still at 0.024. The failure-capture path did its job: checkpoint, seed,
and metadata landed in failures/20260714-001740/ before the halt.

Autopsy from the captured state:
- The runaway cell is at **x = 1** — the column NEXT TO THE INLET, 480
  cells upstream of the cylinder. Innocent bystander physics.
- The velocity along that column alternates cell-to-cell (0.32, 0.16,
  0.50, 0.29, 0.15, 0.52...) — a staggered, grid-scale mode.
- Reference and fused solvers reproduce the blowup IDENTICALLY
  (u_max trajectories match to three digits) — not a kernel bug, a
  boundary-condition property.

Diagnosis: the known low-viscosity fragility of Zou-He. The
reconstruction satisfies mass and momentum at the wall but leaves a
non-hydrodynamic (staggered) component in the boundary populations.
BGK damps that mode at a rate ~ set by the viscosity: at Re = 100
(nu = 0.024) it dies instantly; at Re = 10k (nu = 6e-4, omega = 1.993)
it is essentially undamped, feeds back through the reconstruction each
step, and grows from the 1e-4 seed noise to blowup in 200 steps.

Fix: **regularized boundary conditions** (Latt & Chopard 2008) — keep
the Zou-He mass/momentum solve, then rebuild the whole boundary column
as f = feq(rho_b, u_b) + w_q (9/2) Q_q : Pi_neq, projecting the
non-equilibrium onto its hydrodynamic part and filtering everything
else. Implemented in both solvers (in-register in the kernel).
Verified: Re = 10k now tracks the ramp smoothly to u_max ~ 2.1 u_in
(the starting vortex) with tau_eff engaged; the empty-tunnel regression
still holds to four decimals; fused-vs-reference equivalence and the
cylinder gate re-run follow (open-boundary changes always re-gate —
that is the rule this project bought on day one).

---

## 2026-07-14 — Phase 5, second blood: the local Mach ceiling

Regularized BCs fixed the inlet. The Re = 10k run then died at **step
25,500** — this time in the WAKE, u_max = 0.504. Not a boundary bug:
real physics running past the compressibility envelope.

The diagnosis is a units story, which is the best kind for this project.
The scene used u_lat = 0.1 (freestream Mach 0.17, nominally fine). But a
separated cylinder shear layer accelerates fluid locally to ~3-3.5x
freestream; 3.5 x 0.1 = 0.35 in lattice units -> local Mach ~ 0.6. LBM's
weakly-compressible error grows like Ma^2, and past Ma ~ 0.4 the scheme
simply isn't modeling the right equations anymore. The blowup was the
solver telling the truth about a scene that asked too much of it.

Fix, through the units triangle (NO magic constants): halve u_lat to
0.05 and recover the lost Reynolds number by raising resolution — Re10k
went D = 60 -> 80 cells (2.88M), Re50k went D = 180 -> 340 cells (52M,
3.7 GB, the biggest run of the project). Local Mach now peaks ~0.3.
Both ran the full 200k steps clean: Re10k saturated at u_max = 0.118,
Re50k at 0.178, mass drift < 2e-4, no guard trips.

Baked the lesson forward: the MH45 airfoil scene (Phase 6) also dropped
to u_lat = 0.05 — suction peaks on a lifting section earn the same
caution as a cylinder shoulder.

**Physics payoff.** Energy spectra of the wake (scripts/spectrum.py):
- Re = 10k tracks **k^-3 over ~1.5 decades** — the 2D enstrophy-cascade
  slope, exactly what 2D turbulence should show (Kraichnan 1967). NOT
  k^-5/3; we plotted that line too and labelled it "NOT expected here"
  so the honesty is on the figure itself.
- Re = 50k shows the same k^-3 trend, extended, with more populated
  high-k modes — the wake is visibly turbulent, vortices merging (the
  inverse cascade) while enstrophy fluxes to small scales.

**Gate that matters most:** the Re = 100 cylinder gate re-run with SGS
ENABLED still passes — St = 0.1667, Cd = 1.4385 (vs 1.4357 with SGS
off, a 0.2% shift). The model is near-inert where the grid resolves the
flow, exactly as designed: it doesn't touch the validated result, it
only wakes up where the flow is under-resolved. That is the whole
contract of an LES subgrid model and we can show it holds.

Honesty for the episode, stated plainly: this is 2D LES-flavored
plausibility, not DNS truth. 2D turbulence has the wrong cascade
direction for real 3D turbulence (energy goes UP-scale in 2D, down-scale
in 3D). We are demonstrating a stable, plausible, spectrally-sensible
high-Re flow — not claiming quantitative accuracy at Re = 50k. The 3D
branch is where quantitative high-Re lives.

Deliverable: out/clip_compare (Re=100 laminar street stacked over Re=50k
turbulent wake — same physical cadence, the visual payoff of the whole
phase), plus out/re10k/re10k.mp4 and out/re50k/re50k.mp4.

---

## 2026-07-14 — Phase 6: the airfoil experiment (MH45 at Re = 20,000)

The climax. MH45 (a reflex flying-wing section, 9.85% thick), chord = 400
cells, alpha 0-10 deg step 1, Re = 20k, Smagorinsky LES on the fused
kernel. Each point: 10 convective times of transient, then Cl/Cd averaged
by momentum exchange over 24 convective times (50-105 shedding periods
per point — far past the spec's >= 8). Blockage 2.5%. Full sweep ran
~11 x 6 min; the process is stateless per alpha and the CSV is written
incrementally, which paid off — a session teardown killed it at alpha 8
and the resume picked up cleanly at 9.

**Results (validation/mh45_polar.png, mh45_sweep.csv):**

| alpha | Cl | Cd |
|---|---|---|
| 0 | -0.00 | 0.031 |
| 2 | 0.12 | 0.039 |
| 4 | 0.47 | 0.059 |
| 6 | 0.66 | 0.082 |
| 8 | 0.68 | 0.076 |
| 10 | 0.83 | 0.097 |

**What went RIGHT (the honest win):** the pre-stall lift-curve slope is
**6.76 / rad** — within **8% of thin-airfoil theory's 2*pi = 6.28**, and
the reflex section's near-zero lift at alpha = 0 is exactly what an MH45
should do (the reflexed trailing edge trades zero-alpha lift for pitch
stability). The lift curve is clean and monotonic. If the user's XFOIL
polars land near 2*pi slope (they should, pre-stall), this is a WIN by
the project's stated criterion (slope within ~15%).

**What went as PREDICTED (the honest miss):** Cd is high — 0.031 at
alpha = 0, rising to ~0.097. A real airfoil at Re = 20k sits nearer
0.02-0.03. Two documented causes, both stated before we ever ran it:
(1) the staircased mask (validation/mh45_staircase.png shows the jagged
edge — a real video beat); (2) the boundary layer is
~ c/sqrt(Re) ~ 2.8 cells thick on a 400-cell chord, barely resolved, so
skin friction and separation are both mismodeled upward. Disagreement
WITH EXPLANATION, exactly as the brief demanded — not massaged toward a
prettier number.

**What we DON'T trust (stated plainly):** the stall behavior. Cl plateaus
softly around 6-8 deg then rises again to 10 — that is NOT a
quantitative stall prediction. 2D LES with no transition model has
different separation physics than XFOIL's e^N transition; the stall
angle is untrusted by design and the verdict table says so.

**Error bars are the physics, not the noise.** The Cl std-dev (0.14-0.30)
is comparable to the mean at low alpha. This is real: at Re = 20k the
wake sheds violently and the instantaneous lift genuinely swings that
much. The bars are oscillation amplitude, honestly plotted; the mean is
resolved over 50+ periods.

Verdict table: notes/phase6_verdict.md (auto-regenerates with the XFOIL
overlay the moment data/xfoil_mh45_re20k.csv appears — columns
alpha,cl,cd). Until then the XFOIL rows read PENDING, not a fake pass.

**Footage:** the staircase zoom (the honest mask); the Cl(alpha) /
Cd(alpha) polars with error bars; the alpha-6 vorticity + dye beauty
clips over the section (out/airfoil_a6_*); and, when the user's XFOIL
data arrives, the overlay reveal.

---

## 2026-07-14 — Phase 7: the WebGPU browser toy (approved)

The stretch goal, user-approved. A feature-frozen in-browser wind tunnel:
the same fp32 D2Q9 BGK + Smagorinsky kernel, ported to ONE WebGPU compute
shader (WGSL), fixed 1024x512, mouse-drawn obstacles, vorticity + tracer
rendering, static-page deploy. Lives in `web/`, links back to the repo.

**Design choices carried over from the hard-won Python lessons:**
- Equilibrium velocity inlet + anechoic sponge, NOT Zou-He. The Phase 5
  autopsy showed Zou-He grows a staggered mode at low viscosity; an
  equilibrium (Dirichlet) inlet has no non-equilibrium mode to grow, and
  the sponge pins the outlet so the domain can't pressurize (the Phase 1
  bug). Simplest robust choice for an interactive toy where the user can
  draw pathological geometry.
- Smagorinsky always on (Cs = 0.15), so cranking the viscosity slider
  down to high-Re stays stable instead of blowing up in someone's tab.
- A-B double buffer with a PERSISTENT ping-pong counter — a per-frame
  index desyncs the swap on odd frames (caught in review before it ran).

**Verification without a browser (the honest part).** The in-app browser
preview was unavailable this session, so I verified the toy the way the
rest of the project is verified — by running the real thing and checking
physics, headless, via wgpu-py (the SAME Naga validator Chrome ships):
1. All four shaders compile. This immediately caught a real bug: `macro`
   is a reserved word in WGSL — renamed to `vel`. A browser would have
   surfaced the same error later; the headless validator surfaced it in
   seconds.
2. The step kernel RUNS on the GPU and passes the same gates as
   tests/test_solver.py: empty tunnel holds freestream to 4 decimals
   (max|ux-U| = 0.0000, max|uy| = 0.0000), and a cylinder accelerates the
   flow to 1.4x freestream and sheds a wake. Scripts: web/validate_wgsl.py,
   web/validate_step.py.
3. Live run in a real WebGPU browser: the page reaches its "running ·
   1024×512 · WebGPU" state with a live-updating Reynolds readout
   (Re ≈ 1050, tau 0.512, updating every 15 frames). That only happens
   if the device initialized AND all three pipelines (compute step,
   vorticity render, tracer advect/draw) built without error AND the
   frame loop is executing — so the render/tracer passes are confirmed
   too, not just the step kernel. The one thing left: a pixel screenshot
   (the in-app browser can't read back a WebGPU canvas surface — capture
   times out; a cosmetic capture limitation, not an app fault). Colours
   and tracer density are the only unconfirmed details.

**Footage:** screen-capture of drawing an obstacle and watching the street
form live; the viscosity slider ramping from laminar to turbulent in real
time; a split of the browser toy next to the Python beauty clip of the
same flow — "the exact same kernel, one in CUDA, one in your tab."


---

## 2026-07-14 — 3D moves in: lbm3d/ beside lbm/, separate by design

Restructure (user-directed): the 3D program no longer lives on a branch.
`lbm3d/` + `scenes3d/` + `run3d.py` + `tests3d/` now sit beside the
finished 2D program on main — two versions that run and test separately
(`pytest tests` / `pytest tests3d`; 61 + 35 = 96 combined), never share
solver code, and each carry their own scene and renderer designs. The ONE
shared module is `lbm/units.py`: the Reynolds triangle is dimension-blind
by construction, and one source of truth for the physics rails beats two
copies that can drift. The `3d-d3q19` branch is now historical.

The move was also an upgrade pass. The 3D solver merged in the remaining
validated 2D features it lacked — moving lid (3D Couette linear profile
< 1%), wall_x for the cavity, and Smagorinsky (same Hou closed form; the
formula is dimension-independent, now with the 6-component Pi_neq;
verified inert on resolved Taylor-Green, active in shear). Review of the
branch code found and fixed two latent renderer bugs before they could
bite: the >16M-element CUDA quantile crash (the exact failure that killed
the first 2D Re=50k render — the FULL 3D cylinder is 36.5M cells, so it
WOULD have crashed on its first big run) and frame numbering that
restarted at zero across --resume (silent frame overwrites).

**How 3D is visualized** (lbm3d/render.py, all headless tensor ops):

1. `slice` — omega_z on the mid-span plane. The 2D-comparable read.
2. `three_pane` — the slice stacked over an x-z pane of SPANWISE velocity
   u_z at mid-height: an honest 3D-ness meter, because u_z is identically
   zero for 2D flow.
3. `qcrit` — Q-criterion (Q = (||Omega||^2 - ||S||^2)/2, vortex cores)
   as an emission-absorption volume projection along the span, colored by
   streamwise vorticity omega_x. Counter-rotating mode-A/B vortex pairs
   will render as red/blue braids; solids composite by opacity, so
   occlusion falls out of the math. Verified against analytic fields:
   rigid rotation gives Q = omega^2 exactly, pure shear gives Q = 0.

**A meter must not grade on a curve.** The first three_pane render showed
loud checkerboard "structure" in the u_z pane at Re = 100 — where the
flow must be 2D. Measured: max|u_z| = 1.9e-6, i.e. 0.002% of u_char,
uncorrelated fp32 roundoff — but the pane self-normalized to its own
99.5th percentile, amplifying numerical dust into fake physics. Fix: the
pane now uses an ABSOLUTE scale (full color at 0.15 u_char, the amplitude
of real mode-A structure), locked in by a regression test: a 2D flow must
render blank, injected mode-A-scale u_z must saturate. Same family as the
interior-guard lesson: a diagnostic that defines its own reference will
always show you something.

GPU confirmation: resumed the shed-street checkpoint; `qcrit` renders the
street's cores as discrete volumes (neutral-colored — omega_x ~ 0 because
the flow IS 2D, which is the correct reading), and the three_pane meter
reads blank below a healthy street. Applied the local-Mach lesson to the
3D airfoil scene while touching it (u_lat 0.1 -> 0.05, tau 0.5015).

**Footage:** the three_pane at Re = 100 (meter blank) held next to the
same view at higher Re once SGS runs land (meter alight) — the cleanest
possible on-camera explanation of what "the wake goes 3D" means; the
qcrit street; and the moment the omega_x coloring first shows braids.

---

## 2026-07-14 — 3D validated, and the wake goes 3D (v0.9-3d-validated)

**The 3D gauntlet passes — all three gates, first try** (the 2D program
paid the tuition; 3D got the education):

- Poiseuille3D: L2 = **0.117%** vs the analytic parabola, and the
  profile is spanwise-invariant to the BIT (std = 0.0).
- Ghia mid-plane on the spanwise-periodic cavity: **0.41% / 0.63%**
  (u / v centerlines), spanwise deviation 1.4e-6.
- Full-resolution cylinder (900x450x90 = 36.5M cells, fused kernel):
  **St = 0.1714, Cd = 1.443 ± 0.008** — inside the 2D bands, as they
  must be at Re = 100 — plus the new honesty gate: **|Cz|/Cd = 8e-6**.
  A 3D solver borrowing 2D reference data has to PROVE the flow stayed
  two-dimensional. It did, eight parts per million of spanwise force.
  (St runs a hair above the 2D program's 0.1667 — D = 30 vs 40 cells of
  resolution; both in band, both honest.)

**And then the reason the third dimension exists.** cylinder_re300_modeA:
Re = 300, 59M cells, 200k steps at **3.76 GLUPS sustained including
Q-criterion volume rendering** — the run this whole visualization design
was built for. Above Re ~ 190 the flat vortex street is unstable to
spanwise perturbation (Williamson's modes A and B), and it happened on
camera:

- the qcrit projection shows saturated red/blue streamwise-vorticity
  braids wrapped around every core — compare the same view at Re = 100,
  where the cores render colorless because omega_x ~ 0;
- measured from the final state: **max|u_z| = 60% of u_char** and
  p99 = 17%, versus 0.002% at Re = 100. The spanwise velocity grew by a
  factor of ~30,000. The three_pane meter (absolute scale, full color at
  15% u_char) goes from provably blank to saturated — the cleanest
  on-camera definition of "the wake went 3D" I can imagine.

Performance note for the record: the fused D3Q19 kernel benchmarks at
4.2 GLUPS on the 36.5M-cell grid (67% of the 6.3 GLUPS ceiling, 23x the
readable reference) and Smagorinsky costs ~1% (the constexpr-gated
Pi_neq pass re-reads L2-resident lines, as designed).

**Footage:** the Re=100-vs-Re=300 qcrit pair (colorless cores vs braids);
the three_pane meter waking up; the guards.csv tau_eff column quietly
rising as the wake transitions — the turbulence model reporting the
physics changing under it.
