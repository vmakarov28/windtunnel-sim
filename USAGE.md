# Using the wind tunnel — a practical guide

Everything here works from a fresh clone. No CFD background required:
you describe an experiment in physical units (meters, m/s), the code
does the rest — and refuses configurations it can't simulate honestly.

## 1. Setup

You need Python 3.10+ and (for real runs) an NVIDIA GPU with a CUDA
build of PyTorch. Everything also runs on CPU — fine for the small
`dev_smoke` scene and for units checks, slow for production grids.

```bash
git clone https://github.com/vmakarov28/windtunnel-sim
cd windtunnel-sim
pip install -e .          # pyyaml, numpy, torch, matplotlib, imageio-ffmpeg
pip install pytest wgpu   # optional: test suite + WebGPU shader checks
python -m pytest          # optional: ~100 tests should pass
```

Video assembly needs no separate ffmpeg install — a bundled one ships
via `imageio-ffmpeg`.

> **Windows + NVIDIA note:** if your CUDA torch lives in WSL2 (common on
> Windows 11), run the GPU commands through WSL with the venv's Python
> called by full path, e.g.
> `wsl -e bash -lc "cd /mnt/c/path/to/windtunnel-sim && ~/venvs/windtunnel/bin/python run.py ..."`.
> The Windows-side Python (CPU torch) is fine for everything else:
> dry runs, tests, video assembly.

## 2. Your first run (60 seconds)

Every experiment is a **scene** (a YAML file in `scenes/`) plus a
**seed** — a run is reproducible from those two things alone.

```bash
python run.py --list-scenes                          # what's available
python run.py --scene cylinder_re100 --seed 0 --steps 20000 --frame-every 100
```

PNG frames appear in `out/cylinder_re100-seed0/frames/` as the run goes
(open the folder and watch them arrive — a good health check). Then:

```bash
python scripts/make_video.py out/cylinder_re100-seed0
```

That's the Kármán vortex street — the classic shot.

## 3. The workflow: dry run first, then the real run

**Always dry-run a scene before burning GPU time.** With `--steps 0`
nothing is simulated; the units system just reports what the scene
resolves to:

```bash
python run.py --scene cylinder_re100 --seed 0 --steps 0
```

```
  physical   L = 0.05 m   U = 0.03 m/s   nu = 1.5e-05 m^2/s
  Reynolds   Re = 100
  lattice    N = 40 cells   u_lat = 0.06   ...
  relaxation tau = 0.572
  grid       1200 x 600 = 720,000 cells
  clock      ... steps per convective time L/U
```

Check three things: **Re** is what you intended, the **grid** fits your
GPU, and the **clock** line (steps per convective time L/U) tells you
how many steps you need — 20–40 convective times is a reasonable run.

Then the real run. The most useful flags:

| flag | what it does |
|---|---|
| `--steps N` | how long to run |
| `--frame-every N` | render every N steps (frame rate of your footage) |
| `--solver fused` | the fast Triton kernel (GPU only, ~20× the readable reference) |
| `--preset X` | rendering style — see §7 |
| `--zoom x0,y0,x1,y1` | camera crop, in characteristic lengths |
| `--upscale N` | sharper output frames |
| `--resume ckpt.pt` | continue from a checkpoint |
| `--device cpu` | force CPU (small scenes only) |

## 4. When the tunnel refuses to run

`units.resolve()` rejects untrustworthy configurations **on purpose**:

- `tau < 0.55` (without a turbulence model) — your Reynolds number is
  too high for this resolution. Fix: add `sgs: true` (turns on the
  Smagorinsky subgrid model, needed above Re of a few thousand) and/or
  raise `cells_per_char`.
- `u_lat > 0.1` — compressibility errors would pollute the physics.
  Fix: lower `u_lat` in the scene (high-Re scenes here use `0.05`,
  because separated flow can amplify local velocity ~3×).

Don't work around a refusal by tweaking constants — change the scene to
one the method can actually do. That rule is why the results validate.

## 5. Anatomy of a scene file

```yaml
name: my_experiment
physical:                   # the experiment, in real units
  char_length_m: 0.05       # characteristic length L (cylinder diameter / chord)
  velocity_ms: 0.03         # freestream U   ->  Re = U*L/nu
  nu_m2s: 1.5e-5            # kinematic viscosity (this is air)
lattice:
  cells_per_char: 40        # resolution: cells across L (the accuracy dial)
  u_lat: 0.06               # lattice speed (<= 0.1; 0.05 for high Re)
domain:
  length_chars: 30          # tunnel length, in units of L
  height_chars: 15          # keep obstacle/height blockage under ~8%
sgs: false                  # true = Smagorinsky LES (required at high Re)
boundaries:
  inlet: equilibrium_ramp   # velocity inlet, ramped from rest
  outlet: pressure_sponge   # absorbing outflow (no reflected waves)
  top_bottom: periodic
  ramp_steps: 2000
  sponge_fraction: 0.08
obstacle:                   # see §6
  type: cylinder
  center_x_chars: 8.0
  center_y_chars: 7.3       # slightly off-center: makes shedding start
                            # deterministically instead of waiting on roundoff
```

## 6. Custom shapes

### A cylinder

The `obstacle:` block above — diameter is always 1 characteristic
length (that's what `cells_per_char` resolves), position in units of L.

### The one-command way: generate a whole airfoil scene from numbers

**`scripts/make_airfoil_scene.py` is the easy path** — give it an airfoil,
an angle, and a Reynolds number and it writes a complete, units-checked
scene. It works out the wind speed from Re, turns the turbulence model on
only if the resolution needs it, generates the NACA geometry if you don't
already have it, and *refuses* physically un-simulable combinations the
same way a hand-written scene would.

```bash
# NACA 4412 at Re = 30,000, 6 deg angle of attack, dye footage
python scripts/make_airfoil_scene.py --naca 4412 --re 30000 --alpha 6 \
       --preset dye

# a symmetric section near stall, coarse and fast (fused solver)
python scripts/make_airfoil_scene.py --naca 0012 --re 8000 --alpha 14 \
       --chord-cells 200 --fast --name naca0012_stall

# an airfoil you downloaded — any Selig .dat (UIUC / airfoiltools.com)
python scripts/make_airfoil_scene.py --dat assets/ag16.dat --re 20000 \
       --alpha 3
```

It ends by printing the resolved unit report and the exact three commands
to run next (dry run → real run → assemble). The knobs:

| flag | meaning | default |
|---|---|---|
| `--naca CODE` **or** `--dat FILE` | geometry: a 4-digit code (generated) or a Selig file | one required |
| `--re` | Reynolds number — **this sets the wind speed** | required |
| `--alpha` | angle of attack [deg] | 4 |
| `--chord-cells` | cells across the chord — the resolution / accuracy dial | 256 |
| `--u-lat` | lattice speed, ≤ 0.1 | 0.05 |
| `--preset` | render style baked into the scene (see §8) | vorticity |
| `--sgs auto\|on\|off` | Smagorinsky LES; `auto` = on **iff** tau needs it | auto |
| `--fast` | omit the accurate keys so the fused solver can run it | off = accurate |
| `--domain L H`, `--center CX CY`, `--zoom …` | tunnel size, placement, camera crop (chords) | 8×5, 2.0 2.5, auto |
| `--nu`, `--chord-m`, `--name`, `--points`, `--force` | fluid, physical chord, filename, NACA density, overwrite | air, 0.1 m, auto |

Three things worth understanding:

- **Reynolds sets the speed, not you.** You give `--re`, and the tool
  computes `velocity_ms = Re · nu / chord`. That is *why* experiments here
  match Reynolds numbers instead of raw speeds — the project's first rule.
  Want a different Re? change one number.
- **`auto` SGS is honest.** The generator asks the units system whether the
  bare `tau` clears the plain-BGK floor (0.55). If yes, SGS stays off; if
  no, it enables Smagorinsky (floor 0.501) — and if even that fails, it
  refuses. You never set `tau` or guess whether turbulence modelling is
  needed.
- **Accurate by default, `--fast` for speed.** Without `--fast` the scene
  gets `collision: trt` + `curved_bc: true` and runs on the reference
  solver (true curved walls; see §7). `--fast` drops those so the fused
  Triton kernel (≈20× faster, staircase walls) can run it.

**When it refuses:** e.g. `tau = 0.5008 is below the SGS floor…` means the
Reynolds number is too high for the resolution. Raise `--chord-cells`, or
`--u-lat` (up to 0.1). Nothing is written on a refusal — that's the units
discipline, not a bug.

### Any airfoil (`.dat` file), by hand

If you'd rather write the scene yourself (or need a field the generator
doesn't expose), the obstacle block is just:

```yaml
obstacle:
  type: airfoil_dat
  dat_file: assets/naca4412.dat
  alpha_deg: 4.0            # angle of attack
  center_x_chars: 2.0       # leading edge region placement
  center_y_chars: 2.5
  edge_supersample: 4       # anti-staircase edge sampling — keep it
  curved_bc: true           # optional: Bouzidi true-curve walls (§7)
```

The loader reads **Selig format**: line 1 is the airfoil *name*, then
`x y` pairs (chord normalized 0→1) running trailing edge → upper
surface → leading edge → lower surface → trailing edge. Thousands of
real airfoils ship in this format (UIUC database / airfoiltools.com).

Two traps with downloaded files:

1. **Missing name line.** If line 1 is a coordinate, it gets consumed
   as the name and you silently lose a point — usually the trailing
   edge. Make sure line 1 is text.
2. **Too few points.** 30–40 points is common in the wild and too
   sparse for a 400-cell chord. Prefer 100+ points per surface.

### Or generate a NACA 4-digit section (no download)

```bash
python scripts/make_naca.py 4412        # -> assets/naca4412.dat
python scripts/make_naca.py 0012        # symmetric, any 4-digit code
```

Dense (120 points/surface), cosine-spaced, closed trailing edge, proper
name line — everything the rasterizer wants.

### Worked example: NACA 4412 at Re = 20,000, start to finish

```bash
# 1. generate the scene (geometry + units + accuracy keys, all at once)
python scripts/make_airfoil_scene.py --naca 4412 --re 20000 --alpha 4 \
       --chord-cells 400 --preset dye --name naca4412_re20k_demo

# 2. dry run — confirm Re = 20000, the grid, and tau before burning GPU
python run.py --scene naca4412_re20k_demo --seed 1 --steps 0

# 3. real run (GPU) — 60k steps of dye footage
python run.py --scene naca4412_re20k_demo --seed 1 --steps 60000 \
              --frame-every 100 --upscale 2

# 4. assemble
python scripts/make_video.py out/naca4412_re20k_demo-seed1
```

To sweep angle of attack, generate one scene per angle (they differ only
in `--alpha`), or see `scripts/airfoil_sweep.py` for a whole lift/drag
polar in one run. Honesty notes at this resolution: the lift *slope* is
trustworthy (the MH45 campaign landed within 8% of thin-airfoil theory),
drag reads high (a ~3-cell boundary layer; less so with `curved_bc`),
and near-stall angles are qualitative.

### Measuring lift and drag

Add `--measure-force` to any obstacle run and the tunnel reports the lift
and drag coefficients on the body, computed by **momentum exchange** —
it sums the momentum every population hands the wall as it bounces
(Kruger eq. 5.51), with no pressure integration. The x-component is drag,
the y-component is lift; both are divided by ½·ρ·U²·chord to get `Cd`,
`Cl`:

```bash
python run.py --scene naca4412_re20k_demo --seed 1 --steps 60000 \
              --measure-force
# ... writes out/<scene>-seed1/forces.csv (step,cd,cl) and prints:
#   Cl = +0.42 ± 0.03    Cd = +0.09 ± 0.00    L/D = +4.7
```

The mean is over the converged tail; the ± is the unsteady amplitude
(vortex-shedding buffet). The **trend** of `Cl` with angle and Reynolds
number is reliable; the absolute magnitude is approximate at these
resolutions (see the honesty notes above).

## 7. Accuracy mode (when you can wait longer)

Two opt-in scene keys buy measurably better physics at some speed cost
(they run in the readable reference solver — skip `--solver fused`):

```yaml
collision: trt          # top-level: two-relaxation-time collision
obstacle:
  curved_bc: true       # Bouzidi interpolated walls (cylinder/airfoil)
```

- **`collision: trt`** fixes a subtle BGK artifact: the effective wall
  position drifts with viscosity (0.25 cells at tau = 3). TRT adds a
  second relaxation rate solved from the derived magic parameter
  Λ = 3/16 and pins the wall (drift ÷ 17, measured:
  `python validation/accuracy_tau_sweep.py`).
- **`curved_bc: true`** replaces the staircase wall with interpolated
  bounce-back: every boundary link uses the true distance to the
  surface (analytic circle, or the same polygon the mask came from).
  Setting that distance to ½ reproduces the old rule exactly — the
  upgrade contains the original. This is the main answer to "the
  airfoil is made of steps."
- Both defaults stay off; all validation gates run the default path.
  Grid-convergence comparison: `python validation/cylinder_convergence.py`.
- Known limit (documented in notes/NOTES.md): in body-force-driven
  channels TRT retains a ~1% profile-amplitude artifact. Wind-tunnel
  scenes have no body force.

The other accuracy lever is still resolution: raise `cells_per_char`
and the errors shrink quadratically. "100% accurate" does not exist in
CFD — but every error term here now has a name, a measurement, and a
figure.

## 8. Rendering: the presets and how to choose

Every run writes PNG frames (headless — no window). What those frames
*show* is the preset. There are four:

| preset | the shot | colormap | how it's built |
|---|---|---|---|
| `vorticity` | red/blue spin field — the classic wake picture (**default**) | RdBu | computed from the velocity field at each frame |
| `speed` | velocity magnitude; vortex cores read as dark holes | magma | computed from the velocity field at each frame |
| `dye` | continuum dye/smoke advected from an upstream band — the prettiest | bone | a scalar advected **during the run** |
| `streaklines` | particle streaks, the wind-tunnel-photo look | afmhot | tracer particles advected **during the run** |

**The one thing that trips people up:** `vorticity` and `speed` are
*derived* from the flow — they can be produced from any saved state. But
`dye` and `streaklines` are *carried along as the simulation runs* (dye
is a scalar field advected every step; streaklines are hundreds of
thousands of particles). **You must pick them before the run** — you
can't re-render an old vorticity run as dye. If you want both looks, do
two runs (or one of each preset).

### Choosing a preset — two ways, with precedence

1. **Bake it into the scene** (reproducible default):
   ```yaml
   render:
     preset: dye
     zoom: [1.0, 1.0, 6.0, 4.0]     # camera crop, in chords
   ```
2. **Override at run time** (the flag wins over the scene):
   ```bash
   python run.py --scene my_airfoil --seed 1 --steps 60000 --preset speed
   ```

`--preset` on the command line always beats `render.preset` in the
scene; if neither is set, you get `vorticity`.

### The rest of the render controls

| flag | what it does |
|---|---|
| `--preset X` | style (above) |
| `--zoom x0,y0,x1,y1` | camera crop in characteristic lengths (chords/diameters) |
| `--upscale N` | integer pixel upscale — sharper output frames |
| `--frame-every N` | render cadence: one frame every N steps = your footage frame rate |
| `--tracers N` | particle count for `streaklines` (default 300k) |
| `--overlay-mlups` | burn a live MLUPS/step counter into the frame |
| `--measure-force` | also log lift & drag (see §6) — independent of the preset |

Then assemble the PNGs into an MP4 (bundled ffmpeg, no install):

```bash
python scripts/make_video.py out/<scene>-seed<seed>        # 60 fps H.264
```

### How to render vs how to *simulate* — the solver is separate

The preset is only how the flow is *drawn*; `--solver` is how it's
*computed*, and the two are independent:

- `--solver reference` (default) — the readable PyTorch solver; the only
  one that honours the accuracy keys `collision: trt` and `curved_bc`
  (§7). Use it for accurate airfoil walls.
- `--solver fused` — the Triton kernel, ≈20× faster, but BGK + staircase
  walls (it ignores the accuracy keys). Use it for big/long runs where
  speed matters more than the last percent.

Any preset works with either solver. A scene made with `--fast` has the
accuracy keys omitted, so `--solver fused` is the natural match; an
accurate scene runs on the default reference solver.

## 9. The 3D tunnel

Same pattern, separate program — scenes in `scenes3d/` additionally
need `span_chars` (spanwise depth in units of L):

```bash
python run3d.py --list-scenes
python run3d.py --scene cylinder_re300_modeA --seed 0 --steps 100000 \
                --solver fused --preset qcrit
```

Presets: `slice` (mid-span vorticity — directly comparable to 2D),
`three_pane` (adds a spanwise-velocity pane on an absolute scale — it
stays blank until the flow genuinely goes 3D), `qcrit` (Q-criterion
volume render — vortex cores and braids). Output in `out3d/`. The same
`obstacle:` blocks work; shapes are extruded across the span. Budget
note: 3D grids are big — dry-run and check the GB line first.

## 10. The browser toy

A qualitative, interactive version of the same kernel (no install):

```bash
cd web
python -m http.server 8099    # open http://localhost:8099 in Chrome/Edge 113+
```

Draw obstacles with the mouse (Shift or right-drag erases), sliders for
wind speed and viscosity (= Reynolds number), vorticity/speed fields,
tracers. It trades accuracy for interactivity — the validated numbers
come from `run.py`, the intuition comes from here.

## 11. Tips & troubleshooting

- **"UnitError: tau = ... < 0.55"** → see §4. The scene, not the code.
- **Run went unstable / NaN** → the run halts and saves state + seed +
  the last 120 frames to `failures/<timestamp>/`. That's diagnostic
  gold (and, in this project's history, usually a wrong scene: too
  little resolution for the Re, or `u_lat` too high).
- **Nothing sheds / wake is steady** → some scenes need asymmetry to
  start shedding on a clock (see the cylinder scene's 0.2 L offset);
  symmetric setups can sit on the steady branch for a long time.
- **Airfoil footage looks porous/leaky** → your `.dat` is sparse or
  headerless (§6). Regenerate or densify; `edge_supersample: 4` on.
- **Frames are slow to appear** → check you passed `--solver fused` and
  that torch sees your GPU (`python scripts/check_gpu.py`).
- **Reproducing a result** → same scene + same seed = same run. That's
  the contract; keep it by never editing a scene mid-experiment.
- **Long runs** → `--checkpoint-every 50000` and `--resume` make
  multi-hour runs interruption-proof.
