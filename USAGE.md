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

### Any airfoil (`.dat` file)

```yaml
obstacle:
  type: airfoil_dat
  dat_file: assets/naca4412.dat
  alpha_deg: 4.0            # angle of attack
  center_x_chars: 2.0       # leading edge region placement
  center_y_chars: 2.5
  edge_supersample: 4       # anti-staircase edge sampling — keep it
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
# 1. geometry
python scripts/make_naca.py 4412

# 2. scene: copy the shipped one and edit, or use it as-is
#    (scenes/naca4412_re20k.yaml — chord 400 cells, sgs: true, alpha 4 deg)

# 3. dry run — expect Re = 20000, tau = 0.503, grid 3200 x 2000
python run.py --scene naca4412_re20k --seed 1 --steps 0

# 4. real run (GPU) — 60k steps ~ 7 convective times of dye footage
python run.py --scene naca4412_re20k --seed 1 --steps 60000 \
              --solver fused --preset dye --frame-every 100 \
              --zoom 1.0,1.5,4.5,3.5 --upscale 2

# 5. assemble
python scripts/make_video.py out/naca4412_re20k-seed1
```

Change `alpha_deg` in the scene for other angles (or see
`scripts/airfoil_sweep.py` for a whole polar). Honesty notes at this
resolution: the lift *slope* is trustworthy (the MH45 campaign landed
within 8% of thin-airfoil theory), drag reads high (staircase edges +
a ~3-cell boundary layer), and near-stall angles are not to be trusted.

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

## 8. Rendering presets

| preset | the shot |
|---|---|
| `vorticity` | red/blue spin field — the classic wake picture (default) |
| `speed` | velocity magnitude, vortex cores as dark holes |
| `dye` | continuum dye advection — the prettiest one |
| `streaklines` | particle streaks, wind-tunnel-photo look |

All headless: frames are PNGs, no window needed.

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
