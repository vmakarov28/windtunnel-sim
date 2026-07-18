# WebGPU Wind Tunnel (Phase 7 — the browser toy)

A feature-frozen, in-the-browser version of the wind tunnel: the same
fp32 **D2Q9 BGK + Smagorinsky** kernel from the Python program
(`lbm/fused.py`), ported to a single **WebGPU compute shader**. Fixed
1024×512 grid, **draw obstacles with the mouse**, vorticity + passive
tracer rendering. It links back to the dev-log; the quantitative
validation lives in the main project, not here.

## Run locally

WebGPU needs a secure context, so serve over http (not `file://`):

```
cd web
python -m http.server 8099
# open http://127.0.0.1:8099/ in Chrome or Edge 113+
```

## Files

- `index.html` — page, controls, styling
- `main.js` — WebGPU setup, buffers, ping-pong pipelines, input, frame loop
- `shaders.js` — the WGSL: `STEP` (the LBM kernel), `RENDER` (vorticity/
  speed), and the tracer advect/draw shaders. `STEP` mirrors the validated
  Python kernel line-for-line (same equilibrium, same Hou-1996 subgrid
  closure, same equations with equation references).

## Controls (for screen recording)

- **Draw obstacles**: click-drag on the canvas. **Erase**: hold Shift and
  drag, or drag with the right mouse button. Brush size slider below.
- **Wind speed** slider: inlet velocity (lattice units, capped at 0.1 —
  the same Mach guard the Python solver enforces).
- **Viscosity ν** slider: lower ν = higher Reynolds number = more
  turbulent. The readout above the canvas shows the live Re ≈ value.
- **Field**: Vorticity (red/blue diverging — the classic wake shot) or
  Speed (magnitude — vortex cores show as dark holes).
- **Passive tracers**: white specks advected by the flow.
- **Pause / Clear obstacles / Reset flow** buttons. Reset re-ramps the
  inlet, so restarts are clean footage.
- A starter cylinder is seeded on load so the page opens onto a vortex
  street; Clear obstacles gives you an empty tunnel (which stays calm —
  by design).

## How it is verified (no browser required)

Shaders are extracted **directly from `shaders.js`** by
`shader_source.py` (no Node, no build step — the single source of truth
is the file the browser runs) and checked with the same Naga validator
browsers use, via `wgpu-py` (`pip install wgpu`):

- `python validate_wgsl.py` → extracts and compiles all four shaders
  (catches e.g. reserved-keyword and type errors); fails loudly if
  extraction finds fewer than four.
- `python validate_step.py` → **runs** the step kernel on the GPU and
  asserts the same physics the Python solver is gated on: no NaN, an empty
  tunnel holds freestream, and an obstacle accelerates the flow (~1.4×) and
  sheds a wake.
- The same checks run as part of the main suite: `python -m pytest
  tests/test_web.py` (GPU tests skip gracefully on machines without an
  adapter). `dump_wgsl.mjs` remains as an optional JS-side debugging aid.

## Deploy (static page)

Any static host works. This repo ships a GitHub Pages workflow
(`.github/workflows/deploy-web.yml`) that publishes `web/` on push to
`main` — enable Pages → "GitHub Actions" in the repo settings and it goes
live at `https://vmakarov28.github.io/windtunnel-sim/`.
