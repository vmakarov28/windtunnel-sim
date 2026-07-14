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

## How it was verified (no browser required)

The toy is checked headlessly with the same Naga validator browsers use,
via `wgpu-py` (`pip install wgpu`):

- `node dump_wgsl.mjs` → writes the assembled shaders to `_build/*.wgsl`
- `python validate_wgsl.py` → compiles all four shaders (catches e.g.
  reserved-keyword and type errors)
- `python validate_step.py` → **runs** the step kernel on the GPU and
  asserts the same physics the Python solver is gated on: no NaN, an empty
  tunnel holds freestream, and an obstacle accelerates the flow (~1.4×) and
  sheds a wake.

## Deploy (static page)

Any static host works. This repo ships a GitHub Pages workflow
(`.github/workflows/deploy-web.yml`) that publishes `web/` on push to
`main` — enable Pages → "GitHub Actions" in the repo settings and it goes
live at `https://vmakarov28.github.io/windtunnel-sim/`.
