// Dev helper: write the assembled WGSL strings to _build/*.wgsl so a
// headless validator (naga via wgpu-py) can compile them exactly as the
// browser would. Not shipped; just a verification aid.
import { mkdirSync, writeFileSync } from "node:fs";
import {
  STEP_WGSL, RENDER_WGSL, TRACER_ADVECT_WGSL, TRACER_DRAW_WGSL,
} from "./shaders.js";

mkdirSync("_build", { recursive: true });
const out = {
  step: STEP_WGSL, render: RENDER_WGSL,
  tracer_advect: TRACER_ADVECT_WGSL, tracer_draw: TRACER_DRAW_WGSL,
};
for (const [name, src] of Object.entries(out)) {
  writeFileSync(`_build/${name}.wgsl`, src);
  console.log(`wrote _build/${name}.wgsl (${src.length} chars)`);
}
