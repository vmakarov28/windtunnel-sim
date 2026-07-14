// WebGPU wind-tunnel toy — orchestration.
// Fixed 1024x512 D2Q9 grid, Smagorinsky always on so low-viscosity
// (high-Re) settings stay stable. Draw obstacles with the mouse.
//
// The physics is in shaders.js and mirrors the validated Python kernel;
// this file just wires buffers, pipelines, input, and the frame loop.

import {
  STEP_WGSL, RENDER_WGSL, TRACER_ADVECT_WGSL, TRACER_DRAW_WGSL,
} from "./shaders.js";

const NX = 1024, NY = 512, N = NX * NY;
const N_TRACERS = 30000;

const state = {
  uIn: 0.06,          // inlet speed (lattice units)
  uInTarget: 0.06,
  nu: 0.004,          // lattice viscosity -> omega below
  stepsPerFrame: 6,
  mode: 0,            // 0 vorticity, 1 speed
  tracers: true,
  brush: 16,
  paused: false,
  vortScale: 0.06,
  spongeFrac: 0.10,
  spongeStrength: 0.15,
  D: 70,              // nominal obstacle size for the Re readout
};

const omega = () => 1.0 / (3.0 * state.nu + 0.5);
const reynolds = () =>
  Math.round(state.uIn * state.D / state.nu);

function feqJS(q, rho, ux, uy) {
  const EX = [0, 1, -1, 0, 0, 1, -1, 1, -1];
  const EY = [0, 0, 0, 1, -1, 1, -1, -1, 1];
  const W = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
  const eu = EX[q] * ux + EY[q] * uy;
  const usq = ux * ux + uy * uy;
  return W[q] * rho * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * usq);
}

async function main() {
  const status = document.getElementById("status");
  if (!navigator.gpu) {
    status.textContent =
      "WebGPU not available. Use Chrome or Edge 113+ (or enable WebGPU).";
    status.className = "err";
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { status.textContent = "No WebGPU adapter."; return; }
  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (e) =>
    console.error("WebGPU error:", e.error.message));

  const canvas = document.getElementById("view");
  canvas.width = NX; canvas.height = NY;
  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  // -- buffers --------------------------------------------------------
  const bytesF = N * 9 * 4;
  const fA = device.createBuffer({ size: bytesF, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const fB = device.createBuffer({ size: bytesF, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const maskBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const macroBuf = device.createBuffer({ size: N * 2 * 4, usage: GPUBufferUsage.STORAGE });
  const posBuf = device.createBuffer({ size: N_TRACERS * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const params = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const maskArr = new Uint32Array(N);

  function initFields() {
    // f <- feq(1, (uIn, 0)); a touch of noise seeds shedding
    const f = new Float32Array(N * 9);
    for (let q = 0; q < 9; q++) {
      const base = q * N;
      for (let c = 0; c < N; c++) {
        const jitter = 1 + (Math.random() - 0.5) * 2e-3;
        f[base + c] = feqJS(q, 1.0, state.uIn * jitter, 0.0);
      }
    }
    device.queue.writeBuffer(fA, 0, f);
    device.queue.writeBuffer(fB, 0, f);
    const pos = new Float32Array(N_TRACERS * 2);
    for (let i = 0; i < N_TRACERS; i++) {
      pos[2 * i] = Math.random() * (NX - 2) + 1;
      pos[2 * i + 1] = Math.random() * (NY - 2) + 1;
    }
    device.queue.writeBuffer(posBuf, 0, pos);
  }

  function clearObstacles(seedCylinder) {
    maskArr.fill(0);
    if (seedCylinder) {
      const cx = NX * 0.22, cy = NY * 0.5, r = state.D / 2;
      for (let y = 0; y < NY; y++)
        for (let x = 0; x < NX; x++)
          if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) maskArr[y * NX + x] = 1;
    }
    device.queue.writeBuffer(maskBuf, 0, maskArr);
  }

  // -- pipelines ------------------------------------------------------
  const stepMod = device.createShaderModule({ code: STEP_WGSL });
  const stepPipe = device.createComputePipeline({ layout: "auto", compute: { module: stepMod, entryPoint: "main" } });
  const renderMod = device.createShaderModule({ code: RENDER_WGSL });
  const renderPipe = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: renderMod, entryPoint: "vs" },
    fragment: { module: renderMod, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
  const advMod = device.createShaderModule({ code: TRACER_ADVECT_WGSL });
  const advPipe = device.createComputePipeline({ layout: "auto", compute: { module: advMod, entryPoint: "main" } });
  const drawMod = device.createShaderModule({ code: TRACER_DRAW_WGSL });
  const drawPipe = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: drawMod, entryPoint: "vs" },
    fragment: {
      module: drawMod, entryPoint: "fs",
      targets: [{
        format,
        blend: {  // additive so overlapping tracers glow
          color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
        },
      }],
    },
    primitive: { topology: "point-list" },
  });

  // bind groups (two step groups ping-pong fin/fout)
  const stepAB = device.createBindGroup({ layout: stepPipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: fA } }, { binding: 1, resource: { buffer: fB } },
    { binding: 2, resource: { buffer: maskBuf } }, { binding: 3, resource: { buffer: macroBuf } },
    { binding: 4, resource: { buffer: params } }] });
  const stepBA = device.createBindGroup({ layout: stepPipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: fB } }, { binding: 1, resource: { buffer: fA } },
    { binding: 2, resource: { buffer: maskBuf } }, { binding: 3, resource: { buffer: macroBuf } },
    { binding: 4, resource: { buffer: params } }] });
  const renderBG = device.createBindGroup({ layout: renderPipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: macroBuf } }, { binding: 1, resource: { buffer: maskBuf } },
    { binding: 2, resource: { buffer: params } }] });
  const advBG = device.createBindGroup({ layout: advPipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: posBuf } }, { binding: 1, resource: { buffer: macroBuf } },
    { binding: 2, resource: { buffer: maskBuf } }, { binding: 3, resource: { buffer: params } }] });
  const drawBG = device.createBindGroup({ layout: drawPipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: posBuf } }, { binding: 1, resource: { buffer: params } }] });

  const pbuf = new ArrayBuffer(48);
  const pu32 = new Uint32Array(pbuf), pf32 = new Float32Array(pbuf);
  function writeParams() {
    const spongeStart = Math.floor(NX * (1 - state.spongeFrac));
    pu32[0] = NX; pu32[1] = NY;
    pf32[2] = omega(); pf32[3] = state.uIn; pf32[4] = 0.0225; // Cs^2 = 0.15^2
    pu32[5] = spongeStart; pf32[6] = state.spongeStrength;
    pf32[7] = state.vortScale; pu32[8] = state.mode; pu32[9] = N_TRACERS;
    pf32[10] = (frame % 4096); pf32[11] = 0;
    device.queue.writeBuffer(params, 0, pbuf);
  }

  // -- interaction ----------------------------------------------------
  let painting = false, erase = false;
  function paintAt(ev) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((ev.clientX - rect.left) / rect.width * NX);
    const y = Math.floor((ev.clientY - rect.top) / rect.height * NY);
    const b = state.brush;
    for (let dy = -b; dy <= b; dy++)
      for (let dx = -b; dx <= b; dx++) {
        if (dx * dx + dy * dy > b * b) continue;
        const xx = x + dx, yy = y + dy;
        if (xx < 2 || xx >= NX - 2 || yy < 0 || yy >= NY) continue;
        maskArr[yy * NX + xx] = erase ? 0 : 1;
      }
    device.queue.writeBuffer(maskBuf, 0, maskArr);
  }
  canvas.addEventListener("pointerdown", (e) => {
    painting = true; erase = e.button === 2 || e.shiftKey; paintAt(e);
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => { if (painting) paintAt(e); });
  canvas.addEventListener("pointerup", () => { painting = false; });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  // controls
  const bind = (id, fn) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("input", () => fn(el));
    return el;
  };
  bind("speed", (el) => { state.uInTarget = +el.value; });
  bind("visc", (el) => { state.nu = +el.value; updateReadout(); });
  bind("brush", (el) => { state.brush = +el.value; });
  bind("mode", (el) => { state.mode = el.value === "speed" ? 1 : 0; });
  bind("tracers", (el) => { state.tracers = el.checked; });
  document.getElementById("clear").addEventListener("click", () => clearObstacles(false));
  document.getElementById("reset").addEventListener("click", () => { clearObstacles(true); initFields(); });
  const pauseBtn = document.getElementById("pause");
  pauseBtn.addEventListener("click", () => {
    state.paused = !state.paused; pauseBtn.textContent = state.paused ? "Play" : "Pause";
    if (!state.paused) requestAnimationFrame(loop);
  });

  function updateReadout() {
    const r = document.getElementById("readout");
    if (r) r.textContent =
      `Re ≈ ${reynolds()}   ·   u = ${state.uIn.toFixed(3)}   ·   ν = ${state.nu.toFixed(4)}   ·   τ = ${(1 / omega()).toFixed(3)}`;
  }

  // -- frame loop -----------------------------------------------------
  let frame = 0, stepCount = 0;
  function loop() {
    // gentle ramp of inlet speed toward the slider target
    state.uIn += (state.uInTarget - state.uIn) * 0.02;
    writeParams();

    const enc = device.createCommandEncoder();
    for (let s = 0; s < state.stepsPerFrame; s++) {
      const cp = enc.beginComputePass();
      cp.setPipeline(stepPipe);
      // parity from a PERSISTENT counter so the ping-pong stays in sync
      // across frames (a per-frame index desyncs on odd frames).
      cp.setBindGroup(0, stepCount % 2 === 0 ? stepAB : stepBA);
      cp.dispatchWorkgroups(Math.ceil(NX / 8), Math.ceil(NY / 8));
      cp.end();
      stepCount++;
    }
    // macroBuf is rewritten every step, so the render/tracer passes below
    // always read the freshest velocity regardless of ping-pong parity.
    if (state.tracers) {
      const cp = enc.beginComputePass();
      cp.setPipeline(advPipe); cp.setBindGroup(0, advBG);
      cp.dispatchWorkgroups(Math.ceil(N_TRACERS / 64));
      cp.end();
    }

    const tex = ctx.getCurrentTexture().createView();
    const rp = enc.beginRenderPass({ colorAttachments: [{
      view: tex, clearValue: { r: 0.97, g: 0.97, b: 0.97, a: 1 },
      loadOp: "clear", storeOp: "store" }] });
    rp.setPipeline(renderPipe); rp.setBindGroup(0, renderBG); rp.draw(3);
    if (state.tracers) { rp.setPipeline(drawPipe); rp.setBindGroup(0, drawBG); rp.draw(N_TRACERS); }
    rp.end();

    device.queue.submit([enc.finish()]);
    frame++;
    if (frame % 15 === 0) updateReadout();
    if (!state.paused) requestAnimationFrame(loop);
  }

  // stepsPerFrame must be even so the ping-pong ends back on fA (which the
  // render + tracer bind groups read via macroBuf, written every step).
  state.stepsPerFrame = 6;
  clearObstacles(true);
  initFields();
  updateReadout();
  status.textContent = `running · ${NX}×${NY} · ${adapter.info?.description || "WebGPU"}`;
  requestAnimationFrame(loop);
}

main();
