// WGSL shaders for the WebGPU wind-tunnel toy.
//
// This is a faithful port of the validated fp32 D2Q9 BGK + Smagorinsky
// kernel (lbm/fused.py in the main repo). Same lattice, same equilibrium,
// same subgrid closure, same equations — just running in the browser on
// a fixed grid with mouse-drawn obstacles. It is feature-frozen fun; the
// quantitative validation lives in the Python program.
//
// D2Q9 (Kruger et al. 2017, table 3.1):
//   directions e_q:  0:(0,0) 1:(1,0) 2:(-1,0) 3:(0,1) 4:(0,-1)
//                    5:(1,1) 6:(-1,-1) 7:(1,-1) 8:(-1,1)
//   weights:  w0 = 4/9, axis = 1/9, diagonal = 1/36
//   c_s^2 = 1/3
//
// Storage layout: population q at cell c=(y*nx+x) lives at f[q*N + c]
// (structure-of-arrays, so each direction's reads coalesce). A-B double
// buffer: read `fin`, write `fout`, swap each step.

// Shared prelude: constants + the D2Q9 tables. Prepended to every shader
// that needs the lattice so the numbers live in exactly one place.
export const LATTICE = /* wgsl */ `
struct Params {
  nx : u32,
  ny : u32,
  omega : f32,          // 1/tau (molecular relaxation rate)
  uIn : f32,            // inlet speed (lattice units), ramped on reset
  cs2s : f32,           // Cs^2 for Smagorinsky (0 => plain BGK, exactly)
  spongeStart : u32,    // x index where the anechoic sponge begins
  spongeStrength : f32, // peak blend rate in the sponge
  vortScale : f32,      // vorticity value mapped to full colour
  mode : u32,           // 0 = vorticity, 1 = speed
  nTracers : u32,
  seedT : f32,          // per-frame RNG salt for tracer respawns
  pad0 : f32,
};

const W0 : f32 = 4.0 / 9.0;
const WA : f32 = 1.0 / 9.0;
const WD : f32 = 1.0 / 36.0;

var<private> EX : array<i32, 9> = array<i32,9>(0, 1, -1, 0, 0, 1, -1, 1, -1);
var<private> EY : array<i32, 9> = array<i32,9>(0, 0, 0, 1, -1, 1, -1, -1, 1);
var<private> OPP: array<u32, 9> = array<u32,9>(0u, 2u, 1u, 4u, 3u, 6u, 5u, 8u, 7u);
var<private> WT : array<f32, 9> = array<f32,9>(W0, WA, WA, WA, WA, WD, WD, WD, WD);

// f_q^eq = w_q rho (1 + 3(e.u) + 9/2 (e.u)^2 - 3/2 u^2)   (Kruger eq. 3.54)
fn feq(q : u32, rho : f32, ux : f32, uy : f32, usq : f32) -> f32 {
  let eu = f32(EX[q]) * ux + f32(EY[q]) * uy;
  return WT[q] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq);
}
`;

// The LBM time step: pull-stream + halfway bounce-back + macroscopics +
// Smagorinsky effective relaxation + BGK collision + anechoic sponge.
export const STEP_WGSL = LATTICE + /* wgsl */ `
@group(0) @binding(0) var<storage, read>        fin   : array<f32>;
@group(0) @binding(1) var<storage, read_write>  fout  : array<f32>;
@group(0) @binding(2) var<storage, read>        mask  : array<u32>;
@group(0) @binding(3) var<storage, read_write>  vel : array<f32>; // 2/cell
@group(0) @binding(4) var<uniform>              P     : Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= P.nx || y >= P.ny) { return; }
  let N = P.nx * P.ny;
  let c = y * P.nx + x;

  // Solid cells carry no fluid and no velocity.
  if (mask[c] != 0u) {
    for (var q = 0u; q < 9u; q++) { fout[q * N + c] = 0.0; }
    vel[2u * c] = 0.0;
    vel[2u * c + 1u] = 0.0;
    return;
  }

  // 1. Pull streaming with halfway bounce-back (Kruger eq. 5.26).
  // f_q here came from cell x - e_q; if that neighbour is solid, the
  // population that left this cell toward the wall reflects back as the
  // opposite direction's post-collision value at THIS cell.
  var f : array<f32, 9>;
  for (var q = 0u; q < 9u; q++) {
    let sx = (i32(x) - EX[q] + i32(P.nx)) % i32(P.nx);
    let sy = (i32(y) - EY[q] + i32(P.ny)) % i32(P.ny);  // periodic top/bottom
    let src = u32(sy) * P.nx + u32(sx);
    if (mask[src] != 0u) {
      f[q] = fin[OPP[q] * N + c];
    } else {
      f[q] = fin[q * N + src];
    }
  }

  // 2. Macroscopics.
  var rho = 0.0;
  var mx = 0.0;
  var my = 0.0;
  for (var q = 0u; q < 9u; q++) {
    rho = rho + f[q];
    mx = mx + f32(EX[q]) * f[q];
    my = my + f32(EY[q]) * f[q];
  }
  let rhoInv = select(1.0, 1.0 / rho, rho > 1e-6);
  var ux = mx * rhoInv;
  var uy = my * rhoInv;
  let usq = ux * ux + uy * uy;

  // 3. Smagorinsky effective relaxation from the local non-equilibrium
  // momentum flux (Hou et al. 1996) — no finite differences needed.
  // cs2s = 0 collapses this to omg = omega exactly (plain BGK).
  var pxx = 0.0; var pyy = 0.0; var pxy = 0.0;
  for (var q = 0u; q < 9u; q++) {
    let fneq = f[q] - feq(q, rho, ux, uy, usq);
    pxx = pxx + f32(EX[q] * EX[q]) * fneq;
    pyy = pyy + f32(EY[q] * EY[q]) * fneq;
    pxy = pxy + f32(EX[q] * EY[q]) * fneq;
  }
  let qbar = sqrt(2.0 * (pxx * pxx + 2.0 * pxy * pxy + pyy * pyy));
  let tau0 = 1.0 / P.omega;
  let tauEff = 0.5 * (tau0 + sqrt(tau0 * tau0 + 18.0 * P.cs2s * qbar * rhoInv));
  let omg = 1.0 / tauEff;

  // 4. Equilibrium velocity inlet (Dirichlet) at x = 0: no non-equilibrium
  // mode to grow, so it stays stable at low viscosity without Zou-He.
  if (x == 0u) {
    rho = 1.0; ux = P.uIn; uy = 0.0;
    let u2 = ux * ux;
    for (var q = 0u; q < 9u; q++) { fout[q * N + c] = feq(q, rho, ux, uy, u2); }
    vel[2u * c] = ux; vel[2u * c + 1u] = 0.0;
    return;
  }

  // 5. BGK collision (Kruger eq. 3.9), then the anechoic sponge: the last
  // few percent of the tunnel blend toward the clean freestream so wakes
  // are absorbed instead of reflecting off the outlet (also pins the
  // outlet pressure, so the domain can't slowly pressurize).
  var sigma = 0.0;
  if (x >= P.spongeStart) {
    let s = f32(x - P.spongeStart) / f32(max(1u, P.nx - 1u - P.spongeStart));
    sigma = P.spongeStrength * s * s;
  }
  let u2in = P.uIn * P.uIn;
  for (var q = 0u; q < 9u; q++) {
    var o = f[q] - omg * (f[q] - feq(q, rho, ux, uy, usq));
    if (sigma > 0.0) {
      let tgt = feq(q, 1.0, P.uIn, 0.0, u2in);
      o = o + sigma * (tgt - o);
    }
    fout[q * N + c] = o;
  }
  vel[2u * c] = ux;
  vel[2u * c + 1u] = uy;
}
`;

// Fullscreen vorticity / speed renderer. Reads the vel (ux,uy) buffer,
// computes omega = dv/dx - du/dy by central differences, and maps it
// through an RdBu-style diverging colormap. Solids composite flat grey.
export const RENDER_WGSL = LATTICE + /* wgsl */ `
@group(0) @binding(0) var<storage, read> vel : array<f32>;
@group(0) @binding(1) var<storage, read> mask  : array<u32>;
@group(0) @binding(2) var<uniform>       P     : Params;

struct VSOut { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VSOut {
  // one big triangle covering the viewport
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o : VSOut;
  o.pos = vec4<f32>(p[vi], 0.0, 1.0);
  o.uv = 0.5 * (p[vi] + vec2<f32>(1.0, 1.0));
  return o;
}

fn cellAt(x : i32, y : i32) -> u32 {
  let cx = clamp(x, 0, i32(P.nx) - 1);
  let cy = clamp(y, 0, i32(P.ny) - 1);
  return u32(cy) * P.nx + u32(cx);
}

// diverging blue-white-red, echoing matplotlib RdBu_r
fn diverging(t : f32) -> vec3<f32> {
  let s = clamp(t, 0.0, 1.0);
  let blue = vec3<f32>(0.129, 0.400, 0.674);
  let white = vec3<f32>(0.969, 0.969, 0.969);
  let red = vec3<f32>(0.698, 0.094, 0.168);
  if (s < 0.5) { return mix(blue, white, s * 2.0); }
  return mix(white, red, (s - 0.5) * 2.0);
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let x = i32(in.uv.x * f32(P.nx));
  let y = i32((1.0 - in.uv.y) * f32(P.ny));  // flip so +y is up
  let c = cellAt(x, y);
  if (mask[c] != 0u) { return vec4<f32>(0.42, 0.42, 0.42, 1.0); }

  if (P.mode == 1u) {                        // speed magnitude
    let ux = vel[2u * c]; let uy = vel[2u * c + 1u];
    let sp = clamp(sqrt(ux * ux + uy * uy) / (1.8 * max(P.uIn, 1e-4)), 0.0, 1.0);
    // simple dark->bright ramp
    return vec4<f32>(sp * 0.9 + 0.05, sp * sp * 0.8 + 0.05, sp * 0.4 + 0.08, 1.0);
  }

  // vorticity by central differences of the velocity field
  let cxp = cellAt(x + 1, y); let cxm = cellAt(x - 1, y);
  let cyp = cellAt(x, y + 1); let cym = cellAt(x, y - 1);
  let dvdx = 0.5 * (vel[2u * cxp + 1u] - vel[2u * cxm + 1u]);
  let dudy = 0.5 * (vel[2u * cyp] - vel[2u * cym]);
  let omega = dvdx - dudy;
  return vec4<f32>(diverging(0.5 + 0.5 * omega / max(P.vortScale, 1e-6)), 1.0);
}
`;

// Passive tracers: advected by the velocity, respawned at the inlet
// when they leave the domain or enter a solid. Advection is a compute
// pass; drawing is a point-list render pass with additive blending.
export const TRACER_ADVECT_WGSL = LATTICE + /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> pos   : array<f32>; // 2/particle
@group(0) @binding(1) var<storage, read>       vel : array<f32>;
@group(0) @binding(2) var<storage, read>       mask  : array<u32>;
@group(0) @binding(3) var<uniform>             P     : Params;

fn hash(n : u32) -> f32 {
  var h = n * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  h = (h >> 22u) ^ h;
  return f32(h) / 4294967295.0;
}

fn sampleVel(px : f32, py : f32) -> vec2<f32> {
  // bilinear sample of the velocity field
  let fx = clamp(px, 0.0, f32(P.nx) - 1.001);
  let fy = clamp(py, 0.0, f32(P.ny) - 1.001);
  let x0 = u32(fx); let y0 = u32(fy);
  let x1 = x0 + 1u; let y1 = y0 + 1u;
  let tx = fx - f32(x0); let ty = fy - f32(y0);
  let c00 = y0 * P.nx + x0; let c10 = y0 * P.nx + x1;
  let c01 = y1 * P.nx + x0; let c11 = y1 * P.nx + x1;
  let ux = mix(mix(vel[2u*c00], vel[2u*c10], tx),
               mix(vel[2u*c01], vel[2u*c11], tx), ty);
  let uy = mix(mix(vel[2u*c00+1u], vel[2u*c10+1u], tx),
               mix(vel[2u*c01+1u], vel[2u*c11+1u], tx), ty);
  return vec2<f32>(ux, uy);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.nTracers) { return; }
  var px = pos[2u * i];
  var py = pos[2u * i + 1u];

  // RK2 midpoint advection, scaled up so motion is visible on screen
  let boost = 6.0;
  let v1 = sampleVel(px, py);
  let v2 = sampleVel(px + 0.5 * boost * v1.x, py + 0.5 * boost * v1.y);
  px = px + boost * v2.x;
  py = py + boost * v2.y;

  let cx = clamp(i32(px), 0, i32(P.nx) - 1);
  let cy = clamp(i32(py), 0, i32(P.ny) - 1);
  let solid = mask[u32(cy) * P.nx + u32(cx)] != 0u;
  if (px < 1.0 || px > f32(P.nx) - 2.0 || py < 0.0 || py > f32(P.ny) - 1.0 || solid) {
    px = 1.0 + hash(i * 3u + u32(P.seedT)) * 3.0;          // inlet band
    py = hash(i * 7u + u32(P.seedT) * 5u) * (f32(P.ny) - 2.0) + 1.0;
  }
  pos[2u * i] = px;
  pos[2u * i + 1u] = py;
}
`;

export const TRACER_DRAW_WGSL = LATTICE + /* wgsl */ `
@group(0) @binding(0) var<storage, read> pos : array<f32>;
@group(0) @binding(1) var<uniform>       P   : Params;

@vertex
fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
  let px = pos[2u * i];
  let py = pos[2u * i + 1u];
  let ndcX = px / f32(P.nx) * 2.0 - 1.0;
  let ndcY = (1.0 - py / f32(P.ny)) * 2.0 - 1.0;
  return vec4<f32>(ndcX, ndcY, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 1.0, 1.0, 0.10);  // faint, additive -> streak feel
}
`;
