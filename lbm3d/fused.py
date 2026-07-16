"""Fused D3Q19 collide+stream Triton kernel (A-B pattern) — the 3D speed.

One kernel launch per step: pull-streaming + halfway bounce-back (with
moving-lid momentum) + equilibrium inlet + BGK/Smagorinsky collision +
Guo forcing + the anechoic sponge. 152 B/cell/step of compulsory DRAM
traffic (19 fp32 populations read + written); at ~960 GB/s the ceiling is
~6.3 GLUPS on the RTX 5080.

Design differences from the 2D kernel (lbm/fused.py), both lessons from
that port:
- Directions live in tiny constant GPU tensors (EX/EY/EZ/W/OPP) and the
  kernel loops `tl.static_range(19)` — no 19-fold manual unrolling, same
  compile-time specialization.
- The kernel makes up to three passes over q (pull->macroscopics,
  re-pull->Pi_neq when SGS is on, re-pull->collide+store). Re-pulls hit
  L2 (the lines were just read), so DRAM traffic stays ~152 B/cell.
- ALL boundaries are in-kernel, branch-free selects: the 2D port measured
  ~2 ms/step of Python-side boundary cost before its boundaries moved
  in-kernel. Here there is no torch-side BC work at all.

Ordering note (same algebra as 2D): this solver stores the
POST-COLLISION state and steps stream->BC->collide; the readable
reference (lbm3d/solver.py) stores the post-BC state and steps
collide->stream->BC. Density and momentum are collision invariants, so
the two report identical macroscopic fields step-for-step (up to fp32
op-order) — which is exactly what scripts/check_fused3d.py gates on.

The readable reference is the spec; this file is the speed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .lattice import E, OPP, Q, W
from .solver import Solver

# Triton only sees constexpr globals inside @jit functions (plain
# Python ints are rejected by recent versions — same strictness class
# as WGSL reserving `macro`).
_NQ = tl.constexpr(Q)


@triton.jit
def _pull(A, MASK, LID, here, src, qN, oppN,
          bounce_add, m_l):
    """Pulled population for one direction with halfway bounce-back:
    if the pull-source is solid, take the reflected post-collision
    population at THIS cell (+ moving-wall momentum)."""
    ms = tl.load(MASK + src, mask=m_l, other=1)
    pulled = tl.load(A + qN + src, mask=m_l, other=0.0)
    refl = tl.load(A + oppN + here, mask=m_l, other=0.0)
    lid_src = tl.load(LID + src, mask=m_l, other=0)
    return tl.where(ms != 0, refl + lid_src.to(tl.float32) * bounce_add,
                    pulled)


@triton.jit
def _step_kernel(
    A, B, MASK, LID, SIG,
    EX, EY, EZ, WQ, OPPQ,
    nx, ny, nz, open_bc,
    omega0, fx, fy, fz, u_in, u_lid, cs2s,
    SGS: tl.constexpr, BLOCK: tl.constexpr,
):
    x = tl.program_id(0)
    lane = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    nyz = ny * nz
    m_l = lane < nyz
    y = lane // nz
    z = lane % nz
    here = x * nyz + lane
    N = nx * nyz
    center_solid = tl.load(MASK + here, mask=m_l, other=1)
    is_in = (open_bc != 0) & (x == 0)          # inlet plane (scalar)
    sig = tl.load(SIG + x)                     # sponge strength at this x

    # -- pass 1: pull + bounce-back, accumulate macroscopics --------------
    rho = tl.zeros((BLOCK,), dtype=tl.float32)
    mx = tl.zeros((BLOCK,), dtype=tl.float32)
    my = tl.zeros((BLOCK,), dtype=tl.float32)
    mz = tl.zeros((BLOCK,), dtype=tl.float32)
    for q in tl.static_range(_NQ):
        ex = tl.load(EX + q)
        ey = tl.load(EY + q)
        ez = tl.load(EZ + q)
        wq = tl.load(WQ + q)
        opp = tl.load(OPPQ + q)
        sx = (x - ex + nx) % nx
        sy = (y - ey + ny) % ny
        sz = (z - ez + nz) % nz
        src = sx * nyz + sy * nz + sz
        exf = ex.to(tl.float32)
        fq = _pull(A, MASK, LID, here, src, q * N, opp * N,
                   6.0 * wq * exf * u_lid, m_l)
        rho += fq
        mx += exf * fq
        my += ey.to(tl.float32) * fq
        mz += ez.to(tl.float32) * fq
    rho_safe = tl.where(rho > 1e-12, rho, 1.0)
    ux = (mx + 0.5 * fx) / rho_safe            # Guo half-force shift
    uy = (my + 0.5 * fy) / rho_safe
    uz = (mz + 0.5 * fz) / rho_safe
    usq = ux * ux + uy * uy + uz * uz

    # -- pass 2 (Smagorinsky only): Pi_neq -> effective omega -------------
    # cs2s = 0 reduces exactly to plain BGK, but the SGS constexpr lets
    # Triton drop this pass entirely when the model is off.
    if SGS:
        pxx = tl.zeros((BLOCK,), dtype=tl.float32)
        pyy = tl.zeros((BLOCK,), dtype=tl.float32)
        pzz = tl.zeros((BLOCK,), dtype=tl.float32)
        pxy = tl.zeros((BLOCK,), dtype=tl.float32)
        pxz = tl.zeros((BLOCK,), dtype=tl.float32)
        pyz = tl.zeros((BLOCK,), dtype=tl.float32)
        for q in tl.static_range(_NQ):
            ex = tl.load(EX + q)
            ey = tl.load(EY + q)
            ez = tl.load(EZ + q)
            wq = tl.load(WQ + q)
            opp = tl.load(OPPQ + q)
            sx = (x - ex + nx) % nx
            sy = (y - ey + ny) % ny
            sz = (z - ez + nz) % nz
            src = sx * nyz + sy * nz + sz
            exf = ex.to(tl.float32)
            eyf = ey.to(tl.float32)
            ezf = ez.to(tl.float32)
            fq = _pull(A, MASK, LID, here, src, q * N, opp * N,
                       6.0 * wq * exf * u_lid, m_l)
            cu = 3.0 * (exf * ux + eyf * uy + ezf * uz)
            fneq = fq - wq * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
            pxx += exf * exf * fneq
            pyy += eyf * eyf * fneq
            pzz += ezf * ezf * fneq
            pxy += exf * eyf * fneq
            pxz += exf * ezf * fneq
            pyz += eyf * ezf * fneq
        qbar = tl.sqrt(2.0 * (pxx * pxx + pyy * pyy + pzz * pzz
                              + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz)))
        tau0 = 1.0 / omega0
        tau_eff = 0.5 * (tau0 + tl.sqrt(tau0 * tau0
                                        + 18.0 * cs2s * qbar / rho_safe))
        omg = 1.0 / tau_eff
    else:
        omg = omega0

    # -- pass 3: re-pull, collide (+Guo), sponge, boundaries, store -------
    gpre = 1.0 - 0.5 * omg
    uF = ux * fx + uy * fy + uz * fz
    u_in2 = 1.5 * u_in * u_in
    for q in tl.static_range(_NQ):
        ex = tl.load(EX + q)
        ey = tl.load(EY + q)
        ez = tl.load(EZ + q)
        wq = tl.load(WQ + q)
        opp = tl.load(OPPQ + q)
        sx = (x - ex + nx) % nx
        sy = (y - ey + ny) % ny
        sz = (z - ez + nz) % nz
        src = sx * nyz + sy * nz + sz
        exf = ex.to(tl.float32)
        eyf = ey.to(tl.float32)
        ezf = ez.to(tl.float32)
        fq = _pull(A, MASK, LID, here, src, q * N, opp * N,
                   6.0 * wq * exf * u_lid, m_l)
        cu = 3.0 * (exf * ux + eyf * uy + ezf * uz)
        feq = wq * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
        out = fq - omg * (fq - feq)                       # BGK (eq. 3.9)
        # Guo source (eq. 6.25): (1 - omega/2) w_q [3(e-u).F + 9(e.u)(e.F)]
        eF = exf * fx + eyf * fy + ezf * fz
        out += gpre * wq * (3.0 * (eF - uF) + 3.0 * cu * eF)
        # anechoic sponge + equilibrium inlet share the same target:
        # feq(rho=1, (u_in, 0, 0))
        cu_t = 3.0 * exf * u_in
        tgt = wq * (1.0 + cu_t + 0.5 * cu_t * cu_t - u_in2)
        out += sig * (tgt - out)                          # sponge blend
        out = tl.where(is_in, tgt, out)                   # inlet plane
        out = tl.where(center_solid != 0, 0.0, out)       # solids empty
        tl.store(B + q * N + here, out, mask=m_l)


class FusedSolver(Solver):
    """Same construction, same scenes, same guards — fused stepping.

    Stores the post-collision state (see module docstring); macroscopic
    fields match the readable reference step-for-step.
    """

    BLOCK = 128

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.device.type != "cuda":
            raise RuntimeError("FusedSolver needs a CUDA device")
        dev = self.device
        self._mask_u8 = self.mask.to(torch.uint8).contiguous()
        self._lid_u8 = self._lid_mask.to(torch.uint8).contiguous()
        sig = torch.zeros(self.nx, dtype=torch.float32)
        if self._n_sp:
            sig[self.nx - self._n_sp:] = \
                self._sponge_sigma.cpu().reshape(-1)
        self._sig_x = sig.to(dev)
        self._ex = torch.tensor([e[0] for e in E], dtype=torch.int32,
                                device=dev)
        self._ey = torch.tensor([e[1] for e in E], dtype=torch.int32,
                                device=dev)
        self._ez = torch.tensor([e[2] for e in E], dtype=torch.int32,
                                device=dev)
        self._wq = torch.tensor(W, dtype=torch.float32, device=dev)
        self._opp = torch.tensor(OPP, dtype=torch.int32, device=dev)
        self.f = self.f.contiguous()
        self.f2 = self.f2.contiguous()

    def step(self) -> None:
        a, b = self.f, self.f2
        u_in = self.u_char * self._ramp_factor(self.step_count)

        if self.measure_force:
            # Momentum exchange, fused convention: A is post-collision, and
            # a static obstacle's bounced value equals the outgoing one, so
            # each link transfers e_qbar * 2 f*_qbar — identical accounting
            # to the reference (eq. 5.51).
            force = torch.zeros(3, dtype=torch.float32, device=self.device)
            for q in range(1, Q):
                idx = self._force_idx[q]
                if idx is None:
                    continue
                qb = OPP[q]
                transfer = 2.0 * a[qb].reshape(-1)[idx].sum()
                force[0] += E[qb][0] * transfer
                force[1] += E[qb][1] * transfer
                force[2] += E[qb][2] * transfer
            self.last_force = force

        nyz = self.ny * self.nz
        grid = (self.nx, triton.cdiv(nyz, self.BLOCK))
        _step_kernel[grid](
            a, b, self._mask_u8, self._lid_u8, self._sig_x,
            self._ex, self._ey, self._ez, self._wq, self._opp,
            self.nx, self.ny, self.nz, int(self.inlet_outlet),
            self.omega, self.force[0], self.force[1], self.force[2],
            u_in, self.lid_velocity,
            self.cs_smag ** 2 if self.sgs else 0.0,
            SGS=self.sgs, BLOCK=self.BLOCK,
        )
        self.f, self.f2 = b, a
        self.step_count += 1
