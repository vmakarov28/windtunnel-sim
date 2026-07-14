"""Phase 4 — fused collide+stream Triton kernel (A-B pattern).

One kernel does pull-streaming + halfway bounce-back + BGK collision +
Guo forcing + the anechoic sponge for every interior cell, reading the
post-collision A buffer and writing the post-collision B buffer: 72
bytes/cell/step of compulsory traffic, which at ~960 GB/s puts the
ceiling near 13 GLUPS on the RTX 5080.

Ordering note (why this matches the readable reference EXACTLY): the
reference stores the post-boundary state and steps collide->stream->BC;
the fused solver stores the post-collision state and steps
stream->BC->collide. Those states differ by one collision — but density
and momentum are collision invariants, so rho and u agree step-for-step
to fp32 op-order. The correctness gate compares exactly that.

The two open-boundary columns (Zou-He) stay in PyTorch: 2 columns out of
nx, identical code path to the reference, zero duplication of the
subtle part. Kernel layout: program = (one x, BLOCK cells of y), so
loads coalesce along y (the fastest axis).

Readable reference: lbm/solver.py — that file is the spec; this file is
the speed. (A-B vs A-A pattern discussion: Flatscher's LB-t, ideas only.)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .lattice import E, OPP, Q, W
from .solver import Solver

W0, WA, WD = 4.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0


@triton.jit
def _step_kernel(
    A, B, MASK, LID, SIG,
    nx, ny,
    omega, fx, fy, u_in, u_lid,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    my = offs < ny
    nxy = nx * ny
    here = x * ny + offs

    center_solid = tl.load(MASK + here, mask=my, other=1)

    # -- pull + halfway bounce-back, direction by direction --------------
    # E: 1:(1,0) 2:(-1,0) 3:(0,1) 4:(0,-1) 5:(1,1) 6:(-1,-1) 7:(1,-1)
    #    8:(-1,1); OPP: 1<->2, 3<->4, 5<->6, 7<->8.
    f0 = tl.load(A + here, mask=my, other=0.0)        # rest never streams

    # helper pattern per q (unrolled; each block reads its source cell,
    # bounces + adds moving-wall momentum if the source is solid/lid):
    sx = (x - 1 + nx) % nx
    sy = offs
    src = sx * ny + sy
    ms = tl.load(MASK + src, mask=my, other=1)
    ml = tl.load(LID + src, mask=my, other=0)
    pulled = tl.load(A + 1 * nxy + src, mask=my, other=0.0)
    refl = tl.load(A + 2 * nxy + here, mask=my, other=0.0)
    f1 = tl.where(ms != 0, refl + ml * 6.0 * WA * u_lid, pulled)

    sx = (x + 1) % nx
    src = sx * ny + sy
    ms = tl.load(MASK + src, mask=my, other=1)
    ml = tl.load(LID + src, mask=my, other=0)
    pulled = tl.load(A + 2 * nxy + src, mask=my, other=0.0)
    refl = tl.load(A + 1 * nxy + here, mask=my, other=0.0)
    f2 = tl.where(ms != 0, refl - ml * 6.0 * WA * u_lid, pulled)

    sy = (offs - 1 + ny) % ny
    src = x * ny + sy
    ms = tl.load(MASK + src, mask=my, other=1)
    pulled = tl.load(A + 3 * nxy + src, mask=my, other=0.0)
    refl = tl.load(A + 4 * nxy + here, mask=my, other=0.0)
    f3 = tl.where(ms != 0, refl, pulled)              # e_x = 0: no lid term

    sy = (offs + 1) % ny
    src = x * ny + sy
    ms = tl.load(MASK + src, mask=my, other=1)
    pulled = tl.load(A + 4 * nxy + src, mask=my, other=0.0)
    refl = tl.load(A + 3 * nxy + here, mask=my, other=0.0)
    f4 = tl.where(ms != 0, refl, pulled)

    sx = (x - 1 + nx) % nx
    sy = (offs - 1 + ny) % ny
    src = sx * ny + sy
    ms = tl.load(MASK + src, mask=my, other=1)
    ml = tl.load(LID + src, mask=my, other=0)
    pulled = tl.load(A + 5 * nxy + src, mask=my, other=0.0)
    refl = tl.load(A + 6 * nxy + here, mask=my, other=0.0)
    f5 = tl.where(ms != 0, refl + ml * 6.0 * WD * u_lid, pulled)

    sx = (x + 1) % nx
    sy = (offs + 1) % ny
    src = sx * ny + sy
    ms = tl.load(MASK + src, mask=my, other=1)
    ml = tl.load(LID + src, mask=my, other=0)
    pulled = tl.load(A + 6 * nxy + src, mask=my, other=0.0)
    refl = tl.load(A + 5 * nxy + here, mask=my, other=0.0)
    f6 = tl.where(ms != 0, refl - ml * 6.0 * WD * u_lid, pulled)

    sx = (x - 1 + nx) % nx
    sy = (offs + 1) % ny
    src = sx * ny + sy
    ms = tl.load(MASK + src, mask=my, other=1)
    ml = tl.load(LID + src, mask=my, other=0)
    pulled = tl.load(A + 7 * nxy + src, mask=my, other=0.0)
    refl = tl.load(A + 8 * nxy + here, mask=my, other=0.0)
    f7 = tl.where(ms != 0, refl + ml * 6.0 * WD * u_lid, pulled)

    sx = (x + 1) % nx
    sy = (offs - 1 + ny) % ny
    src = sx * ny + sy
    ms = tl.load(MASK + src, mask=my, other=1)
    ml = tl.load(LID + src, mask=my, other=0)
    pulled = tl.load(A + 8 * nxy + src, mask=my, other=0.0)
    refl = tl.load(A + 7 * nxy + here, mask=my, other=0.0)
    f8 = tl.where(ms != 0, refl - ml * 6.0 * WD * u_lid, pulled)

    # -- macroscopics (Guo half-force shift) ------------------------------
    rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
    rho_safe = tl.where(rho > 1e-12, rho, 1.0)
    ux = (f1 - f2 + f5 - f6 + f7 - f8 + 0.5 * fx) / rho_safe
    uy = (f3 - f4 + f5 - f6 - f7 + f8 + 0.5 * fy) / rho_safe
    usq = ux * ux + uy * uy
    ufdot = ux * fx + uy * fy
    sig = tl.load(SIG + x)
    omg = omega  # scalar BGK rate; SGS makes this a field in Phase 5

    # sponge target: feq(rho=1, (u_in, 0)) — constants per direction
    tusq = u_in * u_in

    # -- collide + Guo + sponge, then store, direction by direction ------
    solid = center_solid != 0

    # q = 0
    feq = W0 * rho * (1.0 - 1.5 * usq)
    guo = (1.0 - 0.5 * omg) * W0 * (3.0 * (-ufdot))
    tgt = W0 * (1.0 - 1.5 * tusq)
    out = f0 - omg * (f0 - feq) + guo
    out = out + sig * (tgt - out)
    tl.store(B + here, tl.where(solid, 0.0, out), mask=my)

    # axis directions
    # q = 1: e=(1,0)
    cu = 3.0 * ux
    feq = WA * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
    guo = (1.0 - 0.5 * omg) * WA * (3.0 * (fx - ufdot) + 3.0 * cu * fx)
    tcu = 3.0 * u_in
    tgt = WA * (1.0 + tcu + 0.5 * tcu * tcu - 1.5 * tusq)
    out = f1 - omg * (f1 - feq) + guo
    out = out + sig * (tgt - out)
    tl.store(B + 1 * nxy + here, tl.where(solid, 0.0, out), mask=my)

    # q = 2: e=(-1,0)
    cu = -3.0 * ux
    feq = WA * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
    guo = (1.0 - 0.5 * omg) * WA * (3.0 * (-fx - ufdot) + 3.0 * cu * (-fx))
    tcu = -3.0 * u_in
    tgt = WA * (1.0 + tcu + 0.5 * tcu * tcu - 1.5 * tusq)
    out = f2 - omg * (f2 - feq) + guo
    out = out + sig * (tgt - out)
    tl.store(B + 2 * nxy + here, tl.where(solid, 0.0, out), mask=my)

    # q = 3: e=(0,1)
    cu = 3.0 * uy
    feq = WA * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
    guo = (1.0 - 0.5 * omg) * WA * (3.0 * (fy - ufdot) + 3.0 * cu * fy)
    tgt = WA * (1.0 - 1.5 * tusq)
    out = f3 - omg * (f3 - feq) + guo
    out = out + sig * (tgt - out)
    tl.store(B + 3 * nxy + here, tl.where(solid, 0.0, out), mask=my)

    # q = 4: e=(0,-1)
    cu = -3.0 * uy
    feq = WA * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
    guo = (1.0 - 0.5 * omg) * WA * (3.0 * (-fy - ufdot) + 3.0 * cu * (-fy))
    tgt = WA * (1.0 - 1.5 * tusq)
    out = f4 - omg * (f4 - feq) + guo
    out = out + sig * (tgt - out)
    tl.store(B + 4 * nxy + here, tl.where(solid, 0.0, out), mask=my)

    # diagonals
    # q = 5: e=(1,1)
    cu = 3.0 * (ux + uy)
    feq = WD * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
    guo = (1.0 - 0.5 * omg) * WD * (3.0 * (fx + fy - ufdot)
                                    + 3.0 * cu * (fx + fy))
    tcu = 3.0 * u_in
    tgt = WD * (1.0 + tcu + 0.5 * tcu * tcu - 1.5 * tusq)
    out = f5 - omg * (f5 - feq) + guo
    out = out + sig * (tgt - out)
    tl.store(B + 5 * nxy + here, tl.where(solid, 0.0, out), mask=my)

    # q = 6: e=(-1,-1)
    cu = -3.0 * (ux + uy)
    feq = WD * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
    guo = (1.0 - 0.5 * omg) * WD * (3.0 * (-fx - fy - ufdot)
                                    + 3.0 * cu * (-fx - fy))
    tcu = -3.0 * u_in
    tgt = WD * (1.0 + tcu + 0.5 * tcu * tcu - 1.5 * tusq)
    out = f6 - omg * (f6 - feq) + guo
    out = out + sig * (tgt - out)
    tl.store(B + 6 * nxy + here, tl.where(solid, 0.0, out), mask=my)

    # q = 7: e=(1,-1)
    cu = 3.0 * (ux - uy)
    feq = WD * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
    guo = (1.0 - 0.5 * omg) * WD * (3.0 * (fx - fy - ufdot)
                                    + 3.0 * cu * (fx - fy))
    tcu = 3.0 * u_in
    tgt = WD * (1.0 + tcu + 0.5 * tcu * tcu - 1.5 * tusq)
    out = f7 - omg * (f7 - feq) + guo
    out = out + sig * (tgt - out)
    tl.store(B + 7 * nxy + here, tl.where(solid, 0.0, out), mask=my)

    # q = 8: e=(-1,1)
    cu = 3.0 * (-ux + uy)
    feq = WD * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
    guo = (1.0 - 0.5 * omg) * WD * (3.0 * (-fx + fy - ufdot)
                                    + 3.0 * cu * (-fx + fy))
    tcu = -3.0 * u_in
    tgt = WD * (1.0 + tcu + 0.5 * tcu * tcu - 1.5 * tusq)
    out = f8 - omg * (f8 - feq) + guo
    out = out + sig * (tgt - out)
    tl.store(B + 8 * nxy + here, tl.where(solid, 0.0, out), mask=my)


class FusedSolver(Solver):
    """Same construction, same scenes, same guards — fused stepping.

    Stores the post-collision state (see module docstring); macroscopic
    fields match the reference solver step-for-step.
    """

    BLOCK = 128

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.device.type != "cuda":
            raise RuntimeError("FusedSolver needs a CUDA device")
        self._mask_u8 = self.mask.to(torch.uint8).contiguous()
        self._lid_u8 = self._lid_mask.to(torch.uint8).contiguous()
        sig = torch.zeros(self.nx, dtype=torch.float32)
        if self._n_sp:
            sig[self.nx - self._n_sp:] = self._sponge_sigma.cpu().squeeze(1)
        self._sig_x = sig.to(self.device)
        self.f = self.f.contiguous()
        self.f2 = self.f2.contiguous()
        if self.inlet_outlet:
            edge_cols = torch.cat([self.mask[:2], self.mask[-2:]])
            assert not bool(edge_cols.any()), \
                "open-boundary columns must be solid-free"

    def step(self) -> None:
        a, b = self.f, self.f2
        u_in = self.u_char * self._ramp_factor(self.step_count)

        if self.measure_force:  # same accounting as the reference (5.51)
            force = torch.zeros(2, dtype=torch.float32, device=self.device)
            for q in range(1, Q):
                idx = self._force_idx[q]
                if idx is None:
                    continue
                qb = OPP[q]
                bounced = a[qb].reshape(-1)[idx] + self._lid_corr[q]
                transfer = (a[qb].reshape(-1)[idx] + bounced).sum()
                force[0] += E[qb][0] * transfer
                force[1] += E[qb][1] * transfer
            self.last_force = force

        grid = (self.nx, triton.cdiv(self.ny, self.BLOCK))
        _step_kernel[grid](
            a, b, self._mask_u8, self._lid_u8, self._sig_x,
            self.nx, self.ny,
            self.omega, self.force[0], self.force[1],
            u_in, self.lid_velocity,
            BLOCK=self.BLOCK,
        )
        if self.inlet_outlet:
            self._fix_boundary_columns(a, b, u_in)
        self.f, self.f2 = b, a
        self.step_count += 1

    # ------------------------------------------------------------------
    def _fix_boundary_columns(self, a: torch.Tensor, b: torch.Tensor,
                              u_in: float) -> None:
        """Redo columns 0 and nx-1 in PyTorch: pull, Zou-He, collide.

        Two columns out of nx — the subtle boundary code stays in one
        place (the reference implementation) conceptually; this mirrors
        it on the post-collision convention."""
        nx, ny = self.nx, self.ny
        for col in (0, nx - 1):
            g = torch.empty((Q, ny), dtype=torch.float32, device=self.device)
            for q in range(Q):
                ex, ey = E[q]
                g[q] = torch.roll(a[q, (col - ex) % nx, :], shifts=ey, dims=0)
            if col == 0:
                rho_in = (g[0] + g[3] + g[4]
                          + 2.0 * (g[2] + g[6] + g[8])) / (1.0 - u_in)
                t = 0.5 * (g[3] - g[4])
                g[1] = g[2] + (2.0 / 3.0) * rho_in * u_in
                g[5] = g[6] - t + (1.0 / 6.0) * rho_in * u_in
                g[7] = g[8] + t + (1.0 / 6.0) * rho_in * u_in
            else:
                u_out = (g[0] + g[3] + g[4]
                         + 2.0 * (g[1] + g[5] + g[7])) - 1.0
                t = 0.5 * (g[3] - g[4])
                g[2] = g[1] - (2.0 / 3.0) * u_out
                g[6] = g[5] + t - (1.0 / 6.0) * u_out
                g[8] = g[7] - t - (1.0 / 6.0) * u_out
            # collide + sponge (no obstacle cells in these columns)
            rho = g.sum(dim=0)
            ux = g[1] - g[2] + g[5] - g[6] + g[7] - g[8]
            uy = g[3] - g[4] + g[5] - g[6] - g[7] + g[8]
            rho_safe = torch.where(rho > 1e-12, rho, torch.ones_like(rho))
            ux, uy = ux / rho_safe, uy / rho_safe
            usq = ux * ux + uy * uy
            sig = float(self._sig_x[col])
            for q in range(Q):
                ex, ey = E[q]
                cu = 3.0 * (ex * ux + ey * uy)
                feq = W[q] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
                out = g[q] - self.omega * (g[q] - feq)
                if sig > 0.0:
                    tcu = 3.0 * ex * u_in
                    tgt = W[q] * (1.0 + tcu + 0.5 * tcu * tcu
                                  - 1.5 * u_in * u_in)
                    out = out + sig * (tgt - out)
                b[q, col, :] = out
