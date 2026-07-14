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

# lattice weights as compile-time constants inside the kernel
W0 = tl.constexpr(4.0 / 9.0)
WA = tl.constexpr(1.0 / 9.0)
WD = tl.constexpr(1.0 / 36.0)


@triton.jit
def _step_kernel(
    A, B, MASK, LID, SIG,
    nx, ny, open_bc,
    omega, fx, fy, u_in, u_lid, cs2s,
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

    # -- Zou-He open boundaries, in-kernel (only the edge programs take
    # these branches; doing this here makes the whole step ONE launch —
    # the PyTorch-column version cost ~2 ms/step of launch overhead).
    # Inlet x = 0: velocity imposed, the +x triple (1,5,7) reconstructed.
    is_in = (x == 0) & (open_bc != 0)
    is_out = (x == nx - 1) & (open_bc != 0)
    t_imb = 0.5 * (f3 - f4)                    # transverse imbalance
    rho_in = (f0 + f3 + f4 + 2.0 * (f2 + f6 + f8)) / (1.0 - u_in)
    f1 = tl.where(is_in, f2 + (2.0 / 3.0) * rho_in * u_in, f1)
    f5 = tl.where(is_in, f6 - t_imb + (1.0 / 6.0) * rho_in * u_in, f5)
    f7 = tl.where(is_in, f8 + t_imb + (1.0 / 6.0) * rho_in * u_in, f7)
    # Outlet x = nx-1: rho = 1 imposed, the -x triple (2,6,8) reconstructed.
    u_out = (f0 + f3 + f4 + 2.0 * (f1 + f5 + f7)) - 1.0
    f2 = tl.where(is_out, f1 - (2.0 / 3.0) * u_out, f2)
    f6 = tl.where(is_out, f5 + t_imb - (1.0 / 6.0) * u_out, f6)
    f8 = tl.where(is_out, f7 - t_imb - (1.0 / 6.0) * u_out, f8)

    # -- macroscopics (Guo half-force shift) ------------------------------
    rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
    rho_safe = tl.where(rho > 1e-12, rho, 1.0)
    ux = (f1 - f2 + f5 - f6 + f7 - f8 + 0.5 * fx) / rho_safe
    uy = (f3 - f4 + f5 - f6 - f7 + f8 + 0.5 * fy) / rho_safe
    usq = ux * ux + uy * uy
    ufdot = ux * fx + uy * fy
    sig = tl.load(SIG + x)

    # -- equilibria (kept in registers, used for Pi_neq AND collision) ----
    cu1 = 3.0 * ux
    cu3 = 3.0 * uy
    cu5 = 3.0 * (ux + uy)
    cu7 = 3.0 * (ux - uy)
    feq0 = W0 * rho * (1.0 - 1.5 * usq)
    feq1 = WA * rho * (1.0 + cu1 + 0.5 * cu1 * cu1 - 1.5 * usq)
    feq2 = WA * rho * (1.0 - cu1 + 0.5 * cu1 * cu1 - 1.5 * usq)
    feq3 = WA * rho * (1.0 + cu3 + 0.5 * cu3 * cu3 - 1.5 * usq)
    feq4 = WA * rho * (1.0 - cu3 + 0.5 * cu3 * cu3 - 1.5 * usq)
    feq5 = WD * rho * (1.0 + cu5 + 0.5 * cu5 * cu5 - 1.5 * usq)
    feq6 = WD * rho * (1.0 - cu5 + 0.5 * cu5 * cu5 - 1.5 * usq)
    feq7 = WD * rho * (1.0 + cu7 + 0.5 * cu7 * cu7 - 1.5 * usq)
    feq8 = WD * rho * (1.0 - cu7 + 0.5 * cu7 * cu7 - 1.5 * usq)

    # -- Smagorinsky effective relaxation (Hou et al. 1996) ---------------
    # cs2s = Cs^2; cs2s = 0 reduces EXACTLY to plain BGK (no branch).
    pxx = (f1 - feq1) + (f2 - feq2) + (f5 - feq5) + (f6 - feq6) \
        + (f7 - feq7) + (f8 - feq8)
    pyy = (f3 - feq3) + (f4 - feq4) + (f5 - feq5) + (f6 - feq6) \
        + (f7 - feq7) + (f8 - feq8)
    pxy = (f5 - feq5) + (f6 - feq6) - (f7 - feq7) - (f8 - feq8)
    qbar = tl.sqrt(2.0 * (pxx * pxx + 2.0 * pxy * pxy + pyy * pyy))
    tau0 = 1.0 / omega
    tau_eff = 0.5 * (tau0 + tl.sqrt(tau0 * tau0
                                    + 18.0 * cs2s * qbar / rho_safe))
    omg = 1.0 / tau_eff

    # sponge target: feq(rho=1, (u_in, 0)) — constants per direction
    tusq = u_in * u_in
    tcu = 3.0 * u_in
    solid = center_solid != 0
    gpre = 1.0 - 0.5 * omg

    # -- collide + Guo + sponge, then store, direction by direction ------
    # q = 0
    guo = gpre * W0 * (3.0 * (-ufdot))
    tgt = W0 * (1.0 - 1.5 * tusq)
    out = f0 - omg * (f0 - feq0) + guo
    out = out + sig * (tgt - out)
    tl.store(B + here, tl.where(solid, 0.0, out), mask=my)

    # q = 1: e=(1,0)
    guo = gpre * WA * (3.0 * (fx - ufdot) + 3.0 * cu1 * fx)
    tgt = WA * (1.0 + tcu + 0.5 * tcu * tcu - 1.5 * tusq)
    out = f1 - omg * (f1 - feq1) + guo
    out = out + sig * (tgt - out)
    tl.store(B + 1 * nxy + here, tl.where(solid, 0.0, out), mask=my)

    # q = 2: e=(-1,0)
    guo = gpre * WA * (3.0 * (-fx - ufdot) - 3.0 * cu1 * (-fx))
    tgt = WA * (1.0 - tcu + 0.5 * tcu * tcu - 1.5 * tusq)
    out = f2 - omg * (f2 - feq2) + guo
    out = out + sig * (tgt - out)
    tl.store(B + 2 * nxy + here, tl.where(solid, 0.0, out), mask=my)

    # q = 3: e=(0,1)
    guo = gpre * WA * (3.0 * (fy - ufdot) + 3.0 * cu3 * fy)
    tgt = WA * (1.0 - 1.5 * tusq)
    out = f3 - omg * (f3 - feq3) + guo
    out = out + sig * (tgt - out)
    tl.store(B + 3 * nxy + here, tl.where(solid, 0.0, out), mask=my)

    # q = 4: e=(0,-1)
    guo = gpre * WA * (3.0 * (-fy - ufdot) - 3.0 * cu3 * (-fy))
    tgt = WA * (1.0 - 1.5 * tusq)
    out = f4 - omg * (f4 - feq4) + guo
    out = out + sig * (tgt - out)
    tl.store(B + 4 * nxy + here, tl.where(solid, 0.0, out), mask=my)

    # q = 5: e=(1,1)
    guo = gpre * WD * (3.0 * (fx + fy - ufdot) + 3.0 * cu5 * (fx + fy))
    tgt = WD * (1.0 + tcu + 0.5 * tcu * tcu - 1.5 * tusq)
    out = f5 - omg * (f5 - feq5) + guo
    out = out + sig * (tgt - out)
    tl.store(B + 5 * nxy + here, tl.where(solid, 0.0, out), mask=my)

    # q = 6: e=(-1,-1)
    guo = gpre * WD * (3.0 * (-fx - fy - ufdot) - 3.0 * cu5 * (-fx - fy))
    tgt = WD * (1.0 - tcu + 0.5 * tcu * tcu - 1.5 * tusq)
    out = f6 - omg * (f6 - feq6) + guo
    out = out + sig * (tgt - out)
    tl.store(B + 6 * nxy + here, tl.where(solid, 0.0, out), mask=my)

    # q = 7: e=(1,-1)
    guo = gpre * WD * (3.0 * (fx - fy - ufdot) + 3.0 * cu7 * (fx - fy))
    tgt = WD * (1.0 + tcu + 0.5 * tcu * tcu - 1.5 * tusq)
    out = f7 - omg * (f7 - feq7) + guo
    out = out + sig * (tgt - out)
    tl.store(B + 7 * nxy + here, tl.where(solid, 0.0, out), mask=my)

    # q = 8: e=(-1,1)
    guo = gpre * WD * (3.0 * (-fx + fy - ufdot) - 3.0 * cu7 * (-fx + fy))
    tgt = WD * (1.0 - tcu + 0.5 * tcu * tcu - 1.5 * tusq)
    out = f8 - omg * (f8 - feq8) + guo
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
        # plain floats for the two BC columns: a float(tensor) here would
        # force a GPU->CPU sync EVERY step (~0.2 ms on WSL2 — measured as
        # the dominant cost of the first benchmark)
        self._sig_col = (float(sig[0]), float(sig[-1]))
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
            self.nx, self.ny, int(self.inlet_outlet),
            self.omega, self.force[0], self.force[1],
            u_in, self.lid_velocity,
            self.cs_smag ** 2 if self.sgs else 0.0,
            BLOCK=self.BLOCK,
        )
        self.f, self.f2 = b, a
        self.step_count += 1

