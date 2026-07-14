#!/usr/bin/env python3
"""Headless physics check of the WGSL LBM step kernel via wgpu-py.

Compiling is necessary but not sufficient — this actually RUNS the browser
kernel on the GPU and checks the same properties the Python solver is gated
on: no NaN, an empty tunnel holds freestream (the "boring tunnel" rule),
and an obstacle sheds a wake with real transverse motion. Same spirit as
tests/test_solver.py, applied to the WebGPU port.
"""
import math
import struct
import sys
from pathlib import Path

import numpy as np
import wgpu

NX, NY = 256, 128
N = NX * NY
U_IN = 0.06
NU = 0.02                     # tau = 0.56, comfortably stable
OMEGA = 1.0 / (3.0 * NU + 0.5)
CS2S = 0.15 * 0.15
STEP = (Path(__file__).resolve().parent / "_build" / "step.wgsl").read_text()

EX = [0, 1, -1, 0, 0, 1, -1, 1, -1]
EY = [0, 0, 0, 1, -1, 1, -1, -1, 1]
W = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]


def feq_field(ux):
    f = np.zeros((9, N), dtype=np.float32)
    for q in range(9):
        eu = EX[q] * ux
        f[q, :] = W[q] * (1.0 + 3*eu + 4.5*eu*eu - 1.5*ux*ux)
    return f.reshape(-1)


def params(sponge_start):
    # nx ny omega uIn cs2s spongeStart spongeStrength vortScale mode
    # nTracers seedT pad0  — must match struct Params in shaders.js
    return struct.pack(
        "<IIfffIffIIff",
        NX, NY, OMEGA, U_IN, CS2S, sponge_start, 0.15, 0.06, 0, 0, 0.0, 0.0)


def run(mask_np, steps):
    dev = wgpu.gpu.request_adapter_sync(
        power_preference="high-performance").request_device_sync()
    S = wgpu.BufferUsage.STORAGE
    f0 = feq_field(U_IN)
    fA = dev.create_buffer_with_data(data=f0, usage=S | wgpu.BufferUsage.COPY_DST)
    fB = dev.create_buffer_with_data(data=f0, usage=S | wgpu.BufferUsage.COPY_DST)
    mask = dev.create_buffer_with_data(
        data=mask_np, usage=S | wgpu.BufferUsage.COPY_DST)
    vel = dev.create_buffer(
        size=N * 2 * 4, usage=S | wgpu.BufferUsage.COPY_SRC)
    pbuf = dev.create_buffer_with_data(
        data=params(int(NX * 0.9)),
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    mod = dev.create_shader_module(code=STEP)
    pipe = dev.create_compute_pipeline(
        layout="auto", compute={"module": mod, "entry_point": "main"})
    lay = pipe.get_bind_group_layout(0)

    def bg(fin, fout):
        return dev.create_bind_group(layout=lay, entries=[
            {"binding": 0, "resource": {"buffer": fin, "offset": 0, "size": fin.size}},
            {"binding": 1, "resource": {"buffer": fout, "offset": 0, "size": fout.size}},
            {"binding": 2, "resource": {"buffer": mask, "offset": 0, "size": mask.size}},
            {"binding": 3, "resource": {"buffer": vel, "offset": 0, "size": vel.size}},
            {"binding": 4, "resource": {"buffer": pbuf, "offset": 0, "size": pbuf.size}},
        ])
    bgAB, bgBA = bg(fA, fB), bg(fB, fA)

    for s in range(steps):
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, bgAB if s % 2 == 0 else bgBA)
        cp.dispatch_workgroups(math.ceil(NX / 8), math.ceil(NY / 8))
        cp.end()
        dev.queue.submit([enc.finish()])

    raw = dev.queue.read_buffer(vel)
    return np.frombuffer(raw, dtype=np.float32).reshape(N, 2)


def main() -> int:
    ok = True

    # 1. empty tunnel must hold freestream (u ~ U_IN, v ~ 0)
    empty = np.zeros(N, dtype=np.uint32)
    v = run(empty, 4000)
    if not np.isfinite(v).all():
        print("FAIL empty: NaN/inf"); return 1
    grid = v.reshape(NY, NX, 2)
    interior = grid[:, 10:-30, :]         # clear of inlet and sponge
    ux_err = float(np.abs(interior[:, :, 0] - U_IN).max())
    uy_err = float(np.abs(interior[:, :, 1]).max())
    print(f"empty tunnel: max|ux-U|={ux_err:.4f}  max|uy|={uy_err:.4f}")
    if ux_err > 0.01 or uy_err > 0.01:
        print("FAIL empty: freestream not held"); ok = False
    else:
        print("PASS empty tunnel holds freestream")

    # 2. cylinder must shed a wake (transverse velocity downstream)
    mask = np.zeros((NY, NX), dtype=np.uint32)
    cy, cx, r = NY // 2, NX // 4, 10
    yy, xx = np.mgrid[0:NY, 0:NX]
    mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 1
    v = run(mask.reshape(-1), 8000)
    if not np.isfinite(v).all():
        print("FAIL cylinder: NaN/inf"); return 1
    grid = v.reshape(NY, NX, 2)
    speed = np.sqrt((grid ** 2).sum(-1))
    umax = float(speed[:, 10:-30].max())
    wake_v = float(np.abs(grid[:, cx + 2*r:cx + 8*r, 1]).max())
    print(f"cylinder: u_max={umax:.4f} (accel over U={U_IN})  "
          f"wake max|v|={wake_v:.4f}")
    if umax <= U_IN * 1.05:
        print("FAIL cylinder: no acceleration around body"); ok = False
    elif wake_v < 1e-3:
        print("FAIL cylinder: no wake"); ok = False
    else:
        print("PASS cylinder accelerates flow and sheds a wake")

    print("\nALL PHYSICS CHECKS PASS" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
