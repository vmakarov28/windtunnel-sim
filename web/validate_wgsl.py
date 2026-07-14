#!/usr/bin/env python3
"""Headless WGSL validation via wgpu-py (same naga validator as browsers).

Compiles each assembled shader; a validation error raises. Exits non-zero
on the first failure so it can gate a commit.
"""
import sys
from pathlib import Path

import wgpu

BUILD = Path(__file__).resolve().parent / "_build"

def main() -> int:
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    ok = True
    for wgsl in sorted(BUILD.glob("*.wgsl")):
        src = wgsl.read_text()
        try:
            device.create_shader_module(code=src)
            print(f"PASS  {wgsl.name}")
        except Exception as e:  # naga validation error
            ok = False
            print(f"FAIL  {wgsl.name}\n{e}\n")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
