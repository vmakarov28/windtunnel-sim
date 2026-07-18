#!/usr/bin/env python3
"""Headless WGSL validation via wgpu-py (same naga validator as browsers).

Shaders are extracted straight from shaders.js (see shader_source.py) —
no Node, no manual dump step, nothing to go stale. Compiles all four;
exits non-zero on any failure OR if extraction finds fewer than four.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wgpu

from shader_source import extract_shaders


def main() -> int:
    shaders = extract_shaders()
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    ok = True
    for name in sorted(shaders):
        try:
            device.create_shader_module(code=shaders[name])
            print(f"PASS  {name}")
        except Exception as e:  # naga validation error
            ok = False
            print(f"FAIL  {name}\n{e}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
