#!/usr/bin/env python3
"""Phase 1/4 toolchain check: is torch seeing the Blackwell card (sm_120)?

Prints torch/CUDA versions, device capability, and a quick D3Q19 step
timing on a mid-size grid -> MLUPS.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    import torch
    print(f"torch {torch.__version__}  cuda_build={torch.version.cuda}")
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE — GPU phases blocked. "
              "(WSL2? driver? torch build?)")
        return 1
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"device: {name}  sm_{cap[0]}{cap[1]}  {mem:.1f} GB")

    from lbm.solver import Solver
    nx, ny = 2048, 1024  # 2.1M cells (the Phase 4 benchmark grid)
    s = Solver(nx, ny, tau=0.6, u_char=0.05, device="cuda",
               inlet_outlet=True, ramp_steps=100)
    for _ in range(20):  # warmup
        s.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_steps = 100
    for _ in range(n_steps):
        s.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mlups = nx * ny * n_steps / dt / 1e6
    print(f"D2Q9 PyTorch reference: {mlups:.0f} MLUPS on {nx}x{ny}")
    print("(Phase 4 fused-kernel ceiling on this card: ~13 GLUPS at "
          "72 B/cell/step over ~960 GB/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
