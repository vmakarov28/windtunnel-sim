#!/usr/bin/env python3
"""3D entry point: python run3d.py --scene <name> --seed <n>.

The 3D sibling of run.py — same reproducibility contract (scene config +
seed and nothing else), separate package (lbm3d/), separate scenes
(scenes3d/), separate output root (out3d/). With --steps 0 (default) it
only prints the resolved unit system.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from lbm3d.config import SceneError, list_scenes, load_scene
from lbm.units import UnitError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="3D GPU lattice-Boltzmann wind tunnel (D3Q19)"
    )
    parser.add_argument("--scene", help="scene name (a file in scenes3d/)")
    parser.add_argument("--seed", type=int, help="RNG seed (required for runs)")
    parser.add_argument("--steps", type=int, default=0,
                        help="time steps to run (0 = just report units)")
    parser.add_argument("--frame-every", type=int, default=100,
                        help="render a frame every N steps")
    parser.add_argument("--guard-every", type=int, default=100,
                        help="run NaN/velocity/mass guards every N steps")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="save a checkpoint every N steps (0 = end only)")
    parser.add_argument("--out", default=None,
                        help="output dir (default out3d/<scene>-seed<seed>)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", default=None,
                        help="checkpoint .pt to restore before running")
    parser.add_argument("--no-ramp", action="store_true",
                        help="impulsive start: skip the inlet velocity ramp")
    parser.add_argument("--preset", default="slice",
                        choices=["slice", "three_pane", "qcrit"],
                        help="slice = omega_z mid-span (2D-comparable); "
                             "three_pane = slice + spanwise-velocity pane "
                             "(the 3D-ness meter); qcrit = Q-criterion "
                             "volume projection (the genuinely-3D shot)")
    parser.add_argument("--solver", default="reference",
                        choices=["reference", "fused"],
                        help="reference = readable PyTorch; fused = Triton "
                             "(one kernel launch per step)")
    parser.add_argument("--list-scenes", action="store_true")
    args = parser.parse_args(argv)

    if args.list_scenes:
        print("\n".join(list_scenes()))
        return 0
    if args.scene is None or args.seed is None:
        parser.error("--scene and --seed are both required (reproducibility)")

    try:
        scene = load_scene(args.scene)
    except (SceneError, UnitError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2

    print(scene.report())
    print(f"  seed       {args.seed}")
    if args.steps <= 0:
        print("\n(no --steps given: units report only)")
        return 0

    # Solver imports live here so the units path works without torch.
    from lbm3d.render import FrameWriter
    from lbm3d.solver import SimulationBlowup, Solver, capture_failure

    if args.solver == "fused":
        from lbm3d.fused import FusedSolver as SolverCls
    else:
        SolverCls = Solver
    solver = SolverCls.from_scene(scene, seed=args.seed, device=args.device,
                                  ramp=not args.no_ramp)
    print(f"  device     {solver.device}   preset {args.preset}"
          + ("   (IMPULSIVE START — no inlet ramp)" if args.no_ramp else ""))

    out = Path(args.out or f"out3d/{scene.name}-seed{args.seed}")
    frames = FrameWriter(out / "frames", preset=args.preset)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "guards.csv", "a", encoding="utf-8")
    if log.tell() == 0:
        log.write("step,u_max,mass_drift,tau_eff_max\n")

    if args.resume:
        solver.restore(args.resume)
        print(f"  resumed    step {solver.step_count} from {args.resume}")

    t0, steps_done, cells = time.perf_counter(), 0, scene.cells
    try:
        for _ in range(args.steps):
            solver.step()
            steps_done += 1
            n = solver.step_count
            if args.guard_every and n % args.guard_every == 0:
                g = solver.check_guards()  # raises SimulationBlowup
                log.write(f"{n},{g['u_max']:.5f},{g['mass_drift']:.3e},"
                          f"{solver.last_tau_eff_max:.4f}\n")
                log.flush()
                if g["u_max"] > 0.3:
                    print(f"  WARNING step {n}: u_max = {g['u_max']:.3f}")
            if args.frame_every and n % args.frame_every == 0:
                frames.write(solver)
            if args.checkpoint_every and n % args.checkpoint_every == 0:
                solver.checkpoint(out / f"checkpoint_{n:08d}.pt")
            if n % 500 == 0:
                mlups = cells * steps_done / (time.perf_counter() - t0) / 1e6
                print(f"  step {n:>7}  {mlups:8.1f} MLUPS  "
                      f"{frames.count} frames", flush=True)
    except SimulationBlowup as e:
        dest = capture_failure(solver, str(e), frames_dir=frames.dir)
        print(f"\nBLOWUP: {e}\ncaptured to {dest} — this is footage.",
              file=sys.stderr)
        log.close()
        return 3

    solver.checkpoint(out / "final.pt")
    log.close()
    dt_wall = time.perf_counter() - t0
    print(f"\ndone: {steps_done} steps, {frames.count} frames, "
          f"{cells * steps_done / dt_wall / 1e6:.1f} MLUPS sustained")
    print(f"assemble: python scripts/make_video.py {frames.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
