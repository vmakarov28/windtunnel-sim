#!/usr/bin/env python3
"""Entry point for every run: python run.py --scene <name> --seed <n>.

All experiments are reproducible from scene config + seed alone.
With --steps 0 (default) this only prints the resolved unit system.
"""

from __future__ import annotations

import argparse
import sys
import time

from lbm.config import SceneError, list_scenes, load_scene
from lbm.units import UnitError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="3D GPU lattice-Boltzmann wind tunnel (D3Q19)"
    )
    parser.add_argument("--scene", help="scene name (a file in scenes/)")
    parser.add_argument("--seed", type=int, help="RNG seed (required for runs)")
    parser.add_argument("--steps", type=int, default=0,
                        help="time steps to run (0 = just report units)")
    parser.add_argument("--frame-every", type=int, default=50,
                        help="render a vorticity frame every N steps")
    parser.add_argument("--guard-every", type=int, default=100,
                        help="run NaN/velocity/mass guards every N steps")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="save a checkpoint every N steps (0 = end only)")
    parser.add_argument("--out", default=None,
                        help="output dir (default out/<scene>-seed<seed>)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", default=None,
                        help="checkpoint .pt to restore before running")
    parser.add_argument("--no-ramp", action="store_true",
                        help="impulsive start: skip the inlet velocity ramp "
                             "(one deliberate failure-reel run, then never again)")
    parser.add_argument("--preset", default=None,
                        choices=["vorticity", "speed", "dye", "streaklines"],
                        help="render preset (default: scene render.preset "
                             "or vorticity)")
    parser.add_argument("--zoom", default=None,
                        help="camera crop x0,y0,x1,y1 in characteristic "
                             "lengths (e.g. 6,5,14,10)")
    parser.add_argument("--upscale", type=int, default=1,
                        help="integer pixel upscale of the output frames")
    parser.add_argument("--tracers", type=int, default=300_000,
                        help="particle count for the streaklines preset")
    parser.add_argument("--overlay-mlups", action="store_true",
                        help="burn a live MLUPS counter into the frames")
    parser.add_argument("--solver", default="reference",
                        choices=["reference", "fused"],
                        help="reference = readable PyTorch; fused = Triton")
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
    import torch  # noqa: F401
    from lbm.cinema import CinemaWriter, Dye, StreaklineBuffer, Tracers
    from lbm.solver import SimulationBlowup, Solver, capture_failure

    if args.solver == "fused":
        from lbm.fused import FusedSolver as SolverCls
    else:
        SolverCls = Solver
    solver = SolverCls.from_scene(scene, seed=args.seed, device=args.device,
                                  ramp=not args.no_ramp)
    print(f"  device     {solver.device}"
          + (" (IMPULSIVE START — no inlet ramp)" if args.no_ramp else ""))

    render_cfg = scene.raw.get("render", {})
    preset = args.preset or render_cfg.get("preset", "vorticity")
    zoom = args.zoom or render_cfg.get("zoom")
    if isinstance(zoom, str):
        zoom = [float(v) for v in zoom.split(",")]
    if zoom is not None:  # characteristic lengths -> cells
        zoom = tuple(v * scene.units.cells for v in zoom)

    extras: dict = {}
    if preset == "dye":
        extras["dye"] = Dye(solver)
    elif preset == "streaklines":
        extras["tracers"] = Tracers(solver, n=args.tracers, seed=args.seed)
        extras["streaks"] = StreaklineBuffer(solver)

    from pathlib import Path
    out = Path(args.out or f"out/{scene.name}-seed{args.seed}")
    frames = CinemaWriter(out / "frames", preset=preset, zoom=zoom,
                          upscale=args.upscale)
    log_path = out / "guards.csv"
    out.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a", encoding="utf-8")
    if log.tell() == 0:
        log.write("step,u_max,mass_drift\n")

    if args.resume:
        solver.restore(args.resume)
        print(f"  resumed    step {solver.step_count} from {args.resume}")

    t0, steps_done, cells = time.perf_counter(), 0, scene.cells
    try:
        for _ in range(args.steps):
            solver.step()
            if "dye" in extras:
                extras["dye"].step(solver)
            if "tracers" in extras:
                extras["tracers"].step(solver)
                extras["streaks"].splat(extras["tracers"])
            steps_done += 1
            n = solver.step_count
            if args.guard_every and n % args.guard_every == 0:
                g = solver.check_guards()  # raises SimulationBlowup
                log.write(f"{n},{g['u_max']:.5f},{g['mass_drift']:.3e}\n")
                log.flush()
                if g["u_max"] > 0.3:
                    print(f"  WARNING step {n}: u_max = {g['u_max']:.3f}")
            if args.frame_every and n % args.frame_every == 0:
                text = None
                if args.overlay_mlups:
                    mlups = (cells * steps_done
                             / (time.perf_counter() - t0) / 1e6)
                    text = (f"{args.solver}: {mlups:,.0f} MLUPS   "
                            f"{scene.nx}x{scene.ny}   step {n}")
                frames.write(solver, extras, overlay=text)
            if args.checkpoint_every and n % args.checkpoint_every == 0:
                solver.checkpoint(out / f"checkpoint_{n:08d}.pt")
            if n % 500 == 0:
                dt_wall = time.perf_counter() - t0
                mlups = cells * steps_done / dt_wall / 1e6
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
    print(f"assemble: ffmpeg -framerate 60 -i {frames.dir}/frame_%06d.png "
          f"-c:v libx264 -pix_fmt yuv420p -crf 18 {out}/{scene.name}.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
