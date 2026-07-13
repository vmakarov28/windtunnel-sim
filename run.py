#!/usr/bin/env python3
"""Entry point for every run: python run.py --scene <name> --seed <n>.

All experiments are reproducible from scene config + seed alone.
Phase 0: loads the scene, resolves and prints its unit system, and exits.
The solver arrives in Phase 1.
"""

from __future__ import annotations

import argparse
import sys

from lbm.config import SceneError, list_scenes, load_scene
from lbm.units import UnitError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="2D GPU lattice-Boltzmann wind tunnel"
    )
    parser.add_argument("--scene", help="scene name (a file in scenes/)")
    parser.add_argument("--seed", type=int, help="RNG seed (required for runs)")
    parser.add_argument(
        "--list-scenes", action="store_true", help="list available scenes"
    )
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
    if scene.description:
        print(f"\n{scene.description}")
    print("\nPhase 0: units + scaffold only. The D2Q9 solver arrives in Phase 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
