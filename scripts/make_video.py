#!/usr/bin/env python3
"""Assemble rendered PNG frames into a 60 fps H.264 mp4.

Uses the system ffmpeg if present, else the static binary bundled with
imageio-ffmpeg (keeps the pipeline headless and admin-free). ProRes and
the other editing presets arrive in Phase 3.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("frames_dir", help="directory of frame_%%06d.png")
    p.add_argument("-o", "--out", default=None, help="output .mp4 path")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--crf", type=int, default=18)
    args = p.parse_args()

    frames = Path(args.frames_dir)
    n = len(list(frames.glob("frame_*.png")))
    if n == 0:
        print(f"no frames in {frames}", file=sys.stderr)
        return 1
    out = Path(args.out or frames.parent / f"{frames.parent.name}.mp4")
    cmd = [
        ffmpeg_exe(), "-y", "-framerate", str(args.fps),
        "-i", str(frames / "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf),
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
        return r.returncode
    print(f"{out}  ({n} frames @ {args.fps} fps = {n / args.fps:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
