#!/usr/bin/env python3
"""Generate a Selig-format .dat for any NACA 4-digit airfoil.

    python scripts/make_naca.py 4412                 # -> assets/naca4412.dat
    python scripts/make_naca.py 0012 --points 150
    python scripts/make_naca.py 2412 --out assets/my2412.dat

Why this exists: .dat files found in the wild are often sparse (30-40
points) or missing the name line — and the Selig convention says line 1
IS the name, so a headerless file silently loses its first coordinate
(usually the trailing edge) to the name field. The rasterizer wants a
dense, watertight outline. This writes a clean one from the formula:
cosine-spaced x (dense at the leading edge, where curvature lives) and
the closed-trailing-edge thickness polynomial, so the outline seals.

Equations: Abbott & von Doenhoff, "Theory of Wing Sections", sec. 6.4.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def naca4(code: str, n: int) -> list[tuple[float, float]]:
    """Selig-ordered outline: TE -> upper -> LE -> lower -> TE."""
    m = int(code[0]) / 100.0          # max camber
    p = int(code[1]) / 10.0           # its chordwise position
    t = int(code[2:]) / 100.0         # thickness

    def camber(x: float) -> tuple[float, float]:
        if m == 0.0:
            return 0.0, 0.0
        if x < p:
            return (m / p**2) * (2*p*x - x*x), (2*m / p**2) * (p - x)
        return ((m / (1-p)**2) * ((1 - 2*p) + 2*p*x - x*x),
                (2*m / (1-p)**2) * (p - x))

    def thickness(x: float) -> float:
        # -0.1036 closes the trailing edge (the open-TE -0.1015 leaves a
        # gap the rasterizer would have to weld)
        return 5*t * (0.2969*math.sqrt(x) - 0.1260*x - 0.3516*x*x
                      + 0.2843*x**3 - 0.1036*x**4)

    xs = [0.5 * (1 - math.cos(math.pi * i / (n - 1))) for i in range(n)]
    upper, lower = [], []
    for x in xs:
        yc, dyc = camber(x)
        yt = thickness(x)
        th = math.atan(dyc)
        upper.append((x - yt*math.sin(th), yc + yt*math.cos(th)))
        lower.append((x + yt*math.sin(th), yc - yt*math.cos(th)))
    # TE -> upper -> LE (shared point once) -> lower -> TE
    return upper[::-1] + lower[1:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("code", help="4-digit NACA code, e.g. 4412")
    ap.add_argument("--points", type=int, default=120,
                    help="points per surface (default 120)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default assets/naca<code>.dat)")
    args = ap.parse_args()
    if len(args.code) != 4 or not args.code.isdigit():
        ap.error("code must be 4 digits, e.g. 4412")

    out = args.out or Path("assets") / f"naca{args.code}.dat"
    pts = naca4(args.code, args.points)
    lines = [f"NACA {args.code} (generated, {args.points} pts/surface, "
             f"closed TE)"]
    lines += [f" {x:.6f}  {y:+.6f}" for x, y in pts]
    out.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"{out}: {len(pts)} points, x in "
          f"[{min(p[0] for p in pts):.4f}, {max(p[0] for p in pts):.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
