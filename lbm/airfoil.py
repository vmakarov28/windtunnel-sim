"""Airfoil geometry: Selig .dat loader + supersampled rasterization.

Selig format: a name line, then x,y pairs tracing the surface from the
trailing edge over the UPPER surface to the leading edge and back along
the lower surface — one closed loop, chord normalized to [0, 1].

Rasterization is the honest part: a smooth curve becomes a staircase of
cells. We supersample (edge_supersample^2 points per cell, count the
fraction inside the polygon, threshold at half coverage) which centers
the staircase on the true surface instead of biasing it fat or thin —
but it is STILL a staircase, and Phase 6's report shows it zoomed-in
because it is one honest reason our Cd runs high.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch


def load_selig(path: str | Path) -> tuple[str, torch.Tensor]:
    """Read a Selig .dat -> (name, (N,2) float32 loop, TE closed)."""
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    name = lines[0].strip()
    pts = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) >= 2:
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            pts.append((x, y))
    if len(pts) < 10:
        raise ValueError(f"{path}: only {len(pts)} coordinate pairs")
    poly = torch.tensor(pts, dtype=torch.float32)
    # close the trailing edge: replace both TE endpoints by their midpoint
    if not torch.allclose(poly[0], poly[-1]):
        te = 0.5 * (poly[0] + poly[-1])
        poly[0] = te
        poly[-1] = te
    return name, poly


def naca4(code: str = "2412", n: int = 80) -> torch.Tensor:
    """Analytic NACA 4-digit section as a Selig-style loop (for tests —
    lets the whole pipeline run before any .dat file exists)."""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    beta = torch.linspace(0.0, math.pi, n)
    x = 0.5 * (1.0 - torch.cos(beta))            # cosine spacing
    yt = 5 * t * (0.2969 * x.sqrt() - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1036 * x**4)   # closed-TE variant
    yc = torch.where(x < p,
                     m / p**2 * (2 * p * x - x**2) if p > 0 else 0.0 * x,
                     m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
    upper = torch.stack([x, yc + yt], 1)
    lower = torch.stack([x, yc - yt], 1)
    return torch.cat([upper.flip(0), lower[1:]], 0)  # TE->LE->TE loop


def rasterize(
    poly: torch.Tensor,           # (N,2) unit-chord loop
    nx: int, ny: int,
    chord_cells: float,
    center_x: float, center_y: float,   # of the quarter-chord point [cells]
    alpha_deg: float = 0.0,
    supersample: int = 4,
) -> torch.Tensor:
    """Rotate by -alpha about quarter-chord, scale, and rasterize to a
    boolean (nx,ny) mask via supersampled point-in-polygon."""
    a = math.radians(-alpha_deg)      # +alpha pitches the nose UP
    ca, sa = math.cos(a), math.sin(a)
    p = poly - torch.tensor([0.25, 0.0])            # about quarter chord
    rot = torch.stack([p[:, 0] * ca - p[:, 1] * sa,
                       p[:, 0] * sa + p[:, 1] * ca], 1)
    verts = rot * chord_cells + torch.tensor([center_x, center_y])

    # bounding box in cells, padded
    x0 = max(int(verts[:, 0].min()) - 2, 0)
    x1 = min(int(verts[:, 0].max()) + 3, nx)
    y0 = max(int(verts[:, 1].min()) - 2, 0)
    y1 = min(int(verts[:, 1].max()) + 3, ny)
    mask = torch.zeros((nx, ny), dtype=torch.bool)
    if x1 <= x0 or y1 <= y0:
        return mask

    ss = supersample
    # supersample points: cell (i,j) covered by ss*ss samples at centers
    xs = torch.arange(x0, x1, dtype=torch.float32)
    ys = torch.arange(y0, y1, dtype=torch.float32)
    off = (torch.arange(ss, dtype=torch.float32) + 0.5) / ss
    px = (xs[:, None] + off[None, :]).reshape(-1)   # (W*ss,)
    py = (ys[:, None] + off[None, :]).reshape(-1)   # (H*ss,)

    # crossing-number test, vectorized over all sample points x edges
    vx, vy = verts[:, 0], verts[:, 1]
    wx, wy = vx.roll(-1), vy.roll(-1)
    inside = torch.zeros((px.numel(), py.numel()), dtype=torch.int8)
    for k in range(len(vx) - 1):
        x_a, y_a, x_b, y_b = vx[k], vy[k], wx[k], wy[k]
        if y_a == y_b:
            continue
        cond = ((y_a <= py) != (y_b <= py))[None, :]    # crosses this row
        t = (py - y_a) / (y_b - y_a)
        x_cross = x_a + t * (x_b - x_a)
        inside += (cond & (px[:, None] < x_cross[None, :])).to(torch.int8)
    frac = (inside % 2).to(torch.float32)
    frac = frac.reshape(x1 - x0, ss, y1 - y0, ss).mean(dim=(1, 3))
    mask[x0:x1, y0:y1] = frac >= 0.5

    # Weld any leaks: where the section is thinner than half a cell (the
    # trailing-edge sliver), coverage can leave an x-column with no solid
    # at all — and a leaking trailing edge is a physics disaster. For
    # exactly those columns (and no others: no global fattening), stamp
    # the cells the outline passes through.
    seg = verts[1:] - verts[:-1]
    n_samp = (seg.abs().max(dim=1).values * 2).ceil().long().clamp(min=1)
    pts = torch.cat([
        verts[k][None, :] + torch.linspace(0, 1, int(n_samp[k]) + 1)[:, None]
        * seg[k][None, :]
        for k in range(len(seg))
    ])
    oxi = pts[:, 0].long().clamp(0, nx - 1)
    oyi = pts[:, 1].long().clamp(0, ny - 1)
    # The span is the OUTLINE's, not the coverage mask's: coverage can
    # drop the last few sub-half-cell trailing-edge columns entirely,
    # silently shortening the chord (caught by the staircase figure —
    # the mask ended 7 cells before the true TE).
    cols = mask.any(dim=1)
    lo, hi = int(oxi.min()), int(oxi.max())
    leaky = ~cols
    leaky[:lo] = leaky[hi + 1:] = False
    weld = leaky[oxi]
    if bool(weld.any()):
        mask[oxi[weld], oyi[weld]] = True
    return mask
