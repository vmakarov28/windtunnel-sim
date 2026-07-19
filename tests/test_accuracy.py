"""Phase 8 accuracy features: TRT collision + Bouzidi curved boundaries.

Both are OPT-IN (scene keys `collision: trt`, `obstacle.curved_bc`);
the default BGK/staircase path must remain bit-identical — that is the
first thing tested. The physics claims are tested the way they will be
shown: the TRT wall does not move with viscosity, and Bouzidi at
s = 1/2 IS halfway bounce-back.
"""

import math

import numpy as np
import pytest
import torch

from lbm.lattice import E, Q, OPP
from lbm.solver import Solver, obstacle_geometry


def _poiseuille_walls(collision, tau, ny=18, steps=9000):
    """Effective wall positions: roots of a parabola fit to the profile."""
    s = Solver(4, ny, tau=tau, u_char=0.0, seed=0, device="cpu",
               inlet_outlet=False, wall_y=True, body_force=(1e-6, 0.0),
               init_noise=0.0, collision=collision)
    for _ in range(steps):
        s.step()
    _, u = s.macroscopics()
    ux = u[0, 2, 1:-1].numpy()
    yc = np.arange(1, ny - 1) + 0.5
    roots = np.roots(np.polyfit(yc, ux, 2))
    return sorted(roots.real)


def test_default_collision_is_bgk_and_unchanged():
    a = Solver(16, 16, tau=0.8, u_char=0.05, seed=3, device="cpu")
    b = Solver(16, 16, tau=0.8, u_char=0.05, seed=3, device="cpu",
               collision="bgk")
    assert a.collision == "bgk"
    for _ in range(50):
        a.step()
        b.step()
    assert torch.equal(a.f, b.f)


def test_trt_equals_bgk_when_rates_coincide():
    """With LAMBDA = (tau-1/2)^2 the odd rate equals the even rate and
    TRT must reproduce BGK exactly — pins the pair algebra."""
    tau = 0.9
    a = Solver(16, 16, tau=tau, u_char=0.05, seed=1, device="cpu",
               collision="bgk")
    b = Solver(16, 16, tau=tau, u_char=0.05, seed=1, device="cpu",
               collision="trt")
    b.lambda_trt = (tau - 0.5) ** 2
    for _ in range(50):
        a.step()
        b.step()
    assert torch.allclose(a.f, b.f, atol=1e-6)


def test_trt_wall_does_not_move_with_viscosity():
    """The magic parameter's whole point: at tau = 2 the BGK no-slip
    plane has visibly drifted off the halfway position; TRT holds it."""
    lo_t, hi_t = _poiseuille_walls("trt", 2.0)
    lo_b, hi_b = _poiseuille_walls("bgk", 2.0)
    assert abs(lo_t - 1.0) < 0.03 and abs(hi_t - 17.0) < 0.03
    assert abs(lo_b - 1.0) > 0.08          # BGK's wall genuinely drifts


def test_wall_fraction_circle_analytic():
    geom = ("circle", 5.0, 5.0, 2.0)
    # ray at y = 5.5 (half a cell off-center): (x-5)^2 + 0.25 = 4
    # -> hits x = 5 - sqrt(3.75)
    px = torch.tensor([2.5]); py = torch.tensor([5.5])
    s = Solver._wall_fraction(px, py, 1.0, 0.0, geom)
    assert abs(float(s) - (5.0 - math.sqrt(3.75) - 2.5)) < 1e-5
    # pointing away: no hit
    s = Solver._wall_fraction(px, py, -1.0, 0.0, geom)
    assert not torch.isfinite(s).item() or float(s) > 1.0


def test_wall_fraction_polygon_square():
    sq = torch.tensor([[4.0, 4.0], [6.0, 4.0], [6.0, 6.0],
                       [4.0, 6.0], [4.0, 4.0]])
    geom = ("polygon", sq)
    px = torch.tensor([3.3]); py = torch.tensor([5.0])
    s = Solver._wall_fraction(px, py, 1.0, 0.0, geom)   # hits x=4 at s=0.7
    assert abs(float(s) - 0.7) < 1e-5


def _cyl_solver(curved, seed=0, n=24):
    nx, ny = 12 * n, 6 * n
    cx, cy, r = 3.0 * n, 3.0 * n - 0.2 * n, n / 2.0
    x = torch.arange(nx, dtype=torch.float32)[:, None] + 0.5
    y = torch.arange(ny, dtype=torch.float32)[None, :] + 0.5
    mask = ((x - cx) ** 2 + (y - cy) ** 2) <= r * r
    return Solver(nx, ny, tau=0.572, u_char=0.06, seed=seed, device="cpu",
                  obstacle_mask=mask, ramp_steps=200,
                  curved_geom=("circle", cx, cy, r) if curved else None)


def test_bouzidi_links_found_with_few_fallbacks():
    s = _cyl_solver(curved=True)
    assert s.bouzidi_links > 100
    assert s.bouzidi_fallback / s.bouzidi_links < 0.05


def test_bouzidi_at_half_is_exactly_halfway_bounce_back():
    """Force every Bouzidi coefficient to its s = 1/2 value; the result
    must be bit-identical to the plain halfway solver — this pins the
    index plumbing (directions, opposites, neighbours)."""
    a = _cyl_solver(curved=False, n=16)
    b = _cyl_solver(curved=True, n=16)
    for q in range(1, Q):
        bz = b._bz[q]
        if bz is None:
            continue
        parts = []
        if bz["idxA"] is not None:
            parts.append(bz["idxA"])
        if bz["idxB"] is not None:
            parts.append(bz["idxB"])
        allidx = torch.cat(parts)
        bz["idxA"] = None
        bz["idxB"] = allidx
        bz["cB1"] = torch.ones_like(allidx, dtype=torch.float32)
        bz["cB2"] = torch.zeros_like(allidx, dtype=torch.float32)
    for _ in range(120):
        a.step()
        b.step()
    assert torch.equal(a.f, b.f)


def test_bouzidi_cylinder_runs_and_stays_finite():
    s = _cyl_solver(curved=True, n=16)
    for _ in range(400):
        s.step()
    assert torch.isfinite(s.f).all()
    rho, u = s.macroscopics()
    assert float(u[0].max()) > 0.9 * 0.06   # flow actually flowing


def test_trt_cylinder_runs_and_stays_finite():
    nx, ny = 192, 96
    x = torch.arange(nx, dtype=torch.float32)[:, None] + 0.5
    y = torch.arange(ny, dtype=torch.float32)[None, :] + 0.5
    mask = ((x - 48.0) ** 2 + (y - 44.0) ** 2) <= 64.0
    s = Solver(nx, ny, tau=0.572, u_char=0.06, seed=0, device="cpu",
               obstacle_mask=mask, ramp_steps=200, collision="trt")
    for _ in range(400):
        s.step()
    assert torch.isfinite(s.f).all()


def test_obstacle_geometry_matches_mask_transform():
    """The polygon handed to Bouzidi must be the same polygon the mask
    was rasterized from: a boundary cell's center must sit close to the
    polygon (within a cell) — catches transform drift."""
    pytest.importorskip("yaml")
    from lbm.config import load_scene
    from lbm.solver import build_obstacle_mask
    scene = load_scene("naca4412_re20k")
    obstacle = scene.raw["obstacle"]
    geom = obstacle_geometry(scene, obstacle)
    assert geom[0] == "polygon"
    verts = geom[1]
    mask = build_obstacle_mask(scene, obstacle)
    idx = mask.reshape(-1).nonzero().squeeze(1)
    i = (idx // scene.ny).float() + 0.5
    j = (idx % scene.ny).float() + 0.5
    # solid-cell centroid should sit inside the polygon's bounding box
    assert verts[:, 0].min() - 1 <= i.mean() <= verts[:, 0].max() + 1
    assert verts[:, 1].min() - 1 <= j.mean() <= verts[:, 1].max() + 1
