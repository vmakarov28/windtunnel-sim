"""Airfoil loader + rasterizer tests, run on an analytic NACA section so
the pipeline is proven before any .dat file exists."""

import math

import pytest

torch = pytest.importorskip("torch")

from lbm.airfoil import load_selig, naca4, rasterize  # noqa: E402


def test_naca_loop_is_closed_and_unit_chord():
    poly = naca4("2412", n=80)
    assert torch.allclose(poly[0], poly[-1], atol=1e-5)   # TE closed
    assert float(poly[:, 0].min()) == pytest.approx(0.0, abs=1e-4)
    assert float(poly[:, 0].max()) == pytest.approx(1.0, abs=1e-4)


def test_selig_roundtrip(tmp_path):
    poly = naca4("0012", n=40)
    lines = ["NACA 0012 test"] + [f" {x:.6f}  {y:.6f}" for x, y in poly]
    p = tmp_path / "naca0012.dat"
    p.write_text("\n".join(lines), encoding="utf-8")
    name, loaded = load_selig(p)
    assert name == "NACA 0012 test"
    assert loaded.shape == poly.shape
    assert torch.allclose(loaded, poly, atol=1e-5)


def test_rasterize_area_matches_analytic():
    # NACA 0012 cross-section area ~ 0.6851 * t * c^2 (t = 0.12).
    poly = naca4("0012", n=200)
    chord = 300.0
    mask = rasterize(poly, 600, 300, chord, 200.0, 150.0, alpha_deg=0.0,
                     supersample=4)
    area = float(mask.sum()) / chord**2
    assert area == pytest.approx(0.6851 * 0.12, rel=0.03)


def test_rasterize_alpha_rotates_the_section():
    poly = naca4("0012", n=120)
    m0 = rasterize(poly, 400, 200, 150.0, 150.0, 100.0, alpha_deg=0.0)
    m10 = rasterize(poly, 400, 200, 150.0, 150.0, 100.0, alpha_deg=10.0)
    # symmetric section at alpha=0 is y-symmetric about its axis;
    # at alpha=10 the nose pitches up: mass shifts up ahead of the pivot
    ys = torch.arange(200, dtype=torch.float32)[None, :]
    cy0 = float((m0 * ys).sum() / m0.sum())
    front10 = m10[:150]                 # cells ahead of quarter-chord
    cy10 = float((front10 * ys).sum() / front10.sum())
    assert abs(cy0 - 100.0) < 1.0
    assert cy10 > 100.5
    # projected height grows roughly with c*sin(alpha)
    h0 = int(m0.any(dim=0).sum())
    h10 = int(m10.any(dim=0).sum())
    assert h10 > h0 + 0.6 * 150.0 * math.sin(math.radians(10.0)) - 3


def test_rasterized_section_is_watertight():
    # The property with physical consequences: no x-column inside the
    # section may be all-fluid (a leaking trailing edge would let flow
    # tunnel through the "solid"). The outline stamp guarantees it even
    # where the section is thinner than a cell, at any alignment/alpha.
    poly = naca4("0012", n=200)
    chord = 80.0  # coarse on purpose: TE is sub-cell thin
    for alpha in (0.0, 4.0, 8.0):
        for dx in (0.0, 0.3, 0.7):
            m = rasterize(poly, 200, 100, chord, 70 + dx, 50 + dx,
                          alpha_deg=alpha, supersample=4)
            cols = m.any(dim=1)
            xs = cols.nonzero().squeeze(1)
            gap_free = bool(cols[xs.min():xs.max() + 1].all())
            assert gap_free, f"leaky mask at alpha={alpha}, dx={dx}"
            # and the fat bias stays bounded
            area = float(m.sum()) / chord**2
            assert abs(area - 0.6851 * 0.12) / (0.6851 * 0.12) < 0.08
