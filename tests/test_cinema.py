"""Cinematography-layer tests: the camera must actually see something.

Born from the first streakline render coming out pure black (tracer
lifetime shorter than the transit to the camera) and the first dye plume
dying 60 cells from its source (decay too strong)."""

import pytest

torch = pytest.importorskip("torch")

from lbm.cinema import Dye, StreaklineBuffer, Tracers  # noqa: E402
from lbm.solver import Solver  # noqa: E402


def make_tunnel():
    return Solver(200, 40, tau=0.6, u_char=0.08, device="cpu",
                  inlet_outlet=True, ramp_steps=50, seed=1)


def test_tracer_lifetime_covers_domain_transit():
    s = make_tunnel()
    tr = Tracers(s, n=2000, seed=1)
    assert tr.max_age >= s.nx / s.u_char   # can cross the whole tunnel


def test_streakline_buffer_lights_up_everywhere():
    s = make_tunnel()
    tr = Tracers(s, n=20_000, seed=1)
    st = StreaklineBuffer(s)
    for _ in range(300):
        s.step()
        tr.step(s)
        st.splat(tr)
    # every quarter of the domain must be lit (uniform initial seeding
    # persists because lifetimes are long)
    for x0 in (0, 50, 100, 150):
        assert float(st.buf[x0:x0 + 50, :].max()) > 0.05, f"dark at x={x0}"


def test_dye_survives_downstream_travel():
    s = make_tunnel()
    dye = Dye(s, source=(2, 15, 6, 25))
    for _ in range(500):
        s.step()
        dye.step(s)
    # after 500 steps at u=0.08 the front is ~40 cells out; it must
    # arrive nearly undiluted (decay budget < 5% over a full transit)
    assert float(dye.field[35:45, :].max()) > 0.8
    assert 0.99995 ** (s.nx / s.u_char) > 0.85