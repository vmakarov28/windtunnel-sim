"""D2Q9 lattice sanity: the isotropy identities Chapman-Enskog requires."""

import pytest

torch = pytest.importorskip("torch")

from lbm.lattice import CS2, E, OPP, Q, W  # noqa: E402


def test_counts_and_weights():
    assert len(E) == len(W) == len(OPP) == Q == 9
    assert sum(W) == pytest.approx(1.0)
    assert W[0] == pytest.approx(4 / 9)
    assert sorted(W[1:5]) == pytest.approx([1 / 9] * 4)
    assert sorted(W[5:]) == pytest.approx([1 / 36] * 4)


def test_first_moment_vanishes():
    for a in range(2):
        assert sum(W[q] * E[q][a] for q in range(Q)) == pytest.approx(0.0)


def test_second_moment_is_isotropic():
    # sum_q w_q e_qa e_qb = c_s^2 delta_ab  — this is what makes c_s^2 = 1/3
    for a in range(2):
        for b in range(2):
            m = sum(W[q] * E[q][a] * E[q][b] for q in range(Q))
            assert m == pytest.approx(CS2 if a == b else 0.0)


def test_third_moment_vanishes():
    for a in range(2):
        for b in range(2):
            for c in range(2):
                m = sum(W[q] * E[q][a] * E[q][b] * E[q][c] for q in range(Q))
                assert m == pytest.approx(0.0)


def test_opposites():
    for q in range(Q):
        ex, ey = E[q]
        ox, oy = E[OPP[q]]
        assert (ox, oy) == (-ex, -ey)
        assert OPP[OPP[q]] == q
