"""D3Q19 lattice constants (Kruger et al. 2017, table 3.1 / fig. 3.5).

The 19 velocities: the rest particle, 6 axis neighbors, and 12 edge
(face-diagonal) neighbors of the unit cube. The corner directions are
omitted — that's the "19" in D3Q19; they carry weight 0 in this quadrature.

Weights:  w_0 = 1/3,  axis = 1/18,  edge-diagonal = 1/36.
Speed of sound: c_s^2 = 1/3 (same as D2Q9).

These satisfy the isotropy conditions the Chapman-Enskog expansion needs
(sum w = 1, sum w e = 0, sum w e_a e_b = c_s^2 delta_ab, ...); they are
tested explicitly in tests/test_lattice.py.
"""

from __future__ import annotations

Q = 19

# Order: rest, then +/- axis pairs, then +/- edge-diagonal pairs.
# Opposite directions are adjacent (2k+1, 2k+2), but OPP is computed, not
# assumed, so reordering can never silently break bounce-back.
E: list[tuple[int, int, int]] = [
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0),
    (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, -1),
    (1, 0, -1), (-1, 0, 1),
    (0, 1, 1), (0, -1, -1),
    (0, 1, -1), (0, -1, 1),
]

W: list[float] = [1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12

# OPP[q] is the index of -E[q]; bounce-back reflects f_q into f_OPP[q].
OPP: list[int] = [E.index((-ex, -ey, -ez)) for (ex, ey, ez) in E]

CS2 = 1.0 / 3.0
