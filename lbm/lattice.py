"""D2Q9 lattice constants (Kruger et al. 2017, table 3.1).

The 9 velocities: rest, 4 axis neighbors, 4 diagonals.
Weights:  w_0 = 4/9,  axis = 1/9,  diagonal = 1/36.
Speed of sound: c_s^2 = 1/3.

These satisfy the isotropy conditions the Chapman-Enskog expansion needs
(sum w = 1, sum w e = 0, sum w e_a e_b = c_s^2 delta_ab, ...); they are
tested explicitly in tests/test_lattice.py.

(The 3D D3Q19 version of this project lives on the `3d-d3q19` branch.)
"""

from __future__ import annotations

Q = 9

# Order: rest, +/- axis pairs, +/- diagonal pairs. OPP is computed, not
# assumed, so reordering can never silently break bounce-back.
E: list[tuple[int, int]] = [
    (0, 0),
    (1, 0), (-1, 0),
    (0, 1), (0, -1),
    (1, 1), (-1, -1),
    (1, -1), (-1, 1),
]

W: list[float] = [4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4

# OPP[q] is the index of -E[q]; bounce-back reflects f_q into f_OPP[q].
OPP: list[int] = [E.index((-ex, -ey)) for (ex, ey) in E]

CS2 = 1.0 / 3.0
