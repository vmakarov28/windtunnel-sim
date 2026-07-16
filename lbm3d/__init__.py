"""3D GPU lattice-Boltzmann wind tunnel (D3Q19).

Lives beside the finished 2D program (`lbm/`) as a SEPARATE package with
its own scenes (`scenes3d/`), entry point (`run3d.py`), tests (`tests3d/`)
and renderer — the two versions run and test independently and never
share solver code. The single shared module is `lbm.units`: the Reynolds
triangle and its guard rails are dimension-blind by construction, and one
source of truth for the physics rails beats two copies that can drift.
"""

__version__ = "0.1.0"
