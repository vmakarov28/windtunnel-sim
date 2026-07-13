"""Unit conversion between physical and lattice units — the Reynolds triangle.

The single most common way an LBM simulation goes wrong is unit conversion,
so every scene in this project is defined in PHYSICAL terms (meters, m/s,
m^2/s) and converted here. Raw tau or lattice velocities are never set by
hand anywhere else in the codebase.

How it works
------------
Two flows are dynamically similar when their Reynolds numbers match:

    Re = U * L / nu          (physical:  m/s * m / (m^2/s), dimensionless)
    Re = u_lat * N / nu_lat  (lattice:   dx = dt = 1 by convention)

Given the physical side (which fixes Re) you pick exactly TWO of the three
lattice quantities

    N      cells across the characteristic length (resolution)
    u_lat  characteristic velocity in lattice units (compressibility)
    tau    BGK relaxation time (viscosity / stability)

and this module solves for the third, via the lattice viscosity relation
(Kruger et al., "The Lattice Boltzmann Method", 2017, eq. 4.17):

    nu_lat = c_s^2 * (tau - 1/2),   c_s^2 = 1/3  for D3Q19 (and D2Q9)

Everything here is dimension-agnostic: the Reynolds triangle and both
guard rails are identical in 2D and 3D.

Hard limits (the whole point of this module)
--------------------------------------------
* tau < TAU_MIN_BGK (0.55) is REFUSED: plain BGK loses accuracy and then
  stability as tau -> 1/2. A Smagorinsky subgrid model (Phase 5) adds local
  eddy viscosity, so with sgs=True the floor drops to TAU_MIN_SGS — the
  *molecular* tau may then approach (but never reach) 1/2.
* u_lat > U_LAT_MAX (0.1) is REFUSED: LBM solves weakly-compressible flow;
  errors grow like O(Ma^2) with Ma = u_lat / c_s. u_lat = 0.1 is Ma ~ 0.17,
  ~3% compressibility error — our ceiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Lattice speed of sound for D3Q19 (Kruger et al. 2017, table 3.1).
CS2 = 1.0 / 3.0
CS = math.sqrt(CS2)

# Kinematic viscosity of air at ~20 C, m^2/s. Default for all scenes.
AIR_NU = 1.5e-5

TAU_MIN_BGK = 0.55   # refuse below this without a turbulence model
TAU_MIN_SGS = 0.501  # with Smagorinsky, molecular tau may approach 1/2
U_LAT_MAX = 0.1      # Ma ~ 0.17 -> ~3% compressibility error; the ceiling

_EPS = 1e-12  # float-tolerance so boundary values (tau=0.55, u=0.1) pass


class UnitError(ValueError):
    """A scene's physical/lattice parameters are inconsistent or unsafe."""


@dataclass(frozen=True)
class LatticeUnits:
    """A fully-resolved, validated unit system for one scene.

    Physical inputs are stored as given; everything else is derived.
    Immutable on purpose: a resolved unit system is a fact, not a knob.
    """

    # -- physical inputs ---------------------------------------------------
    length_m: float     # characteristic length (chord, diameter, height) [m]
    velocity_ms: float  # characteristic velocity (freestream, lid) [m/s]
    nu_m2s: float       # kinematic viscosity [m^2/s]

    # -- lattice quantities (two given, one solved) ------------------------
    cells: float        # cells across the characteristic length
    u_lat: float        # characteristic velocity, lattice units
    tau: float          # BGK relaxation time

    sgs: bool = False   # was the subgrid-model tau floor used?

    # -- derived -----------------------------------------------------------
    @property
    def reynolds(self) -> float:
        return self.velocity_ms * self.length_m / self.nu_m2s

    @property
    def nu_lat(self) -> float:
        # nu = c_s^2 (tau - 1/2)  (Kruger et al. 2017, eq. 4.17)
        return CS2 * (self.tau - 0.5)

    @property
    def mach(self) -> float:
        return self.u_lat / CS

    @property
    def dx_m(self) -> float:
        """Physical size of one lattice cell [m]."""
        return self.length_m / self.cells

    @property
    def dt_s(self) -> float:
        """Physical duration of one time step [s].

        Fixed by matching velocities: u_lat = U_phys * dt / dx.
        """
        return self.u_lat * self.dx_m / self.velocity_ms

    @property
    def steps_per_char_time(self) -> float:
        """Time steps per convective time L/U — the flow's natural clock."""
        return self.cells / self.u_lat

    def report(self, title: str = "unit system") -> str:
        lines = [
            f"--- {title} " + "-" * max(1, 58 - len(title)),
            f"  physical   L = {self.length_m:g} m   U = {self.velocity_ms:g} m/s"
            f"   nu = {self.nu_m2s:g} m^2/s",
            f"  Reynolds   Re = {self.reynolds:.6g}",
            f"  lattice    N = {self.cells:g} cells   u_lat = {self.u_lat:g}"
            f"   nu_lat = {self.nu_lat:.6g}",
            f"  relaxation tau = {self.tau:.6g}"
            + ("   (SGS floor: Smagorinsky required)" if self.sgs else ""),
            f"  Mach       Ma = {self.mach:.4g}  (compressibility error"
            f" ~ {100 * self.mach ** 2:.2g}%)",
            f"  resolution dx = {self.dx_m:.6g} m   dt = {self.dt_s:.6g} s",
            f"  clock      {self.steps_per_char_time:.0f} steps per convective"
            f" time L/U",
            "-" * 64,
        ]
        return "\n".join(lines)


def resolve(
    length_m: float,
    velocity_ms: float,
    nu_m2s: float = AIR_NU,
    *,
    cells: float | None = None,
    u_lat: float | None = None,
    tau: float | None = None,
    sgs: bool = False,
) -> LatticeUnits:
    """Solve the third leg of the Reynolds triangle and validate the result.

    Provide the physical side plus EXACTLY TWO of (cells, u_lat, tau).
    Returns a validated LatticeUnits or raises UnitError with a diagnosis.
    """
    for name, val in [("length_m", length_m), ("velocity_ms", velocity_ms),
                      ("nu_m2s", nu_m2s)]:
        if not (isinstance(val, (int, float)) and val > 0):
            raise UnitError(f"{name} must be a positive number, got {val!r}")

    given = {"cells": cells, "u_lat": u_lat, "tau": tau}
    provided = [k for k, v in given.items() if v is not None]
    if len(provided) != 2:
        raise UnitError(
            "provide exactly two of (cells, u_lat, tau); "
            f"got {provided or 'none'}. The third is solved from Re."
        )
    for k in provided:
        if given[k] <= 0:
            raise UnitError(f"{k} must be positive, got {given[k]!r}")

    re = velocity_ms * length_m / nu_m2s

    # Solve the missing leg:  Re = u_lat * cells / nu_lat,
    # nu_lat = (tau - 1/2) / 3.
    if tau is None:
        nu_lat = u_lat * cells / re
        tau = 0.5 + nu_lat / CS2
    elif u_lat is None:
        u_lat = re * CS2 * (tau - 0.5) / cells
    else:  # cells is None
        cells = re * CS2 * (tau - 0.5) / u_lat

    units = LatticeUnits(
        length_m=length_m, velocity_ms=velocity_ms, nu_m2s=nu_m2s,
        cells=cells, u_lat=u_lat, tau=tau, sgs=sgs,
    )

    # -- the guard rails ---------------------------------------------------
    if u_lat > U_LAT_MAX + _EPS:
        raise UnitError(
            f"u_lat = {u_lat:.4g} exceeds {U_LAT_MAX} (Ma = {units.mach:.3f}, "
            f"compressibility error ~ {100 * units.mach ** 2:.1f}%). "
            "Fix: lower u_lat (more steps) or raise tau/resolution instead."
        )
    tau_min = TAU_MIN_SGS if sgs else TAU_MIN_BGK
    if tau < tau_min - _EPS:
        hint = (
            "Fix: increase cells (resolution) or increase u_lat (up to 0.1); "
            "note that LOWERING u_lat lowers tau further"
            if sgs else
            "Fix: increase cells (resolution), increase u_lat (up to 0.1), "
            "or enable the Smagorinsky model (sgs=true, Phase 5+)"
        )
        raise UnitError(
            f"tau = {tau:.4g} is below the "
            f"{'SGS' if sgs else 'plain-BGK'} floor of {tau_min} "
            f"at Re = {re:.4g} with cells = {cells:g}, u_lat = {u_lat:g}. "
            f"{hint}."
        )

    return units
