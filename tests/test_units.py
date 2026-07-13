"""Unit tests for lbm/units.py — the Reynolds-triangle converter.

These tests are the Phase 0 gate: unit conversion is the #1 LBM failure
mode, so the converter is tested before any physics exists.
"""

import math

import pytest

from lbm.units import (
    AIR_NU, CS2, TAU_MIN_BGK, U_LAT_MAX, UnitError, resolve,
)

# Cylinder Re=100 scene numbers, worked by hand:
#   Re = 0.03 * 0.05 / 1.5e-5 = 100
#   nu_lat = u_lat * N / Re = 0.06 * 40 / 100 = 0.024
#   tau = 0.5 + 3 * nu_lat = 0.572
CYL = dict(length_m=0.05, velocity_ms=0.03, nu_m2s=1.5e-5)


# --- solving each leg of the triangle -----------------------------------

def test_solves_tau_from_cells_and_u_lat():
    u = resolve(**CYL, cells=40, u_lat=0.06)
    assert u.reynolds == pytest.approx(100.0)
    assert u.nu_lat == pytest.approx(0.024)
    assert u.tau == pytest.approx(0.572)


def test_solves_u_lat_from_cells_and_tau():
    u = resolve(**CYL, cells=40, tau=0.572)
    assert u.u_lat == pytest.approx(0.06)


def test_solves_cells_from_u_lat_and_tau():
    u = resolve(**CYL, u_lat=0.06, tau=0.572)
    assert u.cells == pytest.approx(40.0)


def test_all_three_modes_agree():
    a = resolve(**CYL, cells=40, u_lat=0.06)
    b = resolve(**CYL, cells=40, tau=a.tau)
    c = resolve(**CYL, u_lat=0.06, tau=a.tau)
    for u in (b, c):
        assert u.cells == pytest.approx(a.cells)
        assert u.u_lat == pytest.approx(a.u_lat)
        assert u.tau == pytest.approx(a.tau)


# --- physical consistency ------------------------------------------------

def test_reynolds_matches_in_both_unit_systems():
    u = resolve(**CYL, cells=40, u_lat=0.06)
    re_lattice = u.u_lat * u.cells / u.nu_lat
    assert re_lattice == pytest.approx(u.reynolds)


def test_viscosity_conversion_identity():
    # nu_lat must equal nu_phys * dt / dx^2 — this is only true when the
    # Reynolds numbers match, so it is a strong end-to-end check.
    u = resolve(**CYL, cells=40, u_lat=0.06)
    assert u.nu_lat == pytest.approx(u.nu_m2s * u.dt_s / u.dx_m**2)


def test_dx_dt_known_values():
    u = resolve(**CYL, cells=40, u_lat=0.06)
    assert u.dx_m == pytest.approx(0.05 / 40)          # 1.25e-3 m
    assert u.dt_s == pytest.approx(0.06 * 1.25e-3 / 0.03)  # 2.5e-3 s
    assert u.steps_per_char_time == pytest.approx(40 / 0.06)


def test_mach_number():
    u = resolve(**CYL, cells=40, u_lat=0.06)
    assert u.mach == pytest.approx(0.06 / math.sqrt(CS2))


# --- the guard rails -----------------------------------------------------

def test_refuses_tau_below_bgk_floor():
    # Airfoil-scene numbers: Re = 20k, 400 cells, u = 0.1 -> tau = 0.506.
    with pytest.raises(UnitError, match="below the plain-BGK floor"):
        resolve(length_m=0.1, velocity_ms=3.0, cells=400, u_lat=0.1)


def test_sgs_floor_admits_the_airfoil_scene():
    u = resolve(length_m=0.1, velocity_ms=3.0, cells=400, u_lat=0.1, sgs=True)
    assert u.tau == pytest.approx(0.506)
    assert u.sgs


def test_sgs_floor_still_refuses_tau_at_half():
    # tau -> 0.5 exactly means zero viscosity: refused even with SGS.
    with pytest.raises(UnitError, match="below the SGS floor"):
        resolve(length_m=0.1, velocity_ms=3.0, cells=400, u_lat=1e-4, sgs=True)


def test_tau_exactly_at_floor_is_allowed():
    # Engineer tau = 0.55 exactly: nu_lat = (0.55-0.5)/3 = 1/60,
    # so with Re = 120, N = 40: u_lat = Re*nu_lat/N = 0.05.
    u = resolve(length_m=0.05, velocity_ms=0.036, cells=40, u_lat=0.05)
    assert u.reynolds == pytest.approx(120.0)
    assert u.tau == pytest.approx(TAU_MIN_BGK)


def test_refuses_u_lat_above_compressibility_cap():
    with pytest.raises(UnitError, match="compressibility"):
        resolve(**CYL, cells=40, u_lat=0.11)


def test_u_lat_exactly_at_cap_is_allowed():
    u = resolve(**CYL, cells=100, u_lat=U_LAT_MAX)
    assert u.u_lat == U_LAT_MAX


def test_refuses_solved_u_lat_above_cap_too():
    # The cap applies to a SOLVED u_lat as well: huge tau at low
    # resolution forces u_lat = Re*nu_lat/N above 0.1.
    with pytest.raises(UnitError, match="compressibility"):
        resolve(**CYL, cells=40, tau=1.0)


# --- input validation ----------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(),                                  # none given
    dict(cells=40),                          # one given
    dict(cells=40, u_lat=0.06, tau=0.572),   # all three given
])
def test_requires_exactly_two_lattice_inputs(kwargs):
    with pytest.raises(UnitError, match="exactly two"):
        resolve(**CYL, **kwargs)


@pytest.mark.parametrize("bad", [
    dict(length_m=-0.05, velocity_ms=0.03),
    dict(length_m=0.05, velocity_ms=0.0),
    dict(length_m=0.05, velocity_ms=0.03, nu_m2s=-1e-5),
])
def test_refuses_nonpositive_physical_inputs(bad):
    with pytest.raises(UnitError, match="positive"):
        resolve(**{**bad, "cells": 40, "u_lat": 0.06})


def test_refuses_nonpositive_lattice_inputs():
    with pytest.raises(UnitError, match="positive"):
        resolve(**CYL, cells=-40, u_lat=0.06)


def test_default_viscosity_is_air():
    u = resolve(length_m=0.05, velocity_ms=0.03, cells=40, u_lat=0.06)
    assert u.nu_m2s == AIR_NU


def test_result_is_immutable():
    u = resolve(**CYL, cells=40, u_lat=0.06)
    with pytest.raises(Exception):
        u.tau = 1.0  # a resolved unit system is a fact, not a knob
