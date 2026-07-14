"""Smagorinsky subgrid model tests (Phase 5).

The model must be near-inert where the grid resolves the flow (else it
would corrupt the validated laminar results) and active + stabilizing in
under-resolved shear.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from lbm.solver import Solver  # noqa: E402


def test_sgs_is_near_inert_on_resolved_laminar_flow():
    # Same Taylor-Green decay as the BGK gate: with SGS on, the measured
    # decay must still match the molecular-viscosity prediction closely
    # (nu_t ~ (Cs)^2 |S| is tiny for a resolved smooth field).
    n, tau, u0 = 64, 0.8, 0.03
    nu = (tau - 0.5) / 3.0
    k = 2.0 * math.pi / n
    s = Solver(n, n, tau=tau, u_char=u0, device="cpu",
               inlet_outlet=False, init_noise=0.0, sgs=True, cs_smag=0.14)
    x = (torch.arange(n, dtype=torch.float32) + 0.5) * k
    u = torch.zeros((2, n, n))
    u[0] = -u0 * torch.cos(x)[:, None] * torch.sin(x)[None, :]
    u[1] = u0 * torch.sin(x)[:, None] * torch.cos(x)[None, :]
    s._write_equilibrium(s.f, torch.ones((n, n)), u)

    steps = 300
    e0 = float((u * u).sum())
    for _ in range(steps):
        s.step()
    _, u_t = s.macroscopics()
    decay = math.sqrt(float((u_t * u_t).sum()) / e0)
    expected = math.exp(-2.0 * nu * k * k * steps)
    assert decay == pytest.approx(expected, rel=0.025)


def test_sgs_activates_in_shear():
    # A sheared flow must raise tau_eff above the molecular tau somewhere.
    nx, ny = 64, 32
    s = Solver(nx, ny, tau=0.51, u_char=0.08, device="cpu",
               inlet_outlet=False, init_noise=0.0, sgs=True, cs_smag=0.14)
    u = torch.zeros((2, nx, ny))
    y = torch.arange(ny, dtype=torch.float32)
    u[0] = 0.08 * torch.tanh((y - ny / 2) / 2.0)[None, :]  # shear layer
    s._write_equilibrium(s.f, torch.ones((nx, ny)), u)
    for _ in range(20):
        s.step()
    assert s.last_tau_eff_max > 0.51 + 1e-4


def test_sgs_stabilizes_marginal_tau():
    # tau = 0.502 with a bluff body: plain BGK at this margin on a coarse
    # cylinder is on the edge; the model must keep the run finite.
    nx, ny = 200, 80
    x = torch.arange(nx, dtype=torch.float32)[:, None] + 0.5
    y = torch.arange(ny, dtype=torch.float32)[None, :] + 0.5
    mask = ((x - 50.0) ** 2 + (y - 39.0) ** 2 <= 64.0)   # r = 8
    s = Solver(nx, ny, tau=0.502, u_char=0.08, device="cpu",
               obstacle_mask=mask, inlet_outlet=True, ramp_steps=200,
               seed=1, sgs=True, cs_smag=0.14)
    for _ in range(800):
        s.step()
    g = s.check_guards()   # raises on NaN / runaway
    assert not g["has_nan"]
