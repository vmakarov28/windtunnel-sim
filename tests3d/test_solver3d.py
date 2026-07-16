"""Physics and infrastructure tests for the D3Q19 solver (CPU, small grids).

The low-level Solver constructor is used deliberately here: tests need
engineered analytic cases (known tau), which is exactly the exemption the
units discipline grants to tests and validation scripts.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from lbm3d.solver import SimulationBlowup, Solver, capture_failure  # noqa: E402


def make_periodic_box(nx=32, ny=32, nz=8, tau=0.8, u=0.05, **kw):
    return Solver(nx, ny, nz, tau=tau, u_char=u, device="cpu",
                  inlet_outlet=False, init_noise=0.0, **kw)


def tg_field(n, nz, u0):
    """z-invariant Taylor-Green velocity field on an (n, n, nz) grid."""
    k = 2.0 * math.pi / n
    x = (torch.arange(n, dtype=torch.float32) + 0.5) * k
    u = torch.zeros((3, n, n, nz))
    u[0] = (-u0 * torch.cos(x)[:, None] * torch.sin(x)[None, :])[:, :, None]
    u[1] = (u0 * torch.sin(x)[:, None] * torch.cos(x)[None, :])[:, :, None]
    return u, k


def test_uniform_flow_is_a_fixed_point():
    s = make_periodic_box()
    rho = torch.ones((s.nx, s.ny, s.nz))
    u = torch.zeros((3, s.nx, s.ny, s.nz))
    u[0], u[1], u[2] = 0.05, -0.02, 0.01
    s._write_equilibrium(s.f, rho, u)
    for _ in range(10):
        s.step()
    _, u_after = s.macroscopics()
    assert float((u_after[0] - 0.05).abs().max()) < 1e-5
    assert float((u_after[1] + 0.02).abs().max()) < 1e-5
    assert float((u_after[2] - 0.01).abs().max()) < 1e-5


def test_taylor_green_viscous_decay():
    # u decays as exp(-2 nu k^2 t): collision, streaming AND the units
    # relation nu = (tau - 1/2)/3 against an exact Navier-Stokes solution.
    n, tau, u0 = 64, 0.8, 0.03
    nu = (tau - 0.5) / 3.0
    s = Solver(n, n, 4, tau=tau, u_char=u0, device="cpu",
               inlet_outlet=False, init_noise=0.0)
    u, k = tg_field(n, 4, u0)
    s._write_equilibrium(s.f, torch.ones((n, n, 4)), u)
    steps = 300
    e0 = float((u * u).sum())
    for _ in range(steps):
        s.step()
    _, u_t = s.macroscopics()
    decay = math.sqrt(float((u_t * u_t).sum()) / e0)
    assert decay == pytest.approx(math.exp(-2.0 * nu * k * k * steps),
                                  rel=0.02)


def test_sgs_is_near_inert_on_resolved_laminar_flow():
    # Same decay with the Smagorinsky model ON: nu_t ~ Cs^2 |S| is tiny for
    # a resolved smooth field, so the answer must not move (the contract
    # that lets SGS coexist with the validated laminar gates).
    n, tau, u0 = 64, 0.8, 0.03
    nu = (tau - 0.5) / 3.0
    s = Solver(n, n, 4, tau=tau, u_char=u0, device="cpu",
               inlet_outlet=False, init_noise=0.0, sgs=True, cs_smag=0.14)
    u, k = tg_field(n, 4, u0)
    s._write_equilibrium(s.f, torch.ones((n, n, 4)), u)
    steps = 300
    e0 = float((u * u).sum())
    for _ in range(steps):
        s.step()
    _, u_t = s.macroscopics()
    decay = math.sqrt(float((u_t * u_t).sum()) / e0)
    assert decay == pytest.approx(math.exp(-2.0 * nu * k * k * steps),
                                  rel=0.025)


def test_sgs_activates_in_shear():
    nx, ny, nz = 48, 32, 4
    s = Solver(nx, ny, nz, tau=0.51, u_char=0.08, device="cpu",
               inlet_outlet=False, init_noise=0.0, sgs=True, cs_smag=0.14)
    u = torch.zeros((3, nx, ny, nz))
    y = torch.arange(ny, dtype=torch.float32)
    u[0] = (0.08 * torch.tanh((y - ny / 2) / 2.0))[None, :, None]
    s._write_equilibrium(s.f, torch.ones((nx, ny, nz)), u)
    for _ in range(20):
        s.step()
    assert s.last_tau_eff_max > 0.51 + 1e-4


def test_open_boundaries_hold_freestream_3d():
    # REGRESSION for the original 3D bug: an EMPTY tunnel must settle at
    # u = u_in / rho = 1 in the interior (the old fixed-rho inlet + copy
    # outlet pressurized to rho~1.06 and ran ~12% slow).
    s = Solver(160, 48, 8, tau=0.7, u_char=0.08, device="cpu",
               inlet_outlet=True, ramp_steps=150, init_noise=0.0)
    for _ in range(2500):
        s.step()
    rho, u = s.macroscopics()
    assert float((u[0, 10:-14, :, :] - 0.08).abs().max()) < 0.008
    assert float((rho[10:-14, :, :] - 1.0).abs().max()) < 0.005
    assert abs(s.guards()["u_max"] - 0.08) < 0.008


def test_inlet_outlet_cylinder_sheds_wake():
    # Offset cylinder: must accelerate the flow past u_in (the old run
    # never did) and grow transverse wake motion (the street's seed).
    nx, ny, nz = 160, 60, 4
    x = torch.arange(nx, dtype=torch.float32)[:, None] + 0.5
    y = torch.arange(ny, dtype=torch.float32)[None, :] + 0.5
    mask = ((x - 40.0) ** 2 + (y - 28.0) ** 2 <= 36.0)
    mask = mask[:, :, None].expand(nx, ny, nz).clone()
    s = Solver(nx, ny, nz, tau=0.6, u_char=0.08, device="cpu",
               obstacle_mask=mask, inlet_outlet=True, ramp_steps=100, seed=1)
    for _ in range(600):
        s.step()
    g = s.check_guards()
    assert not g["has_nan"]
    assert 0.085 < g["u_max"] < 0.3
    _, u = s.macroscopics()
    assert float(u[1, 50:110, :, 2].abs().max()) > 1e-3


def test_poiseuille_profile_with_guo_forcing_3d():
    # Body-force-driven channel (periodic x, z; bounce-back y): parabola
    # with u_max = F H^2/(8 rho nu), spanwise-invariant.
    nx, ny, nz, tau = 8, 34, 6, 0.8
    nu = (tau - 0.5) / 3.0
    h = ny - 2
    u_target = 0.02
    fx = 8.0 * nu * u_target / h**2
    s = Solver(nx, ny, nz, tau=tau, u_char=u_target, device="cpu",
               inlet_outlet=False, wall_y=True, body_force=(fx, 0.0, 0.0),
               init_noise=0.0)
    for _ in range(8000):
        s.step()
    _, u = s.macroscopics()
    prof = u[0, nx // 2, 1:-1, nz // 2]
    y = torch.arange(1, ny - 1, dtype=torch.float32) - 0.5
    y_hat = y / h
    analytic = 4.0 * u_target * y_hat * (1.0 - y_hat)
    l2 = float(((prof - analytic) ** 2).mean().sqrt()
               / (analytic ** 2).mean().sqrt())
    assert l2 < 0.01, f"3D Poiseuille L2 error {l2:.4f} >= 1%"
    assert float(u[0, nx // 2, ny // 2, :].std()) < 1e-6


def test_couette_profile_with_moving_lid_3d():
    # Top wall slides at U (periodic x, z): linear profile u(y) = U y/H —
    # the moving-lid BC that the spanwise-periodic Ghia cavity gate needs.
    nx, ny, nz, tau, u_lid = 8, 18, 4, 0.8, 0.05
    s = Solver(nx, ny, nz, tau=tau, u_char=u_lid, device="cpu",
               inlet_outlet=False, wall_y=True, lid_velocity=u_lid,
               init_noise=0.0)
    for _ in range(6000):
        s.step()
    _, u = s.macroscopics()
    prof = u[0, nx // 2, 1:-1, nz // 2]
    h = ny - 2
    y_hat = (torch.arange(1, ny - 1, dtype=torch.float32) - 0.5) / h
    err = float((prof - u_lid * y_hat).abs().max()) / u_lid
    assert err < 0.01, f"3D Couette max error {err:.4f} >= 1%"
    assert float(u[0, nx // 2, ny // 2, :].std()) < 1e-6  # spanwise-invariant


def test_momentum_exchange_force_is_drag_dominated():
    nx, ny, nz = 140, 60, 6
    d = 12.0
    x = torch.arange(nx, dtype=torch.float32)[:, None] + 0.5
    y = torch.arange(ny, dtype=torch.float32)[None, :] + 0.5
    mask = ((x - 36.0) ** 2 + (y - 30.0) ** 2 <= (d / 2) ** 2)
    mask = mask[:, :, None].expand(nx, ny, nz).clone()
    u_in = 0.08
    s = Solver(nx, ny, nz, tau=0.65, u_char=u_in, device="cpu",
               obstacle_mask=mask, inlet_outlet=True, ramp_steps=100, seed=1)
    for _ in range(400):
        s.step()
    s.measure_force = True
    fx = fy = fz = 0.0
    n = 200
    for _ in range(n):
        s.step()
        fv = s.last_force.tolist()
        fx += fv[0]; fy += fv[1]; fz += fv[2]
    fx, fy, fz = fx / n, fy / n, fz / n
    cd = fx / (0.5 * u_in * u_in * d * nz)
    assert fx > 0.0
    assert 0.5 < cd < 4.0, f"Cd = {cd:.2f} implausible"
    assert abs(fz) < 0.1 * abs(fx)   # spanwise-uniform body: no z-force


def test_mass_conserved_with_bounce_back():
    nx, ny, nz = 48, 32, 6
    x = torch.arange(nx, dtype=torch.float32)[:, None]
    y = torch.arange(ny, dtype=torch.float32)[None, :]
    mask = ((x - 16.0) ** 2 + (y - 16.0) ** 2 <= 36.0)
    mask = mask[:, :, None].expand(nx, ny, nz).clone()
    s = Solver(nx, ny, nz, tau=0.7, u_char=0.05, device="cpu",
               inlet_outlet=False, init_noise=0.0, obstacle_mask=mask)
    m0 = s.guards()["mass"]
    for _ in range(200):
        s.step()
    g = s.guards()
    assert abs(g["mass"] / m0 - 1.0) < 1e-5
    assert not g["has_nan"]


def test_solid_cells_stay_empty():
    nx, ny, nz = 32, 32, 4
    mask = torch.zeros((nx, ny, nz), dtype=torch.bool)
    mask[10:20, 10:20, :] = True
    s = Solver(nx, ny, nz, tau=0.7, u_char=0.05, device="cpu",
               inlet_outlet=False, init_noise=0.0, obstacle_mask=mask)
    for _ in range(50):
        s.step()
    assert float(s.f[:, s.mask].abs().max()) == 0.0


def test_checkpoint_roundtrip_bitwise():
    a = make_periodic_box(nx=24, ny=16, nz=4)
    for _ in range(30):
        a.step()
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ck.pt"
        a.checkpoint(path)
        for _ in range(20):
            a.step()
        b = make_periodic_box(nx=24, ny=16, nz=4)
        b.restore(path)
        assert b.step_count == 30
        for _ in range(20):
            b.step()
        assert torch.equal(a.f, b.f), "checkpoint restore diverged"


def test_guards_catch_nan_and_capture_failure(tmp_path):
    s = make_periodic_box(nx=16, ny=16, nz=4)
    s.f[3, 5, 5, 1] = float("nan")
    with pytest.raises(SimulationBlowup):
        s.check_guards()
    dest = capture_failure(s, "test NaN", failures_root=tmp_path)
    assert (dest / "checkpoint.pt").exists()
    assert "seed: 0" in (dest / "meta.yaml").read_text(encoding="utf-8")
