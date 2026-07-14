"""Physics and infrastructure tests for the D2Q9 solver (CPU, small grids).

The low-level Solver constructor is used deliberately here: tests need
engineered analytic cases (known tau), which is exactly the exemption the
units discipline grants to tests and validation scripts.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from lbm.solver import SimulationBlowup, Solver, capture_failure  # noqa: E402


def make_periodic_box(nx=32, ny=32, tau=0.8, u=0.05, **kw):
    return Solver(nx, ny, tau=tau, u_char=u, device="cpu",
                  inlet_outlet=False, init_noise=0.0, **kw)


def test_uniform_flow_is_a_fixed_point():
    # f = feq(rho=1, u) must be invariant under collide+stream (periodic).
    s = make_periodic_box()
    rho = torch.ones((s.nx, s.ny))
    u = torch.zeros((2, s.nx, s.ny))
    u[0], u[1] = 0.05, -0.02
    s._write_equilibrium(s.f, rho, u)
    for _ in range(10):
        s.step()
    _, u_after = s.macroscopics()
    assert float((u_after[0] - 0.05).abs().max()) < 1e-5
    assert float((u_after[1] + 0.02).abs().max()) < 1e-5


def test_taylor_green_viscous_decay():
    # u = U (-cos kx sin ky, sin kx cos ky) decays as exp(-2 nu k^2 t).
    # This checks collision, streaming, AND the nu = (tau-1/2)/3 relation —
    # i.e. the whole units story — against an exact Navier-Stokes solution.
    n, tau, u0 = 64, 0.8, 0.03
    nu = (tau - 0.5) / 3.0
    k = 2.0 * math.pi / n
    s = Solver(n, n, tau=tau, u_char=u0, device="cpu",
               inlet_outlet=False, init_noise=0.0)
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
    e_t = float((u_t * u_t).sum())
    decay_measured = math.sqrt(e_t / e0)           # ~ exp(-2 nu k^2 t)
    decay_expected = math.exp(-2.0 * nu * k * k * steps)
    assert decay_measured == pytest.approx(decay_expected, rel=0.02)


def test_open_boundaries_hold_freestream():
    # REGRESSION for the 3D-run bug (NOTES.md 2026-07-13): with velocity
    # imposed at the inlet and pressure anchored at the outlet, an empty
    # tunnel must settle at u = u_in and rho = 1 in the interior — the old
    # rho=1-inlet + copy-outlet pairing pressurized to rho ~ 1.06 and ran
    # 12% slow.
    s = Solver(200, 64, tau=0.7, u_char=0.08, device="cpu",
               inlet_outlet=True, ramp_steps=200, init_noise=0.0)
    for _ in range(2500):
        s.step()
    rho, u = s.macroscopics()
    interior_ux = u[0, 10:-10, :]
    interior_rho = rho[10:-10, :]
    assert float((interior_ux - 0.08).abs().max()) < 0.008   # within 1% u
    assert float((interior_rho - 1.0).abs().max()) < 0.005   # within 0.5%
    g = s.guards()
    assert abs(g["u_max"] - 0.08) < 0.008


def test_poiseuille_profile_with_guo_forcing():
    # Body-force-driven channel: u(y) parabolic, u_max = F H^2 / (8 rho nu).
    # Checks Guo forcing AND halfway bounce-back wall placement (walls sit
    # half a cell outside the outermost fluid nodes).
    nx, ny, tau = 16, 34, 0.8
    nu = (tau - 0.5) / 3.0
    h = ny - 2                      # fluid rows
    u_target = 0.02
    fx = 8.0 * nu * u_target / h**2
    s = Solver(nx, ny, tau=tau, u_char=u_target, device="cpu",
               inlet_outlet=False, wall_y=True, body_force=(fx, 0.0),
               init_noise=0.0)
    for _ in range(8000):           # ~ a few diffusion times H^2/nu
        s.step()
    _, u = s.macroscopics()
    prof = u[0, nx // 2, 1:-1]                       # fluid rows
    y = torch.arange(1, ny - 1, dtype=torch.float32) - 0.5  # wall at y=0.5
    y_hat = y / h
    analytic = 4.0 * u_target * y_hat * (1.0 - y_hat)
    l2 = float(((prof - analytic) ** 2).mean().sqrt()
               / (analytic ** 2).mean().sqrt())
    assert l2 < 0.01, f"Poiseuille L2 error {l2:.4f} >= 1%"


def test_couette_profile_with_moving_lid():
    # Top wall moves at U, bottom fixed: linear profile u(y) = U y/H.
    nx, ny, tau, u_lid = 16, 18, 0.8, 0.05
    s = Solver(nx, ny, tau=tau, u_char=u_lid, device="cpu",
               inlet_outlet=False, wall_y=True, lid_velocity=u_lid,
               init_noise=0.0)
    for _ in range(6000):
        s.step()
    _, u = s.macroscopics()
    prof = u[0, nx // 2, 1:-1]
    h = ny - 2
    y_hat = (torch.arange(1, ny - 1, dtype=torch.float32) - 0.5) / h
    analytic = u_lid * y_hat
    err = float((prof - analytic).abs().max()) / u_lid
    assert err < 0.01, f"Couette max error {err:.4f} >= 1%"


def test_mass_conserved_with_bounce_back():
    # Closed periodic box with a solid cylinder: halfway bounce-back must
    # conserve mass to fp32 roundoff.
    nx, ny = 48, 32
    x = torch.arange(nx, dtype=torch.float32)[:, None]
    y = torch.arange(ny, dtype=torch.float32)[None, :]
    mask = ((x - 16.0) ** 2 + (y - 16.0) ** 2 <= 36.0)  # r = 6 cylinder
    s = Solver(nx, ny, tau=0.7, u_char=0.05, device="cpu",
               inlet_outlet=False, init_noise=0.0, obstacle_mask=mask)
    m0 = s.guards()["mass"]
    for _ in range(200):
        s.step()
    g = s.guards()
    assert abs(g["mass"] / m0 - 1.0) < 1e-5
    assert not g["has_nan"]


def test_solid_cells_stay_empty():
    nx, ny = 32, 32
    mask = torch.zeros((nx, ny), dtype=torch.bool)
    mask[10:20, 10:20] = True
    s = Solver(nx, ny, tau=0.7, u_char=0.05, device="cpu",
               inlet_outlet=False, init_noise=0.0, obstacle_mask=mask)
    for _ in range(50):
        s.step()
    assert float(s.f[:, s.mask].abs().max()) == 0.0


def test_checkpoint_roundtrip_bitwise():
    a = make_periodic_box(nx=24, ny=16)
    for _ in range(30):
        a.step()

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ck.pt"
        a.checkpoint(path)
        for _ in range(20):
            a.step()

        b = make_periodic_box(nx=24, ny=16)
        b.restore(path)
        assert b.step_count == 30
        for _ in range(20):
            b.step()
        # CPU determinism -> continuing from a checkpoint is BITWISE equal
        assert torch.equal(a.f, b.f), "checkpoint restore diverged"


def test_inlet_outlet_smoke_mini_cylinder():
    # Tiny wind tunnel: ramped inlet, sponge outlet, cylinder. Must stay
    # finite, accelerate around the obstacle, and shed vorticity.
    nx, ny = 120, 60
    x = torch.arange(nx, dtype=torch.float32)[:, None] + 0.5
    y = torch.arange(ny, dtype=torch.float32)[None, :] + 0.5
    mask = ((x - 30.0) ** 2 + (y - 29.0) ** 2 <= 25.0)   # r=5, slight offset
    s = Solver(nx, ny, tau=0.6, u_char=0.08, device="cpu",
               obstacle_mask=mask, inlet_outlet=True, ramp_steps=100,
               seed=1)
    for _ in range(600):
        s.step()
    g = s.check_guards()
    assert not g["has_nan"]
    # flow must ACCELERATE around the cylinder (the 3D run never did):
    assert 0.085 < g["u_max"] < 0.3
    _, u = s.macroscopics()
    assert float(u[1, 40:80, :].abs().max()) > 1e-3   # wake sideways motion


def test_guards_catch_nan_and_capture_failure(tmp_path):
    s = make_periodic_box(nx=16, ny=16)
    s.f[3, 5, 5] = float("nan")
    with pytest.raises(SimulationBlowup):
        s.check_guards()
    dest = capture_failure(s, "test NaN", failures_root=tmp_path)
    assert (dest / "checkpoint.pt").exists()
    assert "seed: 0" in (dest / "meta.yaml").read_text(encoding="utf-8")


def test_render_writes_frame(tmp_path):
    pytest.importorskip("matplotlib")
    from lbm.render import FrameWriter
    s = make_periodic_box(nx=32, ny=24)
    for _ in range(5):
        s.step()
    w = FrameWriter(tmp_path)
    p = w.write(s)
    assert p.exists() and p.stat().st_size > 0
