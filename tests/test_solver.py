"""Physics and infrastructure tests for the D3Q19 solver (CPU, small grids).

The low-level Solver constructor is used deliberately here: tests need
engineered analytic cases (known tau), which is exactly the exemption the
units discipline grants to tests and validation scripts.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from lbm.solver import SimulationBlowup, Solver, capture_failure  # noqa: E402


def make_periodic_box(nx=32, ny=32, nz=8, tau=0.8, u=0.05, **kw):
    return Solver(nx, ny, nz, tau=tau, u_char=u, device="cpu",
                  inlet_outlet=False, init_noise=0.0, **kw)


def test_uniform_flow_is_a_fixed_point():
    # f = feq(rho=1, u) must be invariant under collide+stream (periodic).
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
    # u = U (-cos kx sin ky, sin kx cos ky, 0) decays as exp(-2 nu k^2 t).
    # This checks collision, streaming, AND the nu = (tau-1/2)/3 relation —
    # i.e. the whole units story — against an exact Navier-Stokes solution.
    n, tau, u0 = 64, 0.8, 0.03
    nu = (tau - 0.5) / 3.0
    k = 2.0 * math.pi / n
    s = Solver(n, n, 4, tau=tau, u_char=u0, device="cpu",
               inlet_outlet=False, init_noise=0.0)
    x = (torch.arange(n, dtype=torch.float32) + 0.5) * k
    u = torch.zeros((3, n, n, 4))
    u[0] = -u0 * torch.cos(x)[:, None, None] * torch.sin(x)[None, :, None]
    u[1] = u0 * torch.sin(x)[:, None, None] * torch.cos(x)[None, :, None]
    s._write_equilibrium(s.f, torch.ones((n, n, 4)), u)

    steps = 300
    e0 = float((u * u).sum())
    for _ in range(steps):
        s.step()
    _, u_t = s.macroscopics()
    e_t = float((u_t * u_t).sum())
    decay_measured = math.sqrt(e_t / e0)           # ~ exp(-2 nu k^2 t)
    decay_expected = math.exp(-2.0 * nu * k * k * steps)
    assert decay_measured == pytest.approx(decay_expected, rel=0.02)


def test_mass_conserved_with_bounce_back():
    # Closed periodic box with a solid cylinder: halfway bounce-back must
    # conserve mass to fp32 roundoff.
    nx, ny, nz = 48, 32, 6
    x = torch.arange(nx, dtype=torch.float32)[:, None]
    y = torch.arange(ny, dtype=torch.float32)[None, :]
    mask = ((x - 16.0) ** 2 + (y - 16.0) ** 2 <= 36.0)  # r = 6 cylinder
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
    ck = a.__class__  # keep flake quiet

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
        # CPU determinism -> continuing from a checkpoint is BITWISE equal
        assert torch.equal(a.f, b.f), "checkpoint restore diverged"
    assert ck is Solver


def test_open_boundaries_hold_freestream_3d():
    # REGRESSION for the 3D open-boundary bug (NOTES 2026-07-13/14): with an
    # equilibrium velocity inlet and an anechoic sponge outlet, an EMPTY 3D
    # tunnel must settle at u = u_in and rho = 1 in the interior. The old
    # fixed-rho inlet + copy outlet pressurized the box to rho~1.06 and ran
    # ~12% slow, which is why the cylinder never left the symmetric branch.
    s = Solver(160, 48, 8, tau=0.7, u_char=0.08, device="cpu",
               inlet_outlet=True, ramp_steps=150, init_noise=0.0)
    for _ in range(2500):
        s.step()
    rho, u = s.macroscopics()
    interior_ux = u[0, 10:-14, :, :]        # clear of inlet and sponge
    interior_rho = rho[10:-14, :, :]
    assert float((interior_ux - 0.08).abs().max()) < 0.008   # within 1% u
    assert float((interior_rho - 1.0).abs().max()) < 0.005   # within 0.5%
    g = s.guards()
    assert abs(g["u_max"] - 0.08) < 0.008


def test_inlet_outlet_cylinder_sheds_wake():
    # Tiny wind tunnel with an OFFSET cylinder (the shedding trigger): must
    # stay finite, ACCELERATE the flow around the body (the old run never
    # did — it ran slow and symmetric), and grow transverse motion in the
    # wake (the seed of the vortex street).
    nx, ny, nz = 160, 60, 4
    x = torch.arange(nx, dtype=torch.float32)[:, None] + 0.5
    y = torch.arange(ny, dtype=torch.float32)[None, :] + 0.5
    mask = ((x - 40.0) ** 2 + (y - 28.0) ** 2 <= 36.0)   # r=6, offset down
    mask = mask[:, :, None].expand(nx, ny, nz).clone()
    s = Solver(nx, ny, nz, tau=0.6, u_char=0.08, device="cpu",
               obstacle_mask=mask, inlet_outlet=True, ramp_steps=100,
               seed=1)
    for _ in range(600):
        s.step()
    g = s.check_guards()
    assert not g["has_nan"]
    assert 0.085 < g["u_max"] < 0.3          # flow accelerates past U_in
    _, u = s.macroscopics()
    assert float(u[1, 50:110, :, 2].abs().max()) > 1e-3   # wake transverse v


def test_guards_catch_nan_and_capture_failure(tmp_path):
    s = make_periodic_box(nx=16, ny=16, nz=4)
    s.f[3, 5, 5, 1] = float("nan")
    with pytest.raises(SimulationBlowup):
        s.check_guards()
    dest = capture_failure(s, "test NaN", failures_root=tmp_path)
    assert (dest / "checkpoint.pt").exists()
    assert "seed: 0" in (dest / "meta.yaml").read_text(encoding="utf-8")


def test_render_writes_frame(tmp_path):
    pytest.importorskip("matplotlib")
    from lbm.render import FrameWriter
    s = make_periodic_box(nx=32, ny=24, nz=4)
    for _ in range(5):
        s.step()
    w = FrameWriter(tmp_path)
    p = w.write(s)
    assert p.exists() and p.stat().st_size > 0
