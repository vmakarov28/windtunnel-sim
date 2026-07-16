"""3D renderer tests: analytic Q-criterion checks + preset smoke.

Q-criterion is verified against fields whose velocity gradients are
LINEAR, where central differences are exact away from the periodic wrap
(assertions restricted to the interior)."""

import pytest

torch = pytest.importorskip("torch")

from lbm3d.render import (  # noqa: E402
    FrameWriter, q_criterion, streamwise_vorticity,
)
from lbm3d.solver import Solver  # noqa: E402


def coords(n):
    c = torch.arange(n, dtype=torch.float32) - (n - 1) / 2.0
    return c


def test_q_criterion_positive_in_rigid_rotation():
    # u = (-w*y, w*x, 0): pure rotation, S = 0, so Q = w^2 exactly.
    n, w = 24, 0.01
    x, y = coords(n), coords(n)
    u = torch.zeros((3, n, n, 8))
    u[0] = (-w * y)[None, :, None].expand(n, n, 8)
    u[1] = (w * x)[:, None, None].expand(n, n, 8)
    q = q_criterion(u)
    interior = q[4:-4, 4:-4, :]
    assert float(interior.min()) == pytest.approx(w * w, rel=0.01)
    assert float(interior.max()) == pytest.approx(w * w, rel=0.01)


def test_q_criterion_zero_in_pure_shear():
    # u = (k*y, 0, 0): strain and rotation balance, Q = 0 exactly.
    n, k = 24, 0.01
    y = coords(n)
    u = torch.zeros((3, n, n, 8))
    u[0] = (k * y)[None, :, None].expand(n, n, 8)
    q = q_criterion(u)
    assert float(q[4:-4, 4:-4, :].abs().max()) < 1e-9


def test_streamwise_vorticity_of_yz_rotation():
    # u = (0, -w*z, w*y): rotation about x -> omega_x = 2w.
    n, w = 24, 0.01
    y, z = coords(n), coords(8)
    u = torch.zeros((3, n, n, 8))
    u[1] = (-w * z)[None, None, :].expand(n, n, 8)
    u[2] = (w * y)[None, :, None].expand(n, n, 8)
    wx = streamwise_vorticity(u)
    assert float(wx[4:-4, 4:-4, 2:-2].mean()) == pytest.approx(2 * w,
                                                               rel=0.01)


def test_spanwise_pane_is_an_absolute_meter():
    # REGRESSION: the u_z pane self-normalized on its first outing and
    # amplified 1e-6 fp32 roundoff into loud fake 3D structure. With the
    # absolute scale, a 2D flow must render (near-)blank and real spanwise
    # motion must saturate.
    nx, ny, nz = 32, 16, 8
    s = Solver(nx, ny, nz, tau=0.7, u_char=0.08, device="cpu",
               inlet_outlet=False, init_noise=0.0)
    w = FrameWriter.__new__(FrameWriter)   # pane only; skip dir setup
    w.uz_fullscale_frac = 0.15

    # 2D state + roundoff-scale u_z noise -> blank (RdBu midpoint ~0.969)
    u = torch.zeros((3, nx, ny, nz))
    u[0] = 0.08
    u[2] = 2e-6 * (torch.rand(nx, ny, nz) - 0.5)
    s._write_equilibrium(s.f, torch.ones((nx, ny, nz)), u)
    pane = w._spanwise_rgba(s)
    assert abs(pane[..., :3].mean() - 0.969) < 0.01, "2D flow must read blank"

    # real mode-A-scale spanwise motion -> strongly colored
    u[2] = 0.012 * torch.sin(
        torch.arange(nz, dtype=torch.float32) * 2 * 3.14159 / nz
    )[None, None, :].expand(nx, ny, nz)
    s._write_equilibrium(s.f, torch.ones((nx, ny, nz)), u)
    pane = w._spanwise_rgba(s)
    assert pane[..., :3].min() < 0.75, "real u_z must color the meter"


@pytest.mark.parametrize("preset", ["slice", "three_pane", "qcrit"])
def test_presets_write_frames(tmp_path, preset):
    pytest.importorskip("matplotlib")
    nx, ny, nz = 48, 32, 8
    x = torch.arange(nx, dtype=torch.float32)[:, None] + 0.5
    y = torch.arange(ny, dtype=torch.float32)[None, :] + 0.5
    mask = ((x - 12.0) ** 2 + (y - 15.0) ** 2 <= 9.0)
    mask = mask[:, :, None].expand(nx, ny, nz).clone()
    s = Solver(nx, ny, nz, tau=0.6, u_char=0.08, device="cpu",
               obstacle_mask=mask, inlet_outlet=True, ramp_steps=50, seed=1)
    for _ in range(120):
        s.step()
    w = FrameWriter(tmp_path / preset, preset=preset)
    p = w.write(s)
    assert p.exists() and p.stat().st_size > 0


def test_frame_numbering_resumes(tmp_path):
    pytest.importorskip("matplotlib")
    s = Solver(24, 16, 4, tau=0.7, u_char=0.05, device="cpu",
               inlet_outlet=False, init_noise=0.0)
    for _ in range(5):
        s.step()
    w1 = FrameWriter(tmp_path, preset="slice")
    w1.write(s)
    w1.write(s)
    w2 = FrameWriter(tmp_path, preset="slice")   # a --resume in miniature
    assert w2.count == 2
    p = w2.write(s)
    assert p.name == "frame_000002.png"
