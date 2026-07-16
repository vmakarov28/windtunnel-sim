"""Every shipped 3D scene must load and resolve through the units
discipline, and fit the VRAM budget."""

import subprocess
import sys
from pathlib import Path

import pytest

from lbm3d.config import SceneError, list_scenes, load_scene
from lbm.units import UnitError, resolve

ROOT = Path(__file__).resolve().parent.parent

# Practical f-population budget on the RTX 5080 (16 GB): headroom for the
# macroscopic fields, masks, rendering, and the framework itself.
VRAM_BUDGET_GB = 12.0


def test_all_shipped_scenes_resolve():
    names = list_scenes()
    assert len(names) >= 5
    for name in names:
        scene = load_scene(name)
        assert scene.nx > 0 and scene.ny > 0 and scene.nz > 0


def test_all_scenes_fit_in_vram():
    for name in list_scenes():
        scene = load_scene(name)
        assert scene.vram_gb <= VRAM_BUDGET_GB, (
            f"{name}: {scene.vram_gb:.2f} GB of f-populations exceeds the "
            f"{VRAM_BUDGET_GB} GB budget on the 16 GB card"
        )


def test_cylinder_scene_numbers():
    s = load_scene("cylinder_re100")
    assert s.units.reynolds == pytest.approx(100.0)
    assert (s.nx, s.ny, s.nz) == (900, 450, 90)
    assert s.units.tau == pytest.approx(0.563)
    assert s.units.cells / s.ny < 0.08          # blockage < 8%
    # the deterministic shedding trigger must stay in the scene
    assert s.raw["obstacle"]["center_y_chars"] == pytest.approx(7.3)


def test_dev_cylinder_scene_numbers():
    s = load_scene("cylinder_re100_dev")
    assert s.units.reynolds == pytest.approx(100.0)
    assert (s.nx, s.ny, s.nz) == (576, 288, 16)
    assert s.vram_gb < 0.5                      # fast-iteration scene


def test_cavity_scene_numbers():
    s = load_scene("cavity_re100")
    assert (s.nx, s.ny, s.nz) == (256, 256, 16)
    assert s.units.tau == pytest.approx(1.268)


def test_airfoil_scene_needs_sgs():
    s = load_scene("airfoil_mh45_re20k")
    assert s.units.sgs
    with pytest.raises(UnitError, match="below the plain-BGK floor"):
        resolve(
            s.units.length_m, s.units.velocity_ms, s.units.nu_m2s,
            cells=s.units.cells, u_lat=s.units.u_lat, sgs=False,
        )


def test_unknown_scene_raises():
    with pytest.raises(SceneError, match="no 3D scene named"):
        load_scene("does_not_exist")


def test_run3d_smoke():
    out = subprocess.run(
        [sys.executable, str(ROOT / "run3d.py"),
         "--scene", "cylinder_re100_dev", "--seed", "0"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr
    assert "tau = 0.5576" in out.stdout
    assert "576 x 288 x 16" in out.stdout


def test_run3d_requires_seed():
    out = subprocess.run(
        [sys.executable, str(ROOT / "run3d.py"), "--scene",
         "cylinder_re100_dev"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert out.returncode != 0
