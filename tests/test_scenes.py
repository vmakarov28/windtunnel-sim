"""Every shipped scene must load and resolve through the units discipline."""

import subprocess
import sys
from pathlib import Path

import pytest

from lbm.config import SceneError, list_scenes, load_scene
from lbm.units import UnitError, resolve

ROOT = Path(__file__).resolve().parent.parent


def test_all_shipped_scenes_resolve():
    names = list_scenes()
    assert len(names) >= 4  # cylinder, cavity, channel, airfoil
    for name in names:
        scene = load_scene(name)
        assert scene.nx > 0 and scene.ny > 0


def test_cylinder_scene_numbers():
    s = load_scene("cylinder_re100")
    assert s.units.reynolds == pytest.approx(100.0)
    assert (s.nx, s.ny) == (1200, 600)
    assert s.units.cells >= 30              # Phase 1 demo requirement
    blockage = s.units.cells / s.ny
    assert blockage < 0.08                  # < 8% blockage


def test_cavity_scene_numbers():
    s = load_scene("cavity_re100")
    assert (s.nx, s.ny) == (256, 256)
    assert s.units.reynolds == pytest.approx(100.0)
    assert s.units.tau == pytest.approx(1.268)


def test_airfoil_scene_needs_sgs():
    s = load_scene("airfoil_mh45_re20k")
    assert s.units.sgs
    assert s.units.cells >= 400             # Phase 6 chord requirement
    # The same parameters must be REFUSED without the turbulence model:
    with pytest.raises(UnitError, match="below the plain-BGK floor"):
        resolve(
            s.units.length_m, s.units.velocity_ms, s.units.nu_m2s,
            cells=s.units.cells, u_lat=s.units.u_lat, sgs=False,
        )


def test_unknown_scene_raises():
    with pytest.raises(SceneError, match="no scene named"):
        load_scene("does_not_exist")


def test_run_py_smoke():
    out = subprocess.run(
        [sys.executable, str(ROOT / "run.py"),
         "--scene", "cylinder_re100", "--seed", "0"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr
    assert "tau = 0.572" in out.stdout


def test_run_py_requires_seed():
    out = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--scene", "cylinder_re100"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert out.returncode != 0
