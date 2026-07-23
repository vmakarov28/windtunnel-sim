"""The airfoil scene generator (scripts/make_airfoil_scene.py).

Proves the generator stays inside the units discipline: Reynolds sets the
wind speed, SGS is auto-enabled exactly when the tau floor needs it, the
accurate keys are opt-outable, and an un-simulable request is REFUSED
(no file written) — the same guard rails a hand-written scene gets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.make_airfoil_scene as gen           # noqa: E402
from lbm.config import SCENES_DIR, load_scene       # noqa: E402

DAT = "assets/naca4412.dat"                          # ships with the repo


def run_gen(name: str, *args: str) -> int:
    """Invoke the generator's main() with argv, always cleaning the file."""
    argv = ["make_airfoil_scene.py", "--dat", DAT, "--name", name, *args]
    old = sys.argv
    sys.argv = argv
    try:
        return gen.main()
    finally:
        sys.argv = old


@pytest.fixture
def scene_name():
    name = "_pytest_gen"
    yield name
    (SCENES_DIR / f"{name}.yaml").unlink(missing_ok=True)


def test_reynolds_sets_the_speed(scene_name):
    rc = run_gen(scene_name, "--re", "5000", "--alpha", "4",
                 "--chord-cells", "256")
    assert rc == 0
    scene = load_scene(scene_name)
    # velocity is DERIVED: U = Re * nu / chord = 5000 * 1.5e-5 / 0.1
    assert scene.units.velocity_ms == pytest.approx(0.75, rel=1e-6)
    assert round(scene.units.cells) == 256
    assert scene.raw["obstacle"]["alpha_deg"] == 4.0
    # accurate path on by default
    assert scene.raw.get("collision") == "trt"
    assert scene.raw["obstacle"].get("curved_bc") is True


def test_auto_sgs_off_when_tau_clears_floor(scene_name):
    # Re = 500 at chord 256, u_lat 0.05 -> tau ~ 0.577, above the BGK floor
    assert run_gen(scene_name, "--re", "500", "--chord-cells", "256") == 0
    scene = load_scene(scene_name)
    assert scene.raw["sgs"] is False
    assert scene.units.sgs is False


def test_auto_sgs_on_when_tau_needs_it(scene_name):
    # Re = 20000 at chord 256, u_lat 0.05 -> tau ~ 0.502, below BGK floor
    assert run_gen(scene_name, "--re", "20000", "--chord-cells", "256") == 0
    scene = load_scene(scene_name)
    assert scene.raw["sgs"] is True
    assert scene.units.sgs is True


def test_fast_omits_the_accuracy_keys(scene_name):
    assert run_gen(scene_name, "--re", "5000", "--fast") == 0
    scene = load_scene(scene_name)
    assert "collision" not in scene.raw
    assert "curved_bc" not in scene.raw["obstacle"]


def test_refuses_unsimulable_and_writes_nothing(scene_name):
    # Re = 50000 at chord 256, u_lat 0.05 -> tau ~ 0.5008, below even the
    # SGS floor: the units discipline must refuse and leave no file.
    rc = run_gen(scene_name, "--re", "50000", "--chord-cells", "256")
    assert rc == 2
    assert not (SCENES_DIR / f"{scene_name}.yaml").exists()
