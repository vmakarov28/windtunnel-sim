"""Phase 7 (WebGPU browser toy) coverage in the main suite.

The toy broke silently once: its validators read _build/*.wgsl, which is
gitignored output of a manual `node dump_wgsl.mjs` step, so a cleaned
tree passed validate_wgsl.py with ZERO shaders checked. These tests pin
the fix: shaders are extracted straight from web/shaders.js, extraction
itself is asserted (no empty-glob false pass possible), and when a GPU
adapter is available the kernels are compiled and RUN.

GPU tests skip gracefully on machines without wgpu or an adapter; the
extraction tests always run.
"""

import sys
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"
sys.path.insert(0, str(WEB))

from shader_source import extract_shaders  # noqa: E402


def test_extraction_finds_all_four_shaders():
    shaders = extract_shaders()
    assert set(shaders) == {"step", "render", "tracer_advect", "tracer_draw"}
    for name, code in shaders.items():
        assert len(code) > 500, f"{name} suspiciously short — extractor broke"


def test_compute_shaders_carry_the_lattice_prelude():
    shaders = extract_shaders()
    # step and advect both need the D2Q9 constants from LATTICE
    for name in ("step", "tracer_advect"):
        assert "struct Params" in shaders[name], name
        assert "const W0" in shaders[name], name


def test_no_reserved_word_identifiers():
    # `macro` is WGSL-reserved; it bit us once (buffer now named `vel`)
    import re
    for name, code in extract_shaders().items():
        stripped = re.sub(r"//[^\n]*", "", code)
        assert not re.search(r"\bmacro\b", stripped), name


@pytest.fixture(scope="module")
def device():
    wgpu = pytest.importorskip("wgpu")
    try:
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance")
        return adapter.request_device_sync()
    except Exception as e:  # no adapter on this machine (CI, bare VM)
        pytest.skip(f"no WebGPU adapter: {e}")


def test_all_shaders_compile(device):
    for name, code in extract_shaders().items():
        device.create_shader_module(code=code)  # naga raises on error


def test_step_kernel_empty_tunnel_is_boring(device):
    import validate_step
    ok, msg = validate_step.check_empty_tunnel(device)
    assert ok, msg


def test_step_kernel_cylinder_sheds(device):
    import validate_step
    ok, msg = validate_step.check_cylinder_sheds(device)
    assert ok, msg
