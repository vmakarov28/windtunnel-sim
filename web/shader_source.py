"""Extract the assembled WGSL shaders directly from shaders.js.

Single source of truth stays shaders.js (what the browser runs); this
parses its template literals and replicates the `LATTICE + ...`
concatenation, so the validators need no Node and no manual dump step —
the earlier flow (node dump_wgsl.mjs -> _build/*.wgsl -> validate)
silently broke the moment _build/ was cleaned, because _build is
gitignored. Never again: the tests read the real file.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parent

_BLOCK = re.compile(
    r"export const (\w+) = (LATTICE \+ )?/\* wgsl \*/ `([^`]*)`;",
    re.DOTALL,
)


def extract_shaders(path: Path | None = None) -> dict[str, str]:
    """Return {short_name: full_wgsl} for every *_WGSL export."""
    src = (path or WEB / "shaders.js").read_text(encoding="utf-8")
    blocks = {m.group(1): (bool(m.group(2)), m.group(3))
              for m in _BLOCK.finditer(src)}
    if "LATTICE" not in blocks:
        raise RuntimeError("shaders.js: LATTICE block not found — "
                           "extractor out of sync with the file layout")
    lattice = blocks["LATTICE"][1]
    out: dict[str, str] = {}
    for name, (uses_lattice, body) in blocks.items():
        if not name.endswith("_WGSL"):
            continue
        short = name.removesuffix("_WGSL").lower()
        out[short] = (lattice + body) if uses_lattice else body
    expected = {"step", "render", "tracer_advect", "tracer_draw"}
    missing = expected - out.keys()
    if missing:
        raise RuntimeError(f"shaders.js: missing shader blocks {missing}")
    return out


def write_build_dir() -> Path:
    """Optional: materialize _build/*.wgsl (debugging / diffing aid)."""
    build = WEB / "_build"
    build.mkdir(exist_ok=True)
    for name, code in extract_shaders().items():
        (build / f"{name}.wgsl").write_text(code, encoding="utf-8")
    return build


if __name__ == "__main__":
    for name, code in extract_shaders().items():
        print(f"{name}: {len(code)} chars")
    print(f"written to {write_build_dir()}")
