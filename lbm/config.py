"""Scene loading: YAML config -> validated Scene with resolved units.

Every experiment lives in scenes/<name>.yaml and is defined in physical
terms. This module is the only path from a config file to lattice
parameters — everything goes through lbm.units.resolve(), so a scene that
violates the tau / u_lat guard rails cannot even be loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .units import AIR_NU, LatticeUnits, resolve

SCENES_DIR = Path(__file__).resolve().parent.parent / "scenes"


class SceneError(ValueError):
    """A scene file is missing or malformed."""


# D3Q19, fp32, A-B double buffer: 19 populations x 4 bytes x 2 buffers.
# This is the working set that must fit in the RTX 5080's 16 GB.
BYTES_PER_CELL_AB = 19 * 4 * 2


@dataclass(frozen=True)
class Scene:
    name: str
    description: str
    units: LatticeUnits
    nx: int          # grid length, streamwise [cells]
    ny: int          # grid height [cells]
    nz: int          # grid span, periodic [cells]
    raw: dict        # full config dict; solver phases consume the rest

    @property
    def cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def vram_gb(self) -> float:
        """f-population working set (A-B fp32 D3Q19) in GB — the budget."""
        return self.cells * BYTES_PER_CELL_AB / 1e9

    def report(self) -> str:
        grid = (
            f"  grid       {self.nx} x {self.ny} x {self.nz}"
            f" = {self.cells:,} cells   (~{self.vram_gb:.2f} GB f-populations)"
        )
        return self.units.report(title=self.name) + "\n" + grid


def list_scenes() -> list[str]:
    return sorted(p.stem for p in SCENES_DIR.glob("*.yaml"))


def load_scene(name: str) -> Scene:
    """Load scenes/<name>.yaml (or an explicit path) and resolve its units."""
    path = Path(name)
    if path.suffix != ".yaml" or not path.exists():
        path = SCENES_DIR / f"{name}.yaml"
    if not path.exists():
        raise SceneError(
            f"no scene named {name!r}; available: {', '.join(list_scenes())}"
        )

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        phys = cfg["physical"]
        lat = cfg["lattice"]
        dom = cfg["domain"]
        units = resolve(
            length_m=phys["char_length_m"],
            velocity_ms=phys["velocity_ms"],
            nu_m2s=phys.get("nu_m2s", AIR_NU),
            cells=lat["cells_per_char"],
            u_lat=lat["u_lat"],
            sgs=bool(cfg.get("sgs", False)),
        )
        nx = round(dom["length_chars"] * lat["cells_per_char"])
        ny = round(dom["height_chars"] * lat["cells_per_char"])
        nz = round(dom["span_chars"] * lat["cells_per_char"])
    except KeyError as e:
        raise SceneError(f"{path.name}: missing required key {e}") from e

    return Scene(
        name=cfg.get("name", path.stem),
        description=str(cfg.get("description", "")).strip(),
        units=units,
        nx=nx,
        ny=ny,
        nz=nz,
        raw=cfg,
    )
