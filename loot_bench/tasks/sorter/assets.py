"""MjSpec builders and layout for the sorter task category.

The cube and its pieces are STL/XML assets committed under ``xmls/``
(generated once by ``xmls/make_sorter_assets.py`` from the CAD STLs; don't
edit the generated files by hand). ``xmls/sorter_layout.json`` is the single
source for both the cube's hole geometry and the command term's success
metric, so they can never drift apart. Both the cube and every piece are free
bodies; the command term drops them at reset and re-derives the hole frames
from the cube's live pose every step (see ``mdp/commands.py``).

This module is a copy of the STL-sorter half of ``insertion/assets.py``
(pared down to what the sorter needs: no peg/board/socket_layout
machinery, which stays specific to the archived peg/two_sheet insertion
kinds).
"""
# pyright: reportAttributeAccessIssue=false
# (mujoco ships no type stubs for the compiled MjSpec module)

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco

from loot_bench.tasks.common.colors import COLOR_VALUES
from loot_bench.tasks.sorter.variants import Shape

_XML_DIR = Path(__file__).resolve().parent / "xmls"
SORTER_XML = _XML_DIR / "sorter.xml"
SORTER_LAYOUT_JSON = _XML_DIR / "sorter_layout.json"

# Cube placement in the robot workspace (env-local): pinned x with the
# shared +/-1 cm jitter (``tasks/common/fixed_spawns.SPAWN_JITTER``, the
# same number as every other category).
BOARD_POS = (0.375, 0.0, 0.0)

# Uniform shrink per piece shape, so each piece sits in its hole with ~3 mm
# clearance per side at the best yaw. Solved with the generator's own
# ``best_fit`` metric (``xmls/make_sorter_assets.py``).
PIECE_SCALE: dict[str, float] = {
    "circle": 0.9059,  # 3.00 mm clearance
    "square": 0.9031,  # 2.94 mm clearance
    "rect": 0.8124,  # 2.88 mm clearance
    "triangle": 0.8183,  # 2.94 mm clearance
    "hexagon": 0.8951,  # 2.96 mm clearance
    "star": 0.9178,  # 2.94 mm clearance
}


@dataclass(frozen=True)
class SorterHole:
    """One face hole of the sorter cube, in the cube body frame."""

    mouth: tuple[float, float, float]
    axis_out: tuple[float, float, float]
    wall_thickness: float
    bounding_radius: float


@dataclass(frozen=True)
class SorterPiece:
    """One STL piece; centred at its cross-section centroid, axis +z."""

    half_len: float
    cross_bounding_radius: float


@dataclass(frozen=True)
class SorterLayout:
    half_size: float
    holes: dict[Shape, SorterHole]
    pieces: dict[Shape, SorterPiece]


@lru_cache(maxsize=1)
def load_sorter_layout() -> SorterLayout:
    raw = json.loads(SORTER_LAYOUT_JSON.read_text())
    return SorterLayout(
        half_size=raw["cube"]["half_size"],
        holes={
            shape: SorterHole(
                mouth=tuple(h["mouth"]),
                axis_out=tuple(h["axis_out"]),
                wall_thickness=h["wall_thickness"],
                bounding_radius=h["bounding_radius"],
            )
            for shape, h in raw["holes"].items()
        },
        pieces={
            shape: SorterPiece(
                half_len=p["half_len"],
                cross_bounding_radius=p["cross_bounding_radius"],
            )
            for shape, p in raw["pieces"].items()
        },
    )


def get_sorter_spec() -> mujoco.MjSpec:
    """The free-body STL sorter cube (visual mesh + convex parts).

    A ``<freejoint/>`` body authored in ``xmls/sorter.xml``; the command term
    places it on every reset.
    """
    return mujoco.MjSpec.from_file(str(SORTER_XML))


def get_sorter_piece_spec(shape: Shape, color: str) -> mujoco.MjSpec:
    """A free-body STL sorter piece, tinted with its socket colour.

    The piece is shrunk uniformly per ``PIECE_SCALE`` for ~3 mm per-side
    clearance.
    """
    spec = mujoco.MjSpec.from_file(str(_XML_DIR / f"piece_{shape}.xml"))
    rgba = COLOR_VALUES[color]
    for geom in spec.geoms:
        if geom.name in ("piece_geom", "piece_visual"):
            geom.rgba = rgba
    scale = PIECE_SCALE[shape]
    for mesh in spec.meshes:
        mesh.scale = [scale, scale, scale]
    return spec
