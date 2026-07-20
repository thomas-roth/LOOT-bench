"""MjSpec builders and layout for the shape-sorter task category.

The cube and its pieces are STL/XML assets committed under ``xmls/``
(generated once by ``xmls/make_sorter_assets.py`` from the CAD STLs; don't
edit the generated files by hand). ``xmls/sorter_layout.json`` is the single
source for both the cube's hole geometry and the command term's success
metric, so they can never drift apart. Both the cube and every piece are free
bodies; the command term drops them at reset and re-derives the hole frames
from the cube's live pose every step (see ``mdp/commands.py``).

This module is a copy of the STL-sorter half of ``insertion/assets.py``
(pared down to what the shape sorter needs -- no peg/board/socket_layout
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

from g3a_bench.tasks.common.colors import PALETTE
from g3a_bench.tasks.shape_sorter.variants import Shape

_XML_DIR = Path(__file__).resolve().parent / "xmls"
SORTER_XML = _XML_DIR / "sorter.xml"
SORTER_LAYOUT_JSON = _XML_DIR / "sorter_layout.json"

# Cube placement in the robot workspace (env-local).
BOARD_POS = (0.34, 0.0, 0.0)
# Uniform XY jitter of the sorter cube around BOARD_POS on reset.
SORTER_SPAWN_JITTER = (0.02, 0.03)


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
    """The free-body STL shape-sorter cube (visual mesh + convex parts).

    A ``<freejoint/>`` body authored in ``xmls/sorter.xml``; the command term
    places it on every reset.
    """
    return mujoco.MjSpec.from_file(str(SORTER_XML))


def get_sorter_piece_spec(shape: Shape, color: str) -> mujoco.MjSpec:
    """A free-body STL sorter piece, tinted with its socket colour."""
    spec = mujoco.MjSpec.from_file(str(_XML_DIR / f"piece_{shape}.xml"))
    rgba = PALETTE[color]
    for geom in spec.geoms:
        if geom.name in ("piece_geom", "piece_visual"):
            geom.rgba = rgba
    return spec
