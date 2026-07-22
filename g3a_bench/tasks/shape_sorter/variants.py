"""Variant definitions for the shape-sorter task category.

A variant fixes the shape set (every piece must be inserted, unlike the
legacy insertion sorter which only commanded one target piece) and how the
free-standing sorter cube spawns:

- ShapeSorter{3,4,5}FixedOrientation: the cube always spawns at the same
  deterministic pose (only its xy position is jittered).
- ShapeSorter{3,4,5}NoRotation: the cube spawns face-down on one of the faces
  whose hole is *not* needed this variant (plus a random yaw), so no required
  hole ever starts against the table.
- ShapeSorter{3,4,5}Full: the cube spawns face-down on any of its six faces
  (plus a random yaw) -- a required hole may start facing the table, so
  reorienting the cube may be necessary.
- ShapeSorter{3,4,5}RandomPositions: same deterministic cube pose as
  FixedOrientation, but the pieces spawn on a wide ring of cells around the
  cube instead of the near band the other variants use.

Shapes tail-extend across the difficulty axis (3 = the first 3 of the 5-shape
list) so demos for easier tasks can be cut from the hardest task's
recordings (see ``docs/TaskCategories.md`` "General"). Pieces keep the three
darkest domino colors instead of the old black/gray/white tier palette: the
3-shape core uses the lightest of those colors, the 4th shape uses the middle
tone, and the 5th shape uses the darkest tone, so the cutting split stays
aligned with the visual difficulty tiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from g3a_bench.tasks.common.colors import order_color

Shape = Literal["circle", "square", "rect", "triangle", "hexagon"]
SpawnMode = Literal["fixed", "no_rotation", "full"]

# The six flat resting orientations of the sorter cube (see
# ``mdp.commands._flat_quats``), in the order the STL sorter's faces expose
# them: which shape's hole faces the table under each orientation.
FACE_DOWN_SHAPES: tuple[str, ...] = (
    "circle",
    "star",
    "square",
    "rect",
    "hexagon",
    "triangle",
)

# Shapes tail-extend: ShapeSorter3's shapes are shapes 0..2 of ShapeSorter5.
# The easy tier is the cylinder/rect/square core, then the harder hexagon,
# and finally the triangle as the hardest piece.
_SHAPES_BY_COUNT: dict[int, tuple[Shape, ...]] = {
    3: ("circle", "rect", "square"),
    4: ("circle", "rect", "square", "hexagon"),
    5: ("circle", "rect", "square", "hexagon", "triangle"),
}

# The sorter reuses the three darkest domino palette entries in the same
# tier order as the demo split: the first 3 pieces share the lightest tone,
# the 4th gets the middle tone, and the 5th gets the darkest tone.
_SORTER_COLOR_BY_TIER: dict[str, str] = {
    "black": "cobalt",
    "gray": "indigo",
    "white": "violet",
}


def allowed_faces(shapes: tuple[Shape, ...]) -> tuple[int, ...]:
    """Face indices whose hidden (down) shape is *not* one of ``shapes``.

    Used by the ``no_rotation`` spawn mode: dropping the cube on one of these
    faces guarantees no hole the variant actually needs starts against the
    table. The returned set only shrinks as more shapes are required, and is
    always non-empty (the star face is never a required shape).
    """
    return tuple(i for i, s in enumerate(FACE_DOWN_SHAPES) if s not in shapes)


@dataclass(frozen=True)
class ShapeSorterVariantCfg:
    name: str
    """Variant identifier used in the task id (e.g. "ShapeSorter3Full")."""
    shapes: tuple[Shape, ...]
    """One piece/hole per shape, in tail-extending index order."""
    spawn: SpawnMode = "fixed"
    wide_positions: bool = False
    """Scatter pieces on a wide ring around the cube instead of the near band."""
    margin: float = 1.0
    """Scale on the hole-slack success tolerance."""

    @property
    def num_pieces(self) -> int:
        return len(self.shapes)

    @property
    def colors(self) -> tuple[str, ...]:
        return tuple(
            _SORTER_COLOR_BY_TIER[order_color(i, 3, 4)] for i in range(self.num_pieces)
        )


SHAPE_SORTER_VARIANTS: dict[str, ShapeSorterVariantCfg] = {
    v.name: v
    for n in (3, 4, 5)
    for v in (
        ShapeSorterVariantCfg(
            name=f"ShapeSorter{n}FixedOrientation",
            shapes=_SHAPES_BY_COUNT[n],
            spawn="fixed",
        ),
        ShapeSorterVariantCfg(
            name=f"ShapeSorter{n}NoRotation",
            shapes=_SHAPES_BY_COUNT[n],
            spawn="no_rotation",
        ),
        ShapeSorterVariantCfg(
            name=f"ShapeSorter{n}Full",
            shapes=_SHAPES_BY_COUNT[n],
            spawn="full",
        ),
        ShapeSorterVariantCfg(
            name=f"ShapeSorter{n}RandomPositions",
            shapes=_SHAPES_BY_COUNT[n],
            spawn="fixed",
            wide_positions=True,
        ),
    )
}
