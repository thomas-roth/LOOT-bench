"""Shape-sorter variants.

Each fixes the shape set and cube spawn mode:
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
recordings (see ``docs/task_categories.md`` "General").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Shape = Literal["circle", "square", "rect", "triangle", "hexagon"]
SpawnMode = Literal["fixed", "no_rotation", "full"]

# Resting face per orientation, matching mdp.commands flat quats.
FACE_DOWN_SHAPES: tuple[str, ...] = (
    "circle",
    "star",
    "square",
    "rect",
    "hexagon",
    "triangle",
)

# Piece colour per shape (matches the legacy insertion sorter's assignment).
SHAPE_COLOR: dict[Shape, str] = {
    "circle": "blue",
    "square": "red",
    "triangle": "yellow",
    "rect": "orange",
    "hexagon": "green",
}

# Shapes tail-extend: ShapeSorter3's shapes are shapes 0..2 of ShapeSorter5.
_SHAPES_BY_COUNT: dict[int, tuple[Shape, ...]] = {
    3: ("circle", "square", "triangle"),
    4: ("circle", "square", "triangle", "rect"),
    5: ("circle", "square", "triangle", "rect", "hexagon"),
}


def allowed_faces(shapes: tuple[Shape, ...]) -> tuple[int, ...]:
    """Faces whose hidden shape is not required. For no_rotation spawn."""
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
        return tuple(SHAPE_COLOR[s] for s in self.shapes)


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
