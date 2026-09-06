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
recordings (see ``docs/task_categories.md`` "General"). Pieces are colored by
which tier first needs them, so the operator sees the collection order and
where a recording can be cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from g3a_bench.tasks.common.colors import SPLIT_TIERS, order_color

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

# Shapes tail-extend: ShapeSorter3's shapes are shapes 0..2 of ShapeSorter5.
# Ordered by difficulty, so the tail extensions are the hard shapes: the
# circle/rect/square core, then hexagon, then triangle.
_SHAPES_BY_COUNT: dict[int, tuple[Shape, ...]] = {
    3: ("circle", "rect", "square"),
    4: ("circle", "rect", "square", "hexagon"),
    5: ("circle", "rect", "square", "hexagon", "triangle"),
}

# Split tier -> piece color. Positional, not a color match: the light-wood
# sorter cube swallows the black/gray/white ramp, so the tiers take the three
# darkest domino colors in the same order.
_TIER_COLOR: dict[str, str] = dict(
    zip(SPLIT_TIERS, ("cobalt", "indigo", "violet"), strict=True)
)


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
        # Tier boundaries are the easier variants' piece counts (3 and 4).
        return tuple(_TIER_COLOR[order_color(i, 3, 4)] for i in range(self.num_pieces))


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
