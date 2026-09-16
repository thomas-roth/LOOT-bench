"""Shape-sorter variants.

A variant fixes the shape count and one spawn mode:

- ShapeSorter{2,3,4,5}FixedOrientation: the cube always spawns at the same
  deterministic pose (only its xy position is jittered).
- ShapeSorter{2,3,4,5}NoRotation: the cube spawns face-down on one of the faces
  whose hole is *not* needed this variant (plus a random yaw), so no required
  hole ever starts against the table.
- ShapeSorter{2,3,4,5}Full: the cube spawns face-down on any of its six faces
  (plus a random yaw) -- a required hole may start facing the table, so
  reorienting the cube may be necessary.

Shapes tail-extend across the difficulty axis (2 = the first 2 of the 5-shape
list) so demos for easier tasks can be cut from the hardest task's
recordings (see ``docs/task_categories.md`` "General"). Pieces are colored by
which tier first needs them, so the operator sees the collection order and
where a recording can be cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from loot_bench.tasks.common.colors import COLOR_NAMES, order_color

Shape = Literal["circle", "square", "rect", "triangle", "hexagon"]
SpawnMode = Literal["fixed", "no_rotation", "full"]

HARDEST_COUNT = 5
"""The largest shape count: the spawn table, the seed-prefix draws and the
tier boundaries all size for it, so a shorter variant's reset is the longer
one's prefix. Keep this equal to the largest ``num_pieces`` in
``SHAPE_SORTER_VARIANTS``, or a new hardest variant shifts every shorter
variant's spawns."""

# The shared color palette must hold at least ``HARDEST_COUNT`` many colors;
# the assert below fails at import if it does not, rather than at env build on
# an out-of-range index. Tier coloring itself needs only SPLIT_TIERS, but the
# invariant mirrors the domino category so a per-piece color switch stays safe.
assert len(COLOR_NAMES) >= HARDEST_COUNT, (
    "the shared palette is too short to give every piece its own color"
)

HOLE_SLACK = 0.005
"""Radial slack (m) added to every hole's bounding radius (x margin)."""

# Tier boundaries are the easier variants' piece counts (2, 3 and 4).
_FIRST_TIER = 2
_SECOND_TIER = 3
_THIRD_TIER = 4

# Resting face per orientation, matching mdp.commands flat quats. The cube has
# six holes including a star, but no star piece exists: the star socket is a
# dead face, which is why fixed spawn can pin it down and why no_rotation
# always has at least one allowed face.
FACE_DOWN_SHAPES: tuple[str, ...] = (
    "circle",
    "star",
    "square",
    "rect",
    "hexagon",
    "triangle",
)

# Shapes tail-extend: ShapeSorter2's shapes are shapes 0..1 of ShapeSorter5.
# Ordered by difficulty, so the tail extensions are the hard shapes: the
# circle/rect core, then square, then hexagon, then triangle.
_SHAPES_BY_COUNT: dict[int, tuple[Shape, ...]] = {
    2: ("circle", "rect"),
    3: ("circle", "rect", "square"),
    4: ("circle", "rect", "square", "hexagon"),
    5: ("circle", "rect", "square", "hexagon", "triangle"),
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
    margin: float = 1.0
    """Scale on the hole-slack success tolerance."""

    @property
    def num_pieces(self) -> int:
        return len(self.shapes)

    @property
    def colors(self) -> tuple[str, ...]:
        return tuple(
            order_color(i, _FIRST_TIER, _SECOND_TIER, _THIRD_TIER)
            for i in range(self.num_pieces)
        )

    def __post_init__(self):
        assert self.shapes == _SHAPES_BY_COUNT[len(self.shapes)], (
            "shapes must tail-extend: the first k of the hardest task's list"
        )
        assert self.spawn in ("fixed", "no_rotation", "full"), (
            f"unknown spawn mode {self.spawn!r}"
        )
        assert self.margin > 0.0, "margin must be positive"


SHAPE_SORTER_VARIANTS: dict[str, ShapeSorterVariantCfg] = {
    v.name: v
    for n in (2, 3, 4, 5)
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
    )
}

assert max(len(v.shapes) for v in SHAPE_SORTER_VARIANTS.values()) == HARDEST_COUNT, (
    "HARDEST_COUNT must equal the largest variant's shape count"
)
