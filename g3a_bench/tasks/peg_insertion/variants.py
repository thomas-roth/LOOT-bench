"""Variant definitions for the peg-insertion task category.

ARCHIVED (unregistered): superseded by ``tasks/shape_sorter`` for the shape-
sorter family; this category keeps only its original peg/two-sheet kinds,
importable and pyright-clean but not registered or tested. See ``README.md``.

Two kinds of task share one command/observation/reward stack:

- ``peg``        a single peg must be inserted into a single tight-fitting bore
                 on a fixed board. The bore is either vertical (mouth up) or
                 horizontal (mouth facing the robot).
- ``two_sheet``  a peg must be inserted through a hole that only lines up once
                 a movable top sheet has been slid over the fixed bottom sheet.

Geometry conventions (all lengths in metres): a *shape* is a convex prism.
``circle`` is a cylinder, ``square``/``rect`` are boxes, ``triangle``/
``hexagon`` are regular-polygon prisms built as the convex hull of their
corner vertices. ``circumradius``/``nsides`` fully describe the cross
section; the matching bore is a polygonal ring with apothem
``circumradius*cos(pi/nsides) + clearance`` so the peg is a tight fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

Shape = Literal["circle", "square", "rect", "triangle", "hexagon"]
Orientation = Literal["vertical", "horizontal"]
Kind = Literal["peg", "two_sheet"]

# Peg length along its insertion axis (total height = 2 * PEG_HALF_LEN).
PEG_HALF_LEN = 0.03
# Radial gap between a peg and its bore wall (tight fit).
CLEARANCE = 0.0015


@dataclass(frozen=True)
class ShapeGeom:
    """Cross-section description shared by a peg and its matching bore."""

    nsides: int
    """Number of polygon sides. A ``circle`` peg is a true cylinder; its
    nsides only sets how closely the bore apothem approximates the radius
    (the bore wall count is ``assets.CIRCLE_BORE_SEGMENTS``)."""
    circumradius: float
    """Distance from the cross-section centre to a corner."""
    rect_half: tuple[float, float] | None = None
    """For ``rect`` only: (half-x, half-y) of the rectangular cross section."""

    @property
    def inradius(self) -> float:
        """Centre-to-edge distance (apothem) of the peg cross section."""
        if self.rect_half is not None:
            return min(self.rect_half)
        return self.circumradius * math.cos(math.pi / self.nsides)


# Cross sections, each a tight fit in its own matching bore. Every board built
# from these has exactly one hole, so shape pairs need no physical-exclusion
# sizing.
SHAPE_GEOMS: dict[Shape, ShapeGeom] = {
    "circle": ShapeGeom(nsides=24, circumradius=0.0140),
    "square": ShapeGeom(nsides=4, circumradius=0.0130 * math.sqrt(2.0)),
    "rect": ShapeGeom(nsides=4, circumradius=0.0, rect_half=(0.0210, 0.0105)),
    "triangle": ShapeGeom(nsides=3, circumradius=0.0210),
    "hexagon": ShapeGeom(nsides=6, circumradius=0.0150),
}


def bore_apothem(shape: Shape) -> float:
    """Centre-to-wall distance of the bore that the peg slides into."""
    return SHAPE_GEOMS[shape].inradius + CLEARANCE


@dataclass(frozen=True)
class SocketCfg:
    """One bore on a board, plus the peg colour assigned to it."""

    shape: Shape
    color: str
    orientation: Orientation = "vertical"


@dataclass(frozen=True)
class InsertionVariantCfg:
    name: str
    """Variant identifier used in the task id (e.g. "PegCylinder")."""
    kind: Kind
    sockets: tuple[SocketCfg, ...]
    """One per peg; peg ``i`` must go into socket ``i``."""
    bore_depth: float = 0.045
    insert_depth: float = 0.026
    """Required insertion depth (along the bore axis) to count as inserted."""
    lateral_tol: float = 0.010
    """Max lateral offset of the peg from the bore axis when inserted."""
    tilt_tol_deg: float = 20.0
    """Max angle between the peg axis and the bore axis when inserted."""

    @property
    def num_sockets(self) -> int:
        return len(self.sockets)

    @property
    def peg_shapes(self) -> tuple[Shape, ...]:
        return cast("tuple[Shape, ...]", tuple(s.shape for s in self.sockets))

    @property
    def peg_colors(self) -> tuple[str, ...]:
        return tuple(s.color for s in self.sockets)

    @property
    def orientation(self) -> Orientation:
        return self.sockets[0].orientation

    def __post_init__(self):
        assert self.num_sockets >= 1
        if self.kind == "two_sheet":
            assert self.num_sockets == 1, "two_sheet uses a single through-hole"
            assert self.orientation == "vertical"


INSERTION_VARIANTS: dict[str, InsertionVariantCfg] = {
    v.name: v
    for v in (
        # --- Single peg, vertical bore, solid board. ---
        InsertionVariantCfg(
            name="PegCylinder",
            kind="peg",
            sockets=(SocketCfg("circle", "blue"),),
        ),
        InsertionVariantCfg(
            name="PegCuboid",
            kind="peg",
            sockets=(SocketCfg("rect", "red"),),
        ),
        InsertionVariantCfg(
            name="PegHex",
            kind="peg",
            sockets=(SocketCfg("hexagon", "green"),),
        ),
        # Same peg but the bore faces the robot (horizontal insertion).
        InsertionVariantCfg(
            name="PegCylinderH",
            kind="peg",
            sockets=(SocketCfg("circle", "blue", "horizontal"),),
        ),
        # --- Two-sheet alignment: slide the top sheet, then insert. ---
        InsertionVariantCfg(
            name="TwoSheet",
            kind="two_sheet",
            sockets=(SocketCfg("circle", "blue"),),
        ),
    )
}
