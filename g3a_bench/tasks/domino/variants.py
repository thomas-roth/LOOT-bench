"""Variant definitions for the domino task category.

A variant fixes the domino count and up to three boolean modifiers:

- Domino{2,3,4}: n identical dominoes standing upright in a row.
- Domino{n}FixedRow: row position is fixed at workspace center (consistent
  across resets, useful for teleoperation training). Without this, the row's
  start position and orientation are drawn uniformly at random each reset.
- Domino{n}DifferentDominoes: dominoes get progressively thinner and lighter
  along the row (same height/width), so upright-standing/knock-over physics
  vary along the row.
- Domino{n}KnockOver: once every domino stands at its row slot, toppling the
  whole line as a chain is also required for success — every domino must
  fall along the row axis, all leaning the same way, in row order (sideways
  or out-of-order toppling doesn't count); a `phase` observation flags the
  build (0) vs. topple (1) half of the episode.

Modifiers are combinable: e.g., Domino3FixedRowDifferentDominoes combines
fixed row position with different masses; Domino4FixedRowKnockOver adds
knock-over to fixed rows.
"""

from dataclasses import dataclass

BASE_HALF_EXTENTS = (0.0075, 0.02, 0.035)
"""Half extents (hx, hy, hz) of a base domino."""

BASE_MASS = 0.10

SPACING = 0.05
"""Row slot spacing (m); < domino height (0.07) so toppling one knocks over
the next."""

HARDEST_COUNT = 4
"""The row-placement clamp always sizes the reach band for this count, so
the row's start-position/yaw distribution does not depend on the variant's
actual n (seed-prefix rule)."""

DOMINO_COLORS: tuple[str, ...] = ("sky", "cobalt", "indigo", "violet")
"""Per-index identity color (``tasks/common/colors.PALETTE`` keys), index 0
through ``HARDEST_COUNT - 1``. All registered variants use ``ordered=True``
(fixed domino-index -> slot-index assignment, see ``mdp/commands.py``), but
nothing previously let a teleoperator tell which physical domino is which
index -- every domino rendered identically, and spawn position is a uniform
random draw independent of index (see docs/DominoTask-Analysis.md). This is
a colorblind-safe cool-hue arc with monotonically decreasing brightness
(sky = lightest/index 0, violet = darkest/index 3), so the sequence reads
from hue position and from brightness alone -- deliberately excludes green,
which is reserved for the placed/success marker states."""

_THICKNESS_DECAY = 0.8


def domino_half_extents(index: int, different: bool) -> tuple[float, float, float]:
    hx, hy, hz = BASE_HALF_EXTENTS
    if different:
        hx *= _THICKNESS_DECAY**index
    return (hx, hy, hz)


def domino_mass(index: int, different: bool) -> float:
    mass = BASE_MASS
    if different:
        mass *= _THICKNESS_DECAY**index
    return mass


@dataclass(frozen=True)
class DominoVariantCfg:
    name: str
    """Variant identifier used in the task id (e.g. "Domino3")."""
    num_dominoes: int
    different: bool = False
    """Progressively thinner/lighter dominoes along the row."""
    knock_over: bool = False
    """Require toppling the completed line after it is built."""
    ordered: bool = False
    """Identity slot assignment; False allows proximity re-matching
    (interchangeable dominoes only, i.e. never combined with `different`)."""
    margin: float = 1.0
    """Scale on the position/tilt success tolerances."""
    fixed_row: bool = False
    """Fixed row position and orientation; otherwise sampled randomly each reset."""

    @property
    def half_extents(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            domino_half_extents(i, self.different) for i in range(self.num_dominoes)
        )

    @property
    def masses(self) -> tuple[float, ...]:
        return tuple(domino_mass(i, self.different) for i in range(self.num_dominoes))

    def __post_init__(self):
        assert not (self.different and not self.ordered), (
            "differently-weighted dominoes cannot be re-matched by proximity"
        )


DOMINO_VARIANTS: dict[str, DominoVariantCfg] = {
    v.name: v
    for n in (2, 3, 4)
    for fixed in (False, True)
    for v in (
        DominoVariantCfg(
            name=f"Domino{n}{'FixedRow' if fixed else ''}",
            num_dominoes=n,
            ordered=True,
            fixed_row=fixed,
        ),
        DominoVariantCfg(
            name=f"Domino{n}{'FixedRow' if fixed else ''}DifferentDominoes",
            num_dominoes=n,
            different=True,
            ordered=True,
            fixed_row=fixed,
        ),
        DominoVariantCfg(
            name=f"Domino{n}{'FixedRow' if fixed else ''}KnockOver",
            num_dominoes=n,
            knock_over=True,
            ordered=True,
            fixed_row=fixed,
        ),
    )
}
