"""Variant definitions for the domino task category.

A variant fixes the domino count and one modifier:

- Domino{2,3,4,5,6}: n identical dominoes standing upright in a row, at the
  same fixed row every reset (centred at x = 0.40, along +y).
- Domino{n}RandomRow: the row centre's y and the row yaw are instead drawn
  uniformly at random each reset (only x is pinned at 0.40).
- Domino{n}DifferentDominoes: dominoes get progressively thinner and lighter
  along the row (same height/width), so upright-standing/knock-over physics
  vary along the row.
- Domino{n}KnockOver: once every domino stands at its row slot, toppling the
  whole line as a chain is also required for success: every domino must
  fall along the row axis, all leaning the same way, in row order (sideways
  or out-of-order toppling doesn't count); a `phase` observation flags the
  build (0) vs. topple (1) half of the episode.

The row is fixed by default because a teleoperator building demos benefits
from a layout that does not move between takes; ``RandomRow`` is the one
variant that opts back into a sampled row centre and yaw. The dominoes spawn
from the fixed spawn table (with jitter) in every variant.
"""

from dataclasses import dataclass

from loot_bench.tasks.common.colors import COLOR_NAMES

BASE_HALF_EXTENTS = (0.0075, 0.015, 0.035)
"""Half extents (hx, hy, hz) of a base domino."""

BASE_MASS = 0.10

SPACING = 0.05
"""Row slot spacing (m); < domino height (0.07) so toppling one knocks over
the next."""

HARDEST_COUNT = 6
"""The row is always laid out for this count — the random row's y-range and
the fixed row's centring both size for it — so the row centre does not depend
on the variant's actual n (seed-prefix rule). Keep this equal to the largest
``num_dominoes`` in ``DOMINO_VARIANTS``, or a new hardest variant shifts every
shorter variant's slots."""

# The shared color palette must hold at least ``HARDEST_COUNT`` many colors;
# the assert below fails at import if it does not, rather than at env build on
# an out-of-range index.
assert len(COLOR_NAMES) >= HARDEST_COUNT, (
    "the shared palette is too short to give every domino its own color"
)

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
    random_row: bool = False
    """Sample the row centre's y and the row yaw each reset; only x stays
    pinned. Default is the fixed row."""

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
    for n in (2, 3, 4, 5, 6)
    for v in (
        DominoVariantCfg(name=f"Domino{n}", num_dominoes=n, ordered=True),
        DominoVariantCfg(
            name=f"Domino{n}RandomRow",
            num_dominoes=n,
            ordered=True,
            random_row=True,
        ),
        DominoVariantCfg(
            name=f"Domino{n}DifferentDominoes",
            num_dominoes=n,
            different=True,
            ordered=True,
        ),
        DominoVariantCfg(
            name=f"Domino{n}KnockOver", num_dominoes=n, knock_over=True, ordered=True
        ),
    )
}
