"""Henge variants.

A *level* is one block lying flat (the **lintel**) carried at a fixed height
by whatever stands under it; a variant asks for 1, 2 or 3 levels stacked on
top of each other. Two builds satisfy a level and the task encodes neither:
a trilithon (two uprights, one lintel) or a T (one centred upright). The
trilithon costs a third block and stands; the T is faster and tips when the
next level is built on it. The physics decides, which is the point of the
multi-level tasks (ADR-020).

There is one block type, a square prism used both standing (an upright,
contributing ``BLOCK_LENGTH`` to a pile's height) and lying (a lintel or a
flat, contributing ``BLOCK_WIDTH``). Every variant gets exactly
``BLOCKS_PER_LEVEL`` blocks per level, tail-extending across the level axis
(1Henge's blocks are blocks 0..2 of 2Henges, which are blocks 0..5 of
3Henges), so ``loot-split-demos`` can cut the easier tasks out of a 3Henges
recording. ``order_color`` paints those cut boundaries onto the blocks, one
colour tier per level.

**The four geometry/tolerance constants are one coupled system.** A lintel
spanning two piles of unequal height rests at their mean, so unintended
builds of uprights and flats can land inside ``HEIGHT_TOL`` of a target;
the tilt gate ``TILT_COS_MAX`` is what excludes them. Change any one of
``BLOCK_LENGTH``, ``BLOCK_WIDTH``, ``HEIGHT_TOL`` or ``TILT_COS_MAX`` and
:func:`assert_no_alternative_build` re-checks the whole system at import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from loot_bench.tasks.common.colors import order_color

LEVELS: tuple[int, ...] = (1, 2, 3)
"""Levels per task; the ``{n}Henge(s)`` half of the task id."""

BLOCKS_PER_LEVEL = 3
"""Two uprights and one lintel: the trilithon budget.

Not the minimum (a single pile ``U, F, U, F, U, F`` completes three levels
with six blocks). Chosen because it is the reference build's cost, it puts
the colour tiers exactly on level boundaries, and it keeps the tail
extension clean."""

BLOCK_HALF_EXTENTS: tuple[float, float, float] = (0.010, 0.010, 0.0325)
"""(hx, hy, hz); the long axis is local z, so a spawned block stands.

Square cross-section: blocks spawn at a random yaw and must grasp the same
standing and lying. The size is pinned by three constraints at once
(trilithon pad clearance, reach at three levels, the cheat lattice below),
see ADR-020; it is not a hand-picked "nice" block."""

BLOCK_LENGTH = 2.0 * BLOCK_HALF_EXTENTS[2]
"""``L`` = 0.065 m: what an upright adds to a pile's height."""

BLOCK_WIDTH = 2.0 * BLOCK_HALF_EXTENTS[0]
"""``w`` = 0.020 m: what a flat block adds to a pile's height."""

BLOCK_DENSITY = 1562.5
"""kg/m^3, the standard cube's 100 g at a 0.04 m edge. Kept so the block's
mass (40.6 g) follows its shape rather than being a second free number."""

BLOCK_MASS = BLOCK_DENSITY * 8.0 * math.prod(BLOCK_HALF_EXTENTS)

LEVEL_HEIGHT = BLOCK_LENGTH + BLOCK_WIDTH
"""One level, 0.085 m: an upright's length plus the lintel lying on it."""

HEIGHT_TOL = 0.003
"""Max |lintel centre height - level target| (m), times the variant margin.

Provisional, taken from the ~3 mm settle in ``docs/sim_settings.md`` and to
be replaced by the henge settle measurement. It is coupled to
``TILT_COS_MAX``: the achievable heights of upright/flat piles form a
lattice of pitch ``w/4`` = 5 mm on one pile and ``w/8`` = 2.5 mm across a
pile *pair*, so raising this means tightening the tilt gate to match, and
it cannot reach ``w/4`` at all (a zero-tilt single pile then lands inside
it and no tilt gate excludes a level lintel)."""

TILT_COS_MAX = 0.10
"""Max |cos(block long axis, world z)| for a block to count as horizontal.

0.10 is 5.7 deg. This is not a "looks level enough" threshold, and it is
deliberately tighter than the 0.26 (15 deg) the other categories use: a
lintel spanning two piles of unequal top height is tilted by
``atan(dz / (L - w))``, so this gate bounds ``dz`` to
``(L - w) * tan(asin(0.10))`` = 4.5 mm and thereby excludes every
upright/flat pile pair that would otherwise sit inside ``HEIGHT_TOL`` (at
0.26 four such builds fit the block budget). Defined here and nowhere else:
``mdp/success.py`` imports it, and ``loot-split-demos`` recomputes success
through that kernel, so a second copy that drifted would make an offline
relabel disagree with the environment."""


def alternative_builds(
    length: float,
    width: float,
    height_tol: float,
    tilt_cos_max: float,
    levels: int,
) -> list[tuple[int, tuple[int, int], tuple[int, int], int]]:
    """Unintended upright/flat builds the criterion would accept, if any.

    Enumerates a lintel resting on one or two piles of ``a`` uprights and
    ``b`` flats (pile top ``a * length + b * width``) inside the block budget
    of ``levels`` levels, and returns every ``(level, pile_a, pile_b, cost)``
    whose lintel is inside ``height_tol`` of that level's target *and* inside
    the tilt gate, other than the intended ``(k, k - 1)`` pile under level
    ``k``. Pure arithmetic, so the regression tests can drive it with
    known-bad inputs; :func:`assert_no_alternative_build` runs it at import
    on the module constants.

    - A single pile is the ``pile_a == pile_b`` diagonal and costs
      ``a + b + 1`` blocks, not ``2 (a + b) + 1``: charging it twice would
      skip in-budget T-shaped builds as over budget.
    - The lintel centre is ``(top_a + top_b) / 2 + width / 2``.
    - The lintel's tilt satisfies ``tan(tilt) = |top_a - top_b| / span``
      (``tan``, not ``sin``: the pile separation is the horizontal run under
      the tilted lintel, not the distance along it).
    - The span is ``length - width``, the widest the pile centres can sit and
      still both be overhung, and the conservative choice: a tilted lintel
      touches the same relative edge of both piles, so contact separation
      equals centre separation. Taking ``length`` instead would drop the
      2.5 mm cheat's tilt from 6.34 to 4.4 deg and let it through the gate.
    """
    budget = BLOCKS_PER_LEVEL * levels
    max_delta = (length - width) * math.tan(math.asin(tilt_cos_max))
    piles = {
        (a, b): a * length + b * width
        for a in range(budget + 1)
        for b in range(budget + 1)
        if 0 < a + b <= budget
    }
    found: list[tuple[int, tuple[int, int], tuple[int, int], int]] = []
    for k in range(1, levels + 1):
        target = k * (length + width) - width / 2
        intended = (k, k - 1)
        for pile_a, top_a in piles.items():
            for pile_b, top_b in piles.items():
                if pile_a > pile_b:
                    continue  # unordered pair, listed once
                if pile_a == pile_b:
                    cost = sum(pile_a) + 1
                else:
                    cost = sum(pile_a) + sum(pile_b) + 1
                if cost > budget or abs(top_a - top_b) > max_delta:
                    continue
                if abs((top_a + top_b) / 2 + width / 2 - target) > height_tol:
                    continue
                if pile_a == pile_b == intended:
                    continue
                found.append((k, pile_a, pile_b, cost))
    return found


def assert_no_alternative_build(levels: int, height_tol: float) -> None:
    """Refuse to import if the geometry and the tolerances have drifted apart.

    Runs at import (from every variant's ``__post_init__``) rather than as a
    test, because the constants it guards are the kind that get "tuned" in
    place: a looser ``HEIGHT_TOL`` after a settle measurement, a wider
    ``TILT_COS_MAX`` copied from another category, a block resized for
    reach. Any of those can open a build that scores without a henge under
    it, and a unit test that nobody runs would not stop the task from
    registering.
    """
    builds = alternative_builds(
        BLOCK_LENGTH, BLOCK_WIDTH, height_tol, TILT_COS_MAX, levels
    )
    assert not builds, (
        f"a {levels}-level henge is also reachable by an upright/flat build "
        f"other than the intended one: {builds} (level, pile A (uprights, "
        "flats), pile B, blocks). BLOCK_LENGTH, BLOCK_WIDTH, HEIGHT_TOL and "
        "TILT_COS_MAX are one coupled system; see ADR-020"
    )


@dataclass(frozen=True)
class HengeVariantCfg:
    name: str
    """Variant identifier, used only for the experiment-name string."""
    levels: int
    """How many levels to stack; the ``{n}Henge(s)`` half of the task id."""
    block_colors: tuple[str, ...]
    """One palette colour per block, in tail-extending index order."""
    margin: float = 1.0
    """Scale on the height tolerance (default = tight, for data collection)."""

    @property
    def num_blocks(self) -> int:
        return len(self.block_colors)

    @property
    def height_tol(self) -> float:
        return HEIGHT_TOL * self.margin

    def level_centre_height(self, k: int) -> float:
        """Target lintel centre height above the table for level ``k`` (1-based).

        The centre, not the top face: a lintel resting on two uneven uprights
        sits at their mean, so its centre is exact geometry to first order,
        while its box-AABB top at the gate's 5.7 deg limit reads 3.2 mm high,
        comparable to ``HEIGHT_TOL`` itself (ADR-020).
        """
        assert 1 <= k <= self.levels, f"level {k} of a {self.levels}-level henge"
        return k * LEVEL_HEIGHT - BLOCK_WIDTH / 2

    @property
    def level_centre_heights(self) -> tuple[float, ...]:
        return tuple(self.level_centre_height(k) for k in range(1, self.levels + 1))

    def __post_init__(self):
        assert self.levels >= 1, "a henge has at least one level"
        assert len(self.block_colors) == BLOCKS_PER_LEVEL * self.levels, (
            f"{self.name}: {len(self.block_colors)} blocks for {self.levels} levels"
        )
        assert_no_alternative_build(self.levels, self.height_tol)


_TIER_BOUNDS = (BLOCKS_PER_LEVEL, 2 * BLOCKS_PER_LEVEL)
"""The easier variants' block counts, which is where a recording is cut."""


def _build_variants() -> dict[str, HengeVariantCfg]:
    out: dict[str, HengeVariantCfg] = {}
    for levels in LEVELS:
        v = HengeVariantCfg(
            name=f"Henge{levels}",
            levels=levels,
            block_colors=tuple(
                order_color(i, *_TIER_BOUNDS) for i in range(BLOCKS_PER_LEVEL * levels)
            ),
        )
        out[v.name] = v
    return out


HENGE_VARIANTS: dict[str, HengeVariantCfg] = _build_variants()
