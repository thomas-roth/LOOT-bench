"""Variant definitions for the stacking task category.

A variant fixes the block set and the target height:

- Stack{3,4,5}: n identical cubes, target height n * 0.04.
- Stack{n}DifferentBlocks: mixed forms (rect bridging base, tall block) with
  more material than needed, so block choice becomes a strategy.
- Stack{n}TwoStacks: 2n cubes; two separate stacks must reach the target
  height with xy separation.

Block lists are *tail-extending* across the difficulty axis (Stack3's blocks
are blocks 0..2 of Stack4, etc.) so demos for easier tasks can be cut from
the hardest task's recordings (see ``docs/TaskCategories.md`` "General").
"""

from dataclasses import dataclass
from typing import Literal

Form = Literal["cube", "rect", "tall"]

# Half extents (hx, hy, hz) and mass per form.
FORM_HALF_EXTENTS: dict[Form, tuple[float, float, float]] = {
    "cube": (0.02, 0.02, 0.02),
    "rect": (0.03, 0.02, 0.015),
    "tall": (0.02, 0.02, 0.03),
}

FORM_MASS: dict[Form, float] = {
    "cube": 0.10,
    "rect": 0.12,
    "tall": 0.06,
}

# Height of one standard cube; the target height is measured in cubes.
CUBE_HEIGHT = 2.0 * FORM_HALF_EXTENTS["cube"][2]

HEIGHT_TOL = 0.01
"""Max distance between a block's top face and the target height (× margin)."""

MIN_STACK_SEPARATION = 0.06
"""Min xy distance between the two reached blocks for TwoStacks success."""


def form_height(form: Form) -> float:
    return 2.0 * FORM_HALF_EXTENTS[form][2]


@dataclass(frozen=True)
class BlockCfg:
    form: Form = "cube"
    color: str = "red"


@dataclass(frozen=True)
class StackingVariantCfg:
    name: str
    """Variant identifier used in the task id (e.g. "Stack3")."""
    target_blocks: int
    """Target height in standard cubes (target_height = target_blocks * 0.04)."""
    blocks: tuple[BlockCfg, ...]
    """All block entities, in tail-extending index order."""
    two_stacks: bool = False
    """If True, two separated blocks must both reach the target height."""
    margin: float = 1.0
    """Scale on the height tolerance (default = tight, for data collection)."""

    @property
    def target_height(self) -> float:
        return self.target_blocks * CUBE_HEIGHT

    @property
    def height_tol(self) -> float:
        return HEIGHT_TOL * self.margin

    def __post_init__(self):
        # Enough material to reach the target height (twice for TwoStacks).
        total = sum(form_height(b.form) for b in self.blocks)
        needed = (2 if self.two_stacks else 1) * self.target_height
        assert total + self.height_tol >= needed, "not enough block material"


def _cubes(n: int, color: str = "red") -> tuple[BlockCfg, ...]:
    return tuple(BlockCfg(form="cube", color=color) for _ in range(n))


def _different_blocks(n: int) -> tuple[BlockCfg, ...]:
    """Mixed set: rect bridging bases + tall block first, cubes appended."""
    return (
        BlockCfg(form="rect", color="purple"),
        BlockCfg(form="rect", color="purple"),
        BlockCfg(form="tall", color="yellow"),
    ) + _cubes(n - 1)


STACKING_VARIANTS: dict[str, StackingVariantCfg] = {
    v.name: v
    for n in (3, 4, 5)
    for v in (
        StackingVariantCfg(
            name=f"Stack{n}",
            target_blocks=n,
            blocks=_cubes(n),
        ),
        StackingVariantCfg(
            name=f"Stack{n}DifferentBlocks",
            target_blocks=n,
            blocks=_different_blocks(n),
        ),
        StackingVariantCfg(
            name=f"Stack{n}TwoStacks",
            target_blocks=n,
            blocks=_cubes(2 * n),
            two_stacks=True,
        ),
    )
}
