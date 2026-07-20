"""Variant definitions for the stack task category.

A variant fixes the block set and the target height:

- Stack{3,4,5}: 2n identical cubes, target height n * 0.04. Block count is
  strictly more than the minimum needed, so height, not block count, is the
  binding constraint.
- Stack{n}DifferentBlocks: a mixed prefix of four forms (tall, cylinder,
  plate x2) followed by cubes, always with more material than needed, so
  block choice becomes a real strategy.
- Stack{n}TwoStacks: 2n+2 cubes; two separate stacks must each reach the
  target height with xy separation.
- Stack{n}TwoStacksDifferentBlocks: a doubled mixed prefix (tall, cylinder
  x2, plate x2) followed by cubes; two separate stacks, each buildable from
  a mix of forms.

All form heights are multiples of 0.02 m, and every variant's block set
contains a subset whose heights sum exactly to the target height (a multiple
of 0.04 m), so a target is always exactly reachable, not just exceedable.

Block lists are *tail-extending* across the difficulty axis within each
family (Stack3's blocks are blocks 0..2 of Stack4, etc.) so demos for easier
tasks can be cut from the hardest task's recordings (see
``docs/TaskCategories.md`` "General").
"""

from dataclasses import dataclass
from typing import Literal

Form = Literal["cube", "plate", "tall", "cylinder"]

# Half extents (hx, hy, hz) per form; the cylinder uses hx = hy = radius.
# All heights are multiples of 0.02 so mixed sets can reach every target
# height (a multiple of 0.04) exactly. Trade-offs: cube = balanced
# baseline; plate = wide stabilizing cap/bridge, little height; tall =
# most height per placement but slender and tippy; cylinder = good height
# and yaw-free grasping, but rolls when tipped.
FORM_HALF_EXTENTS: dict[Form, tuple[float, float, float]] = {
    "cube": (0.02, 0.02, 0.02),
    "plate": (0.03, 0.02, 0.01),
    "tall": (0.015, 0.015, 0.04),
    "cylinder": (0.02, 0.02, 0.03),
}

FORM_MASS: dict[Form, float] = {
    "cube": 0.10,
    "plate": 0.09,
    "tall": 0.06,
    "cylinder": 0.08,
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
class StackVariantCfg:
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


# Mixed prefix: 0.08 + 0.06 + 2 * 0.02 = 0.18 m of material.
_MIXED = (
    BlockCfg(form="tall", color="yellow"),
    BlockCfg(form="cylinder", color="cyan"),
    BlockCfg(form="plate", color="purple"),
    BlockCfg(form="plate", color="purple"),
)
# Doubled for TwoStacks: 0.32 m.
_MIXED_TWO = _MIXED + (
    BlockCfg(form="tall", color="yellow"),
    BlockCfg(form="cylinder", color="cyan"),
)


def _different_blocks(n: int) -> tuple[BlockCfg, ...]:
    return _MIXED + _cubes(n - 1)


def _two_stacks_different_blocks(n: int) -> tuple[BlockCfg, ...]:
    return _MIXED_TWO + _cubes(2 * n - 6)


STACK_VARIANTS: dict[str, StackVariantCfg] = {
    v.name: v
    for n in (3, 4, 5)
    for v in (
        StackVariantCfg(name=f"Stack{n}", target_blocks=n, blocks=_cubes(2 * n)),
        StackVariantCfg(
            name=f"Stack{n}DifferentBlocks",
            target_blocks=n,
            blocks=_different_blocks(n),
        ),
        StackVariantCfg(
            name=f"Stack{n}TwoStacks",
            target_blocks=n,
            blocks=_cubes(2 * n + 2),
            two_stacks=True,
        ),
        StackVariantCfg(
            name=f"Stack{n}TwoStacksDifferentBlocks",
            target_blocks=n,
            blocks=_two_stacks_different_blocks(n),
            two_stacks=True,
        ),
    )
}
