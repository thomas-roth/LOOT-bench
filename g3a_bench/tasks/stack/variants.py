"""Variant definitions for the stack task category.

A variant fixes two independent axes:

- **Stack{3,4,5}** — the target height, measured in blocks: ``target_blocks``
  blocks stacked on top of each other. Each variant carries ``2n`` identical
  blocks, strictly more than needed, so height (not block count) is the
  binding constraint and support structures stay viable.
- **{Easy,Medium,Hard}** — the block's height:width ratio (0.5 / 1.0 / 2.0,
  see ``DIFFICULTY_HALF_EXTENTS``). All blocks of a variant are identical, so
  the ratio alone decides how big the landing area is, how high the center of
  mass of each placement sits, and how tall the finished stack gets.

All blocks of a variant have the same height, so the target height is
reachable by exactly ``target_blocks`` blocks and by no other count (the
height tolerance stays below half a block height).

Block lists are *tail-extending* across the height axis within one difficulty
(Stack3's blocks are blocks 0..5 of Stack4, etc.) so demos for easier tasks
can be cut from the hardest task's recordings (see ``docs/task_categories.md``
"General"). Blocks are colored by their split-prefix pool (`order_color`): the
Stack3-prefix blocks are black, the Stack4-only extension gray, the
Stack5-only extension white — so the collect-in-order/split boundary is
visible while the base can still be built from any block in the (black)
Stack3 pool.
"""

from dataclasses import dataclass
from typing import Literal

from g3a_bench.tasks.common.colors import order_color

Difficulty = Literal["Easy", "Medium", "Hard"]

DIFFICULTIES: tuple[Difficulty, ...] = ("Easy", "Medium", "Hard")

TARGET_BLOCKS: tuple[int, ...] = (3, 4, 5)
"""Target heights in blocks; the Stack{n} half of the task id."""

# Half extents (hx, hy, hz) per difficulty. The footprint is square (blocks
# spawn at a random yaw, so both horizontal axes must be graspable), which
# makes the difficulty exactly the height:width ratio: 0.5 flat and wide
# (large landing area, low center of mass, short stack), 1.0 the standard
# cube, 2.0 tall and slender (small top face, tippy, and a Stack5 tower
# reaches 0.30 m). Volumes stay within 20% of each other, so the difficulty
# is the shape, not the size of a block.
DIFFICULTY_HALF_EXTENTS: dict[Difficulty, tuple[float, float, float]] = {
    "Easy": (0.025, 0.025, 0.0125),
    "Medium": (0.02, 0.02, 0.02),
    "Hard": (0.015, 0.015, 0.03),
}

BLOCK_DENSITY = 1562.5
"""kg/m^3 — the standard cube's 100 g at 0.04 m edge length. Mass follows the
shape instead of being fixed, so the slender Hard block is also the lightest."""

HEIGHT_TOL = 0.01
"""Max distance between a block's top face and the target height (x margin)."""


def block_half_extents(difficulty: Difficulty) -> tuple[float, float, float]:
    return DIFFICULTY_HALF_EXTENTS[difficulty]


def block_height(difficulty: Difficulty) -> float:
    return 2.0 * DIFFICULTY_HALF_EXTENTS[difficulty][2]


def block_mass(difficulty: Difficulty) -> float:
    hx, hy, hz = DIFFICULTY_HALF_EXTENTS[difficulty]
    return BLOCK_DENSITY * 8.0 * hx * hy * hz


@dataclass(frozen=True)
class BlockCfg:
    difficulty: Difficulty = "Medium"
    color: str = "red"

    @property
    def half_extents(self) -> tuple[float, float, float]:
        return block_half_extents(self.difficulty)

    @property
    def height(self) -> float:
        return block_height(self.difficulty)

    @property
    def mass(self) -> float:
        return block_mass(self.difficulty)


@dataclass(frozen=True)
class StackVariantCfg:
    name: str
    """Variant identifier used in the task id (e.g. "Stack3Easy")."""
    difficulty: Difficulty
    """Block height:width ratio; the {Easy,Medium,Hard} half of the task id."""
    target_blocks: int
    """Target height in blocks (target_height = target_blocks * block_height)."""
    blocks: tuple[BlockCfg, ...]
    """All block entities, in tail-extending index order."""
    margin: float = 1.0
    """Scale on the height tolerance (default = tight, for data collection)."""

    @property
    def block_height(self) -> float:
        return block_height(self.difficulty)

    @property
    def target_height(self) -> float:
        return self.target_blocks * self.block_height

    @property
    def height_tol(self) -> float:
        return HEIGHT_TOL * self.margin

    def __post_init__(self):
        # More blocks than needed: block count must not be the constraint.
        assert len(self.blocks) > self.target_blocks, "not enough blocks"
        assert all(b.difficulty == self.difficulty for b in self.blocks)
        # Below half a block height, so only a stack of exactly target_blocks
        # blocks sits within tolerance of the target.
        assert self.height_tol < self.block_height / 2.0, "tolerance too loose"


def _build_variants() -> dict[str, StackVariantCfg]:
    # Block counts grow with the target height, so the Stack3/Stack4 counts are
    # exactly the split-prefix boundaries used to color the pools.
    counts = {n: 2 * n for n in TARGET_BLOCKS}
    low, medium = counts[3], counts[4]
    out: dict[str, StackVariantCfg] = {}
    for difficulty in DIFFICULTIES:
        for n in TARGET_BLOCKS:
            v = StackVariantCfg(
                name=f"Stack{n}{difficulty}",
                difficulty=difficulty,
                target_blocks=n,
                blocks=tuple(
                    BlockCfg(difficulty=difficulty, color=order_color(i, low, medium))
                    for i in range(counts[n])
                ),
            )
            out[v.name] = v
    return out


STACK_VARIANTS: dict[str, StackVariantCfg] = _build_variants()
