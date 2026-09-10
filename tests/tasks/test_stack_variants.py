"""Invariants of the stack variant catalog (spares, prefix rule, spawn table)."""

import pytest
import torch

from loot_bench.tasks.common.fixed_spawns import sample_fixed_spawns
from loot_bench.tasks.stack.mdp.commands import STACK_SPAWN_CELLS
from loot_bench.tasks.stack.variants import (
    DIFFICULTIES,
    STACK_VARIANTS,
    TARGET_BLOCKS,
    Difficulty,
    block_half_extents,
    block_height,
)

_FAMILIES = {d: [f"Stack{n}{d}" for n in TARGET_BLOCKS] for d in DIFFICULTIES}


def test_catalog_has_all_families():
    assert set(STACK_VARIANTS) == {n for names in _FAMILIES.values() for n in names}


def test_aspect_ratio_orders_the_difficulties():
    def ratio(difficulty: Difficulty) -> float:
        hx, hy, _ = block_half_extents(difficulty)
        assert hx == hy, "footprint must be square (blocks spawn at random yaw)"
        return block_height(difficulty) / (2.0 * hx)

    assert ratio("Easy") < ratio("Medium") < ratio("Hard")


@pytest.mark.parametrize("variant", STACK_VARIANTS.values(), ids=lambda v: v.name)
def test_more_blocks_than_needed(variant):
    # strictly more: block count must not be the constraint
    assert len(variant.blocks) > variant.target_blocks


@pytest.mark.parametrize("variant", STACK_VARIANTS.values(), ids=lambda v: v.name)
def test_only_the_target_block_count_reaches_the_target(variant):
    # All blocks share one height, so N blocks hit the target exactly and no
    # other count comes within tolerance.
    h, tol = variant.block_height, variant.height_tol
    assert variant.target_height == pytest.approx(variant.target_blocks * h)
    for n in range(len(variant.blocks) + 1):
        reached = abs(n * h - variant.target_height) <= tol
        assert reached is (n == variant.target_blocks)


@pytest.mark.parametrize("family", _FAMILIES.values(), ids=list(_FAMILIES))
def test_blocks_tail_extend_across_target_height(family):
    for easier, harder in zip(family, family[1:], strict=False):
        e, h = STACK_VARIANTS[easier].blocks, STACK_VARIANTS[harder].blocks
        assert h[: len(e)] == e


@pytest.mark.parametrize("variant", STACK_VARIANTS.values(), ids=lambda v: v.name)
def test_block_count_is_two_n(variant):
    assert len(variant.blocks) == 2 * variant.target_blocks


def test_spawn_table_holds_hardest_block_set():
    hardest = max(len(v.blocks) for v in STACK_VARIANTS.values())
    assert hardest == 2 * max(TARGET_BLOCKS)
    assert len(STACK_SPAWN_CELLS) >= hardest


@pytest.mark.parametrize("family", _FAMILIES.values(), ids=list(_FAMILIES))
def test_easier_reset_is_prefix_of_harder(family):
    # Object i takes cell i, so Stack3's reset is a prefix of Stack4's and
    # Stack4's a prefix of Stack5's.
    cells = torch.tensor(STACK_SPAWN_CELLS)
    counts = [len(STACK_VARIANTS[name].blocks) for name in family]
    assert counts == [2 * n for n in TARGET_BLOCKS]
    for easier, harder in zip(counts, counts[1:], strict=False):
        torch.manual_seed(11)
        xy_easy, yaw_easy = sample_fixed_spawns(cells, 4, easier)
        torch.manual_seed(11)
        xy_hard, yaw_hard = sample_fixed_spawns(cells, 4, harder)
        assert torch.equal(xy_easy, xy_hard[:, :easier])
        assert torch.equal(yaw_easy, yaw_hard[:, :easier])
