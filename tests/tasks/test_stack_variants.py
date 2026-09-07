"""Invariants of the stack variant catalog (spares, prefix rule, scatter fit)."""

import math

import pytest

from loot_bench.tasks.common.scatter import ScatterGrid
from loot_bench.tasks.stack.mdp.commands import StackCommandCfg
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
def test_scatter_grid_fits_block_set(variant):
    cfg_defaults = StackCommandCfg.__dataclass_fields__
    ws_x = cfg_defaults["workspace_x"].default
    ws_y = cfg_defaults["workspace_y"].default
    radius = max(math.hypot(*b.half_extents[:2]) for b in variant.blocks)
    grid = ScatterGrid.tiling(ws_x, ws_y, object_radius=radius, device="cpu")
    assert grid.num_cells >= len(variant.blocks)
