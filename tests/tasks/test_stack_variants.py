"""Invariants of the stack variant catalog (material, prefix rule, scatter fit)."""

from itertools import combinations

import pytest

from g3a_bench.tasks.common.scatter import ScatterGrid
from g3a_bench.tasks.stack.mdp.commands import StackCommandCfg
from g3a_bench.tasks.stack.variants import (
    FORM_HALF_EXTENTS,
    STACK_VARIANTS,
    form_height,
)

_FAMILIES = {
    "base": ["Stack3", "Stack4", "Stack5"],
    "different": [
        "Stack3DifferentBlocks",
        "Stack4DifferentBlocks",
        "Stack5DifferentBlocks",
    ],
    "two_stacks": ["Stack3TwoStacks", "Stack4TwoStacks", "Stack5TwoStacks"],
    "two_stacks_different": [
        "Stack3TwoStacksDifferentBlocks",
        "Stack4TwoStacksDifferentBlocks",
        "Stack5TwoStacksDifferentBlocks",
    ],
}


def test_catalog_has_all_families():
    assert set(STACK_VARIANTS) == {n for names in _FAMILIES.values() for n in names}


@pytest.mark.parametrize("variant", STACK_VARIANTS.values(), ids=lambda v: v.name)
def test_more_material_than_needed(variant):
    total = sum(form_height(b.form) for b in variant.blocks)
    needed = (2 if variant.two_stacks else 1) * variant.target_height
    assert total > needed  # strictly more: block count must not be the constraint


@pytest.mark.parametrize("variant", STACK_VARIANTS.values(), ids=lambda v: v.name)
def test_exact_height_combination_exists(variant):
    heights = [form_height(b.form) for b in variant.blocks]
    hits = {
        round(sum(c), 4)
        for r in range(1, len(heights) + 1)
        for c in combinations(heights, r)
    }
    assert round(variant.target_height, 4) in hits


@pytest.mark.parametrize("family", _FAMILIES.values(), ids=list(_FAMILIES))
def test_blocks_tail_extend_across_difficulty(family):
    for easier, harder in zip(family, family[1:], strict=False):
        e, h = STACK_VARIANTS[easier].blocks, STACK_VARIANTS[harder].blocks
        assert h[: len(e)] == e


@pytest.mark.parametrize("variant", STACK_VARIANTS.values(), ids=lambda v: v.name)
def test_scatter_grid_fits_block_set(variant):
    import math

    cfg_defaults = StackCommandCfg.__dataclass_fields__
    ws_x = cfg_defaults["workspace_x"].default
    ws_y = cfg_defaults["workspace_y"].default
    radius = max(math.hypot(*FORM_HALF_EXTENTS[b.form][:2]) for b in variant.blocks)
    grid = ScatterGrid.tiling(ws_x, ws_y, object_radius=radius, device="cpu")
    assert grid.num_cells >= len(variant.blocks)
