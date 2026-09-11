"""Invariants of the henge variant catalog: block budgets, colour tiers, level
heights, the geometry/tolerance assertion, and the spawn table."""

import math

import pytest
import torch

from loot_bench.tasks.common.fixed_spawns import (
    min_pairwise_separation,
    sample_fixed_spawns,
    spawn_cell_in_envelope,
    spawn_pitch,
)
from loot_bench.tasks.common.success import bore_axis_cos_tilt
from loot_bench.tasks.stack.mdp.commands import HENGE_SPAWN_CELLS, henge_spawn_is_clear
from loot_bench.tasks.stack.mdp.success import henge_levels_complete
from loot_bench.tasks.stack.variants import (
    BLOCK_HALF_EXTENTS,
    BLOCK_LENGTH,
    BLOCK_WIDTH,
    BLOCKS_PER_LEVEL,
    HEIGHT_TOL,
    HENGE_VARIANTS,
    LEVEL_HEIGHT,
    LEVELS,
    TILT_COS_MAX,
    alternative_builds,
)

_VARIANTS = list(HENGE_VARIANTS.values())
_HARDEST = max(_VARIANTS, key=lambda v: v.levels)


def test_catalog_has_one_variant_per_level_count():
    assert sorted(v.levels for v in _VARIANTS) == list(LEVELS)


def test_block_is_a_square_prism_standing_on_its_long_axis():
    hx, hy, hz = BLOCK_HALF_EXTENTS
    assert hx == hy, "footprint must be square (blocks spawn at random yaw)"
    assert hz > hx, "the long axis is local z, so a spawned block stands"
    assert math.isclose(BLOCK_LENGTH, 0.065)
    assert math.isclose(BLOCK_WIDTH, 0.020)
    assert math.isclose(LEVEL_HEIGHT, 0.085)


@pytest.mark.parametrize("variant", _VARIANTS, ids=lambda v: v.name)
def test_three_blocks_per_level_in_colour_tiers(variant):
    assert variant.num_blocks == BLOCKS_PER_LEVEL * variant.levels
    tiers = ("navy", "teal", "white")
    expected = tuple(tiers[i // BLOCKS_PER_LEVEL] for i in range(variant.num_blocks))
    assert variant.block_colors == expected


@pytest.mark.parametrize("variant", _VARIANTS, ids=lambda v: v.name)
def test_blocks_tail_extend_across_levels(variant):
    assert _HARDEST.block_colors[: variant.num_blocks] == variant.block_colors


def test_level_centre_heights():
    assert _HARDEST.level_centre_heights == pytest.approx((0.075, 0.160, 0.245))
    for variant in _VARIANTS:
        assert (
            variant.level_centre_heights
            == _HARDEST.level_centre_heights[: variant.levels]
        )


# --- the geometry/tolerance assertion ------------------------------------- #

_GOOD = (BLOCK_LENGTH, BLOCK_WIDTH, HEIGHT_TOL, TILT_COS_MAX)


@pytest.mark.parametrize("levels", LEVELS)
def test_shipped_constants_admit_no_alternative_build(levels):
    assert alternative_builds(*_GOOD, levels) == []


@pytest.mark.parametrize(
    "length,width,tol,gate,levels,expected",
    [
        # 0.26 is the gate the other categories use, and it is wrong here:
        # two piles of unequal height carry a lintel 2.5 mm under level 1
        # at 6.34 deg, inside the budget from 2Henges up.
        (0.065, 0.020, 0.003, 0.26, 2, [(1, (0, 3), (1, 0), 5)]),
        (
            0.065,
            0.020,
            0.003,
            0.26,
            3,
            [(1, (0, 3), (1, 0), 5), (2, (1, 4), (2, 1), 9)],
        ),
        # Twice the tolerance opens a zero-tilt single pile of three flats.
        (
            0.065,
            0.020,
            0.006,
            0.10,
            2,
            [(1, (0, 3), (0, 3), 4), (2, (1, 4), (1, 4), 6)],
        ),
        # An integer ratio lets two flats replace an upright exactly.
        (0.040, 0.020, 0.003, 0.10, 1, [(1, (0, 2), (0, 2), 3)]),
        # The ceiling on the tolerance: at w/4 a *level* lintel scores on a
        # pile of flats, and no gate value excludes a level lintel.
        (
            0.065,
            0.020,
            0.0055,
            0.01,
            2,
            [(1, (0, 3), (0, 3), 4), (2, (1, 4), (1, 4), 6)],
        ),
    ],
)
def test_bad_constants_are_caught(length, width, tol, gate, levels, expected):
    assert alternative_builds(length, width, tol, gate, levels) == expected


@pytest.mark.parametrize(
    "length,width,tol,gate",
    [
        (0.065, 0.020, 0.003, 0.26),
        (0.065, 0.020, 0.006, 0.10),
        (0.065, 0.020, 0.0055, 0.01),
    ],
)
def test_bad_constants_pass_at_one_level_because_the_budget_does_the_work(
    length, width, tol, gate
):
    # The cheats above all cost four blocks or more and a 1-level budget is
    # three, so a test that only drives one level would never see them.
    assert alternative_builds(length, width, tol, gate, 1) == []


def _quat_about_y(angle: float) -> torch.Tensor:
    return torch.tensor([math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0])


@pytest.mark.parametrize("variant", _VARIANTS, ids=lambda v: v.name)
@pytest.mark.parametrize("uprights", [1, 2], ids=["T", "trilithon"])
def test_intended_builds_pass_the_criterion(variant, uprights):
    # The assertion excludes alternatives; this checks it did not exclude
    # the target: every level of a T and of a trilithon scores.
    hx, _hy, hz = BLOCK_HALF_EXTENTS
    centre_z, quats = [], []
    for level in range(variant.levels):
        base = level * LEVEL_HEIGHT
        for _ in range(uprights):
            centre_z.append(base + hz)
            quats.append(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        centre_z.append(base + BLOCK_LENGTH + hx)
        quats.append(_quat_about_y(math.pi / 2))
    complete = henge_levels_complete(
        torch.tensor([centre_z]),
        torch.stack(quats).unsqueeze(0),
        0.0,
        torch.tensor(variant.level_centre_heights),
        variant.height_tol,
    )
    assert complete.all()
    lintel_cos = bore_axis_cos_tilt(quats[-1], torch.tensor([0.0, 0.0, 1.0]))
    assert float(lintel_cos) <= TILT_COS_MAX


# --- spawn table ----------------------------------------------------------- #

_RADIUS = math.hypot(*BLOCK_HALF_EXTENTS[:2])


def test_spawn_table_holds_the_hardest_block_set():
    assert len(HENGE_SPAWN_CELLS) >= _HARDEST.num_blocks


def test_spawn_separation_clears_the_gripper_pitch_with_margin():
    pitch = spawn_pitch(_RADIUS, _RADIUS)
    assert pitch == pytest.approx(0.088, abs=1e-3)
    assert min_pairwise_separation(torch.tensor(HENGE_SPAWN_CELLS)) >= 0.12 - 1e-6


@pytest.mark.parametrize("cell", HENGE_SPAWN_CELLS, ids=str)
def test_every_cell_inside_the_envelope_and_clear_of_the_build_zone(cell):
    assert spawn_cell_in_envelope(cell[0], cell[1], _RADIUS)
    assert henge_spawn_is_clear(cell[0], cell[1], _RADIUS)


def test_easier_reset_is_prefix_of_harder():
    # Object i takes cell i, so 1Henge's reset is a prefix of 2Henges' and
    # 2Henges' a prefix of 3Henges'.
    cells = torch.tensor(HENGE_SPAWN_CELLS)
    counts = [v.num_blocks for v in sorted(_VARIANTS, key=lambda v: v.levels)]
    for easier, harder in zip(counts, counts[1:], strict=False):
        torch.manual_seed(11)
        xy_easy, yaw_easy = sample_fixed_spawns(cells, 4, easier)
        torch.manual_seed(11)
        xy_hard, yaw_hard = sample_fixed_spawns(cells, 4, harder)
        assert torch.equal(xy_easy, xy_hard[:, :easier])
        assert torch.equal(yaw_easy, yaw_hard[:, :easier])
