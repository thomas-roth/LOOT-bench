"""Unit tests for ``tasks/common/fixed_spawns.py``.

One shared envelope, one pitch rule, and a sampler that places object ``i``
on cell ``i`` with jitter, in index order. Table validation here is
parameterised over every live category's table (stack, domino, sorter);
Place's envelope draws are covered by ``test_place_spawns.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from loot_bench.tasks.common.fixed_spawns import (
    GRIPPER_CLEARANCE,
    SPAWN_JITTER,
    SPAWN_REACH_RADIUS,
    SPAWN_TABLE_HALF_WIDTH,
    SPAWN_X_MIN,
    min_pairwise_separation,
    sample_fixed_spawns,
    spawn_cell_in_envelope,
    spawn_pitch,
    spawn_pose_in_envelope,
    validate_spawn_table,
)
from loot_bench.tasks.domino.mdp.commands import (
    DOMINO_LANE_EDGE,
    DOMINO_ROW_TILT,
    DOMINO_ROW_X,
    DOMINO_ROW_Y_RANGE,
    DOMINO_SPAWN_CELLS,
    domino_row_slots,
    domino_spawn_is_clear,
)
from loot_bench.tasks.domino.variants import BASE_HALF_EXTENTS, HARDEST_COUNT
from loot_bench.tasks.shape_sorter import assets as sorter_assets
from loot_bench.tasks.shape_sorter.assets import BOARD_POS, load_sorter_layout
from loot_bench.tasks.shape_sorter.mdp import commands as sorter_commands
from loot_bench.tasks.shape_sorter.mdp.commands import (
    SORTER_CORRIDOR_HALF_WIDTH,
    SORTER_SPAWN_CELLS,
    sorter_spawn_is_clear,
)
from loot_bench.tasks.shape_sorter.variants import HARDEST_COUNT as SORTER_HARDEST_COUNT
from loot_bench.tasks.shape_sorter.variants import SHAPE_SORTER_VARIANTS
from loot_bench.tasks.stack.mdp.commands import (
    BUILD_HALO_MARGIN,
    BUILD_ZONE_HALF_WIDTH,
    BUILD_ZONE_X,
    HENGE_SPAWN_CELLS,
    henge_spawn_is_clear,
)
from loot_bench.tasks.stack.variants import BLOCK_HALF_EXTENTS, HENGE_VARIANTS


def _example_tables() -> dict[str, dict]:
    """Synthetic category table specs exercising the validator's shape.

    Toy inputs for the unit checks below; the real per-category tables live
    in :func:`_category_tables` (cells + radius + pitch + max objects +
    work-zone predicate) and run through the same validator.
    """

    def clear_of_centre_strip(x: float, y: float) -> bool:
        # Toy work zone: the centreline corridor |y| < 0.05 is blocked.
        return abs(y) >= 0.05

    return {
        "dummy-pair": {
            "cells": torch.tensor([[0.22, 0.10], [0.22, -0.10]]),
            "object_radius": 0.02,
            "pitch": spawn_pitch(0.02, 0.02),
            "max_objects": 2,
            "is_clear": clear_of_centre_strip,
        },
        "dummy-triple": {
            "cells": torch.tensor([[0.30, 0.10], [0.30, -0.10], [0.40, 0.10]]),
            "object_radius": 0.02,
            "pitch": spawn_pitch(0.02, 0.02),
            "max_objects": 3,
            "is_clear": clear_of_centre_strip,
        },
    }


class TestSpawnPitch:
    def test_formula(self):
        assert spawn_pitch(0.02, 0.03) == pytest.approx(
            2 * SPAWN_JITTER + 0.02 + 0.03 + GRIPPER_CLEARANCE
        )

    def test_stack_pitch_matches_prd_band(self):
        r = math.hypot(0.02, 0.02)
        assert spawn_pitch(r, r) == pytest.approx(0.117, abs=1e-3)

    def test_domino_pitch_matches_prd_band(self):
        r = math.hypot(0.0075, 0.015)
        assert spawn_pitch(r, r) == pytest.approx(0.094, abs=1e-3)

    def test_symmetric_in_radii(self):
        assert spawn_pitch(0.01, 0.03) == pytest.approx(spawn_pitch(0.03, 0.01))

    def test_jitter_param_shifts_by_two(self):
        assert spawn_pitch(0.02, 0.02, jitter=0.0) == pytest.approx(
            spawn_pitch(0.02, 0.02) - 2 * SPAWN_JITTER
        )


class TestEnvelope:
    def test_centreline_far_reach_passes(self):
        assert spawn_pose_in_envelope(0.45, 0.0, 0.02)

    def test_beyond_reach_fails(self):
        assert not spawn_pose_in_envelope(0.50, 0.0, 0.02)

    def test_near_bound(self):
        assert spawn_pose_in_envelope(SPAWN_X_MIN, 0.0, 0.02)
        assert not spawn_pose_in_envelope(SPAWN_X_MIN - 1e-3, 0.0, 0.02)

    def test_half_width_boundary(self):
        r = 0.02
        assert spawn_pose_in_envelope(0.25, SPAWN_TABLE_HALF_WIDTH - r, r)
        assert not spawn_pose_in_envelope(0.25, SPAWN_TABLE_HALF_WIDTH - r + 1e-3, r)

    def test_radial_bound_uses_hypot(self):
        # Far corner of the old rectangle band: hypot matters, not x alone.
        assert not spawn_pose_in_envelope(0.45, 0.25, 0.02)
        assert pytest.approx(SPAWN_REACH_RADIUS) == 0.49

    def test_cell_checks_all_four_jitter_corners(self):
        # x = 0.21 is the closest a centre may sit: corner at 0.20 passes.
        assert spawn_cell_in_envelope(0.21, 0.0, 0.02, jitter=0.01)
        assert not spawn_cell_in_envelope(0.205, 0.0, 0.02, jitter=0.01)

    def test_sorter_edge_cell_passes_but_next_step_fails(self):
        r = 0.029445
        assert spawn_cell_in_envelope(0.21, 0.31, r, jitter=0.01)
        assert not spawn_cell_in_envelope(0.21, 0.32, r, jitter=0.01)


class TestSampleFixedSpawns:
    def _cells(self) -> torch.Tensor:
        return torch.tensor([[0.22, 0.10], [0.22, -0.10], [0.35, 0.18]])

    def test_shapes_and_yaw_range(self):
        cells = self._cells()
        xy, yaw = sample_fixed_spawns(cells, 8, 3)
        assert xy.shape == (8, 3, 2)
        assert yaw.shape == (8, 3)
        assert bool((yaw >= 0).all()) and bool((yaw < 2 * math.pi).all())

    def test_object_i_takes_cell_i_within_jitter(self):
        cells = self._cells()
        xy, _ = sample_fixed_spawns(cells, 64, 3, jitter=0.01)
        delta = (xy - cells.unsqueeze(0)).abs()
        assert bool((delta <= 0.01 + 1e-6).all())

    def test_zero_jitter_pins_cells(self):
        cells = self._cells()
        xy, _ = sample_fixed_spawns(cells, 4, 3, jitter=0.0)
        assert torch.equal(xy, cells.unsqueeze(0).expand(4, -1, -1))

    def test_deterministic_under_seed(self):
        cells = self._cells()
        torch.manual_seed(3)
        a = sample_fixed_spawns(cells, 8, 3)
        torch.manual_seed(3)
        b = sample_fixed_spawns(cells, 8, 3)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    def test_seed_prefix_property(self):
        # An easier task's draws are a prefix of a harder task's under the
        # same seed: object i draws once, in index order.
        cells = self._cells()
        torch.manual_seed(7)
        xy2, yaw2 = sample_fixed_spawns(cells, 16, 2)
        torch.manual_seed(7)
        xy3, yaw3 = sample_fixed_spawns(cells, 16, 3)
        assert torch.equal(xy2, xy3[:, :2])
        assert torch.equal(yaw2, yaw3[:, :2])

    def test_more_objects_than_cells_raises(self):
        cells = torch.zeros(2, 2)
        with pytest.raises(AssertionError):
            sample_fixed_spawns(cells, 1, 3)


class TestValidateSpawnTable:
    def test_min_pairwise_separation(self):
        cells = torch.tensor([[0.0, 0.0], [0.12, 0.0], [0.0, 0.12]])
        assert min_pairwise_separation(cells) == pytest.approx(0.12)

    def test_passing_table_reports_no_errors(self):
        spec = _example_tables()["dummy-pair"]
        assert (
            validate_spawn_table(
                spec["cells"],
                object_radius=spec["object_radius"],
                pitch=spec["pitch"],
                max_objects=spec["max_objects"],
                is_clear=spec["is_clear"],
            )
            == []
        )

    def test_too_short_table_reports(self):
        spec = _example_tables()["dummy-pair"]
        errors = validate_spawn_table(
            spec["cells"][:1],
            object_radius=spec["object_radius"],
            pitch=spec["pitch"],
            max_objects=spec["max_objects"],
            is_clear=spec["is_clear"],
        )
        assert any("at least" in e for e in errors)

    def test_too_close_cells_report(self):
        cells = torch.tensor([[0.30, 0.10], [0.30, 0.11]])
        errors = validate_spawn_table(
            cells,
            object_radius=0.02,
            pitch=spawn_pitch(0.02, 0.02),
            max_objects=2,
        )
        assert any("separation" in e for e in errors)

    def test_cell_outside_envelope_reports(self):
        cells = torch.tensor([[0.10, 0.0], [0.30, 0.10]])
        errors = validate_spawn_table(
            cells, object_radius=0.02, pitch=0.05, max_objects=2
        )
        assert any("envelope" in e for e in errors)

    def test_cell_in_work_zone_reports(self):
        spec = _example_tables()["dummy-pair"]
        blocked = torch.tensor([[0.30, 0.0], [0.30, 0.10]])
        errors = validate_spawn_table(
            blocked,
            object_radius=spec["object_radius"],
            pitch=0.05,
            max_objects=2,
            is_clear=spec["is_clear"],
        )
        assert any("work zone" in e for e in errors)

    @pytest.mark.parametrize(
        "spec", _example_tables().values(), ids=list(_example_tables())
    )
    def test_example_tables_pass(self, spec: dict):
        # The pattern the category tables reuse: supply the table and the
        # checks (separation vs pitch, envelope at all jitter corners,
        # work-zone clearance, table length) run unchanged.
        assert (
            validate_spawn_table(
                spec["cells"],
                object_radius=spec["object_radius"],
                pitch=spec["pitch"],
                max_objects=spec["max_objects"],
                is_clear=spec["is_clear"],
            )
            == []
        )


def _category_tables() -> dict[str, dict]:
    """Real per-category spawn tables, same spec shape as :func:`_example_tables`.

    Stack's blocks spawn standing, so the footprint is the prism's square
    cross-section.
    """
    stack_radius = math.hypot(*BLOCK_HALF_EXTENTS[:2])
    domino_radius = math.hypot(*BASE_HALF_EXTENTS[:2])
    layout = load_sorter_layout()
    sorter_radius = max(
        layout.pieces[shape].cross_bounding_radius
        for shape in ("circle", "rect", "square", "hexagon", "triangle")
    )
    return {
        "stack": {
            "cells": torch.tensor(HENGE_SPAWN_CELLS),
            "object_radius": stack_radius,
            "pitch": spawn_pitch(stack_radius, stack_radius),
            "max_objects": max(v.num_blocks for v in HENGE_VARIANTS.values()),
            "is_clear": lambda x, y: henge_spawn_is_clear(x, y, stack_radius),
        },
        "domino": {
            "cells": torch.tensor(DOMINO_SPAWN_CELLS),
            "object_radius": domino_radius,
            "pitch": spawn_pitch(domino_radius, domino_radius),
            "max_objects": HARDEST_COUNT,
            "is_clear": lambda x, y: domino_spawn_is_clear(x, y, domino_radius),
        },
        "sorter": {
            "cells": torch.tensor(SORTER_SPAWN_CELLS),
            "object_radius": sorter_radius,
            "pitch": spawn_pitch(sorter_radius, sorter_radius),
            "max_objects": SORTER_HARDEST_COUNT,
            "is_clear": lambda x, y: sorter_spawn_is_clear(x, y, sorter_radius),
        },
    }


class TestCategoryTables:
    @pytest.mark.parametrize(
        "spec", _category_tables().values(), ids=list(_category_tables())
    )
    def test_category_tables_pass_shared_validation(self, spec: dict):
        # Every live category supplies its table in the same spec shape, and
        # the checks (separation vs pitch, envelope at all jitter corners,
        # work-zone clearance, table length) run unchanged over each.
        assert (
            validate_spawn_table(
                spec["cells"],
                object_radius=spec["object_radius"],
                pitch=spec["pitch"],
                max_objects=spec["max_objects"],
                is_clear=spec["is_clear"],
            )
            == []
        )

    def test_stack_table_holds_nine_cells_for_nine_blocks(self):
        spec = _category_tables()["stack"]
        assert spec["cells"].shape == (9, 2)
        assert spec["max_objects"] == 9

    def test_stack_build_zone_predicate(self):
        # Build zone x in [0.24, 0.32], |y| <= 0.07, plus a halo of object
        # radius + jitter + margin, is blocked; everything else is clear.
        spec = _category_tables()["stack"]
        radius = spec["object_radius"]
        halo = radius + SPAWN_JITTER + BUILD_HALO_MARGIN
        assert BUILD_ZONE_X == (0.24, 0.32)
        assert BUILD_ZONE_HALF_WIDTH == 0.07
        assert not henge_spawn_is_clear(0.28, 0.0, radius)
        for cell in spec["cells"].tolist():
            assert henge_spawn_is_clear(cell[0], cell[1], radius)
        x0, x1 = BUILD_ZONE_X
        assert not henge_spawn_is_clear(x1 + halo - 1e-3, 0.0, radius)
        assert henge_spawn_is_clear(x1 + halo + 1e-3, 0.0, radius)
        assert not henge_spawn_is_clear(x0 - halo + 1e-3, 0.0, radius)
        assert not henge_spawn_is_clear(
            0.28, BUILD_ZONE_HALF_WIDTH + halo - 1e-3, radius
        )
        assert henge_spawn_is_clear(0.28, BUILD_ZONE_HALF_WIDTH + halo + 1e-3, radius)


class TestDominoRowLane:
    def _lane_extremes(self) -> tuple[float, float]:
        """Min row-slot x over every permitted row y, tilt, branch and count,
        and max spawn x over every cell jitter corner."""
        cells = torch.tensor(DOMINO_SPAWN_CELLS)
        max_spawn_x = float(cells[:, 0].max()) + SPAWN_JITTER
        ys = torch.linspace(-DOMINO_ROW_Y_RANGE, DOMINO_ROW_Y_RANGE, 9)
        tilts = torch.linspace(-DOMINO_ROW_TILT, DOMINO_ROW_TILT, 9)
        grid_y, grid_tilt = torch.meshgrid(ys, tilts, indexing="ij")
        centre = torch.stack([torch.full_like(grid_y, DOMINO_ROW_X), grid_y], dim=-1)
        min_slot_x = math.inf
        for sign in (1.0, -1.0):
            yaw = sign * math.pi / 2 + grid_tilt
            for count in range(2, HARDEST_COUNT + 1):
                slots = domino_row_slots(centre, yaw, count)
                min_slot_x = min(min_slot_x, float(slots[..., 0].min()))
        return min_slot_x, max_spawn_x

    def test_lane_free_of_spawns_for_any_row_y_and_tilt(self):
        # Spawns and row separate in x by construction: no keep-out mask, no
        # rejection loop, no sampled fallback.
        domino_radius = math.hypot(*BASE_HALF_EXTENTS[:2])
        min_slot_x, max_spawn_x = self._lane_extremes()
        assert min_slot_x - max_spawn_x >= 2 * domino_radius

    def test_lane_edge_matches_geometry(self):
        # The 0.340 near edge is the worst-case slot x minus the domino
        # footprint; every jittered spawn (plus footprint) stays left of it.
        domino_radius = math.hypot(*BASE_HALF_EXTENTS[:2])
        min_slot_x, max_spawn_x = self._lane_extremes()
        assert min_slot_x - domino_radius >= DOMINO_LANE_EDGE - 1e-6
        assert max_spawn_x + domino_radius <= DOMINO_LANE_EDGE + 1e-6

    def test_hardest_row_endpoints_inside_envelope(self):
        # The sampled row-centre y range is sized so the longest row stays on
        # the table and inside reach at any permitted tilt.
        domino_radius = math.hypot(*BASE_HALF_EXTENTS[:2])
        ys = torch.linspace(-DOMINO_ROW_Y_RANGE, DOMINO_ROW_Y_RANGE, 9)
        tilts = torch.linspace(-DOMINO_ROW_TILT, DOMINO_ROW_TILT, 9)
        grid_y, grid_tilt = torch.meshgrid(ys, tilts, indexing="ij")
        centre = torch.stack([torch.full_like(grid_y, DOMINO_ROW_X), grid_y], dim=-1)
        for sign in (1.0, -1.0):
            slots = domino_row_slots(
                centre, sign * math.pi / 2 + grid_tilt, HARDEST_COUNT
            )
            for x, y in slots.reshape(-1, 2).tolist():
                assert spawn_pose_in_envelope(x, y, domino_radius)


def _sorter_radius() -> float:
    layout = load_sorter_layout()
    return max(
        layout.pieces[shape].cross_bounding_radius
        for shape in ("circle", "rect", "square", "hexagon", "triangle")
    )


class TestSorterTable:
    def test_table_holds_five_cells_for_five_pieces(self):
        cells = torch.tensor(SORTER_SPAWN_CELLS)
        assert cells.shape == (SORTER_HARDEST_COUNT, 2)
        assert (
            max(v.num_pieces for v in SHAPE_SORTER_VARIANTS.values())
            == SORTER_HARDEST_COUNT
        )

    def test_min_separation_matches_prd(self):
        cells = torch.tensor(SORTER_SPAWN_CELLS)
        assert min_pairwise_separation(cells) == pytest.approx(0.120)

    def test_corridor_half_width_matches_prd(self):
        assert pytest.approx(0.03) == SORTER_CORRIDOR_HALF_WIDTH

    def test_centred_cell_would_block_the_corridor(self):
        # This is why the table has no y = 0 cell: the rect hole's
        # centreline approach runs through it.
        assert not sorter_spawn_is_clear(0.21, 0.0, _sorter_radius())

    def test_corridor_clear_at_every_jitter_corner(self):
        cells = torch.tensor(SORTER_SPAWN_CELLS)
        radius = _sorter_radius()
        for i in range(cells.shape[0]):
            cx, cy = float(cells[i, 0]), float(cells[i, 1])
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    assert sorter_spawn_is_clear(
                        cx + sx * SPAWN_JITTER, cy + sy * SPAWN_JITTER, radius
                    ), f"cell {i} blocks the corridor at a jitter corner"

    def test_cube_pinned_with_shared_jitter(self):
        # The cube sits at x = 0.375 with the shared +/-1 cm jitter, the
        # same number as every other category.
        assert tuple(BOARD_POS[:2]) == pytest.approx((0.375, 0.0))
        assert not hasattr(sorter_assets, "SORTER_SPAWN_JITTER")

    def test_random_positions_variation_is_gone(self):
        assert "ShapeSorter3RandomPositions" not in SHAPE_SORTER_VARIANTS
        assert "ShapeSorter4RandomPositions" not in SHAPE_SORTER_VARIANTS
        assert "ShapeSorter5RandomPositions" not in SHAPE_SORTER_VARIANTS
        assert all(
            not getattr(v, "wide_positions", False)
            for v in SHAPE_SORTER_VARIANTS.values()
        )
        assert not hasattr(sorter_commands, "_WIDE_CELLS")
