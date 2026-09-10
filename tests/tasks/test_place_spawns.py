"""Unit tests for Place's envelope spawns (ticket 04).

Place is exempt from the fixed spawn tables: when a variant axis says
random, the cube or target is drawn uniformly over the shared envelope
with a fixed, reproducible number of draws per object in index order (no
rejection loop, so the seed-prefix convention holds); when it says fixed,
the pose is pinned. A single deterministic push-apart keeps the
cube-target separation at or above ``PLACE_MIN_SEPARATION``.
"""

import math

import pytest
import torch

from loot_bench.tasks.common.fixed_spawns import (
    SPAWN_JITTER,
    SPAWN_REACH_RADIUS,
    SPAWN_TABLE_HALF_WIDTH,
    SPAWN_X_MIN,
    spawn_pose_in_envelope,
)
from loot_bench.tasks.place.mdp.commands import (
    FIXED_SPAWN_XY,
    FIXED_TARGET_XY,
    MAX_TRAVEL_DIST,
    PLACE_MIN_SEPARATION,
    PLACE_OBJECT_RADIUS,
    PLACE_X_MAX,
    draw_place_poses,
    enforce_place_separation,
    place_half_width,
    place_xy_from_draws,
)


class TestPlacePins:
    def test_fixed_spawn_pin(self):
        assert FIXED_SPAWN_XY == (0.25, 0.0)

    def test_fixed_target_pin(self):
        assert FIXED_TARGET_XY == (0.40, 0.0)

    def test_pins_hold_the_minimum_separation(self):
        dist = math.hypot(
            FIXED_TARGET_XY[0] - FIXED_SPAWN_XY[0],
            FIXED_TARGET_XY[1] - FIXED_SPAWN_XY[1],
        )
        assert dist >= PLACE_MIN_SEPARATION
        assert pytest.approx(0.10) == PLACE_MIN_SEPARATION

    def test_fixed_reset_is_exactly_the_pins(self):
        torch.manual_seed(0)
        cube_xy, cube_yaw, target_xy = draw_place_poses(
            8, random_spawn=False, random_target=False
        )
        assert torch.equal(cube_xy, torch.tensor(FIXED_SPAWN_XY).expand(8, 2))
        assert torch.equal(cube_yaw, torch.zeros(8))
        assert torch.equal(target_xy, torch.tensor(FIXED_TARGET_XY).expand(8, 2))


class TestPlaceEnvelopeMapping:
    def test_half_width_formula(self):
        r = PLACE_OBJECT_RADIUS
        x = torch.tensor([0.45, 0.20])
        yw = place_half_width(x, r)
        assert yw[0].item() == pytest.approx(math.sqrt(0.49**2 - 0.45**2))
        assert yw[1].item() == pytest.approx(SPAWN_TABLE_HALF_WIDTH - r - SPAWN_JITTER)

    def test_draw_corners_map_to_envelope_corners(self):
        r = PLACE_OBJECT_RADIUS
        y_max = SPAWN_TABLE_HALF_WIDTH - r - SPAWN_JITTER
        xy = place_xy_from_draws(torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]), r)
        assert xy[0, 0].item() == pytest.approx(SPAWN_X_MIN)
        assert xy[0, 1].item() == pytest.approx(-y_max)
        assert xy[1, 0].item() == pytest.approx(PLACE_X_MAX)
        assert xy[2, 0].item() == pytest.approx(0.5 * (SPAWN_X_MIN + PLACE_X_MAX))
        assert xy[2, 1].item() == pytest.approx(0.0)

    def test_random_draws_stay_inside_the_envelope(self):
        torch.manual_seed(11)
        for _ in range(5):
            xy = place_xy_from_draws(torch.rand(512, 2), PLACE_OBJECT_RADIUS)
            for x, y in xy.tolist():
                assert spawn_pose_in_envelope(x, y, PLACE_OBJECT_RADIUS)

    def test_reach_radius_matches_shared_envelope(self):
        assert pytest.approx(0.49) == SPAWN_REACH_RADIUS


class TestPlaceSeparation:
    def test_coincident_poses_push_to_exactly_the_minimum(self):
        cube = torch.tensor([[0.30, 0.05]])
        target = torch.tensor([[0.30, 0.05]])
        cube_out, target_out = enforce_place_separation(cube, target)
        dist = float((target_out - cube_out).norm())
        assert dist == pytest.approx(PLACE_MIN_SEPARATION)

    def test_close_poses_push_apart_symmetrically(self):
        cube = torch.tensor([[0.30, 0.0]])
        target = torch.tensor([[0.34, 0.0]])
        cube_out, target_out = enforce_place_separation(cube, target)
        # The midpoint is preserved: each side moves half the shortfall.
        mid_before = (cube + target) / 2.0
        mid_after = (cube_out + target_out) / 2.0
        assert torch.allclose(mid_after, mid_before)
        assert float((target_out - cube_out).norm()) == pytest.approx(
            PLACE_MIN_SEPARATION
        )

    def test_pinned_side_never_moves(self):
        # A Fixed axis stays exactly pinned: the random side takes the
        # whole shortfall.
        cube = torch.tensor([[0.36, 0.0]])
        target = torch.tensor([[0.40, 0.0]])
        cube_out, target_out = enforce_place_separation(
            cube, target, move_cube=True, move_target=False
        )
        assert torch.equal(target_out, target)
        assert float((target_out - cube_out).norm()) == pytest.approx(
            PLACE_MIN_SEPARATION
        )
        cube_out, target_out = enforce_place_separation(
            cube, target, move_cube=False, move_target=True
        )
        assert torch.equal(cube_out, cube)
        assert float((target_out - cube_out).norm()) == pytest.approx(
            PLACE_MIN_SEPARATION
        )

    def test_distant_poses_are_untouched(self):
        cube = torch.tensor([[0.25, 0.0]])
        target = torch.tensor([[0.40, 0.0]])
        cube_out, target_out = enforce_place_separation(cube, target)
        assert torch.equal(cube_out, cube)
        assert torch.equal(target_out, target)

    def test_fuzzed_resets_never_start_solved(self):
        for seed in range(5):
            torch.manual_seed(seed)
            cube_xy, _, target_xy = draw_place_poses(
                512, random_spawn=True, random_target=True
            )
            dist = (target_xy - cube_xy).norm(dim=-1)
            assert bool((dist >= PLACE_MIN_SEPARATION - 1e-6).all())

    def test_pinned_axis_survives_any_draw(self):
        # The push-apart never moves a pinned axis, whatever the draws are.
        for seed in range(5):
            torch.manual_seed(seed)
            _, _, target_xy = draw_place_poses(
                512, random_spawn=True, random_target=False
            )
            assert torch.equal(target_xy, torch.tensor(FIXED_TARGET_XY).expand(512, 2))
            torch.manual_seed(seed)
            cube_xy, _, _ = draw_place_poses(
                512, random_spawn=False, random_target=True
            )
            assert torch.equal(cube_xy, torch.tensor(FIXED_SPAWN_XY).expand(512, 2))


class TestPlaceDrawCounts:
    def test_deterministic_under_seed(self):
        torch.manual_seed(3)
        first = draw_place_poses(8, random_spawn=True, random_target=True)
        torch.manual_seed(3)
        second = draw_place_poses(8, random_spawn=True, random_target=True)
        for a, b in zip(first, second, strict=True):
            assert torch.equal(a, b)

    def test_random_cube_matches_across_target_axes(self):
        # The cube draws first, in index order, whether or not the target
        # axis is random: RandS-RandT and RandS-FixedT stage the same cube
        # draws. (With the push-apart on, the final poses couple by design,
        # so this checks the draws with the push disabled.)
        torch.manual_seed(7)
        cube_both, _, _ = draw_place_poses(
            16, random_spawn=True, random_target=True, min_separation=0.0
        )
        torch.manual_seed(7)
        cube_pinned_target, _, _ = draw_place_poses(
            16, random_spawn=True, random_target=False, min_separation=0.0
        )
        assert torch.equal(cube_both, cube_pinned_target)

    def test_random_target_matches_across_spawn_axes(self):
        # The target draws after the cube's fixed draw count even when the
        # spawn is pinned: FixedS-RandT and RandS-RandT sample the same
        # target draws (push disabled, as above).
        torch.manual_seed(7)
        _, _, target_both = draw_place_poses(
            16, random_spawn=True, random_target=True, min_separation=0.0
        )
        torch.manual_seed(7)
        _, _, target_pinned_spawn = draw_place_poses(
            16, random_spawn=False, random_target=True, min_separation=0.0
        )
        assert torch.equal(target_both, target_pinned_spawn)

    def test_fixed_draw_counts_whatever_the_variant(self):
        # No rejection loop: cube always draws rand(n, 3), target always
        # rand(n, 2), in that order — even for pinned axes.
        real_rand = torch.rand
        calls: list[tuple[int, ...]] = []

        def spy(*shape: int, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(tuple(shape))
            return real_rand(*shape, **kwargs)

        torch.rand = spy  # type: ignore[method-assign]
        try:
            for random_spawn in (True, False):
                for random_target in (True, False):
                    calls.clear()
                    draw_place_poses(
                        4,
                        random_spawn=random_spawn,
                        random_target=random_target,
                    )
                    assert calls == [(4, 3), (4, 2)]
        finally:
            torch.rand = real_rand  # type: ignore[method-assign]

    def test_max_travel_covers_the_envelope(self):
        # The progress normalizer must swallow any post-push-apart pair.
        torch.manual_seed(21)
        worst = 0.0
        for _ in range(10):
            cube_xy, _, target_xy = draw_place_poses(
                512, random_spawn=True, random_target=True
            )
            worst = max(worst, float((target_xy - cube_xy).norm(dim=-1).max()))
        assert worst <= MAX_TRAVEL_DIST
