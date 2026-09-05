"""Unit tests for ``tasks/common/scatter.py``."""

import math

import pytest
import torch

from g3a_bench.tasks.common.scatter import ScatterGrid


class TestSamplePoses:
    """The sequential Fisher-Yates sampler (seed-prefix safe)."""

    def _grid(self) -> ScatterGrid:
        return ScatterGrid.tiling(
            (0.22, 0.42), (-0.18, 0.18), object_radius=0.028, device="cpu"
        )

    def test_tiling_pitch_and_jitter_are_collision_free(self):
        grid = self._grid()
        # Pitch per axis >= one object diameter + gap; jitter fills the rest.
        assert grid.num_cells >= 2
        xs = torch.unique(grid.cells[:, 0])
        ys = torch.unique(grid.cells[:, 1])
        min_pitch = 2 * 0.028 + 0.004
        if len(xs) > 1:
            assert float(xs[1] - xs[0]) >= min_pitch - 1e-6
        if len(ys) > 1:
            assert float(ys[1] - ys[0]) >= min_pitch - 1e-6
        assert bool((grid.jitter >= 0).all())

    def test_distinct_cells_and_bounds(self):
        grid = self._grid()
        count = min(grid.num_cells, 8)
        xy, yaw = grid.sample_poses(64, count)
        assert xy.shape == (64, count, 2) and yaw.shape == (64, count)
        assert bool((yaw >= 0).all()) and bool((yaw < 2 * math.pi).all())
        # Each sample within jitter (linf) of its nearest cell centre, and
        # cells are distinct within an env.
        cell_idx = (
            torch.cdist(xy.reshape(-1, 2), grid.cells).argmin(-1).reshape(64, count)
        )
        delta = (xy - grid.cells[cell_idx]).abs()
        assert bool((delta <= grid.jitter + 1e-6).all())
        for env_cells in cell_idx:
            assert len(set(env_cells.tolist())) == count

    def test_deterministic_under_seed(self):
        grid = self._grid()
        torch.manual_seed(3)
        a = grid.sample_poses(8, 4)
        torch.manual_seed(3)
        b = grid.sample_poses(8, 4)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    def test_seed_prefix_property(self):
        # The easier task's draws (3 objects) are a prefix of the harder
        # task's (5 objects) under the same seed. The demo-cutoff rule.
        grid = self._grid()
        torch.manual_seed(7)
        xy3, yaw3 = grid.sample_poses(16, 3)
        torch.manual_seed(7)
        xy5, yaw5 = grid.sample_poses(16, 5)
        assert torch.equal(xy3, xy5[:, :3])
        assert torch.equal(yaw3, yaw5[:, :3])

    def test_more_objects_than_cells_raises(self):
        grid = ScatterGrid(cells=torch.zeros(2, 2), jitter=torch.zeros(2))
        with pytest.raises(AssertionError):
            grid.sample_poses(1, 3)


class TestBlockedNear:
    """The goal-position keep-out mask (``blocked_near`` + ``blocked``)."""

    def _grid(self) -> ScatterGrid:
        return ScatterGrid.tiling(
            (0.22, 0.42), (-0.18, 0.18), object_radius=0.022, device="cpu"
        )

    def _points(self, num_envs: int) -> torch.Tensor:
        # A 4-slot row along +y, one row per env at a different x.
        x = torch.linspace(0.26, 0.38, num_envs).view(-1, 1)
        y = torch.linspace(-0.08, 0.07, 4).view(1, -1)
        return torch.stack([x.expand(-1, 4), y.expand(num_envs, -1)], dim=-1)

    def test_samples_keep_min_distance_from_points(self):
        grid, num_envs, count, min_dist = self._grid(), 128, 4, 0.06
        points = self._points(num_envs)
        blocked = grid.blocked_near(points, min_dist, keep_free=count)
        xy, _ = grid.sample_poses(num_envs, count, blocked=blocked)
        dist = (xy[:, :, None, :] - points[:, None, :, :]).norm(dim=-1)
        assert float(dist.min()) >= min_dist

    def test_jitter_cannot_breach_the_keep_out(self):
        # Cells are blocked by centre distance *plus* the worst-case jitter.
        grid, min_dist = self._grid(), 0.06
        points = self._points(16)
        blocked = grid.blocked_near(points, min_dist, keep_free=0)
        centre_dist = (
            (grid.cells[None, :, None, :] - points[:, None, :, :]).norm(dim=-1).amin(-1)
        )
        allowed = centre_dist[~blocked]
        assert float(allowed.min()) >= min_dist + float(grid.jitter.norm())

    def test_keep_free_survives_an_all_covering_keep_out(self):
        # A keep-out over the whole region still leaves keep_free cells — the
        # farthest ones — so sampling degrades instead of failing.
        grid, count = self._grid(), 4
        points = grid.cells.mean(0).view(1, 1, 2).expand(8, 1, -1)
        blocked = grid.blocked_near(points, 10.0, keep_free=count)
        assert int((~blocked).sum(dim=1).min()) == count
        grid.sample_poses(8, count, blocked=blocked)  # does not raise

    def test_unblocked_mask_matches_no_mask(self):
        # An empty keep-out must not shift the draw-to-pose mapping.
        grid = self._grid()
        torch.manual_seed(5)
        xy_a, yaw_a = grid.sample_poses(16, 4)
        torch.manual_seed(5)
        xy_b, yaw_b = grid.sample_poses(
            16, 4, blocked=torch.zeros(16, grid.num_cells, dtype=torch.bool)
        )
        assert torch.equal(xy_a, xy_b) and torch.equal(yaw_a, yaw_b)

    def test_seed_prefix_property_under_a_shared_mask(self):
        # Same keep-out (domino blocks against the hardest row regardless of
        # n) -> the shorter draw stays a prefix of the longer one.
        grid = self._grid()
        blocked = grid.blocked_near(self._points(16), 0.06, keep_free=4)
        torch.manual_seed(7)
        xy2, yaw2 = grid.sample_poses(16, 2, blocked=blocked)
        torch.manual_seed(7)
        xy4, yaw4 = grid.sample_poses(16, 4, blocked=blocked)
        assert torch.equal(xy2, xy4[:, :2]) and torch.equal(yaw2, yaw4[:, :2])

    def test_too_few_unblocked_cells_raises(self):
        grid = self._grid()
        blocked = torch.ones(4, grid.num_cells, dtype=torch.bool)
        blocked[:, 0] = False
        with pytest.raises(AssertionError):
            grid.sample_poses(4, 2, blocked=blocked)
