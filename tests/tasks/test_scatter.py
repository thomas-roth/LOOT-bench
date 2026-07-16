"""Unit tests for ``tasks/common/scatter.py``."""

import math

import pytest
import torch

from g3a_bench.tasks.common.scatter import ScatterGrid, random_yaw


class TestScatterGrid:
    def test_from_region_cell_centers_and_jitter(self):
        grid = ScatterGrid.from_region(
            (0.0, 0.3), (-0.3, 0.3), nx=3, ny=6, device="cpu"
        )
        assert grid.num_cells == 18
        # Cell centers sit at the midpoints of a 3x6 partition.
        assert grid.cells[0].tolist() == pytest.approx([0.05, -0.25])
        assert grid.cells[-1].tolist() == pytest.approx([0.25, 0.25])
        # Jitter is 0.4 of the half-cell size per axis.
        assert grid.jitter.tolist() == pytest.approx([0.02, 0.02])

    def test_sample_draws_distinct_cells(self):
        grid = ScatterGrid.from_region(
            (0.0, 0.3), (-0.3, 0.3), nx=3, ny=6, device="cpu"
        )
        xy = grid.sample(64, 10)
        assert xy.shape == (64, 10, 2)
        # Recover each sample's cell by nearest center; cells must be distinct
        # within an env (that is the collision-free-by-construction property).
        d = torch.cdist(xy.reshape(-1, 2), grid.cells).argmin(dim=-1).reshape(64, 10)
        for env_cells in d:
            assert len(set(env_cells.tolist())) == 10

    def test_sample_stays_within_jitter_of_a_center(self):
        grid = ScatterGrid.from_region(
            (0.0, 0.3), (-0.3, 0.3), nx=3, ny=6, device="cpu"
        )
        xy = grid.sample(64, 18)
        delta = (xy.unsqueeze(2) - grid.cells.view(1, 1, -1, 2)).abs()
        nearest = delta.max(dim=-1).values.min(dim=-1).values  # linf to nearest
        assert bool((nearest <= grid.jitter.max() + 1e-6).all())

    def test_zero_jitter_returns_exact_cells(self):
        cells = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        grid = ScatterGrid(cells=cells, jitter=torch.zeros(2))
        xy = grid.sample(16, 3)
        for env_xy in xy:
            got = {tuple(round(v, 6) for v in p.tolist()) for p in env_xy}
            want = {tuple(round(v, 6) for v in p.tolist()) for p in cells}
            assert got == want

    def test_sample_more_than_cells_raises(self):
        grid = ScatterGrid(cells=torch.zeros(2, 2), jitter=torch.zeros(2))
        with pytest.raises(AssertionError):
            grid.sample(1, 3)


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
        # task's (5 objects) under the same seed — the demo-cutoff rule.
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


def test_random_yaw_range_and_shape():
    yaw = random_yaw(32, 5, device="cpu")
    assert yaw.shape == (32, 5)
    assert bool((yaw >= 0).all()) and bool((yaw < 2 * math.pi).all())
