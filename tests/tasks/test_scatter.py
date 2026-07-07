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


def test_random_yaw_range_and_shape():
    yaw = random_yaw(32, 5, device="cpu")
    assert yaw.shape == (32, 5)
    assert bool((yaw >= 0).all()) and bool((yaw < 2 * math.pi).all())
