"""Grid-based spawn sampling with jittered, distinct cell centers.

Distinct cells guarantee sampled *centers* never coincide -- not that
finite-sized objects can't overlap (jitter + object extent can still exceed
the half-cell pitch). Callers must size jitter/objects accordingly. The same
primitive covers the workspace piece scatter (3x6 grid), the near-band push
scatter (1x4), the parts-tray peg scatter (2x4), the sorter's fixed 60 mm
cells (zero jitter), and the cube stack-base grid — pass explicit
``cells``/``jitter`` for irregular layouts, or use :meth:`from_region`.
"""

from __future__ import annotations

import math

import torch


class ScatterGrid:
    """Sample distinct, jittered xy positions from a fixed set of cells."""

    def __init__(self, cells: torch.Tensor, jitter: torch.Tensor):
        """
        Args:
            cells: (C, 2) cell-centre xy positions.
            jitter: (2,) max |offset| per axis applied around each centre.
        """
        self.cells = cells
        self.jitter = jitter

    @classmethod
    def from_region(
        cls,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        *,
        nx: int,
        ny: int,
        device: str | torch.device,
        jitter_frac: float = 0.4,
    ) -> ScatterGrid:
        """Partition a rectangle into nx*ny cells, jitter a fraction of each
        half-cell so neighbouring samples cannot collide."""
        x0, x1 = x_range
        y0, y1 = y_range
        cx = torch.linspace(x0, x1, nx + 1, device=device)
        cy = torch.linspace(y0, y1, ny + 1, device=device)
        cells = torch.cartesian_prod((cx[:-1] + cx[1:]) / 2, (cy[:-1] + cy[1:]) / 2)
        jitter = jitter_frac * torch.tensor(
            [(x1 - x0) / nx / 2, (y1 - y0) / ny / 2], device=device
        )
        return cls(cells, jitter)

    @property
    def num_cells(self) -> int:
        return self.cells.shape[0]

    def sample(self, num_envs: int, count: int) -> torch.Tensor:
        """Draw ``count`` distinct jittered cells per env, (num_envs, count, 2)."""
        assert count <= self.num_cells, (
            f"cannot place {count} objects on {self.num_cells} cells"
        )
        pick = torch.argsort(
            torch.rand(num_envs, self.num_cells, device=self.cells.device), dim=1
        )[:, :count]
        xy = self.cells[pick]
        return xy + (torch.rand_like(xy) - 0.5) * 2.0 * self.jitter


def random_yaw(num_envs: int, count: int, device: str | torch.device) -> torch.Tensor:
    """Uniform yaw in [0, 2*pi), (num_envs, count)."""
    return torch.rand(num_envs, count, device=device) * 2 * math.pi
