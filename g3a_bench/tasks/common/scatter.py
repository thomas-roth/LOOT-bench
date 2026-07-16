"""Grid-based spawn sampling with jittered, distinct cell centers.

:meth:`ScatterGrid.tiling` sizes the grid so cells are at least one object
diameter apart and jitter never closes that gap — samples are collision-free
by construction. :meth:`sample_poses` draws them with a *sequential*
Fisher-Yates shuffle: object ``i`` consumes exactly one ``rand(E, 4)`` draw
(cell pick among the cells left by objects ``0..i-1``, jitter x/y, yaw), in
index order. Fixed draws per object, independent of the total object count,
so a task's resets are a seed-prefix of a harder task's (see
``docs/TaskCategories.md`` "General": demos for easier tasks are cut from
the hardest task's recordings). Pass explicit ``cells``/``jitter`` for
irregular layouts.
"""

from __future__ import annotations

import math

import torch

# Minimum clearance between two objects in neighboring cells (m).
_GAP = 0.004


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
    def tiling(
        cls,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        *,
        object_radius: float,
        device: str | torch.device,
    ) -> ScatterGrid:
        """Finest grid whose pitch fits one object per cell without overlap.

        Pitch is at least ``2 * object_radius + _GAP`` per axis; jitter fills
        the remaining half-cell (``half_pitch - object_radius - _GAP/2``), so
        cells nearly tile the region while neighbors can never collide.

        Args:
            object_radius: bounding-circle radius of the largest object.
        """
        x0, x1 = x_range
        y0, y1 = y_range
        min_pitch = 2.0 * object_radius + _GAP
        nx = max(1, int((x1 - x0) / min_pitch))
        ny = max(1, int((y1 - y0) / min_pitch))
        cx = torch.linspace(x0, x1, nx + 1, device=device)
        cy = torch.linspace(y0, y1, ny + 1, device=device)
        cells = torch.cartesian_prod((cx[:-1] + cx[1:]) / 2, (cy[:-1] + cy[1:]) / 2)
        jitter = torch.tensor(
            [
                max(0.0, (x1 - x0) / nx / 2 - object_radius - _GAP / 2),
                max(0.0, (y1 - y0) / ny / 2 - object_radius - _GAP / 2),
            ],
            device=device,
        )
        return cls(cells, jitter)

    @property
    def num_cells(self) -> int:
        return self.cells.shape[0]

    def sample_poses(
        self, num_envs: int, count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``count`` distinct jittered cells plus a yaw per object.

        Sequential Fisher-Yates: object ``i`` consumes one ``rand(E, 4)``
        draw — cell pick among the remaining cells, jitter x/y, yaw — so the
        draw count and the draw-to-pose mapping are independent of ``count``
        (seed-prefix rule, see module docstring).

        Returns:
            ``(xy, yaw)``: (num_envs, count, 2) and (num_envs, count).
        """
        num_cells = self.num_cells
        assert count <= num_cells, f"cannot place {count} objects on {num_cells} cells"
        dev = self.cells.device
        perm = torch.arange(num_cells, device=dev).expand(num_envs, -1).clone()
        env = torch.arange(num_envs, device=dev)
        xy = torch.empty(num_envs, count, 2, device=dev)
        yaw = torch.empty(num_envs, count, device=dev)
        for i in range(count):
            u = torch.rand(num_envs, 4, device=dev)
            j = i + (u[:, 0] * (num_cells - i)).long().clamp_max(num_cells - i - 1)
            picked = perm[env, j]
            perm[env, j] = perm[env, i]
            perm[env, i] = picked
            xy[:, i] = self.cells[picked] + (u[:, 1:3] - 0.5) * 2.0 * self.jitter
            yaw[:, i] = u[:, 3] * 2.0 * math.pi
        return xy, yaw
