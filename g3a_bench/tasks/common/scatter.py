"""Grid-based spawn sampling with jittered, distinct cell centers.

:meth:`ScatterGrid.tiling` sizes the grid so cells are at least one object
diameter apart and jitter never closes that gap. Samples are collision-free
by construction. :meth:`sample_poses` draws them with a *sequential*
Fisher-Yates shuffle: object ``i`` consumes exactly one ``rand(E, 4)`` draw
(cell pick among the cells left by objects ``0..i-1``, jitter x/y, yaw), in
index order. Fixed draws per object, independent of the total object count,
so a task's resets are a seed-prefix of a harder task's (see
``docs/TaskCategories.md`` "General": demos for easier tasks are cut from
the hardest task's recordings). Pass explicit ``cells``/``jitter`` for
irregular layouts.

:meth:`blocked_near` builds a per-env keep-out mask for ``sample_poses`` so
objects never spawn on (or right next to) their own goal positions; it
consumes no randomness, keeping the seed-prefix rule intact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_from_euler_xyz

# Minimum clearance between two objects in neighboring cells (m).
_GAP = 0.004
# Clearance above the origin's z-plane an upright-spawned object rests at (m).
_SPAWN_CLEARANCE = 0.002


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

    def blocked_near(
        self, points: torch.Tensor, min_distance: float, *, keep_free: int = 0
    ) -> torch.Tensor:
        """Per-env keep-out mask of the cells too close to ``points``.

        A cell is blocked when *any* pose it can produce — centre plus the
        worst-case jitter corner — would land within ``min_distance`` of a
        point. The ``keep_free`` cells farthest from every point are never
        blocked, so a keep-out that swallows the region still leaves
        :meth:`sample_poses` enough cells (it then falls back to the least
        bad ones instead of failing).

        Args:
            points: (num_envs, K, 2) keep-out centres, same frame as the cells.
            min_distance: min distance from a sampled pose to any point.
            keep_free: cells to leave free per env (pass the object count).

        Returns:
            (num_envs, C) bool mask for :meth:`sample_poses`'s ``blocked``.
        """
        delta = self.cells[None, :, None, :] - points[:, None, :, :]  # (E, C, K, 2)
        nearest = delta.norm(dim=-1).amin(dim=-1)  # (E, C)
        blocked = nearest < min_distance + float(torch.linalg.norm(self.jitter))
        rank = nearest.argsort(dim=1, descending=True).argsort(dim=1)
        return blocked & (rank >= keep_free)

    def sample_poses(
        self, num_envs: int, count: int, blocked: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``count`` distinct jittered cells plus a yaw per object.

        Sequential Fisher-Yates: object ``i`` consumes one ``rand(E, 4)``
        draw (cell pick among remaining cells, jitter x/y, yaw), so the
        draw count and the draw-to-pose mapping are independent of ``count``
        (seed-prefix rule, see module docstring).

        Args:
            blocked: optional (num_envs, C) mask of cells to skip, e.g. from
                :meth:`blocked_near`. Each env must keep >= ``count`` cells.

        Returns:
            ``(xy, yaw)``: (num_envs, count, 2) and (num_envs, count).
        """
        num_cells = self.num_cells
        assert count <= num_cells, f"cannot place {count} objects on {num_cells} cells"
        dev = self.cells.device
        if blocked is None:
            perm = torch.arange(num_cells, device=dev).expand(num_envs, -1).clone()
            avail = torch.full((num_envs,), num_cells, device=dev, dtype=torch.long)
        else:
            # Stable sort: allowed cells first, both groups in index order.
            perm = blocked.to(torch.uint8).argsort(dim=1, stable=True)
            avail = (~blocked).sum(dim=1)
            assert int(avail.min()) >= count, (
                f"cannot place {count} objects on {int(avail.min())} unblocked cells"
            )
        env = torch.arange(num_envs, device=dev)
        xy = torch.empty(num_envs, count, 2, device=dev)
        yaw = torch.empty(num_envs, count, device=dev)
        for i in range(count):
            u = torch.rand(num_envs, 4, device=dev)
            span = avail - i  # cells still free in this env, >= 1
            j = i + torch.minimum((u[:, 0] * span).long(), span - 1)
            picked = perm[env, j]
            perm[env, j] = perm[env, i]
            perm[env, i] = picked
            xy[:, i] = self.cells[picked] + (u[:, 1:3] - 0.5) * 2.0 * self.jitter
            yaw[:, i] = u[:, 3] * 2.0 * math.pi
        return xy, yaw


def write_upright_spawn(
    entities: Sequence[Entity],
    xy: torch.Tensor,
    spawn_yaw: torch.Tensor,
    half_heights: torch.Tensor,
    origins: torch.Tensor,
    env_ids: torch.Tensor,
) -> None:
    """Write each entity upright at its scattered ``(xy, spawn_yaw)`` cell.

    Sits each entity on the origin's z-plane with ``_SPAWN_CLEARANCE`` and
    zeros its velocity. ``xy``/``spawn_yaw`` are :meth:`ScatterGrid.sample_poses`'s
    per-object output (object-major, index order); ``half_heights`` is the
    per-object half height along z; ``origins`` is the per-env world offset
    for the ``env_ids`` batch.
    """
    n = len(env_ids)
    dev = xy.device
    zeros = torch.zeros(n, device=dev)
    zero_vel = torch.zeros(n, 6, device=dev)
    for i, entity in enumerate(entities):
        z = (half_heights[i] + _SPAWN_CLEARANCE + origins[:, 2]).unsqueeze(-1)
        pos = torch.cat([xy[:, i, :] + origins[:, :2], z], dim=-1)
        quat = quat_from_euler_xyz(zeros, zeros, spawn_yaw[:, i])
        pose = torch.cat([pos, quat], dim=-1)
        entity.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        entity.write_root_link_velocity_to_sim(zero_vel, env_ids=env_ids)
