"""Upright-spawn writer shared by every task category.

:func:`write_upright_spawn` sits entities upright on their sampled poses and
zeros their velocity. The poses come from the category's fixed spawn table
via :func:`fixed_spawns.sample_fixed_spawns` (object ``i`` on cell ``i``,
with jitter, in index order) or, for Place, from its envelope draw — either
way the writer only places what it is handed.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_from_euler_xyz

# Clearance above the origin's z-plane an upright-spawned object rests at (m).
_SPAWN_CLEARANCE = 0.002


def write_upright_spawn(
    entities: Sequence[Entity],
    xy: torch.Tensor,
    spawn_yaw: torch.Tensor,
    half_heights: torch.Tensor,
    origins: torch.Tensor,
    env_ids: torch.Tensor,
) -> None:
    """Write each entity upright at its ``(xy, spawn_yaw)`` pose.

    Sits each entity on the origin's z-plane with ``_SPAWN_CLEARANCE`` and
    zeros its velocity. ``xy``/``spawn_yaw`` are per-object outputs in index
    order (object ``i`` on cell ``i``); ``half_heights`` is the per-object
    half height along z; ``origins`` is the per-env world offset for the
    ``env_ids`` batch.
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
