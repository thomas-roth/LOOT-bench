"""Observation terms for the stack task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.stack.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def block_poses(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """All block poses in the env-local world frame, flattened (E, N*7).

    Object-major (block index order): each block contributes
    (pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z). Dropping the
    trailing blocks' 7-dim slices adapts the observation to an easier task
    (demo cutoff).
    """
    command = _command(env, command_name)
    pos = command.block_pos_w - env.scene.env_origins.unsqueeze(1)
    return torch.cat([pos, command.block_quat_w], dim=-1).reshape(env.num_envs, -1)
