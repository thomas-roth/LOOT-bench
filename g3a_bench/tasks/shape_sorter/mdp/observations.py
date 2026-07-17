"""Observation terms for the shape-sorter task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.shape_sorter.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def cube_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """The sorter cube's pose in the env-local world frame (E, 7).

    A single pose for the whole rigid cube/hole assembly -- every hole's
    position and orientation is a fixed, learnable offset from it, so no
    per-hole target observation is added.
    """
    command = _command(env, command_name)
    pos = command.cube_pos_w - env.scene.env_origins
    return torch.cat([pos, command.cube_quat_w], dim=-1)


def piece_poses(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """All piece poses in the env-local world frame, flattened (E, N*7).

    Object-major (shape index order): each piece contributes (pos xyz,
    quat wxyz). Dropping the trailing pieces' 7-dim slices adapts the
    observation to an easier task (demo cutoff).
    """
    command = _command(env, command_name)
    pos = command.piece_pos_w - env.scene.env_origins.unsqueeze(1)
    return torch.cat([pos, command.piece_quat_w], dim=-1).reshape(env.num_envs, -1)


def socket_inserted(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Per-socket latched flags (E, N)."""
    return _command(env, command_name).latched.float()
