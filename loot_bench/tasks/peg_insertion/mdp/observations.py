"""Observation terms for the peg-insertion task category (archived, unregistered)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from loot_bench.tasks.peg_insertion.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def board_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Board pose, env-local (E, 7). Constant per task."""
    command = _command(env, command_name)
    pos = command.board_pos_w - env.scene.env_origins
    return torch.cat([pos, command.board_quat_w], dim=-1)


def peg_poses(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Peg poses, env-local and flattened (E, N*7), in socket order."""
    command = _command(env, command_name)
    pos = command.peg_pos_w - env.scene.env_origins.unsqueeze(1)
    return torch.cat([pos, command.peg_quat_w], dim=-1).reshape(env.num_envs, -1)


def sheet_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Top-sheet pose, env-local (E, 7). Zeros if no sheet."""
    command = _command(env, command_name)
    if command.sheet is None:
        return torch.zeros(env.num_envs, 7, device=env.device)
    pos = command.sheet_pos_w - env.scene.env_origins
    return torch.cat([pos, command.sheet_quat_w], dim=-1)


def socket_inserted_flags(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Per-socket inserted flags (E, S)."""
    return _command(env, command_name).inserted
