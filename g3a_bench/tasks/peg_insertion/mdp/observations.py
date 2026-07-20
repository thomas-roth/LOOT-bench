"""Observation terms for the peg-insertion task category (archived, unregistered)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.peg_insertion.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def board_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """The insertion board's pose in the env-local world frame (E, 7).

    Board is fixed and constant per task; a single pose for the whole
    rigid board/socket assembly, matching shape_sorter's cube_pose — every
    bore's position/orientation is a fixed, learnable offset from it.
    """
    command = _command(env, command_name)
    pos = command.board_pos_w - env.scene.env_origins
    return torch.cat([pos, command.board_quat_w], dim=-1)


def peg_poses(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """All peg poses in the env-local world frame, flattened (E, N*7).

    Object-major (socket index order): each peg contributes (pos xyz,
    quat wxyz). Dropping the trailing pegs' 7-dim slices adapts the
    observation to an easier task (demo cutoff).
    """
    command = _command(env, command_name)
    pos = command.peg_pos_w - env.scene.env_origins.unsqueeze(1)
    return torch.cat([pos, command.peg_quat_w], dim=-1).reshape(env.num_envs, -1)


def sheet_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Top-sheet pose in the env-local world frame (E, 7).

    Two-sheet variants only; zeros if no sheet is present.
    """
    command = _command(env, command_name)
    if command.sheet is None:
        return torch.zeros(env.num_envs, 7, device=env.device)
    pos = command.sheet_pos_w - env.scene.env_origins
    return torch.cat([pos, command.sheet_quat_w], dim=-1)


def socket_inserted_flags(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Per-socket inserted flags (E, S)."""
    return _command(env, command_name).inserted
