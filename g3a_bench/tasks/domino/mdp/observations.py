"""Observation terms for the domino task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.domino.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def domino_poses(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """All domino poses in the env-local world frame, flattened (E, G*7).

    Object-major (domino index order): each domino contributes
    (pos xyz, quat wxyz). Dropping the trailing dominoes' 7-dim slices
    adapts the observation to an easier task (demo cutoff).
    """
    command = _command(env, command_name)
    pos = command.domino_pos_w - env.scene.env_origins.unsqueeze(1)
    return torch.cat([pos, command.domino_quat_w], dim=-1).reshape(env.num_envs, -1)


def slot_placed(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Per-slot placed indicator, (E, G)."""
    return _command(env, command_name).slot_at_goal


def phase_obs(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(E, 1) build (0) / topple (1) phase, KnockOver variants only."""
    return _command(env, command_name).phase.unsqueeze(-1)
