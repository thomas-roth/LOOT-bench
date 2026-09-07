"""Observation terms for the domino task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from loot_bench.tasks.common.observations import flatten_object_poses
from loot_bench.tasks.domino.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def domino_poses(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """All domino poses in the env-local world frame, flattened (E, G*7)."""
    command = _command(env, command_name)
    return flatten_object_poses(
        command.domino_pos_w, command.domino_quat_w, env.scene.env_origins
    )


def slot_placed(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Per-slot placed indicator, (E, G)."""
    return _command(env, command_name).slot_at_goal


def phase_obs(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(E, 1) build (0) / topple (1) phase, KnockOver variants only."""
    return _command(env, command_name).phase.unsqueeze(-1)
