"""Observation terms for the stack task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from loot_bench.tasks.common.observations import flatten_object_poses
from loot_bench.tasks.stack.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def block_poses(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """All block poses in the env-local world frame, flattened (E, N*7)."""
    command = _command(env, command_name)
    return flatten_object_poses(
        command.block_pos_w, command.block_quat_w, env.scene.env_origins
    )
