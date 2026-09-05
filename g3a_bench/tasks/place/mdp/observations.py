"""Observation terms for the place task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.common.observations import flatten_pose
from g3a_bench.tasks.place.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def cube_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Cube pose in the env-local world frame (E, 7)."""
    command = _command(env, command_name)
    flattened_pose = flatten_pose(
        command.cube_pos_w, command.cube_quat_w, env.scene.env_origins
    )
    return flattened_pose
