"""Observation terms for the ball task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.ball.mdp.commands import get_command as _command
from g3a_bench.tasks.common.observations import flatten_pose

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def ball_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Ball position in the env-local world frame (E, 3).

    No quaternion: a sphere's orientation carries no task-relevant signal.
    """
    command = _command(env, command_name)
    return command.ball_pos_w - env.scene.env_origins


def field_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Field pose in the env-local world frame (E, 7)."""
    command = _command(env, command_name)
    return flatten_pose(
        command.field_pos_w, command.field_quat_w, env.scene.env_origins
    )


def wall_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Wall pose in the env-local world frame (E, 7).

    Only registered for variants with an independently-moving wall entity
    (``ThrowWallVariation``); other variants bake the wall into the field.
    """
    command = _command(env, command_name)
    return flatten_pose(command.wall_pos_w, command.wall_quat_w, env.scene.env_origins)


def ball_velocity(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Ball linear velocity in the world frame (E, 3)."""
    return _command(env, command_name).ball_vel_w
