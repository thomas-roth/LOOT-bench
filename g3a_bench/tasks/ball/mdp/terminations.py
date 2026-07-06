"""Termination terms for the ball task category."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from g3a_bench.tasks.ball.mdp.commands import BallCommand
from g3a_bench.tasks.ball.variants import (
    BALL_RADIUS,
    WALL_HALF_X,
    WALL_HALF_Y,
    WALL_HEIGHT,
    WALL_X,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def ball_at_target(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Ball in target, at rest, and released."""
    command = cast(BallCommand, env.command_manager.get_term(command_name))
    return command.success.bool()


def ball_out_of_bounds(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_radius: float = 1.2,
) -> torch.Tensor:
    """Ball strayed far from the env origin (e.g. knocked off the table)."""
    command = cast(BallCommand, env.command_manager.get_term(command_name))
    rel = command.ball_pos_w - env.scene.env_origins
    return torch.norm(rel[:, :2], dim=-1) > max_radius


def ball_hits_ground(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Ball collides with ground (below surface level)."""
    command = cast(BallCommand, env.command_manager.get_term(command_name))
    return command.ball_pos_w[:, 2] < command.surface_z - 0.05


def ball_hits_wall(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Ball collides with throw wall."""
    command = cast(BallCommand, env.command_manager.get_term(command_name))
    if not command.variant.has_wall:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    rel = command.ball_pos_w - env.scene.env_origins
    wall_x_min = WALL_X - WALL_HALF_X
    wall_x_max = WALL_X + WALL_HALF_X
    wall_y_min = -WALL_HALF_Y
    wall_y_max = WALL_HALF_Y
    wall_z_max = WALL_HEIGHT
    in_wall_x = (rel[:, 0] >= wall_x_min) & (rel[:, 0] <= wall_x_max)
    in_wall_y = (rel[:, 1] >= wall_y_min) & (rel[:, 1] <= wall_y_max)
    in_wall_z = (rel[:, 2] >= 0) & (rel[:, 2] <= wall_z_max + BALL_RADIUS)
    return in_wall_x & in_wall_y & in_wall_z
