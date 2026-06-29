"""Termination terms for the ball task category."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from g3a_bench.tasks.ball.mdp.commands import BallCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def ball_at_target(
    env: ManagerBasedRlEnv,
    command_name: str,
    hold_time: float = 0.0,
) -> torch.Tensor:
    """Ball in the target at rest (instant reset when goal reached)."""
    command = cast(BallCommand, env.command_manager.get_term(command_name))
    at_target = command.at_target
    at_rest = command.success_steps * env.step_dt >= hold_time
    return at_target & at_rest


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
    from g3a_bench.tasks.ball.variants import BALL_RADIUS, WALL_HEIGHT, WALL_X

    wall_x_min = WALL_X - 0.02
    wall_x_max = WALL_X + 0.02
    wall_y_min = -0.25
    wall_y_max = 0.25
    wall_z_max = WALL_HEIGHT
    in_wall_x = (command.ball_pos_w[:, 0] >= wall_x_min) & (
        command.ball_pos_w[:, 0] <= wall_x_max
    )
    in_wall_y = (command.ball_pos_w[:, 1] >= wall_y_min) & (
        command.ball_pos_w[:, 1] <= wall_y_max
    )
    in_wall_z = (command.ball_pos_w[:, 2] >= 0) & (
        command.ball_pos_w[:, 2] <= wall_z_max + BALL_RADIUS
    )
    return in_wall_x & in_wall_y & in_wall_z
