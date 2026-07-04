"""Reward terms for the ball task category."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from g3a_bench.tasks.ball.mdp.commands import BallCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, name: str) -> BallCommand:
    return cast(BallCommand, env.command_manager.get_term(name))


def success_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Bonus when the ball is in target, at rest, and released."""
    command = _command(env, command_name)
    d_xy = torch.norm(command.ball_pos_w[:, :2] - command.target_pos_w[:, :2], dim=-1)
    within = d_xy < command.target_radius
    speed = torch.norm(command.ball_vel_w, dim=-1)
    at_rest = speed < command.variant.at_rest_speed
    is_released = command.is_ball_released()
    return (within & at_rest & is_released).float()
