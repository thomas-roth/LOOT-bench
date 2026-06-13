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
    hold_time: float = 0.3,
) -> torch.Tensor:
    """Ball in the target at rest for ``hold_time`` seconds."""
    command = cast(BallCommand, env.command_manager.get_term(command_name))
    return command.success_steps * env.step_dt >= hold_time


def ball_out_of_bounds(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_radius: float = 1.2,
) -> torch.Tensor:
    """Ball strayed far from the env origin (e.g. knocked off the table)."""
    command = cast(BallCommand, env.command_manager.get_term(command_name))
    rel = command.ball_pos_w - env.scene.env_origins
    return torch.norm(rel[:, :2], dim=-1) > max_radius
