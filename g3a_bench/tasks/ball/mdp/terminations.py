"""Termination terms for the ball task category.

``ball_hits_wall`` / ``ball_off_platform`` are only registered for the
variants where they can occur (throw / hole); see ``make_ball_env_cfg``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.ball.mdp.commands import get_command as _command
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
    return _command(env, command_name).success.bool()


def ball_out_of_bounds(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_radius: float = 1.2,
) -> torch.Tensor:
    """Ball strayed far from the env origin (e.g. knocked off the table)."""
    command = _command(env, command_name)
    rel = command.ball_pos_w - env.scene.env_origins
    return torch.norm(rel[:, :2], dim=-1) > max_radius


def ball_off_platform(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Ball dropped below the platform top outside the pit (fell off an edge).

    The pit is a square of half-width ``target_radius``, so the "outside"
    test uses the Chebyshev distance: a ball dropping into the pit near a
    corner stays inside it, while a ball on the floor around the platform
    is beyond it.
    """
    command = _command(env, command_name)
    below = command.ball_pos_w[:, 2] < command.surface_z
    d_cheb = (
        (command.ball_pos_w[:, :2] - command.target_pos_w[:, :2])
        .abs()
        .max(dim=-1)
        .values
    )
    return below & (d_cheb > command.target_radius)


def ball_hits_wall(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Ball entered the throw wall's box (touched instead of cleared it)."""
    command = _command(env, command_name)
    rel = command.ball_pos_w - env.scene.env_origins
    in_wall_x = (rel[:, 0] >= WALL_X - WALL_HALF_X) & (
        rel[:, 0] <= WALL_X + WALL_HALF_X
    )
    in_wall_y = (rel[:, 1] >= -WALL_HALF_Y) & (rel[:, 1] <= WALL_HALF_Y)
    in_wall_z = (rel[:, 2] >= 0) & (rel[:, 2] <= WALL_HEIGHT + BALL_RADIUS)
    return in_wall_x & in_wall_y & in_wall_z
