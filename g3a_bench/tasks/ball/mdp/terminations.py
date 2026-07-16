"""Termination terms for the ball task category.

``ball_off_platform`` / ``ball_hits_wall`` are only registered for the
variants where they can occur (minigolf / throw); see ``make_ball_env_cfg``.
Both read the command's *live* field/wall pose instead of env-origin
constants, since the field (and, for ThrowWallVariation, the wall) moves
every reset.
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
    """Ball dropped below the platform's base level outside the pit.

    ``spawn_z`` is the platform's base elevation -- constant across every
    minigolf layout, including Ramp's raised target section -- so a ball
    anywhere on the platform (low section, incline, or raised section) stays
    at or above it; the Chebyshev distance to the (possibly elevated) target
    keeps the pit itself, which legitimately drops below this level, exempt.
    """
    command = _command(env, command_name)
    below = command.ball_pos_w[:, 2] < command.spawn_z
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
    """Ball entered the throw wall's box (touched instead of cleared it).

    Uses the command's live ``wall_pos_w``: for ThrowWallVariation the wall
    is a separately-moving fixed entity; for the other throw variants it's
    baked into the field, offset by the field's own live pose.
    """
    command = _command(env, command_name)
    rel = command.ball_pos_w - command.wall_pos_w
    in_wall_x = rel[:, 0].abs() <= WALL_HALF_X
    in_wall_y = rel[:, 1].abs() <= WALL_HALF_Y
    in_wall_z = (rel[:, 2] >= 0) & (rel[:, 2] <= WALL_HEIGHT + BALL_RADIUS)
    return in_wall_x & in_wall_y & in_wall_z
