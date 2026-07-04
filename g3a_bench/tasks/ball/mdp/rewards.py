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
    return _command(env, command_name).success
