"""Reward terms for the ball task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.ball.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def success_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Bonus when the ball is in target, at rest, and released."""
    return _command(env, command_name).success
