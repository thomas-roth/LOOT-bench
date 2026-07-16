"""Reward terms for the stack task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.stack.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def progress_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Fraction of the target height reached, in [0, 1]."""
    return _command(env, command_name).progress


def success_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Sparse bonus while the stack (or both stacks) stand at target height."""
    return _command(env, command_name).success
