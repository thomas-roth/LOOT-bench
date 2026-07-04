"""Reward terms for the object-insertion task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.insertion.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def inserted_count_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Fraction of pegs correctly inserted."""
    command = _command(env, command_name)
    return command.inserted.sum(dim=1) / command.num_sockets


def success_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Sparse bonus while all pegs are inserted."""
    return _command(env, command_name).inserted.all(dim=1).float()
