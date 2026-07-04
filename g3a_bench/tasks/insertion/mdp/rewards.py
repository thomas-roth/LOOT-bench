"""Reward terms for the object-insertion task category."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from g3a_bench.tasks.insertion.mdp.commands import InsertionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, name: str) -> InsertionCommand:
    return cast(InsertionCommand, env.command_manager.get_term(name))


def inserted_count_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Fraction of pegs correctly inserted."""
    command = _command(env, command_name)
    return command.inserted.sum(dim=1) / command.num_sockets


def success_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Sparse bonus while all pegs are inserted."""
    return _command(env, command_name).inserted.all(dim=1).float()
