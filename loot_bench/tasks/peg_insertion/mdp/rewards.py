"""Reward terms for the peg-insertion task category (archived, unregistered)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from loot_bench.tasks.peg_insertion.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def inserted_count_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Fraction of sockets currently filled."""
    return _command(env, command_name).scored_fraction()


def success_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Sparse bonus while the task is complete, pieces at rest and released."""
    return _command(env, command_name).success
