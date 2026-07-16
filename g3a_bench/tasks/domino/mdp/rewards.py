"""Reward terms for the domino task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.domino.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def progress_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Placement fraction (build phase) / toppling fraction (topple phase)."""
    return _command(env, command_name).progress


def success_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Sparse bonus while the row (and, for KnockOver, the fall) is complete."""
    return _command(env, command_name).success
