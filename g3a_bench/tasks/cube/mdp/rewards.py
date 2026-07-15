"""Reward terms for the cube task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.cube.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def placed_slots_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Fraction of placed slots; rewards progress and keeping pieces placed."""
    command = _command(env, command_name)
    return command.slot_at_goal.sum(dim=1) / command.num_slots


def success_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Sparse bonus while the full arrangement is complete."""
    return _command(env, command_name).success
