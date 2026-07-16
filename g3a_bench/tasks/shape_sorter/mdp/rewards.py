"""Reward terms for the shape-sorter task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.shape_sorter.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def progress_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Fraction of pieces latched into their own hole, in [0, 1]."""
    return _command(env, command_name).progress


def success_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Sparse bonus while every piece is latched and the gripper is open."""
    return _command(env, command_name).success
