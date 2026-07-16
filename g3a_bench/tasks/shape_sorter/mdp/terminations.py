"""Termination terms for the shape-sorter task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.shape_sorter.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def sorter_complete(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Every piece latched and the gripper released."""
    return _command(env, command_name).success.bool()
