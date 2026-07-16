"""Termination terms for the domino task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g3a_bench.tasks.domino.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def domino_task_complete(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Placement (and, for KnockOver, toppling), velocity, and gripper
    conditions all met."""
    return _command(env, command_name).success.bool()
