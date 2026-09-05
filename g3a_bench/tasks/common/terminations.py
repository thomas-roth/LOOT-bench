"""Termination-term factories shared by every task category."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.command_manager import CommandTerm


def command_success_termination(
    env: ManagerBasedRlEnv,
    command_name: str,
    get_command: Callable[[ManagerBasedRlEnv, str], CommandTerm],
) -> torch.Tensor:
    """Task's success condition (task-done, at-rest, and released) all met."""
    return get_command(env, command_name).success.bool()
