"""Reward-term factories shared by every task category.

Every category's ``progress``/``success`` reward is just the command term's
``.progress``/``.success`` buffer, read through the category's own typed
``get_command`` accessor.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.command_manager import CommandTerm


def command_progress_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    get_command: Callable[[ManagerBasedRlEnv, str], CommandTerm],
) -> torch.Tensor:
    """Fraction of the task completed, in [0, 1]."""
    return get_command(env, command_name).progress  # type: ignore[attr-defined]


def command_success_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    get_command: Callable[[ManagerBasedRlEnv, str], CommandTerm],
) -> torch.Tensor:
    """Sparse bonus while the task's success condition holds."""
    return get_command(env, command_name).success  # type: ignore[attr-defined]
