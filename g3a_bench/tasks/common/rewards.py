"""Reward-term factories shared by every task category.

**The convention.** A task's whole reward is one term in ``[-1, 0]``, and it
reaches exactly ``0`` only at success. Categories with a progress signal shape
the interval; categories without one (ball) sit at ``-1`` until they score.

Why the sign, and why exactly zero: see ``ADR-006``.

**The reward lags the observation by one step, on purpose.** mjlab's ``step()``
runs the termination and reward managers *before* the command manager, so both
read the success and progress buffers as they stood at the end of the previous
step. Success reached during step ``k`` is therefore paid at step ``k + 1``,
which is also the step that terminates: the last transition of a successful
episode carries ``r == 0`` together with ``terminated``, exactly the absorbing
state offline RL wants. The lag is identical when recording and when
evaluating, so it cancels; do not "fix" it on one side alone.

Rewards are not scaled by the timestep (``scale_rewards_by_dt=False`` in
``env_cfg.make_manipulation_env_cfg``). mjlab's default would multiply every
reward by ``step_dt``, which would silently rescale a recorded dataset if the
decimation ever changed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.command_manager import CommandTerm

MIN_SHORTFALL = 1e-3
"""How far below zero a non-success reward has to stay.

A task can report full progress while its success gate has not latched: the
objects are in place but still moving, so the at-rest condition has yet to
hold. Without this floor that step would read exactly ``0.0`` and ``--sparse``
would relabel it a success.
"""


def shaped_task_reward(progress: torch.Tensor, success: torch.Tensor) -> torch.Tensor:
    """The task reward: ``progress - 1``, floored, and exactly ``0`` at success.

    Args:
        progress: fraction of the task completed. Clamped to ``[0, 1]``, so a
            category that overshoots cannot buy reward past the goal.
        success: whether the success gate has latched. Bool or float.

    Returns:
        A tensor shaped like ``progress``, in ``[-1, 0]``, equal to ``0``
        exactly where ``success`` holds.
    """
    shaped = (progress.clamp(0.0, 1.0) - 1.0).clamp(max=-MIN_SHORTFALL)
    return torch.where(success.bool(), torch.zeros_like(shaped), shaped)


def command_task_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    get_command: Callable[[ManagerBasedRlEnv, str], CommandTerm],
) -> torch.Tensor:
    """The whole reward for a category that reports progress."""
    command = get_command(env, command_name)
    return shaped_task_reward(command.progress, command.success)


def command_sparse_task_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    get_command: Callable[[ManagerBasedRlEnv, str], CommandTerm],
) -> torch.Tensor:
    """The whole reward for a category with no progress signal.

    ``-1`` every step until the success gate latches, then ``0``.
    """
    success = get_command(env, command_name).success
    return shaped_task_reward(torch.zeros_like(success, dtype=torch.float), success)
