"""Reward terms for the place task category."""

from __future__ import annotations

from functools import partial

from loot_bench.tasks.common.rewards import (
    command_sparse_task_reward,
    command_task_reward,
)
from loot_bench.tasks.place.mdp.commands import get_command

sparse_task_reward = partial(command_sparse_task_reward, get_command=get_command)
"""Sparse member of the shared convention: -1 every step until the cube is
placed, 0 on the step that scores."""

dense_task_reward = partial(command_task_reward, get_command=get_command)
"""Shaped member of the shared convention: the fraction of the spawn->target
travel completed, in [-1, 0], floored at ``-MIN_SHORTFALL`` off success."""

task_reward = sparse_task_reward
"""The category's whole reward, in [-1, 0], zero only at success.

Sparse by default: the travel fraction says little once the policy can reach,
and the shaped arm is opt-in through ``make_place_env_cfg(dense_reward=True)``
(or ``LOOT_PLACE_DENSE_REWARD=1``, e.g. ``train_qc.sh --dense``).
Zero when the cube sits within tolerance of the target, at rest, with the
gripper open.
"""
