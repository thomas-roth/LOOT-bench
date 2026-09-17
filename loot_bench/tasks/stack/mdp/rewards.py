"""Reward terms for the pi (stack) task category."""

from __future__ import annotations

from functools import partial

from loot_bench.tasks.common.rewards import command_task_reward
from loot_bench.tasks.stack.mdp.commands import get_command

task_reward = partial(command_task_reward, get_command=get_command)
"""The category's whole reward, in [-1, 0], zero only at success.

Shaped by the fraction of levels standing (``pi_progress``), not by any
height. Zero when every level has its lintel, the blocks are at rest, and
the gripper has released. The step where the last lintel lands but is still
settling reads ``-MIN_SHORTFALL``, not ``0``: progress is already 1 there
and the floor is what keeps it from looking like a success.
"""
