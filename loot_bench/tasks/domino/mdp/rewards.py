"""Reward terms for the domino task category."""

from __future__ import annotations

from functools import partial

from loot_bench.tasks.common.rewards import command_task_reward
from loot_bench.tasks.domino.mdp.commands import get_command

task_reward = partial(command_task_reward, get_command=get_command)
"""The category's whole reward, in [-1, 0], zero only at success.

Shaped by the placement fraction in the build phase, and by the toppling
fraction in the topple phase. Zero when the row is complete, and for
KnockOver, when the fall is complete too.
"""
