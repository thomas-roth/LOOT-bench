"""Reward terms for the shape-sorter task category."""

from __future__ import annotations

from functools import partial

from g3a_bench.tasks.common.rewards import command_task_reward
from g3a_bench.tasks.shape_sorter.mdp.commands import get_command

task_reward = partial(command_task_reward, get_command=get_command)
"""The category's whole reward, in [-1, 0], zero only at success.

Shaped by the fraction of pieces latched into their own hole. Zero when every
piece is latched and the gripper is open.
"""
