"""Reward terms for the place task category."""

from __future__ import annotations

from functools import partial

from g3a_bench.tasks.common.rewards import command_task_reward
from g3a_bench.tasks.place.mdp.commands import get_command

task_reward = partial(command_task_reward, get_command=get_command)
"""The category's whole reward, in [-1, 0], zero only at success.

Shaped by the fraction of the spawn->target travel completed. Zero when the
cube sits within tolerance of the target, at rest, with the gripper open.
"""
