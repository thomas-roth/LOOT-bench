"""Termination terms for the domino task category."""

from __future__ import annotations

from functools import partial

from loot_bench.tasks.common.terminations import command_success_termination
from loot_bench.tasks.domino.mdp.commands import get_command

domino_task_complete = partial(command_success_termination, get_command=get_command)
"""Placement (and, for KnockOver, toppling), velocity, and gripper conditions
all met."""
