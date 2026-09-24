"""Termination terms for the sorter task category."""

from __future__ import annotations

from functools import partial

from loot_bench.tasks.common.terminations import command_success_termination
from loot_bench.tasks.sorter.mdp.commands import get_command

sorter_complete = partial(command_success_termination, get_command=get_command)
"""Every piece latched and the gripper released."""
