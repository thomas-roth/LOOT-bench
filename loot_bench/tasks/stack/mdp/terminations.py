"""Termination terms for the henge (stack) task category."""

from __future__ import annotations

from functools import partial

from loot_bench.tasks.common.terminations import command_success_termination
from loot_bench.tasks.stack.mdp.commands import get_command

henge_complete = partial(command_success_termination, get_command=get_command)
"""Every level standing, velocity, and gripper conditions all met."""
