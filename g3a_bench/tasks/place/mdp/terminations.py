"""Termination terms for the place task category."""

from __future__ import annotations

from functools import partial

from g3a_bench.tasks.common.terminations import command_success_termination
from g3a_bench.tasks.place.mdp.commands import get_command

place_task_complete = partial(command_success_termination, get_command=get_command)
"""Position, velocity, and gripper conditions all met."""
