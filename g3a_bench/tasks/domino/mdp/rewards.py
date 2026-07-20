"""Reward terms for the domino task category."""

from __future__ import annotations

from functools import partial

from g3a_bench.tasks.common.rewards import (
    command_progress_reward,
    command_success_reward,
)
from g3a_bench.tasks.domino.mdp.commands import get_command

progress_reward = partial(command_progress_reward, get_command=get_command)
"""Placement fraction (build phase) / toppling fraction (topple phase)."""

success_reward = partial(command_success_reward, get_command=get_command)
"""Sparse bonus while the row (and, for KnockOver, the fall) is complete."""
