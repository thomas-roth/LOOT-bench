"""Reward terms for the stack task category."""

from __future__ import annotations

from functools import partial

from g3a_bench.tasks.common.rewards import (
    command_progress_reward,
    command_success_reward,
)
from g3a_bench.tasks.stack.mdp.commands import get_command

progress_reward = partial(command_progress_reward, get_command=get_command)
"""Fraction of the target height reached, in [0, 1]."""

success_reward = partial(command_success_reward, get_command=get_command)
"""Sparse bonus while the stack (or both stacks) stand at target height."""
