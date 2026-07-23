"""Reward terms for the place task category."""

from __future__ import annotations

from functools import partial

from g3a_bench.tasks.common.rewards import (
    command_progress_reward,
    command_success_reward,
)
from g3a_bench.tasks.place.mdp.commands import get_command

progress_reward = partial(command_progress_reward, get_command=get_command)
"""Continuous distance-to-target shaping, in [0, 1]."""

success_reward = partial(command_success_reward, get_command=get_command)
"""Sparse bonus while the place criterion (position + at-rest + released) holds."""
