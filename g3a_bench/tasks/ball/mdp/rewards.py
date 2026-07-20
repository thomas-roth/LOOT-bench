"""Reward terms for the ball task category.

No ``progress`` reward: unlike the other categories, ball's task rewards are
``success`` only (see ``ball_env_cfg.py``).
"""

from __future__ import annotations

from functools import partial

from g3a_bench.tasks.ball.mdp.commands import get_command
from g3a_bench.tasks.common.rewards import command_success_reward

success_reward = partial(command_success_reward, get_command=get_command)
"""Bonus when the ball is in target, at rest, and released."""
