"""Reward terms for the ball task category.

Ball has no progress signal: a ball is either in the target zone or not,
and distance says little about whether the throw will land. So the reward
is the sparse member of the shared convention: -1 every step until the ball
scores, 0 on the step that scores.
"""

from __future__ import annotations

from functools import partial

from g3a_bench.tasks.ball.mdp.commands import get_command
from g3a_bench.tasks.common.rewards import command_sparse_task_reward

task_reward = partial(command_sparse_task_reward, get_command=get_command)
"""The category's whole reward, in [-1, 0].

Zero when the ball is in the target zone, at rest, and released.
"""
