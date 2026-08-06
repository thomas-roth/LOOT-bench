"""Reward-term factories shared by every task category.

Every category's ``progress``/``success`` reward is just the command term's
``.progress``/``.success`` buffer, read through the category's own typed
``get_command`` accessor.

Reward convention
-----------------
Every category term is normalized to ``[0, 1]`` -- ``progress`` is a completion
fraction, ``success`` an indicator -- and the episode terminates the step
``success`` fires.  On top of those, ``make_manipulation_env_cfg`` appends the
shared :data:`REWARD_OFFSET_TERM`, a constant ``-reward_max`` (see
:func:`reward_max`), so the env's per-step reward is

    ``sum(task terms) - reward_max``  in ``[-reward_max, 0]``

reaching ``0`` at success.  Exactly ``0`` for every category except **stack**:
its success test is a two-sided tolerance band (``|height - target| <= 0.01``)
while ``stack_progress`` is ``(best / target_height).clamp(0, 1)``, so an
in-band *undershoot* still counts as success while progress is short by up to
``tol / target_height`` -- 0.083 for Stack-Low, 0.050 for Stack-High.  The
absorbing state is worth that much below 0 instead of 0.  Negligible against a
value scale of ``[-reward_max / (1 - discount), 0]`` (~0.04% at discount 0.99),
but it is an approximation, not an identity.  Domino and sorter are exact
(their progress is the same all-or-nothing fraction their success tests on);
place and ball are exact (no progress term).

Two reasons this convention and not the raw positive sum:

- With termination on success, a positive reward makes *stalling* optimal: a
  policy that holds a near-complete state and never releases collects reward
  every step until the time-out (which bootstraps), while finishing ends the
  episode.  Shifting by the attainable maximum inverts that -- stalling costs
  ``-1`` per step, finishing is worth ``0``.
- It matches the convention every offline-RL benchmark QC was tuned against
  (OGBench ``success - 1``, D4RL AntMaze / RoboMimic ``reward - 1``), which
  puts Q-values on the scale QC's ``alpha``/lr defaults expect.

The offset is only meaningful with ``scale_rewards_by_dt=False`` (also set by
``make_manipulation_env_cfg``): mjlab's default multiplies every term by the
control step, which is right for locomotion rate-terms but turns our
dimensionless fraction/indicator into ``1/50``-sized values, and would make
the constant offset here 50x too large.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.command_manager import CommandTerm
    from mjlab.managers.reward_manager import RewardTermCfg

REWARD_OFFSET_TERM = "max_reward_offset"
"""Key of the shared constant term that shifts a task's reward to ``<= 0``."""


def constant_one(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(E,) of ones -- the carrier for :data:`REWARD_OFFSET_TERM`.

    mjlab has no notion of a bias, so the offset is expressed as a constant
    term whose *weight* is the (negative) shift.
    """
    return torch.ones(env.num_envs, device=env.device)


def reward_max(task_rewards: Mapping[str, RewardTermCfg]) -> float:
    """Largest per-step reward a task's own terms can produce.

    Relies on the invariant that every category term is in ``[0, 1]``, so the
    maximum is just the sum of the weights (2.0 for the ``progress`` +
    ``success`` categories, 1.0 for the ``success``-only ones).  A category
    that adds a term outside ``[0, 1]`` -- an unbounded shaping term, or a
    penalty -- breaks this and must not be added without revisiting the offset.

    ``REWARD_OFFSET_TERM`` itself is excluded so the function stays correct if
    called on an already-assembled reward dict.
    """
    return float(
        sum(
            term.weight
            for name, term in task_rewards.items()
            if name != REWARD_OFFSET_TERM
        )
    )


def command_progress_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    get_command: Callable[[ManagerBasedRlEnv, str], CommandTerm],
) -> torch.Tensor:
    """Fraction of the task completed, in [0, 1]."""
    return get_command(env, command_name).progress  # type: ignore[attr-defined]


def command_success_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    get_command: Callable[[ManagerBasedRlEnv, str], CommandTerm],
) -> torch.Tensor:
    """Sparse bonus while the task's success condition holds."""
    return get_command(env, command_name).success  # type: ignore[attr-defined]
