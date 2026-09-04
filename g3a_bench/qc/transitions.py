"""Recorded rows -> the transition arrays ``qc`` trains on.

A recording is one row per timestep: ``(obs_t, action_t, reward_t)``. Offline
RL wants ``(s, a, r, s', done)``, which is not the same shape of thing --
``s'`` is the *next* row, and the last row of an episode has no next row.

Everything here is pure array work, so it runs without a simulator and without
lerobot. :func:`assemble_transitions` is the whole contract; ``convert.py``
does the file handling around it.

Two columns carry the meaning, and they are **not** the same column:

* ``terminals`` marks an **episode boundary**. ``qc``'s ``sample_sequence``
  keys its ``valid`` mask off this alone, so an unmarked boundary lets a
  sampled action chunk walk out of one episode and into the next.
* ``masks`` marks whether to **bootstrap** past the transition (``1``) or not
  (``0``). Only an absorbing state gets ``0``.

Every episode end is a boundary; only a success is absorbing. Collapsing the
two, as ``live_bridge/lerobot_to_qc.py`` did, writing ``masks = 1 -
terminals``, tells the learner that running out of time is worth exactly as
much as reaching the goal, which is the one lie an offline critic cannot
recover from.

The timeout case is why this is not a one-liner. Its final row is a boundary
that is *not* absorbing, so it needs an ``s'`` to bootstrap into, and the
recorder never stored one. The old converter zero-filled it and set
``masks = 0``, so two faults hid each other: the zeros were never read because
the mask was wrong. Here the unusable row is **dropped**, and the row before it
bootstraps into the dropped row's observation, which is a real state.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from g3a_bench.qc.episode import SUCCESS_REWARD

QC_KEYS = frozenset(
    {"observations", "next_observations", "actions", "rewards", "terminals", "masks"}
)
"""The keys ``qc``'s ``Dataset.create`` may be given.

:func:`assemble_transitions` returns metadata alongside these. Filter to
``QC_KEYS`` before building the dataset: ``qc``'s replay buffer requires every
offline key to appear in the online transition dicts too, and those only ever
carry these six.
"""

MIN_REWARD = -1.0
"""The floor of the reward convention (``tasks/common/rewards.py``)."""


def assemble_transitions(
    episode_index: NDArray[np.integer],
    observations: NDArray[np.floating],
    actions: NDArray[np.floating],
    rewards: NDArray[np.floating],
) -> dict[str, NDArray]:
    """Build ``qc``'s transition arrays from one recording's columns.

    Args:
        episode_index: which episode each row belongs to. Rows of one episode
            must be contiguous and in time order; the caller sorts.
        observations: ``(N, obs_dim)``, the state each action was chosen from.
        actions: ``(N, action_dim)``, as the policy will emit them.
        rewards: ``(N,)``, following ``tasks/common/rewards.py``, inside
            ``[-1, 0]``, and exactly ``0`` only on the success step.

    Returns:
        The six :data:`QC_KEYS` plus an ``episode_index`` column for analysis.
        Shorter than the input: every non-absorbing episode loses its final
        row, which has no successor state to bootstrap into.

    Raises:
        ValueError: if the columns disagree in length, if an episode's rows are
            not contiguous, if a reward leaves ``[-1, 0]``, if a success reward
            appears anywhere but an episode's final row, or if no row survives.
    """
    episode_index = np.asarray(episode_index)
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)

    n = len(episode_index)
    lengths = {len(observations), len(actions), len(rewards), n}
    if len(lengths) != 1:
        raise ValueError(
            "episode_index, observations, actions and rewards must have the "
            f"same length, got {sorted(lengths)}"
        )
    if n == 0:
        raise ValueError("no rows to convert")

    starts, stops = _episode_bounds(episode_index)
    _check_rewards(rewards, stops)

    # Successor is row i+1; rows running off an episode end are dropped, never
    # redirected across a boundary.
    successor = np.minimum(np.arange(n) + 1, n - 1)
    keep = np.zeros(n, dtype=bool)
    terminals = np.zeros(n, dtype=np.float32)
    masks = np.ones(n, dtype=np.float32)

    for start, stop in zip(starts, stops, strict=True):
        absorbing = bool(rewards[stop - 1] == SUCCESS_REWARD)
        if absorbing:
            last = stop - 1
            # Never bootstrapped (masks = 0), but still passed through the critic:
            # point at self, not zeros, to stay in distribution.
            successor[last] = last
            masks[last] = 0.0
        else:
            # No stored successor for this row; drop it.
            last = stop - 2
            if last < start:
                # A one-row non-absorbing episode is entirely unusable.
                continue
        keep[start : last + 1] = True
        terminals[last] = 1.0

    if not keep.any():
        raise ValueError(
            "no usable transitions: every episode is a single non-absorbing row"
        )

    return {
        "observations": observations[keep],
        "next_observations": observations[successor][keep],
        "actions": actions[keep],
        "rewards": rewards[keep],
        "terminals": terminals[keep],
        "masks": masks[keep],
        # Metadata. Strip before Dataset.create(), see QC_KEYS.
        "episode_index": episode_index[keep],
    }


def _episode_bounds(
    episode_index: NDArray[np.integer],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Half-open ``[start, stop)`` row bounds of each episode, in row order."""
    changes = np.flatnonzero(episode_index[1:] != episode_index[:-1]) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [len(episode_index)]))
    if len(np.unique(episode_index)) != len(starts):
        raise ValueError(
            "each episode's rows must be contiguous, but at least one episode "
            "index appears in more than one run, sort by (episode, frame) "
            "first"
        )
    return starts, stops


def _check_rewards(rewards: NDArray[np.floating], stops: NDArray[np.intp]) -> None:
    """Fail loudly on a recording that predates the reward convention.

    A dataset whose rewards mean something else trains happily and scores
    nothing, so this is worth catching at conversion time.
    """
    if np.any(rewards > SUCCESS_REWARD):
        bad = int(np.argmax(rewards > SUCCESS_REWARD))
        raise ValueError(
            f"reward {rewards[bad]:.6g} at row {bad} is above zero; the task "
            "reward is in [-1, 0] with 0 only at success"
        )
    if np.any(rewards < MIN_REWARD):
        bad = int(np.argmax(rewards < MIN_REWARD))
        raise ValueError(
            f"reward {rewards[bad]:.6g} at row {bad} is below -1; the task "
            "reward is in [-1, 0]"
        )

    # Success ends the episode, so zero elsewhere means a bad recording.
    is_success = rewards == SUCCESS_REWARD
    is_success[stops - 1] = False
    if np.any(is_success):
        bad = int(np.argmax(is_success))
        raise ValueError(
            f"success reward at row {bad} is mid-episode; success ends an "
            "episode, so it can only appear on an episode's final row"
        )
