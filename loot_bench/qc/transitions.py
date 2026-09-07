"""Recorded rows -> the transition arrays ``qc`` trains on.

A recording is one row per timestep: ``(obs_t, action_t, reward_t)``. Offline
RL wants ``(s, a, r, s', done)``, which is not the same shape of thing:
``s'`` is the *next* row, and the last row of an episode has no next row.

Everything here is pure array work, so it runs without a simulator and without
lerobot. :func:`assemble_transitions` is the whole contract; ``convert.py``
does the file handling around it.

Two columns carry the meaning, and they are **not** the same column
(``ADR-010``):

* ``terminals`` marks an **episode boundary**.
* ``masks`` marks whether to **bootstrap** past the transition (``1``) or not
  (``0``).

Every episode end is a boundary; only a success is absorbing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

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
"""The floor of the reward convention (``tasks/common/rewards.py``).

Defined here instead of in ``tasks/common/rewards.py`` because torch is not
needed here.
"""

SUCCESS_REWARD = 0.0
"""The ceiling of the reward convention (``tasks/common/rewards.py``).

Defined here instead of in ``tasks/common/rewards.py`` because torch is not
needed here.
"""


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

    row_count = len(episode_index)
    lengths = {len(observations), len(actions), len(rewards), row_count}
    if len(lengths) != 1:
        raise ValueError(
            "episode_index, observations, actions and rewards must have the "
            f"same length, got {sorted(lengths)}"
        )
    if row_count == 0:
        raise ValueError("no rows to convert")

    starts, stops = _episode_bounds(episode_index)
    _check_rewards(rewards, stops)

    # Successor is row i+1; rows running off an episode end are dropped, never
    # redirected across a boundary.
    successor = np.minimum(np.arange(row_count) + 1, row_count - 1)
    keep = np.zeros(row_count, dtype=bool)
    terminals = np.zeros(row_count, dtype=np.float32)
    masks = np.ones(row_count, dtype=np.float32)

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
        "episode_index": episode_index[keep],  # Metadata, see QC_KEYS.
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
        first_bad_row = int(np.argmax(rewards > SUCCESS_REWARD))
        raise ValueError(
            f"reward {rewards[first_bad_row]:.6g} at row {first_bad_row} is "
            "above zero; the task reward is in [-1, 0] with 0 only at success"
        )
    if np.any(rewards < MIN_REWARD):
        first_bad_row = int(np.argmax(rewards < MIN_REWARD))
        raise ValueError(
            f"reward {rewards[first_bad_row]:.6g} at row {first_bad_row} is "
            "below -1; the task reward is in [-1, 0]"
        )

    # Success ends the episode, so zero elsewhere means a bad recording.
    is_success = rewards == SUCCESS_REWARD
    is_success[stops - 1] = False
    if np.any(is_success):
        first_bad_row = int(np.argmax(is_success))
        raise ValueError(
            f"success reward at row {first_bad_row} is mid-episode; success "
            "ends an episode, so it can only appear on an episode's final row"
        )
