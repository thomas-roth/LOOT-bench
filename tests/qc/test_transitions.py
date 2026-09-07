"""Tests for the pure transition assembly.

Every assertion here is about episode structure, so none of it needs lerobot
or a simulator. The failure mode for all of it is silent: a wrong ``masks``
column trains an agent to like timeouts, and a wrong ``terminals`` column lets
a sampled sequence run out of one episode and into the next. Neither raises.
"""

import numpy as np
import pytest

from loot_bench.qc.transitions import QC_KEYS, assemble_transitions


def _episode(
    ep: int, obs_values: list[float], rewards: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One episode's columns. Observations are 2-wide, actions 3-wide."""
    n = len(obs_values)
    assert len(rewards) == n
    episode_index = np.full(n, ep, dtype=np.int64)
    observations = np.array([[v, -v] for v in obs_values], dtype=np.float32)
    actions = np.array([[v, v, v] for v in obs_values], dtype=np.float32) * 0.1
    return episode_index, observations, actions, np.array(rewards, dtype=np.float32)


def _concat(*episodes) -> dict[str, np.ndarray]:
    return {
        "episode_index": np.concatenate([e[0] for e in episodes]),
        "observations": np.concatenate([e[1] for e in episodes]),
        "actions": np.concatenate([e[2] for e in episodes]),
        "rewards": np.concatenate([e[3] for e in episodes]),
    }


# An episode that reached the goal: the last reward is exactly 0.0.
SUCCESS = _episode(0, [1.0, 2.0, 3.0], [-0.7, -0.4, 0.0])
# An episode that ran out of time: every reward is below zero.
TIMEOUT = _episode(1, [10.0, 20.0, 30.0], [-0.9, -0.8, -0.6])


class TestSuccessfulEpisode:
    def test_keeps_every_recorded_row(self):
        out = assemble_transitions(**_concat(SUCCESS))
        assert len(out["observations"]) == 3

    def test_final_transition_is_absorbing(self):
        """terminated -> do not bootstrap past it. This is the whole point of
        a success: the value after the goal is zero, not V(s')."""
        out = assemble_transitions(**_concat(SUCCESS))
        assert out["terminals"][-1] == 1.0
        assert out["masks"][-1] == 0.0

    def test_interior_transitions_bootstrap(self):
        out = assemble_transitions(**_concat(SUCCESS))
        assert list(out["masks"][:-1]) == [1.0, 1.0]
        assert list(out["terminals"][:-1]) == [0.0, 0.0]

    def test_next_observations_are_the_following_row(self):
        out = assemble_transitions(**_concat(SUCCESS))
        assert np.array_equal(out["next_observations"][0], out["observations"][1])
        assert np.array_equal(out["next_observations"][1], out["observations"][2])


class TestTimeoutEpisode:
    """A timeout is not an absorbing state, and its last row is unusable.

    The recorder stores (obs_t, action_t, reward_t) and never the state that
    followed the final action, so the last recorded row of a non-absorbing
    episode has no next observation to bootstrap from. It is dropped rather
    than zero-filled: with masks=1 the critic would read those zeros.
    """

    def test_drops_the_unusable_final_row(self):
        out = assemble_transitions(**_concat(TIMEOUT))
        assert len(out["observations"]) == 2

    def test_last_kept_row_bootstraps_into_a_real_state(self):
        out = assemble_transitions(**_concat(TIMEOUT))
        assert out["terminals"][-1] == 1.0, "still an episode boundary"
        assert out["masks"][-1] == 1.0, "but not absorbing"
        # The dropped row's observation is exactly what it bootstraps into.
        assert np.array_equal(out["next_observations"][-1], TIMEOUT[1][2])

    def test_no_next_observation_is_left_zeroed_where_it_is_read(self):
        out = assemble_transitions(**_concat(TIMEOUT))
        read = out["masks"] == 1.0
        assert not np.any(np.all(out["next_observations"][read] == 0.0, axis=1))


class TestEpisodeBoundaries:
    def test_next_observations_never_cross_into_the_next_episode(self):
        out = assemble_transitions(**_concat(SUCCESS, TIMEOUT))
        # Row 2 is the success terminal; row 3 starts the timeout episode.
        boundary = int(np.flatnonzero(out["terminals"] > 0)[0])
        assert not np.array_equal(
            out["next_observations"][boundary], out["observations"][boundary + 1]
        )

    def test_every_episode_end_is_marked_terminal(self):
        """qc's sample_sequence keys `valid` off terminals alone. An unmarked
        boundary lets a sampled window walk into the next episode."""
        out = assemble_transitions(**_concat(SUCCESS, TIMEOUT))
        assert int(out["terminals"].sum()) == 2

    def test_last_row_of_the_dataset_is_terminal(self):
        """qc's add_history asserts terminal_locs[-1] == size - 1."""
        out = assemble_transitions(**_concat(SUCCESS, TIMEOUT))
        assert out["terminals"][-1] == 1.0

    def test_masks_are_zero_only_on_success(self):
        out = assemble_transitions(**_concat(SUCCESS, TIMEOUT))
        assert np.array_equal(out["masks"] == 0.0, out["rewards"] == 0.0)

    def test_a_single_row_timeout_episode_contributes_nothing(self):
        short = _episode(2, [5.0], [-0.5])
        out = assemble_transitions(**_concat(SUCCESS, short))
        assert len(out["observations"]) == 3
        assert not np.any(out["episode_index"] == 2)

    def test_interleaved_episodes_are_rejected(self):
        """Rows must arrive grouped by episode; the caller sorts."""
        cols = _concat(SUCCESS, TIMEOUT)
        cols["episode_index"] = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
        with pytest.raises(ValueError, match="contiguous"):
            assemble_transitions(**cols)


class TestRewardConvention:
    """The dataset has to agree with tasks/common/rewards.py, and a recording
    made before that convention landed has to fail loudly rather than train."""

    def test_positive_reward_is_rejected(self):
        cols = _concat(_episode(0, [1.0, 2.0], [0.5, 0.0]))
        with pytest.raises(ValueError, match="above zero"):
            assemble_transitions(**cols)

    def test_reward_below_minus_one_is_rejected(self):
        cols = _concat(_episode(0, [1.0, 2.0], [-1.5, 0.0]))
        with pytest.raises(ValueError, match="below -1"):
            assemble_transitions(**cols)

    def test_success_before_the_final_row_is_rejected(self):
        """Success terminates the episode, so a zero anywhere else means the
        recording and the task disagree about what ends an episode."""
        cols = _concat(_episode(0, [1.0, 2.0, 3.0], [0.0, -0.4, 0.0]))
        with pytest.raises(ValueError, match="mid-episode"):
            assemble_transitions(**cols)


class TestOutputContract:
    def test_publishes_every_key_qc_requires(self):
        out = assemble_transitions(**_concat(SUCCESS, TIMEOUT))
        assert set(out) >= QC_KEYS

    def test_scalar_columns_are_one_dimensional(self):
        """Matching qc's online loop, which pushes python floats into these
        buffers. A (N, 1) column would broadcast instead of failing."""
        out = assemble_transitions(**_concat(SUCCESS, TIMEOUT))
        n = len(out["observations"])
        for key in ("rewards", "terminals", "masks"):
            assert out[key].shape == (n,)
            assert out[key].dtype == np.float32

    def test_observation_and_action_widths_are_preserved(self):
        out = assemble_transitions(**_concat(SUCCESS, TIMEOUT))
        assert out["observations"].shape[1:] == (2,)
        assert out["next_observations"].shape[1:] == (2,)
        assert out["actions"].shape[1:] == (3,)
