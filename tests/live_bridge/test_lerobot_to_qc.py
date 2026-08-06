import numpy as np
import pytest

from g3a_bench.scripts.live_bridge.lerobot_to_qc import (
    _check_absorbing_frames_scored_success,
    _check_recorded_reward_range,
    _compute_terminals_and_masks,
)


class TestCheckAbsorbingFramesScoredSuccess:
    """``masks`` comes from the operator's keypress, ``rewards`` from the env.

    Nothing else cross-checks them, and a ``masks == 0`` frame is used as the
    Q-target directly (no bootstrap) — so a frame the operator called a success
    but the env never scored would teach the critic the goal is worth
    ``-reward_max``.
    """

    @staticmethod
    def _episodes(n_frames: int) -> np.ndarray:
        return np.zeros(n_frames, dtype=np.int64)

    def test_accepts_a_terminal_the_env_scored(self):
        rewards = np.array([-2.0, -1.5, 0.0], dtype=np.float32)
        masks = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        _check_absorbing_frames_scored_success(rewards, masks, self._episodes(3))

    def test_accepts_stacks_in_band_undershoot(self):
        """Stack's success band lets progress fall up to 0.083 short of 1."""
        rewards = np.array([-2.0, -1.5, -0.083], dtype=np.float32)
        masks = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        _check_absorbing_frames_scored_success(rewards, masks, self._episodes(3))

    def test_rejects_a_terminal_the_env_never_scored(self):
        """Operator pressed the success key before the gate latched."""
        rewards = np.array([-2.0, -1.5, -1.4], dtype=np.float32)
        masks = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="never fired"):
            _check_absorbing_frames_scored_success(rewards, masks, self._episodes(3))

    def test_names_the_offending_episodes(self):
        rewards = np.array([-1.0, 0.0, -1.0, -1.0], dtype=np.float32)
        masks = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        episodes = np.array([0, 0, 7, 7], dtype=np.int64)
        with pytest.raises(ValueError, match=r"episode\(s\) \[7\]"):
            _check_absorbing_frames_scored_success(rewards, masks, episodes)

    def test_no_absorbing_frames_is_a_no_op(self):
        """All-failure datasets keep masks == 1 everywhere."""
        rewards = np.array([-2.0, -2.0], dtype=np.float32)
        masks = np.ones(2, dtype=np.float32)
        _check_absorbing_frames_scored_success(rewards, masks, self._episodes(2))


class TestCheckRecordedRewardRange:
    """Rewards are passed through untouched, so the recording must already be
    on the task's convention: ``progress + success - reward_max`` in
    ``[-reward_max, 0]``. Neither failure is fixable by rescaling."""

    def test_accepts_a_recording_on_the_current_convention(self):
        rewards = np.array([-2.0, -1.7, -1.0, 0.0], dtype=np.float32)
        _check_recorded_reward_range(rewards, 2.0, "G3A-Stack-Low-Dk1_uni")

    def test_rejects_positive_rewards(self):
        """Pre-convention recordings are the raw term sum scaled by step_dt."""
        rewards = np.array([0.0029, 0.0399], dtype=np.float32)
        with pytest.raises(ValueError, match="predates the negative reward"):
            _check_recorded_reward_range(rewards, 1.0, "G3A-Place-Easy-Dk1_uni")

    def test_rejects_rewards_below_the_floor(self):
        """The task had more reward terms when this was recorded than it does now."""
        rewards = np.array([-1.9, -1.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="more reward terms"):
            _check_recorded_reward_range(rewards, 1.0, "G3A-Place-Easy-Dk1_uni")

    def test_empty_recording_is_a_no_op(self):
        _check_recorded_reward_range(np.array([], dtype=np.float32), 2.0, "t")


class TestComputeTerminalsAndMasks:
    def test_success_episode_gets_zero_mask_at_last_frame(self):
        # One 3-frame episode ending in success.
        episode_idx = np.array([0, 0, 0])
        success = np.array([True, True, True])

        terminals, masks = _compute_terminals_and_masks(episode_idx, success)

        np.testing.assert_array_equal(terminals, [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(masks, [1.0, 1.0, 0.0])

    def test_timeout_episode_keeps_mask_one_at_last_frame(self):
        # One 3-frame episode that ran out of time / was aborted (not success).
        episode_idx = np.array([0, 0, 0])
        success = np.array([False, False, False])

        terminals, masks = _compute_terminals_and_masks(episode_idx, success)

        # terminals still marks the trajectory boundary...
        np.testing.assert_array_equal(terminals, [0.0, 0.0, 1.0])
        # ...but masks must NOT be zeroed, since no true terminal state was reached.
        np.testing.assert_array_equal(masks, [1.0, 1.0, 1.0])

    def test_mixed_episodes_only_success_gets_zero_mask(self):
        # Episode 0 (2 frames): times out. Episode 1 (2 frames): succeeds.
        episode_idx = np.array([0, 0, 1, 1])
        success = np.array([False, False, True, True])

        terminals, masks = _compute_terminals_and_masks(episode_idx, success)

        np.testing.assert_array_equal(terminals, [0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(masks, [1.0, 1.0, 1.0, 0.0])

    def test_all_success_matches_old_buggy_behavior_exactly(self):
        # Regression guard: when every episode succeeds (the recorder writes
        # success/failure episodes to *separate* directories, so a dataset built
        # from only a success-dir has this property for every episode), the new
        # masks must equal the old (masks = 1 - terminals) computation exactly —
        # i.e. this fix changes nothing for success-only datasets.
        episode_idx = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        success = np.array([True] * 9)

        terminals, masks = _compute_terminals_and_masks(episode_idx, success)

        np.testing.assert_array_equal(masks, 1.0 - terminals)

    def test_single_frame_episode(self):
        episode_idx = np.array([0])
        success = np.array([True])

        terminals, masks = _compute_terminals_and_masks(episode_idx, success)

        np.testing.assert_array_equal(terminals, [1.0])
        np.testing.assert_array_equal(masks, [0.0])

    def test_output_dtypes(self):
        episode_idx = np.array([0, 0, 1, 1])
        success = np.array([True, True, False, False])

        terminals, masks = _compute_terminals_and_masks(episode_idx, success)

        assert terminals.dtype == np.float32
        assert masks.dtype == np.float32
