import numpy as np

from g3a_bench.scripts.live_bridge.lerobot_to_qc import _compute_terminals_and_masks


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
