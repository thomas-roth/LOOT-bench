"""Tests for the BC gate's pure kernels.

The gate itself needs JAX and the qc checkout, so it runs from the qc venv and
never under pytest. Everything it decides *before* the fit is plain array work,
and every one of those decisions is silent when it is wrong: a leaked held-out
episode, a window that walks across an episode boundary, or a standardizer
dividing by a constant dimension all produce a number rather than an error, and
the number is what a run gets started on.
"""

import numpy as np
import pytest
from bc_gate import (
    INTERIOR_SKIP,
    build_chunks,
    episode_bounds,
    first_frames,
    interior_frames,
    load_columns,
    split_episodes,
    standardizer,
    training_starts,
)

# Three episodes of four rows each, ending on a terminal.
EPISODE_INDEX = np.repeat([0, 1, 2], 4)
TERMINALS = np.tile([0.0, 0.0, 0.0, 1.0], 3).astype(np.float32)
ACTIONS = np.arange(12, dtype=np.float32)[:, None] * np.ones((1, 3), dtype=np.float32)


class TestBuildChunks:
    def test_masks_across_a_boundary(self):
        """Steps at or after a terminal inside the window carry no gradient."""
        _, valid = build_chunks(ACTIONS, TERMINALS, horizon=3)

        # A window wholly inside episode 0.
        assert valid[0].tolist() == [1.0, 1.0, 1.0]
        # A window starting on episode 0's last row: the two steps after the
        # boundary belong to episode 1 and must be masked.
        assert valid[3].tolist() == [1.0, 0.0, 0.0]
        # The window after it starts in episode 1 and is clean again.
        assert valid[4].tolist() == [1.0, 1.0, 1.0]

    def test_targets_are_the_following_actions(self):
        chunk, _ = build_chunks(ACTIONS, TERMINALS, horizon=3)
        assert chunk.shape == (10, 3, 3)
        np.testing.assert_array_equal(chunk[2], ACTIONS[2:5])

    def test_rejects_a_recording_shorter_than_a_chunk(self):
        with pytest.raises(ValueError, match="cannot fill a chunk"):
            build_chunks(ACTIONS[:2], TERMINALS[:2], horizon=5)


class TestEpisodeBounds:
    def test_are_half_open_and_keyed_by_id(self):
        assert episode_bounds(EPISODE_INDEX) == {0: (0, 4), 1: (4, 8), 2: (8, 12)}

    def test_rejects_interleaved_rows(self):
        """Unsorted rows would give every episode a bound covering only one run."""
        with pytest.raises(ValueError, match="contiguous"):
            episode_bounds(np.array([0, 1, 0, 1]))


class TestSplitEpisodes:
    def test_is_a_partition(self):
        train, held = split_episodes(list(range(10)), holdout=3, seed=0)

        assert len(held) == 3
        assert len(train) == 7
        assert sorted(train + held) == list(range(10))

    def test_is_seeded(self):
        assert split_episodes(list(range(10)), 3, seed=0) == split_episodes(
            list(range(10)), 3, seed=0
        )

    def test_does_not_hold_out_everything(self):
        with pytest.raises(ValueError, match="still fit anything"):
            split_episodes([0, 1, 2], holdout=3, seed=0)


class TestTrainingStarts:
    def test_drops_every_window_touching_a_holdout(self):
        """Not only the windows that start in one. A window can cross into it."""
        starts = training_starts(EPISODE_INDEX, horizon=3, held=[1])

        windows = EPISODE_INDEX[starts[:, None] + np.arange(3)[None, :]]
        assert not (windows == 1).any()
        # Episode 0 keeps the two windows that end before row 4, and episode 2 the
        # two that begin at or after row 8.
        assert starts.tolist() == [0, 1, 8, 9]


class TestFirstFrames:
    def test_reads_the_opening_row(self):
        bounds = episode_bounds(EPISODE_INDEX)
        assert first_frames(bounds, [0, 2]).tolist() == [0, 8]


class TestInteriorFrames:
    def test_skips_the_start_pose_and_the_chunk_tail(self):
        """The opening frames look alike, and a full chunk has to fit."""
        bounds = {0: (0, 20)}
        rows = interior_frames(bounds, [0], horizon=5, per_episode=40)

        assert rows.min() == INTERIOR_SKIP
        assert rows.max() < 20 - 5

    def test_caps_each_episode(self):
        bounds = {0: (0, 500), 1: (500, 520)}
        rows = interior_frames(bounds, [0, 1], horizon=5, per_episode=10)

        assert (rows < 500).sum() <= 10
        assert (rows >= 500).sum() <= 10

    def test_needs_one_long_enough_episode(self):
        with pytest.raises(ValueError, match="no interior to read"):
            interior_frames({0: (0, 6)}, [0], horizon=5, per_episode=40)


class TestStandardizer:
    def test_reads_only_the_fitted_rows(self):
        observations = np.array([[0.0], [2.0], [100.0]], dtype=np.float32)
        mean, std = standardizer(observations, np.array([0, 1]))

        assert mean.tolist() == [1.0]
        assert std.tolist() == [1.0]

    def test_leaves_a_constant_dimension_alone(self):
        """The DK1 start pose is deterministic, so about 23 dimensions never move."""
        observations = np.array([[5.0, 1.0], [5.0, 3.0]], dtype=np.float32)
        _, std = standardizer(observations, np.array([0, 1]))

        assert std[0] == 1.0
        assert std[1] == pytest.approx(1.0)


class TestLoadColumns:
    def test_names_what_is_missing(self, tmp_path):
        path = tmp_path / "partial.npz"
        np.savez(path, observations=np.zeros((2, 3)), actions=np.zeros((2, 2)))

        with pytest.raises(ValueError, match="terminals, episode_index"):
            load_columns(path)
