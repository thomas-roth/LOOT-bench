"""Unit tests for ``loot_bench/qc/episode.py``."""

from loot_bench.qc.episode import CONTACT_THRESHOLD, EpisodeStats


class TestEpisodeStats:
    def test_counts_and_sums(self):
        stats = EpisodeStats()
        for r in (-1.0, -1.0, -0.5):
            stats.record(r)
        assert stats.length == 3
        assert stats.info()["episode_length"] == 3.0
        assert abs(stats.info()["episode_return"] - (-2.5)) < 1e-9

    def test_no_success_before_zero_reward(self):
        stats = EpisodeStats()
        for r in (-1.0, -0.5, -1e-3):
            stats.record(r)
        assert stats.info()["success"] == 0.0

    def test_zero_reward_is_success(self):
        stats = EpisodeStats()
        stats.record(-1.0)
        stats.record(0.0)
        assert stats.info()["success"] == 1.0

    def test_success_latches(self):
        """The case that makes the latch worth having.

        A success gate can unlatch: a block settles a hair and the at-rest
        test stops holding, while the episode ends a step later on the
        termination that already fired. Reading the final step alone would
        score that episode a failure.
        """
        stats = EpisodeStats()
        stats.record(-1.0)
        stats.record(0.0)
        stats.record(-0.2)
        assert stats.info()["success"] == 1.0

    def test_info_keys_are_scalars(self):
        """qc averages flatten(info) across episodes, so no arrays or dicts."""
        stats = EpisodeStats()
        stats.record(-1.0)
        assert all(isinstance(v, float) for v in stats.info().values())

    def test_publishes_pad_contact_force_flat_and_scalar(self):
        """qc tests the flat key and then indexes it. Nested alone is a KeyError."""
        stats = EpisodeStats()
        stats.record(-1.0, 0.42)
        info = stats.info()
        assert info["pad_contact_force"] == 0.42
        assert isinstance(info["pad_contact_force"], float)

    def test_pad_contact_force_carries_the_latest_step(self):
        """qc thresholds it per step to count grips, so it must not accumulate."""
        stats = EpisodeStats()
        stats.record(-1.0, 3.0)
        stats.record(-1.0, 0.0)
        assert stats.info()["pad_contact_force"] == 0.0

    def test_omits_pad_contact_force_without_a_sensor(self):
        """A task that cannot measure contact reports nothing, not 0.0."""
        stats = EpisodeStats()
        stats.record(-1.0)
        assert "pad_contact_force" not in stats.info()

    def test_object_contact_steps_accumulate(self):
        """The per-episode total, which is what survives qc's averaging."""
        stats = EpisodeStats()
        stats.record(-1.0, object_contact_force=2.0)
        stats.record(-1.0, object_contact_force=0.0)
        stats.record(-1.0, object_contact_force=2.0)
        info = stats.info()
        assert info["object_contact_force"] == 2.0
        assert info["object_contact_steps"] == 2.0
        assert info["object_touched"] == 1.0

    def test_object_contact_below_threshold_does_not_count(self):
        """A brush is not a touch: the threshold matches qc's own."""
        stats = EpisodeStats()
        stats.record(-1.0, object_contact_force=CONTACT_THRESHOLD)
        info = stats.info()
        assert info["object_contact_steps"] == 0.0
        assert info["object_touched"] == 0.0

    def test_self_contact_does_not_reach_the_object_keys(self):
        """The whole point: a gripper closed on air loads the pads, not the cube.

        `grip_contact` has no secondary filter and reports the pads pressing
        on each other. `object_contact_force` is filtered to the cube, so the same
        step reads a hard zero there.
        """
        stats = EpisodeStats()
        stats.record(-1.0, pad_contact_force=2.5, object_contact_force=0.0)
        info = stats.info()
        assert info["pad_contact_force"] == 2.5
        assert info["object_touched"] == 0.0

    def test_omits_object_contact_without_a_sensor(self):
        """A category with no single scored object publishes none of the three."""
        stats = EpisodeStats()
        stats.record(-1.0, pad_contact_force=1.0)
        info = stats.info()
        assert "object_contact_force" not in info
        assert "object_contact_steps" not in info
        assert "object_touched" not in info

    def test_fresh_episode_is_empty(self):
        info = EpisodeStats().info()
        assert info == {
            "success": 0.0,
            "episode_length": 0.0,
            "episode_return": 0.0,
        }
