"""Unit tests for ``g3a_bench/qc/episode.py``."""

from g3a_bench.qc.episode import EpisodeStats


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

        A success gate can unlatch -- a block settles a hair and the at-rest
        test stops holding -- while the episode ends a step later on the
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

    def test_does_not_publish_gripper_contact(self):
        """Publishing it in nested form alone is a KeyError in qc's evaluate."""
        assert "gripper_contact" not in EpisodeStats().info()

    def test_fresh_episode_is_empty(self):
        info = EpisodeStats().info()
        assert info == {
            "success": 0.0,
            "episode_length": 0.0,
            "episode_return": 0.0,
        }
