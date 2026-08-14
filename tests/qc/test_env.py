"""Contract tests for the qc gym adapter, run against a real task.

These build a simulator, so they are slower than the pure-kernel tests. They
earn it: every assertion here is a promise ``qc`` relies on and cannot check
for itself, and the failure mode when one breaks is a training curve that is
flat rather than an exception.
"""

import numpy as np
import pytest

from g3a_bench.qc import G3aEnv, is_g3a_task

TASK_ID = "G3A-Stack-Low-Dk1_uni"

ROBOT_TERMS = ("joint_pos", "joint_vel", "ee_pose")
"""Observation terms that are a property of the robot alone."""


@pytest.fixture(scope="module")
def env():
    e = G3aEnv(TASK_ID, device="cpu", seed=0, start_pose_noise=0.05)
    yield e
    e.close()


def _term_slices(env: G3aEnv) -> dict[str, slice]:
    """Map each actor-observation term to its slice of the flat vector.

    Derived rather than hardcoded: a task that gains a term would otherwise
    shift every index and turn these tests into assertions about the wrong
    numbers, silently.
    """
    manager = env.unwrapped_env.observation_manager
    names = manager.active_terms["actor"]
    dims = manager.group_obs_term_dim["actor"]
    out: dict[str, slice] = {}
    start = 0
    for name, dim in zip(names, dims, strict=True):
        width = int(np.prod(dim))
        out[name] = slice(start, start + width)
        start += width
    return out


def _split(env: G3aEnv, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split observations into (robot columns, scene columns)."""
    slices = _term_slices(env)
    robot = np.concatenate([rows[:, slices[t]] for t in ROBOT_TERMS], axis=1)
    scene = rows[:, slices["block_poses"]]
    return robot, scene


def _peak_to_peak(block: np.ndarray) -> float:
    """Largest per-column range.

    Never the standard deviation. The observations are float32, and
    ``np.std`` over bit-identical samples returns ~1e-6 because the
    accumulated mean is an ulp off -- so a non-zero std does not prove
    variation, and a reset that randomizes nothing would pass a std test.
    The range is exact under any accumulation order.
    """
    return float((block.max(axis=0) - block.min(axis=0)).max())


class TestTaskPrefix:
    def test_recognizes_g3a_tasks(self):
        assert is_g3a_task(TASK_ID)
        assert is_g3a_task("G3A-Ball-Throw-U-Dk1_uni")

    def test_rejects_other_env_names(self):
        assert not is_g3a_task("antmaze-large-diverse-v2")
        assert not is_g3a_task("cube-triple-play-singletask-task1-v0")


class TestSpaces:
    def test_action_space_is_the_unit_box(self, env):
        assert env.action_space.shape == (7,)
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)

    def test_observation_space_matches_what_reset_returns(self, env):
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32


class TestStepContract:
    def test_step_returns_the_gym_five_tuple(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.zeros(7, np.float32))
        assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reward_obeys_the_convention(self, env):
        """r in [-1, 0]. qc's --sparse depends on nothing else touching 0."""
        env.reset()
        for _ in range(5):
            _, reward, *_ = env.step(np.zeros(7, np.float32))
            assert -1.0 <= reward <= 0.0

    def test_actions_are_clipped_not_rejected(self, env):
        """qc clips to [-1, 1] itself, but a stray value must not raise."""
        env.reset()
        env.step(np.full(7, 5.0, np.float32))
        env.step(np.full(7, -5.0, np.float32))

    def test_info_carries_a_scalar_success(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.zeros(7, np.float32))
        assert isinstance(info["success"], float)
        assert info["success"] in (0.0, 1.0)

    def test_info_omits_gripper_contact(self, env):
        """Emitting it nested-only is a KeyError inside qc's evaluate()."""
        env.reset()
        _, _, _, _, info = env.step(np.zeros(7, np.float32))
        assert "gripper_contact" not in info

    def test_observation_changes_as_the_arm_moves(self, env):
        """Guards against handing back a stale or copied-once buffer."""
        first, _ = env.reset()
        obs = first
        for _ in range(5):
            obs, *_ = env.step(np.full(7, 0.3, np.float32))
        assert not np.allclose(first, obs)

    def test_returned_observation_is_not_a_view_on_sim_memory(self, env):
        env.reset()
        obs_a, *_ = env.step(np.zeros(7, np.float32))
        snapshot = obs_a.copy()
        env.step(np.full(7, 0.5, np.float32))
        assert np.array_equal(obs_a, snapshot), "earlier obs was mutated in place"


class TestEpisodeBoundaries:
    def test_does_not_auto_reset(self):
        """The promise qc's evaluation loop rests on.

        With auto-reset on, the observation returned alongside ``done`` is the
        *next* episode's first state, which would corrupt the last transition
        pushed to the replay buffer and the info the success metric reads.
        """
        env = G3aEnv(TASK_ID, device="cpu", seed=1, start_pose_noise=0.0)
        try:
            assert env.unwrapped_env.cfg.auto_reset is False
        finally:
            env.close()

    def test_reset_clears_episode_stats(self, env):
        env.reset()
        for _ in range(3):
            env.step(np.zeros(7, np.float32))
        _, info = env.reset()
        assert info["episode_length"] == 0.0
        assert info["episode_return"] == 0.0
        assert info["success"] == 0.0

    def test_episode_length_counts_steps(self, env):
        env.reset()
        for i in range(4):
            _, _, _, _, info = env.step(np.zeros(7, np.float32))
            assert info["episode_length"] == float(i + 1)

    def test_zero_start_pose_noise_repeats_the_same_start(self):
        """Replay verification needs a bit-identical start."""
        env = G3aEnv(TASK_ID, device="cpu", seed=3, start_pose_noise=0.0)
        try:
            rows = np.stack([env.reset()[0] for _ in range(8)])
            robot, scene = _split(env, rows)
            assert _peak_to_peak(robot) == 0.0
            # The control that makes the line above mean something: the scene
            # is still being scattered, so the measurement demonstrably would
            # have caught robot variation had there been any.
            assert _peak_to_peak(scene) > 0.0
        finally:
            env.close()

    def test_nonzero_start_pose_noise_varies_the_start(self):
        env = G3aEnv(TASK_ID, device="cpu", seed=4, start_pose_noise=0.05)
        try:
            rows = np.stack([env.reset()[0] for _ in range(8)])
            robot, scene = _split(env, rows)
            assert _peak_to_peak(robot) > 1e-3
            assert _peak_to_peak(scene) > 0.0
        finally:
            env.close()
