"""Contract tests for the qc gym adapter, run against a real task.

These build a simulator, so they are slower than the pure-kernel tests. They
earn it: every assertion here is a promise ``qc`` relies on and cannot check
for itself, and the failure mode when one breaks is a training curve that is
flat rather than an exception.
"""

import numpy as np
import pytest

from loot_bench.qc import LootEnv, is_loot_task
from loot_bench.qc.dataset import STAMP_KEYS
from loot_bench.qc.episode import CONTACT_THRESHOLD
from loot_bench.tasks.common.actions import JOINT_POS_ACTION_NAME

TASK_ID = "LOOT-Stack-1Pi-Dk1"

PLACE_TASK_ID = "LOOT-Place-RandS-RandT-Dk1"
"""The one category that declares the object-filtered contact sensor.

Place scores exactly one free object, so "contact with the object" is a
well-defined geom match there. Stack has three blocks and declares no such
sensor, which the tests below use as the negative control.
"""

ROBOT_TERMS = ("joint_pos", "joint_vel", "ee_pose", "pad_contact_force")
"""Observation terms that are a property of the robot alone."""


@pytest.fixture(scope="module")
def env():
    e = LootEnv(TASK_ID, device="cpu", seed=0, start_pose_noise=0.05)
    yield e
    e.close()


@pytest.fixture(scope="module")
def place_env():
    e = LootEnv(PLACE_TASK_ID, device="cpu", seed=0, start_pose_noise=0.05)
    yield e
    e.close()


def _term_slices(env: LootEnv) -> dict[str, slice]:
    """Map each actor-observation term to its slice of the flat vector.

    Derived rather than hardcoded: a task that gains a term would otherwise
    shift every index and turn these tests into assertions about the wrong
    numbers, silently.
    """
    manager = env.mjlab_env.observation_manager
    names = manager.active_terms["actor"]
    dims = manager.group_obs_term_dim["actor"]
    out: dict[str, slice] = {}
    start = 0
    for name, dim in zip(names, dims, strict=True):
        width = int(np.prod(dim))
        out[name] = slice(start, start + width)
        start += width
    return out


def _split(env: LootEnv, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split observations into (robot columns, scene columns)."""
    slices = _term_slices(env)
    robot = np.concatenate([rows[:, slices[t]] for t in ROBOT_TERMS], axis=1)
    scene = rows[:, slices["block_poses"]]
    return robot, scene


def _hold_action(env: LootEnv) -> np.ndarray:
    """The action that commands the robot's *current* joint pose.

    Inverted from the action term's own scale and offset rather than from a
    second copy of the mapping, so it stays correct for any robot whose limits
    differ (see ``tasks/common/actions.py``). Stepping this repeatedly parks
    the arm where the reset left it, which is what lets a test move the gripper
    alone.
    """
    term = env.mjlab_env.action_manager.get_term(JOINT_POS_ACTION_NAME)
    pos = term._entity.data.joint_pos[:, term._target_ids]
    action = (pos - term._offset) / term._scale
    return action[0].detach().cpu().numpy().astype(np.float32)


def _gripper_index(env: LootEnv) -> int:
    """Which action column drives the gripper, read off the term's joint names."""
    names = list(
        env.mjlab_env.action_manager.get_term(JOINT_POS_ACTION_NAME)._target_names
    )
    return next(i for i, name in enumerate(names) if "gripper" in name)


def _peak_to_peak(block: np.ndarray) -> float:
    """Largest per-column range.

    Never the standard deviation. The observations are float32, and
    ``np.std`` over bit-identical samples returns ~1e-6 because the
    accumulated mean is an ulp off. So a non-zero std does not prove
    variation, and a reset that randomizes nothing would pass a std test.
    The range is exact under any accumulation order.
    """
    return float((block.max(axis=0) - block.min(axis=0)).max())


class TestTaskPrefix:
    def test_recognizes_loot_tasks(self):
        assert is_loot_task(TASK_ID)
        assert is_loot_task("LOOT-Ball-Throw-U-Dk1")

    def test_rejects_other_env_names(self):
        assert not is_loot_task("antmaze-large-diverse-v2")
        assert not is_loot_task("cube-triple-play-singletask-task1-v0")


class TestSpaces:
    def test_action_space_is_the_unit_box(self, env):
        assert env.action_space.shape == (7,)
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)

    def test_action_convention_reports_every_stamp_key(self, env):
        # Guards convert.action_convention against drifting from
        # dataset.STAMP_KEYS: both build the same dict shape from different
        # sources, so nothing but a test catches a field added to one side.
        assert set(env.action_convention()) == STAMP_KEYS

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

    def test_info_carries_a_flat_pad_contact_force(self, env):
        """qc counts grips out of this key; nested-only would be a KeyError.

        It is the only contact evidence an evaluation gets, so a policy that
        scores zero can still be told apart from one that never touched
        anything.
        """
        env.reset()
        _, _, _, _, info = env.step(np.zeros(7, np.float32))
        assert isinstance(info["pad_contact_force"], float)
        assert info["pad_contact_force"] >= 0.0

    def test_reset_info_carries_pad_contact_force_too(self, env):
        """A key that appears only on some steps is harder to trust."""
        _, info = env.reset()
        assert "pad_contact_force" in info

    def test_a_task_without_the_sensor_omits_the_object_keys(self, env):
        """The negative control for :class:`TestObjectContact`.

        Stack scores three blocks, so it declares no object sensor. Reporting
        0.0 there would read as "never touched a block" and mean "never asked".
        """
        env.reset()
        _, _, _, _, info = env.step(np.zeros(7, np.float32))
        assert "object_contact_force" not in info
        assert "object_touched" not in info

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


class TestObjectContact:
    """The cube-filtered sensor: the only key that answers "touched the cube".

    ``pad_contact_force`` cannot answer it. That sensor has no secondary match,
    so it reports the pads meeting anything -- the table, another block -- not
    just the object ``qc`` scores. It no longer reports a gripper closed on
    empty air: ``dk1.xml`` excludes the finger-pad pair.
    """

    def test_reset_info_carries_the_object_keys(self, place_env):
        _, info = place_env.reset()
        assert isinstance(info["object_contact_force"], float)
        assert info["object_contact_steps"] == 0.0
        assert info["object_touched"] == 0.0

    def test_a_gripper_closed_on_air_reports_no_contact(self, place_env):
        """Neither sensor may fire on an empty grasp.

        The arm holds its reset pose, which parks the gripper about 0.2 m above
        the table while the cube lies on it, so the pads cannot reach the cube
        whatever the scatter sampled. Closing there used to load the two pads
        against each other and report grip force with nothing held, which made
        ``pad_contact_force`` count the very thing a failing policy does all
        episode. ``dk1.xml`` excludes the pair, so both keys stay at zero.
        """
        place_env.reset()
        action = _hold_action(place_env)
        action[_gripper_index(place_env)] = -1.0  # Fully closed.
        info: dict[str, float] = {}
        for _ in range(60):
            _, _, _, _, info = place_env.step(action)
        assert info["pad_contact_force"] <= CONTACT_THRESHOLD, (
            "an empty gripper reported grip force; the finger-pad <exclude> in "
            "dk1.xml is gone, or this robot's pads meet somewhere else"
        )
        assert info["object_contact_force"] == 0.0
        assert info["object_contact_steps"] == 0.0
        assert info["object_touched"] == 0.0


class TestEpisodeBoundaries:
    def test_does_not_auto_reset(self):
        """The promise qc's evaluation loop rests on.

        With auto-reset on, the observation returned alongside ``done`` is the
        *next* episode's first state, which would corrupt the last transition
        pushed to the replay buffer and the info the success metric reads.
        """
        env = LootEnv(TASK_ID, device="cpu", seed=1, start_pose_noise=0.0)
        try:
            assert env.mjlab_env.cfg.auto_reset is False
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
        env = LootEnv(TASK_ID, device="cpu", seed=3, start_pose_noise=0.0)
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
        env = LootEnv(TASK_ID, device="cpu", seed=4, start_pose_noise=0.05)
        try:
            rows = np.stack([env.reset()[0] for _ in range(8)])
            robot, scene = _split(env, rows)
            assert _peak_to_peak(robot) > 1e-3
            assert _peak_to_peak(scene) > 0.0
        finally:
            env.close()
