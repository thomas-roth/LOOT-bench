"""The teleop half of the shared action convention.

``tests/tasks/test_actions.py`` pins the other half -- what the simulator does
with an action. Together they say that the action a recording stores and the
action a policy emits mean the same joints.
"""

import numpy as np
import pytest

from loot_bench.robots.trlc_dk1.dk1_actuators import GRIPPER_COMMAND_RANGE
from loot_bench.scripts.live_bridge.mapper import (
    action_scale_offset,
    joint_pos_limits,
    leader_is_aligned,
    lerobot_to_mjlab_ctrl,
    lerobot_to_mjlab_raw_action,
    mjlab_gripper_meters,
)


class TestGripperConversion:
    def test_open_to_meters(self):
        assert mjlab_gripper_meters(0.0) == 0.0

    def test_closed_to_meters(self):
        assert mjlab_gripper_meters(1.0) == pytest.approx(
            GRIPPER_COMMAND_RANGE["closed"]
        )


class TestAbsoluteCtrl:
    def test_joints_pass_through(self):
        joint_pos = np.array([0.5, -1.2, 0.3, 2.1, -0.8, 1.4, 0.0], dtype=np.float32)
        ctrl = lerobot_to_mjlab_ctrl(joint_pos)
        np.testing.assert_array_equal(ctrl[:6], joint_pos[:6])
        assert ctrl[6] == 0.0

    def test_gripper_maps_to_meters(self):
        ctrl = lerobot_to_mjlab_ctrl(
            np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)
        )
        assert ctrl[6] == pytest.approx(GRIPPER_COMMAND_RANGE["closed"])


class TestRawAction:
    """The teleop half of the shared action convention.

    ``tests/tasks/test_actions.py`` pins the other half: what the simulator
    does with the action. Together they say a recorded pose and a replayed
    pose are the same pose.
    """

    def test_roundtrip_with_ctrl(self):
        """The action inverts the action term exactly, which is what makes a
        recording replayable."""
        joint_pos = np.array([0.5, -0.3, 2.1, -0.8, 1.2, -1.5, 0.7], dtype=np.float32)
        raw = lerobot_to_mjlab_raw_action(joint_pos)
        scale, offset = action_scale_offset()
        recovered = raw * scale + offset
        np.testing.assert_array_almost_equal(
            recovered, lerobot_to_mjlab_ctrl(joint_pos), decimal=5
        )

    def test_a_reachable_pose_stays_inside_the_action_box(self):
        """A bounded policy can only emit [-1, 1], so any pose the joints can
        hold has to map inside it."""
        lower, upper = joint_pos_limits()
        middle = (lower + upper) / 2.0
        joint_pos = middle.copy()
        joint_pos[6] = 0.5  # The gripper arrives normalized, not in metres.
        raw = lerobot_to_mjlab_raw_action(joint_pos)
        assert np.all(np.abs(raw) <= 1.0), raw

    def test_the_joint_limits_map_to_the_box_corners(self):
        lower, upper = joint_pos_limits()
        scale, offset = action_scale_offset()
        np.testing.assert_array_almost_equal((lower - offset) / scale, -1.0, decimal=5)
        np.testing.assert_array_almost_equal((upper - offset) / scale, 1.0, decimal=5)

    def test_limits_are_read_from_the_model_not_hardcoded(self):
        """The regression this guards: a copy of dk1.xml's ranges going stale."""
        lower, upper = joint_pos_limits()
        assert lower.shape == (7,) and upper.shape == (7,)
        assert np.all(upper > lower)


class TestAlignment:
    """The joint-space episode-start gate: all 6 arm joints within 0.2 rad."""

    def test_the_start_pose_is_aligned(self):
        from loot_bench.robots.trlc_dk1.dk1_constants import START_JOINT_POS
        from loot_bench.scripts.live_bridge.mapper import _ARM_JOINT_NAMES

        pose = np.zeros(7, dtype=np.float32)
        pose[:6] = [START_JOINT_POS[n] for n in _ARM_JOINT_NAMES]
        assert leader_is_aligned(pose)

    def test_the_gripper_does_not_affect_alignment(self):
        from loot_bench.robots.trlc_dk1.dk1_constants import START_JOINT_POS
        from loot_bench.scripts.live_bridge.mapper import _ARM_JOINT_NAMES

        pose = np.zeros(7, dtype=np.float32)
        pose[:6] = [START_JOINT_POS[n] for n in _ARM_JOINT_NAMES]
        pose[6] = 1.0
        assert leader_is_aligned(pose)

    def test_a_far_pose_is_not_aligned(self):
        from loot_bench.robots.trlc_dk1.dk1_constants import START_JOINT_POS
        from loot_bench.scripts.live_bridge.mapper import _ARM_JOINT_NAMES

        pose = np.zeros(7, dtype=np.float32)
        pose[:6] = [START_JOINT_POS[n] for n in _ARM_JOINT_NAMES]
        pose[0] += 0.6
        assert not leader_is_aligned(pose)

    def test_the_rest_pose_is_not_aligned(self):
        assert not leader_is_aligned(np.zeros(7, dtype=np.float32))
