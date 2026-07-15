import numpy as np
import pytest

from g3a_bench.robots.trlc_dk1.dk1_actuators import GRIPPER_COMMAND_RANGE
from g3a_bench.scripts.live_bridge.mapper import (
    ACTION_SCALE_ACTUATED,
    HOME_JOINT_POS_ACTUATED,
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
    def test_home_position_returns_near_zero(self):
        joint_pos = HOME_JOINT_POS_ACTUATED.copy()
        raw = lerobot_to_mjlab_raw_action(joint_pos)
        np.testing.assert_array_almost_equal(raw, 0.0, decimal=5)

    def test_roundtrip_with_ctrl(self):
        joint_pos = np.array([0.5, -0.3, 2.1, -0.8, 1.2, -1.5, 0.7], dtype=np.float32)
        raw = lerobot_to_mjlab_raw_action(joint_pos)
        recovered = HOME_JOINT_POS_ACTUATED + raw * ACTION_SCALE_ACTUATED
        np.testing.assert_array_almost_equal(
            recovered, lerobot_to_mjlab_ctrl(joint_pos), decimal=5
        )
