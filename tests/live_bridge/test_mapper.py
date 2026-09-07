"""The teleop half of the shared action convention.

``tests/tasks/test_actions.py`` pins the other half -- what the simulator does
with an action. Together they say that the action a recording stores and the
action a policy emits mean the same motion.
"""

import numpy as np
import pytest

from loot_bench.robots.trlc_dk1.dk1_actuators import GRIPPER_COMMAND_RANGE
from loot_bench.robots.trlc_dk1.dk1_constants import START_JOINT_POS
from loot_bench.scripts.live_bridge.mapper import (
    ACTUATED_JOINT_NAMES,
    GRIPPER_ACTION_INDEX,
    gripper_scale_offset,
    joint_pos_limits,
    leader_grasp_pose,
    leader_is_aligned,
    leader_to_delta_action,
    lerobot_to_mjlab_ctrl,
    mjlab_gripper_meters,
    reset_grasp_pose,
)
from loot_bench.tasks.common.actions import DELTA_POS_SCALE


def _start_pose(gripper: float = 0.0) -> np.ndarray:
    """The leader pose an aligned operator holds: the episode's start pose."""
    pose = np.zeros(7, dtype=np.float32)
    pose[:6] = [START_JOINT_POS[name] for name in ACTUATED_JOINT_NAMES[:6]]
    pose[GRIPPER_ACTION_INDEX] = gripper
    return pose


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


class TestLeaderGraspPose:
    def test_the_start_pose_lands_where_the_home_pose_is_documented(self):
        """FK against the number ``dk1_constants`` states: q2=q3=1.7,
        q4=-pi/2 puts the downward grasp site at about (0.30, 0, 0.20)."""
        pos, quat = leader_grasp_pose(_start_pose())
        np.testing.assert_allclose(pos, [0.30, 0.0, 0.20], atol=0.02)
        assert quat.shape == (4,)
        assert float(np.linalg.norm(quat)) == pytest.approx(1.0, abs=1e-5)

    def test_the_gripper_does_not_move_the_grasp_site(self):
        """The site rides on the palm, not the fingers -- so opening the hand
        must not read as a commanded end-effector motion."""
        open_pos, _ = leader_grasp_pose(_start_pose(gripper=0.0))
        shut_pos, _ = leader_grasp_pose(_start_pose(gripper=1.0))
        np.testing.assert_array_almost_equal(open_pos, shut_pos, decimal=6)


class TestLeaderToDeltaAction:
    """The mapping cannot disagree with the action term, because both call
    ``tasks/common/actions.py``. What is left to pin is the leader-arm side:
    the FK, the frame, the clamp, and the gripper channel."""

    def test_a_leader_at_the_current_pose_asks_for_nothing(self):
        pose = _start_pose()
        current_pos, current_quat = leader_grasp_pose(pose)
        action = leader_to_delta_action(pose, current_pos, current_quat)
        np.testing.assert_array_almost_equal(action[:6], 0.0, decimal=5)

    def test_one_box_of_travel_is_one_unit(self):
        """The definition of the action unit, read from the teleop side."""
        pose = _start_pose()
        current_pos, current_quat = leader_grasp_pose(pose)
        behind = current_pos - np.array([DELTA_POS_SCALE, 0.0, 0.0], dtype=np.float32)
        action = leader_to_delta_action(pose, behind, current_quat)
        assert action[0] == pytest.approx(1.0, abs=1e-4)
        np.testing.assert_array_almost_equal(action[1:6], 0.0, decimal=4)

    def test_a_pose_beyond_the_box_is_clamped_not_scaled(self):
        """The clamp is the convention: a leader three boxes away still asks
        for one, because that is all a bounded policy could have asked for."""
        pose = _start_pose()
        current_pos, current_quat = leader_grasp_pose(pose)
        far = current_pos - np.array([3.0 * DELTA_POS_SCALE, 0.0, 0.0], np.float32)
        action = leader_to_delta_action(pose, far, current_quat)
        assert action[0] == pytest.approx(1.0, abs=1e-6)

    def test_every_action_is_one_a_bounded_policy_could_emit(self):
        pose = _start_pose(gripper=1.0)
        current_pos, current_quat = leader_grasp_pose(_start_pose())
        far = current_pos + np.array([0.4, -0.3, 0.5], dtype=np.float32)
        action = leader_to_delta_action(pose, far, current_quat)
        assert action.shape == (7,)
        assert np.all(np.abs(action) <= 1.0), action

    def test_the_gripper_channel_inverts_the_term(self):
        """The one channel that is still absolute. It has to invert the
        gripper term's own scale and offset, not a second copy of them."""
        scale, offset = gripper_scale_offset()
        pose = _start_pose(gripper=1.0)
        current_pos, current_quat = leader_grasp_pose(pose)
        action = leader_to_delta_action(pose, current_pos, current_quat)
        recovered = action[GRIPPER_ACTION_INDEX] * scale + offset
        assert recovered == pytest.approx(GRIPPER_COMMAND_RANGE["closed"], abs=1e-5)

    def test_an_open_gripper_inverts_too(self):
        scale, offset = gripper_scale_offset()
        pose = _start_pose(gripper=0.0)
        current_pos, current_quat = leader_grasp_pose(pose)
        action = leader_to_delta_action(pose, current_pos, current_quat)
        recovered = action[GRIPPER_ACTION_INDEX] * scale + offset
        assert recovered == pytest.approx(GRIPPER_COMMAND_RANGE["open"], abs=1e-5)


class TestAlignment:
    """The gate that replaced the opening snap.

    ADR-016 measured 66 of 99 demos opening with the arm jumping toward an
    already-parked leader -- a motion no bounded policy can reproduce. The gate
    is one action box wide, so recording may only start where the first stored
    action is a real action.
    """

    def test_the_reset_pose_is_aligned(self):
        assert leader_is_aligned(_start_pose())

    def test_the_gripper_does_not_affect_alignment(self):
        """It is an absolute channel with no jump to gate."""
        assert leader_is_aligned(_start_pose(gripper=1.0))

    def test_a_leader_one_box_past_the_reset_pose_is_not_aligned(self):
        pose = _start_pose()
        pose[0] += 0.6  # Sweeps the whole arm well past one box at that reach.
        assert not leader_is_aligned(pose)

    def test_the_rest_pose_is_not_aligned(self):
        """The pose the leader hangs in when the operator lets go. Recording
        from there is the failure the gate exists for."""
        assert not leader_is_aligned(np.zeros(7, dtype=np.float32))

    def test_the_gate_compares_against_the_start_pose(self):
        reset_pos, _ = reset_grasp_pose()
        start_pos, _ = leader_grasp_pose(_start_pose())
        np.testing.assert_array_almost_equal(reset_pos, start_pos, decimal=6)


class TestJointLimits:
    def test_limits_are_read_from_the_model_not_hardcoded(self):
        """The regression this guards: a copy of dk1.xml's ranges going stale."""
        lower, upper = joint_pos_limits()
        assert lower.shape == (7,) and upper.shape == (7,)
        assert np.all(upper > lower)

    def test_the_gripper_stroke_maps_to_the_box_edges(self):
        lower, upper = joint_pos_limits()
        scale, offset = gripper_scale_offset()
        i = GRIPPER_ACTION_INDEX
        assert (lower[i] - offset) / scale == pytest.approx(-1.0, abs=1e-5)
        assert (upper[i] - offset) / scale == pytest.approx(1.0, abs=1e-5)
