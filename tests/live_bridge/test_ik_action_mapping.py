"""Unit tests for the task-space (differential-IK) teleop action mapping.

Covers the pure kernels behind the ``-Ik`` recording path (see
``scripts/live_bridge/mapper.py``): the leader forward-kinematics helper, the
leader→env-action delta, and the IK action normalizer round-trip. None of these
need a running env or hardware.
"""

import numpy as np
import torch

from g3a_bench.scripts.live_bridge.mapper import (
    IK_RAW_ACTION_MAX,
    IK_RAW_ACTION_MIN,
    ActionNormalizer,
    leader_ee_pose_base,
    leader_to_ik_env_action,
)

# A non-trivial leader arm pose (radians) + open gripper.
_LEADER = np.array([0.2, 1.4, 1.5, -1.2, 0.1, 0.3, 0.0], dtype=np.float32)


def _compose_world(base_pos, base_quat, pos_base, quat_base):
    from mjlab.utils.lab_api.math import quat_apply, quat_mul

    def t(a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32)).unsqueeze(0)

    w_pos = (t(base_pos) + quat_apply(t(base_quat), t(pos_base)))[0].numpy()
    w_quat = quat_mul(t(base_quat), t(quat_base))[0].numpy()
    return w_pos, w_quat


def test_leader_fk_is_deterministic_and_shaped():
    pos, quat = leader_ee_pose_base(_LEADER)
    assert pos.shape == (3,)
    assert quat.shape == (4,)
    np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-5)
    # Pure function of the arm joints: the gripper column must not move the site.
    other_gripper = _LEADER.copy()
    other_gripper[6] = 0.7
    pos2, quat2 = leader_ee_pose_base(other_gripper)
    np.testing.assert_allclose(pos, pos2, atol=1e-6)
    np.testing.assert_allclose(quat, quat2, atol=1e-6)


def test_ik_action_zero_delta_when_ee_matches_leader():
    """When the sim grasp pose already equals the leader's FK pose, the arm
    delta block must be (near) zero — the operator is commanding no motion."""
    base_pos = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identity
    fk_pos, fk_quat = leader_ee_pose_base(_LEADER)
    ee_pos, ee_quat = _compose_world(base_pos, base_quat, fk_pos, fk_quat)

    act = leader_to_ik_env_action(
        _LEADER,
        base_pos_w=base_pos,
        base_quat_w=base_quat,
        ee_pos_w=ee_pos,
        ee_quat_w=ee_quat,
        delta_pos_scale=0.05,
        delta_ori_scale=0.5,
    )
    assert act.shape == (7,)
    np.testing.assert_allclose(act[:6], 0.0, atol=1e-5)


def test_ik_action_points_toward_leader_target():
    """An end-effector sitting behind the leader target (smaller x) must get a
    positive x delta driving it forward, scaled and clipped to [-1, 1]."""
    base_pos = np.zeros(3, dtype=np.float32)
    base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    fk_pos, fk_quat = leader_ee_pose_base(_LEADER)
    des_pos, _ = _compose_world(base_pos, base_quat, fk_pos, fk_quat)

    ee_pos = des_pos.copy()
    ee_pos[0] -= 0.02  # sim EE is 2 cm behind the target in +x
    act = leader_to_ik_env_action(
        _LEADER,
        base_pos_w=base_pos,
        base_quat_w=base_quat,
        ee_pos_w=ee_pos,
        ee_quat_w=fk_quat,
        delta_pos_scale=0.05,
        delta_ori_scale=0.5,
    )
    # 0.02 m error / 0.05 m scale = 0.4, within range (not clipped).
    np.testing.assert_allclose(act[0], 0.4, atol=1e-4)
    assert act[:6].min() >= -1.0 and act[:6].max() <= 1.0


def test_ik_action_clips_large_delta():
    base_pos = np.zeros(3, dtype=np.float32)
    base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    fk_pos, fk_quat = leader_ee_pose_base(_LEADER)
    des_pos, _ = _compose_world(base_pos, base_quat, fk_pos, fk_quat)
    ee_pos = des_pos.copy()
    ee_pos[1] += 1.0  # a metre off — far beyond one step's reach
    act = leader_to_ik_env_action(
        _LEADER,
        base_pos_w=base_pos,
        base_quat_w=base_quat,
        ee_pos_w=ee_pos,
        ee_quat_w=fk_quat,
        delta_pos_scale=0.05,
        delta_ori_scale=0.5,
    )
    assert act[1] == -1.0  # saturated toward the target


def test_ik_normalizer_round_trip():
    norm = ActionNormalizer.ik()
    rng = np.random.default_rng(0)
    # Sample env actions: arm in [-1, 1], gripper in its physical range.
    arm = rng.uniform(-1.0, 1.0, size=(32, 6)).astype(np.float32)
    grip = rng.uniform(IK_RAW_ACTION_MIN[6], IK_RAW_ACTION_MAX[6], size=(32, 1)).astype(
        np.float32
    )
    env_action = np.concatenate([arm, grip], axis=1)
    normalized = norm.normalize(env_action)
    assert normalized.min() >= -1.0 - 1e-6 and normalized.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(norm.denormalize(normalized), env_action, atol=1e-5)
    # Arm dims are identity: normalize leaves them unchanged.
    np.testing.assert_allclose(normalized[:, :6], arm, atol=1e-5)
