import mujoco
import numpy as np

from g3a_bench.robots.trlc_dk1.dk1_actuators import (
    ARM_JOINTS,
    DK1_ACTION_SCALE_UNIMANUAL,
    GRIPPER_COMMAND_RANGE,
)
from g3a_bench.robots.trlc_dk1.dk1_constants import (
    HOME_JOINT_POS,
    get_unimanual_dk1_robot_cfg,
)

_ARM_JOINT_NAMES = list(ARM_JOINTS.keys())

HOME_JOINT_POS_ACTUATED = np.array(
    [HOME_JOINT_POS[n] for n in _ARM_JOINT_NAMES] + [HOME_JOINT_POS["gripper_left"]],
    dtype=np.float32,
)
ACTION_SCALE_ACTUATED = np.array(
    [DK1_ACTION_SCALE_UNIMANUAL[n] for n in _ARM_JOINT_NAMES]
    + [DK1_ACTION_SCALE_UNIMANUAL["gripper_left"]],
    dtype=np.float32,
)

# This leader's joint1 was hand-calibrated (DM_CAN.set_zero_position — see
# trlc-dk1/examples/calibration_follower.py) to a "zero" that doesn't match
# the MJCF asset's own joint1=0 reference (arm reaching forward, see
# CLAUDE.md's home-pose note). Sim-visualization-only correction — never
# apply this to the real leader_action, which is sent to the real follower
# (whose calibration already agrees with the leader) and recorded to disk.
# Re-tune (or remove) if the leader is re-calibrated.

# Physical joint limits from dk1.xml, in ctrl space (radians for joints, meters for gripper).
# Order: joint1..joint6, gripper_left — matches HOME_JOINT_POS / ACTION_SCALE.
JOINT_POS_LIMITS_LO = np.array(
    [-2.0943951, -0.0872665, -0.0872665, -1.95, -1.5707963, -2.0943951, -0.045],
    dtype=np.float32,
)
JOINT_POS_LIMITS_HI = np.array(
    [2.0943951, 3.1415927, 4.7123890, 1.5707963, 1.5707963, 2.0943951, 0.001],
    dtype=np.float32,
)

# Physical limits expressed in raw_action space: (ctrl_limit - HOME) / ACTION_SCALE.
# Use these to normalize teleop sim-actions to [-1, 1] with a fixed, session-independent
# transform that matches the full reachable joint range.
PHYSICAL_RAW_ACTION_MIN: np.ndarray = (
    JOINT_POS_LIMITS_LO - HOME_JOINT_POS_ACTUATED
) / ACTION_SCALE_ACTUATED
PHYSICAL_RAW_ACTION_MAX: np.ndarray = (
    JOINT_POS_LIMITS_HI - HOME_JOINT_POS_ACTUATED
) / ACTION_SCALE_ACTUATED


def mjlab_gripper_meters(norm_val: float) -> float:
    """Map real-robot gripper [0,1] → prismatic meters [0.0, ≈-0.0411].

    Inverts the controller's normalization through the pinion radius.
    """
    return float(
        np.interp(
            norm_val,
            [0.0, 1.0],
            [GRIPPER_COMMAND_RANGE["open"], GRIPPER_COMMAND_RANGE["closed"]],
        )
    )


class ActionNormalizer:
    """Maps raw actions to the policy's [-1, 1] and back over a fixed range.

    One home for the convention previously duplicated across lerobot_to_qc
    (normalize), mjlab_env and play_qcfql (denormalize). Defaults to the
    physical DK1 joint range; construct from a dataset's stored action_min/max
    to honour a data-normalized dataset.
    """

    def __init__(self, action_min: np.ndarray, action_max: np.ndarray):
        self.action_min = np.asarray(action_min, dtype=np.float32)
        self.action_max = np.asarray(action_max, dtype=np.float32)

    @classmethod
    def physical(cls) -> "ActionNormalizer":
        return cls(PHYSICAL_RAW_ACTION_MIN, PHYSICAL_RAW_ACTION_MAX)

    @classmethod
    def ik(cls) -> "ActionNormalizer":
        """Bounds for the task-space (``-Ik``) env action: identity ``[-1, 1]``
        on the 6 arm delta dims, physical DK1 range on the gripper dim (see
        ``IK_RAW_ACTION_MIN``/``MAX``)."""
        return cls(IK_RAW_ACTION_MIN, IK_RAW_ACTION_MAX)

    def normalize(self, raw_action: np.ndarray) -> np.ndarray:
        """raw action → [-1, 1]. Guards zero-range (constant) dims against /0."""
        span = self.action_max - self.action_min
        rng = np.where(span > 1e-6, span, 1.0)
        return 2.0 * (raw_action - self.action_min) / rng - 1.0

    def denormalize(self, action: np.ndarray) -> np.ndarray:
        """[-1, 1] → raw action. Clips first: policy output must saturate at
        the physical limit, not extrapolate past it."""
        clipped = np.clip(action, -1.0, 1.0)
        span = self.action_max - self.action_min
        return (clipped + 1.0) / 2.0 * span + self.action_min


def lerobot_to_mjlab_ctrl(joint_pos: np.ndarray) -> np.ndarray:
    """Convert real-robot joint_pos (7,) → MJLab data.ctrl (7,).

    Joints 0-5 (radians) pass through unchanged.
    Gripper (index 6) goes from normalised [0,1] to prismatic meters
    (see mjlab_gripper_meters).
    """
    ctrl = np.empty(7, dtype=np.float32)
    ctrl[:6] = joint_pos[:6]
    ctrl[6] = mjlab_gripper_meters(float(joint_pos[6]))
    return ctrl


_GRAVITY_STIFFNESS = np.array(
    [ARM_JOINTS[n].kp for n in _ARM_JOINT_NAMES], dtype=np.float32
)
_gravity_model: mujoco.MjModel | None = None
_gravity_data: mujoco.MjData | None = None
_gravity_dofadr: np.ndarray | None = None


def _gravity_scratch():
    """Lazily build a standalone (non-rendering) DK1 model used only to
    evaluate gravity torque at a candidate target pose. Built on first use so
    importing this module stays cheap for callers that never enable
    gravity compensation (e.g. record_replay.py, tests).
    """
    global _gravity_model, _gravity_data, _gravity_dofadr
    if _gravity_model is None:
        from mjlab.entity import Entity

        entity = Entity(get_unimanual_dk1_robot_cfg())
        _gravity_model = entity.compile()
        _gravity_data = mujoco.MjData(_gravity_model)
        _gravity_dofadr = np.array(
            [
                _gravity_model.jnt_dofadr[
                    mujoco.mj_name2id(_gravity_model, mujoco.mjtObj.mjOBJ_JOINT, n)
                ]
                for n in _ARM_JOINT_NAMES
            ]
        )
    return _gravity_model, _gravity_data, _gravity_dofadr


def gravity_compensate_ctrl(ctrl: np.ndarray) -> np.ndarray:
    """Push the commanded arm target past gravity sag so the PD actuator
    actually settles at ``ctrl`` instead of short of it.

    The actuators are soft PD position controllers (~2 Hz, tuned for RL
    residual steps) with no gravity feedforward, so a large one-shot teleop
    target — e.g. lifting the arm well above home — sags under gravity and
    never reaches the commanded angle. Standard fix: command
    ``target + g(target) / kp`` instead of ``target``, where ``g(target)`` is
    the gravity torque needed to hold that pose statically (kp is the
    actuator's own PD stiffness, so this exactly cancels the steady-state
    error). Gripper (index 6) passes through unchanged.
    """
    model, data, dofadr = _gravity_scratch()
    data.qvel[:] = 0.0
    for name, value in zip(_ARM_JOINT_NAMES, ctrl[:6], strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)
    gravity_torque = data.qfrc_bias[dofadr]

    compensated = ctrl.copy()
    compensated[:6] = ctrl[:6] + gravity_torque / _GRAVITY_STIFFNESS
    return compensated


def lerobot_to_mjlab_raw_action(
    joint_pos: np.ndarray,
    gravity_compensate: bool = False,
) -> np.ndarray:
    """Convert to raw action for JointPositionActionCfg(use_default_offset=True).

    MJLab computes:  target = HOME_ACTUATED_JOINT_POS + raw_action * ACTION_SCALE
    where the result recovers the original joint positions from lerobot_to_mjlab_ctrl.

    Args:
        gravity_compensate: If True, push the target past gravity sag (see
            gravity_compensate_ctrl) before computing raw_action. Opt-in and
            off by default — record_replay.py and tests rely on the exact
            uncompensated mapping.
    """
    ctrl = lerobot_to_mjlab_ctrl(joint_pos)
    if gravity_compensate:
        ctrl = gravity_compensate_ctrl(ctrl)
    return (ctrl - HOME_JOINT_POS_ACTUATED) / ACTION_SCALE_ACTUATED


# ── Task-space (differential-IK) action mapping ──────────────────────────────
#
# The ``-Ik`` task variant's env action is
#     [dx, dy, dz, drx, dry, drz, gripper]   (7-D)
#     └──── arm end-effector delta pose ────┘ └grip┘
# (see ``tasks/common/ik_action.py``). The arm block is a *relative* delta the
# env applies to the current grasp-site pose each control step; the gripper
# block is the *same* physical raw joint action as under joint control, so the
# gripper behaves identically in both variants. These helpers turn a live
# teleop leader pose into that env action, and normalize it to the ``[-1, 1]``
# policy space (arm dims already live there, so only the gripper dim is scaled
# by its physical range).

IK_ARM_ACTION_DIM = 6

# IK env-action normalization bounds. The 6 arm delta dims are policy-native
# (the env's ``delta_pos_scale``/``delta_ori_scale`` already map [-1, 1] to a
# metre/radian step), so their bounds are the identity [-1, 1]; the gripper dim
# reuses the physical DK1 range so the same 1-DoF joint action calibration
# holds across both variants.
IK_RAW_ACTION_MIN: np.ndarray = np.concatenate(
    [-np.ones(IK_ARM_ACTION_DIM, dtype=np.float32), PHYSICAL_RAW_ACTION_MIN[6:7]]
).astype(np.float32)
IK_RAW_ACTION_MAX: np.ndarray = np.concatenate(
    [np.ones(IK_ARM_ACTION_DIM, dtype=np.float32), PHYSICAL_RAW_ACTION_MAX[6:7]]
).astype(np.float32)


# Standalone (non-rendering) DK1 model used only for forward kinematics of the
# teleop leader pose. Built lazily on first use (same rationale as the gravity
# scratch model above) and reused across calls.
_fk_model: mujoco.MjModel | None = None
_fk_data: mujoco.MjData | None = None
_fk_grasp_site_id: int | None = None
_fk_arm_qposadr: np.ndarray | None = None


def _fk_scratch() -> tuple[mujoco.MjModel, mujoco.MjData, int, np.ndarray]:
    global _fk_model, _fk_data, _fk_grasp_site_id, _fk_arm_qposadr
    if _fk_model is None:
        from mjlab.entity import Entity

        from g3a_bench.robots.trlc_dk1.dk1_constants import DK1_GRASP_SITE

        entity = Entity(get_unimanual_dk1_robot_cfg())
        _fk_model = entity.compile()
        _fk_data = mujoco.MjData(_fk_model)
        # The compiled standalone entity names its site "grasp_site" (no scene
        # prefix), but match by suffix to stay robust to any prefixing.
        site_id = -1
        for sid in range(_fk_model.nsite):
            name = mujoco.mj_id2name(_fk_model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
            if name == DK1_GRASP_SITE or name.endswith("/" + DK1_GRASP_SITE):
                site_id = sid
                break
        assert site_id >= 0, f"grasp site {DK1_GRASP_SITE!r} not found in DK1 model"
        _fk_grasp_site_id = site_id
        _fk_arm_qposadr = np.array(
            [
                _fk_model.jnt_qposadr[
                    mujoco.mj_name2id(_fk_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in _ARM_JOINT_NAMES
            ]
        )
    assert _fk_data is not None and _fk_grasp_site_id is not None
    assert _fk_arm_qposadr is not None
    return _fk_model, _fk_data, _fk_grasp_site_id, _fk_arm_qposadr


def leader_ee_pose_base(joint_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward-kinematics grasp-site pose for a leader joint reading.

    Returns ``(pos, quat_wxyz)`` of the grasp site in the robot base frame,
    given a real-robot ``joint_pos`` (7,) (only the 6 arm joints are used; the
    gripper does not move the grasp site).
    """
    model, data, site_id, arm_qposadr = _fk_scratch()
    data.qpos[arm_qposadr] = joint_pos[:6]
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[site_id].astype(np.float32).copy()
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return pos, quat.astype(np.float32)


def leader_to_ik_env_action(
    joint_pos: np.ndarray,
    *,
    base_pos_w: np.ndarray,
    base_quat_w: np.ndarray,
    ee_pos_w: np.ndarray,
    ee_quat_w: np.ndarray,
    delta_pos_scale: float,
    delta_ori_scale: float,
) -> np.ndarray:
    """Env action (7,) for the ``-Ik`` variant from a live teleop step.

    Drives the sim grasp site toward the leader's commanded grasp pose: the arm
    block is the relative delta ``compute_pose_error(current_ee, desired_ee)``
    (the exact inverse of the env's ``apply_delta_pose``) divided by the env's
    delta scales and clipped to ``[-1, 1]``; the gripper block is the same
    physical raw joint action used under joint control.

    Args:
        joint_pos: real-robot leader reading (7,).
        base_pos_w, base_quat_w: robot base link pose in the (env-local) world
            frame — used to lift the base-frame FK pose into that frame so the
            delta matches what the env's world-frame IK term consumes.
        ee_pos_w, ee_quat_w: current sim grasp-site pose in the same frame.
        delta_pos_scale, delta_ori_scale: the env's IK cfg scales, so the
            recorded delta lands in policy units.
    """
    import torch
    from mjlab.utils.lab_api.math import compute_pose_error, quat_apply, quat_mul

    def _t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, dtype=np.float32)).unsqueeze(0)

    fk_pos_base, fk_quat_base = leader_ee_pose_base(joint_pos)
    base_p, base_q = _t(base_pos_w), _t(base_quat_w)
    # desired grasp pose in world = base_pose ∘ FK(base frame)
    desired_pos = base_p + quat_apply(base_q, _t(fk_pos_base))
    desired_quat = quat_mul(base_q, _t(fk_quat_base))

    pos_err, rot_err = compute_pose_error(
        _t(ee_pos_w), _t(ee_quat_w), desired_pos, desired_quat, "axis_angle"
    )
    arm = np.concatenate(
        [
            (pos_err[0].numpy() / delta_pos_scale),
            (rot_err[0].numpy() / delta_ori_scale),
        ]
    ).astype(np.float32)
    arm = np.clip(arm, -1.0, 1.0)

    gripper = lerobot_to_mjlab_raw_action(joint_pos, gravity_compensate=False)[6]
    return np.concatenate([arm, [gripper]]).astype(np.float32)
