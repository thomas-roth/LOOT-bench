"""Task-space (differential-IK) action variant for the manipulation tasks.

The default action space is joint-position control (one
:class:`JointPositionActionCfg` over every actuator, see
``tasks/common/env_cfg.py``). This module builds the *alternative* action
dict used by the ``-Ik`` task variants: a 6-DoF end-effector delta-pose
command driving the arm through mjlab's damped-least-squares differential IK,
plus a separate 1-DoF joint-position term for the gripper (mjlab's IK term
controls the arm only).

The action layout mirrors OGBench-style task-space manipulation but keeps the
full wrist orientation (the DK1 teleop leader is 6-DoF, so constraining to a
top-down + yaw action would discard operator information):

    action = [dx, dy, dz, drx, dry, drz, gripper]   (7-D)
             └──────── arm_ik (6-D) ────────┘ └grip┘

``arm_ik`` is a *relative* delta applied to the current end-effector pose each
control step: translation components are scaled by
``manipulator.ik_delta_pos_scale`` (metres) and rotation components by
``manipulator.ik_delta_ori_scale`` (radians), so a policy/​demonstration action
in ``[-1, 1]`` maps to a bounded per-step motion. The concatenation order is
fixed by the dict order below (arm first, gripper last) and every consumer --
the recorded dataset, the ``last_action`` observation, replay -- must match it.
"""

from mjlab.envs.mdp.actions import DifferentialIKActionCfg, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg

from g3a_bench.robots.interface import ManipulatorInterface

ARM_IK_ACTION_NAME = "arm_ik"
GRIPPER_ACTION_NAME = "gripper"


def _gripper_scale(manipulator: ManipulatorInterface) -> float | dict[str, float]:
    """Scale for the gripper joint-position term, read from the robot's
    per-actuator ``action_scale`` so the IK variant's gripper behaves exactly
    like it does under joint control."""
    scale = manipulator.action_scale
    if isinstance(scale, dict):
        return {name: scale[name] for name in manipulator.gripper_actuator_names}
    return scale


def make_ik_actions(manipulator: ManipulatorInterface) -> dict[str, ActionTermCfg]:
    """Build the ``{arm_ik, gripper}`` action dict for the IK task variant.

    Reads the arm/gripper actuator split, the grasp site (IK target frame) and
    the delta scales from the robot's :class:`ManipulatorInterface`. Raises if
    the robot declares no IK actuators (only robots with
    ``arm_actuator_names`` set support an IK variant).
    """
    if not manipulator.arm_actuator_names or not manipulator.gripper_actuator_names:
        raise ValueError(
            f"Robot {manipulator.name!r} has no IK action support: set "
            "arm_actuator_names and gripper_actuator_names on its "
            "ManipulatorInterface to register an -Ik variant."
        )
    return {
        ARM_IK_ACTION_NAME: DifferentialIKActionCfg(
            entity_name="robot",
            actuator_names=manipulator.arm_actuator_names,
            frame_type="site",
            frame_name=manipulator.grasp_site,
            use_relative_mode=True,
            position_weight=1.0,
            orientation_weight=1.0,
            delta_pos_scale=manipulator.ik_delta_pos_scale,
            delta_ori_scale=manipulator.ik_delta_ori_scale,
        ),
        GRIPPER_ACTION_NAME: JointPositionActionCfg(
            entity_name="robot",
            actuator_names=manipulator.gripper_actuator_names,
            scale=_gripper_scale(manipulator),
            use_default_offset=True,
        ),
    }
