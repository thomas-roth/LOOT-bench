"""Wire a robot's :class:`ManipulatorInterface` into a base task cfg.

This is the single seam between the (robot-agnostic) task categories and a
(task-agnostic) concrete robot: every robot-specific value is read from the
interface, so adding a robot needs only its own ``get_<robot>_interface()``
and zero task edits. Each category's ``config/__init__.py`` names its own
contact sensors, if any; everything else is identical across categories.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg

from g3a_bench.robots.interface import ManipulatorInterface
from g3a_bench.tasks.common.actions import (
    EE_DELTA_ACTION_NAME,
    GRIPPER_ACTION_NAME,
    EndEffectorDeltaActionCfg,
    NormalizedJointPositionActionCfg,
)
from g3a_bench.tasks.common.commands import GripperGateCommandCfg
from g3a_bench.tasks.common.events import reset_joints_to_pose


def wire_manipulator_into_env(
    cfg: ManagerBasedRlEnvCfg,
    manipulator: ManipulatorInterface,
    *,
    ee_contact_sensors: tuple[str, ...] = (),
    grip_contact_sensor: str | None = None,
    object_contact_sensor: str | None = None,
) -> None:
    """Fill the robot placeholders left by a ``make_<cat>_env_cfg`` in place.

    Adds the robot as the ``"robot"`` scene entity and sets the viewer body.

    Both action terms are filled from the robot's declared actuator split: the
    end-effector delta term gets ``arm_actuators``, the robot's grasp site as
    its IK frame, and its ``ik_max_dq``; the gripper term gets
    ``gripper_actuators``. This is the only place that knows a robot has an arm
    and a gripper -- a task file never names either, and a robot declaring an
    empty split fails when the term is built rather than at the first step of a
    training run.

    The grasp site is also written to the ``asset_cfg`` of the
    ``ee_pose`` observation term (present in every category via the shared
    env cfg) and to any command cfg carrying a ``grasp_cfg``. Command cfgs
    deriving from :class:`GripperGateCommandCfg` get the robot's gripper
    joints/threshold. Sensors named in ``ee_contact_sensors`` get the
    end-effector subtree as their primary match pattern. The sensors named
    ``grip_contact_sensor`` and ``object_contact_sensor``, if any, get the
    finger-pad geom pattern as their primary match pattern (see
    ``tasks.common.commands.ContactReleaseGate``). Only their *secondary*
    match differs, and that is the task's business rather than the robot's:
    the task declares which object it scores, the robot contributes the pads.

    A robot declaring ``reset_joint_pos`` gets the ``reset_robot_joints`` event
    swapped from mjlab's keyframe-relative offset reset to the absolute-pose
    reset in ``tasks.common.events``. The task's own sample ranges carry over
    unchanged, so ``start_pose_noise`` still means the same thing either way.
    """
    cfg.scene.entities["robot"] = manipulator.entity_cfg_fn()

    ee_delta = cfg.actions[EE_DELTA_ACTION_NAME]
    assert isinstance(ee_delta, EndEffectorDeltaActionCfg)
    ee_delta.actuator_names = manipulator.arm_actuators
    ee_delta.frame_name = manipulator.grasp_site
    # Derived per-robot, never a literal. It is a torque cap rather than a rate
    # limit -- see ManipulatorInterface.ik_max_dq, which also records why
    # ADR-016's velocity-limit derivation cannot hold the arm up.
    ee_delta.max_dq = manipulator.ik_max_dq

    gripper = cfg.actions[GRIPPER_ACTION_NAME]
    assert isinstance(gripper, NormalizedJointPositionActionCfg)
    gripper.actuator_names = manipulator.gripper_actuators

    if manipulator.reset_joint_pos is not None:
        event = cfg.events["reset_robot_joints"]
        event.func = reset_joints_to_pose
        event.params = {
            "pose": dict(manipulator.reset_joint_pos),
            "position_range": event.params["position_range"],
            "velocity_range": event.params["velocity_range"],
            "asset_cfg": event.params["asset_cfg"],
        }

    grasp = (manipulator.grasp_site,)
    for group in cfg.observations.values():
        term = group.terms.get("ee_pose")
        if term is not None:
            term.params["asset_cfg"].site_names = grasp

    for command in cfg.commands.values():
        grasp_cfg = getattr(command, "grasp_cfg", None)
        if isinstance(grasp_cfg, SceneEntityCfg):
            grasp_cfg.site_names = grasp
        if isinstance(command, GripperGateCommandCfg):
            command.gripper_joints = manipulator.gripper_joints
            command.gripper_open_threshold = manipulator.gripper_open_threshold

    if ee_contact_sensors:
        assert cfg.scene.sensors is not None
        for sensor in cfg.scene.sensors:
            if sensor.name in ee_contact_sensors:
                assert isinstance(sensor, ContactSensorCfg)
                sensor.primary.pattern = manipulator.ee_subtree

    pad_sensors = {
        name
        for name in (grip_contact_sensor, object_contact_sensor)
        if name is not None
    }
    if pad_sensors:
        assert cfg.scene.sensors is not None
        for sensor in cfg.scene.sensors:
            if sensor.name in pad_sensors:
                assert isinstance(sensor, ContactSensorCfg)
                sensor.primary.pattern = manipulator.pad_geoms

    cfg.viewer.body_name = manipulator.viewer_body
