"""Wire a robot's :class:`ManipulatorInterface` into a base task cfg.

This is the single seam between the (robot-agnostic) task categories and a
(task-agnostic) concrete robot: every robot-specific value is read from the
interface, so adding a robot needs only its own ``get_<robot>_interface()``
and zero task edits. Each category's ``config/__init__.py`` names its own
contact sensors, if any; everything else is identical across categories.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg

from g3a_bench.robots.interface import ManipulatorInterface
from g3a_bench.tasks.common.commands import GripperGateCommandCfg


def wire_manipulator_into_env(
    cfg: ManagerBasedRlEnvCfg,
    manipulator: ManipulatorInterface,
    *,
    ee_contact_sensors: tuple[str, ...] = (),
    grip_contact_sensor: str | None = None,
) -> None:
    """Fill the robot placeholders left by a ``make_<cat>_env_cfg`` in place.

    Adds the robot as the ``"robot"`` scene entity and sets the action scale
    and viewer body. The grasp site is written to the ``asset_cfg`` of the
    ``ee_pose`` observation term (present in every category via the shared
    env cfg) and to any command cfg carrying a ``grasp_cfg``. Command cfgs
    deriving from :class:`GripperGateCommandCfg` get the robot's gripper
    joints/threshold. Sensors named in ``ee_contact_sensors`` get the
    end-effector subtree as their primary match pattern. The sensor named
    ``grip_contact_sensor``, if any, gets the finger-pad geom pattern as its
    primary match pattern (see ``tasks.common.commands.ContactReleaseGate``).
    """
    cfg.scene.entities["robot"] = manipulator.entity_cfg_fn()

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = manipulator.action_scale

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

    if grip_contact_sensor is not None:
        assert cfg.scene.sensors is not None
        for sensor in cfg.scene.sensors:
            if sensor.name == grip_contact_sensor:
                assert isinstance(sensor, ContactSensorCfg)
                sensor.primary.pattern = manipulator.pad_geoms

    cfg.viewer.body_name = manipulator.viewer_body
