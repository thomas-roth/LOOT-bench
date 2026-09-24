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

from loot_bench.robots.interface import ManipulatorInterface
from loot_bench.tasks.common.events import reset_joints_to_pose


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

    The action term needs nothing from the robot: it reads each joint's travel
    off the compiled model (see ``tasks.common.actions``). The grasp site is
    written to the ``asset_cfg`` of the
    ``ee_pose`` observation term (present in every category via the shared
    env cfg) and to any command cfg carrying a ``grasp_cfg``. Sensors named
    in ``ee_contact_sensors`` get the
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

    pad_sensors = (grip_contact_sensor, object_contact_sensor)
    for sensor in cfg.scene.sensors:
        if sensor.name in ee_contact_sensors:
            pattern = manipulator.ee_subtree
        elif sensor.name in pad_sensors:
            pattern = manipulator.pad_geoms
        else:
            continue
        assert isinstance(sensor, ContactSensorCfg)
        sensor.primary.pattern = pattern

    cfg.viewer.body_name = manipulator.viewer_body
