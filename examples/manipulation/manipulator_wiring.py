"""Wire a robot's :class:`ManipulatorInterface` into a base manipulation cfg.

This is the single seam between a (robot-agnostic) manipulation task and a
(task-agnostic) concrete robot: every robot-specific value is read from the
interface, so adding a robot needs only its own ``get_<robot>_interface()``
and zero task edits.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactSensorCfg

from g3a_bench.robots.interface import ManipulatorInterface

_FRICTION_EVENTS = (
    "fingertip_friction_slide",
    "fingertip_friction_spin",
    "fingertip_friction_roll",
)


def wire_manipulator_into_env(
    cfg: ManagerBasedRlEnvCfg, manipulator: ManipulatorInterface
) -> None:
    """Apply ``manipulator`` to ``cfg`` in place.

    Adds the robot as the ``"robot"`` scene entity and sets every robot-specific
    knob (action scale, grasp site, finger-pad friction geoms, end-effector
    contact subtree, viewer body). The caller owns task objects (e.g. cubes),
    which it must place in ``cfg.scene.entities`` before calling this.
    """
    cfg.scene.entities["robot"] = manipulator.entity_cfg_fn()

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = manipulator.action_scale

    grasp = (manipulator.grasp_site,)
    cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = grasp
    cfg.rewards["lift"].params["asset_cfg"].site_names = grasp

    for event in _FRICTION_EVENTS:
        cfg.events[event].params["asset_cfg"].geom_names = manipulator.finger_pad_geoms

    assert cfg.scene.sensors is not None
    for sensor in cfg.scene.sensors:
        if sensor.name == "ee_ground_collision":
            assert isinstance(sensor, ContactSensorCfg)
            sensor.primary.pattern = manipulator.ee_subtree

    cfg.viewer.body_name = manipulator.viewer_body
