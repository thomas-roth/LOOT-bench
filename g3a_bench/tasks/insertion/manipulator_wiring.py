"""Wire a robot's :class:`ManipulatorInterface` into a base insertion cfg.

This is the single seam between the (robot-agnostic) insertion task and a
(task-agnostic) concrete robot: every robot-specific value is read from the
interface, so adding a robot needs only its own ``get_<robot>_interface()`` and
zero task edits. Mirrors ``examples/manipulation/manipulator_wiring.py`` but for
the insertion-specific term layout (grasp-site terms differ from the cube
example, and the ``two_sheet`` variant adds extra grasp-site terms).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactSensorCfg

from g3a_bench.robots.interface import ManipulatorInterface

# Term keys whose ``asset_cfg`` points at the grasp site. The two_sheet variant
# adds "sheet_rel_grasp"; absent keys are skipped so the same wiring fits every
# variant.
_GRASP_OBS_TERMS = ("pegs_rel_grasp", "sheet_rel_grasp")


def wire_manipulator_into_env(
    cfg: ManagerBasedRlEnvCfg, manipulator: ManipulatorInterface
) -> None:
    """Fill the robot placeholders left by ``make_insertion_env_cfg`` in place."""
    cfg.scene.entities["robot"] = manipulator.entity_cfg_fn()

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = manipulator.action_scale

    grasp = (manipulator.grasp_site,)
    for group in cfg.observations.values():
        for key in _GRASP_OBS_TERMS:
            term = group.terms.get(key)
            if term is not None:
                term.params["asset_cfg"].site_names = grasp

    geoms = manipulator.finger_pad_geoms
    cfg.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = geoms

    assert cfg.scene.sensors is not None
    for sensor in cfg.scene.sensors:
        if sensor.name == "ee_werkbox_collision":
            assert isinstance(sensor, ContactSensorCfg)
            sensor.primary.pattern = manipulator.ee_subtree

    cfg.viewer.body_name = manipulator.viewer_body
