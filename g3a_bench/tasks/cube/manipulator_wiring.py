"""Wire a ManipulatorInterface into a base cube cfg.

Mirrors ``tasks/insertion/manipulator_wiring.py`` for the cube-specific
term layout (``pieces_rel_grasp`` obs, ``staged_next_slot`` reward).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactSensorCfg

from g3a_bench.robots.interface import ManipulatorInterface

_GRASP_OBS_TERMS = ("pieces_rel_grasp",)
_GRASP_REWARD_TERMS = ("staged_next_slot",)


def wire_manipulator_into_env(
    cfg: ManagerBasedRlEnvCfg, manipulator: ManipulatorInterface
) -> None:
    """Apply ``manipulator`` to ``cfg`` in place.

    Fills robot entity, action scale, grasp site on ``pieces_rel_grasp`` obs
    and ``staged_next_slot`` reward, finger-pad friction geoms, end-effector
    contact subtree, and viewer body.
    """
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
    for key in _GRASP_REWARD_TERMS:
        reward = cfg.rewards.get(key)
        if reward is not None:
            reward.params["asset_cfg"].site_names = grasp

    cfg.events["fingertip_friction_slide"].params[
        "asset_cfg"
    ].geom_names = manipulator.finger_pad_geoms

    assert cfg.scene.sensors is not None
    for sensor in cfg.scene.sensors:
        if sensor.name == "ee_ground_collision":
            assert isinstance(sensor, ContactSensorCfg)
            sensor.primary.pattern = manipulator.ee_subtree

    cfg.viewer.body_name = manipulator.viewer_body
