"""Wire a ManipulatorInterface into a base cube cfg.

Mirrors ``tasks/insertion/manipulator_wiring.py`` for the cube-specific
term layout (``pieces_rel_grasp`` obs).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from g3a_bench.robots.interface import ManipulatorInterface

_GRASP_OBS_TERMS = ("pieces_rel_grasp",)


def wire_manipulator_into_env(
    cfg: ManagerBasedRlEnvCfg, manipulator: ManipulatorInterface
) -> None:
    """Apply ``manipulator`` to ``cfg`` in place.

    Fills robot entity, action scale, grasp site on ``pieces_rel_grasp`` obs,
    finger-pad friction geoms, and viewer body.
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

    cfg.events["fingertip_friction_slide"].params[
        "asset_cfg"
    ].geom_names = manipulator.finger_pad_geoms

    cfg.viewer.body_name = manipulator.viewer_body
