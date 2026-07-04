"""Wire a robot's :class:`ManipulatorInterface` into a base ball cfg.

This is the single seam between the (robot-agnostic) ball task and a
(task-agnostic) concrete robot: every robot-specific value is read from the
interface, so adding a robot needs only its own ``get_<robot>_interface()`` and
zero task edits. Mirrors ``insertion/manipulator_wiring.py`` but for the
ball-specific term layout: the ball task is joint-position only and carries no
sensors, so it skips the EEF-action and camera/contact-sensor wiring.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from g3a_bench.robots.interface import ManipulatorInterface

# Term keys whose ``asset_cfg`` points at the grasp site. The grasp reward terms
# are action-specific (golf adds "reach"/"contact_time", push adds "push_guide",
# throw adds "reach"); absent keys are skipped so the same wiring fits every
# variant.
_GRASP_OBS_TERMS = ("ball_rel_grasp",)
_GRASP_REWARD_TERMS = ("reach", "contact_time", "push_guide")


def wire_manipulator_into_env(
    cfg: ManagerBasedRlEnvCfg, manipulator: ManipulatorInterface
) -> None:
    """Apply ``manipulator`` to ``cfg`` in place.

    Pulls the robot's ``EntityCfg`` and every robot-specific knob from the
    interface and replaces the placeholders left by ``make_ball_env_cfg``: adds
    the robot as the ``"robot"`` scene entity, and sets the action scale, grasp
    site, finger-pad friction geoms, and viewer body. The caller owns the task
    objects (field, ball), which it must place in ``cfg.scene.entities`` before
    calling.
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

    geoms = manipulator.finger_pad_geoms
    cfg.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = geoms

    cfg.viewer.body_name = manipulator.viewer_body
