"""Wire a robot's :class:`ManipulatorInterface` into a base insertion cfg.

This is the single seam between the (robot-agnostic) insertion task and a
(task-agnostic) concrete robot: every robot-specific value is read from the
interface, so adding a robot needs only its own ``get_<robot>_interface()`` and
zero task edits. Mirrors ``examples/manipulation/manipulator_wiring.py`` but for
the insertion-specific term layout (grasp-site terms differ from the cube
example, and the ``two_sheet`` variant adds extra grasp-site terms).
"""

import re

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import DifferentialIKActionCfg, JointPositionActionCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg

from g3a_bench.robots.interface import ManipulatorInterface
from g3a_bench.tasks.insertion.insertion_env_cfg import _CAMERA_OBS_NAME

# Term keys whose ``asset_cfg`` points at the grasp site. The two_sheet variant
# adds "sheet_rel_grasp"/"sheet_alignment"; absent keys are skipped so the same
# wiring fits every variant.
_GRASP_OBS_TERMS = ("pegs_rel_grasp", "sheet_rel_grasp")
_GRASP_REWARD_TERMS = ("staged_next_socket", "sheet_alignment")


def wire_manipulator_into_env(
    cfg: ManagerBasedRlEnvCfg, manipulator: ManipulatorInterface
) -> None:
    """Apply ``manipulator`` to ``cfg`` in place.

    Pulls the robot's ``EntityCfg`` and every robot-specific knob from the
    interface and replaces the placeholders left by ``make_insertion_env_cfg``:
    adds the robot as the ``"robot"`` scene entity, and sets the action scale,
    grasp site, finger-pad friction geoms, end-effector contact subtree, and
    viewer body. The caller owns the task objects (board, sheet, pegs), which it
    must place in ``cfg.scene.entities`` before calling.
    """
    cfg.scene.entities["robot"] = manipulator.entity_cfg_fn()

    _wire_actions(cfg, manipulator)

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

    assert cfg.scene.sensors is not None
    for sensor in cfg.scene.sensors:
        if sensor.name == "ee_ground_collision":
            assert isinstance(sensor, ContactSensorCfg)
            sensor.primary.pattern = manipulator.ee_subtree
        elif sensor.name == _CAMERA_OBS_NAME:
            assert isinstance(sensor, CameraSensorCfg)
            assert manipulator.camera_name is not None, (
                f"obs_mode='camera' needs a camera_name on '{manipulator.name}'"
            )
            sensor.camera_name = manipulator.camera_name

    cfg.viewer.body_name = manipulator.viewer_body


def _wire_actions(cfg: ManagerBasedRlEnvCfg, manipulator: ManipulatorInterface) -> None:
    """Fill robot-specific action fields for whichever action mode is active.

    ``"joint"`` mode has a single ``"joint_pos"`` term; ``"eef"`` mode has an
    ``"arm"`` differential-IK term plus a ``"gripper"`` joint term (see
    ``make_insertion_env_cfg``).
    """
    if "joint_pos" in cfg.actions:  # joint (angle-based) control.
        action = cfg.actions["joint_pos"]
        assert isinstance(action, JointPositionActionCfg)
        action.scale = manipulator.action_scale
        return

    # EEF control: IK drives the arm via the grasp site, gripper stays direct.
    arm = cfg.actions["arm"]
    assert isinstance(arm, DifferentialIKActionCfg)
    arm.actuator_names = manipulator.arm_actuators
    arm.frame_name = manipulator.grasp_site

    gripper = cfg.actions["gripper"]
    assert isinstance(gripper, JointPositionActionCfg)
    gripper.actuator_names = manipulator.gripper_actuators
    gripper.scale = _scale_for(manipulator.action_scale, manipulator.gripper_actuators)


def _scale_for(
    action_scale: dict[str, float] | float, actuators: tuple[str, ...]
) -> dict[str, float] | float:
    """Restrict a per-actuator scale dict to ``actuators`` (pass scalars through).

    The full ``action_scale`` dict covers every actuator; a per-term action only
    accepts scales for its own targets, so keys outside ``actuators`` are dropped.
    """
    if not isinstance(action_scale, dict):
        return action_scale
    return {
        name: val
        for name, val in action_scale.items()
        if any(re.fullmatch(pat, name) for pat in actuators)
    }
