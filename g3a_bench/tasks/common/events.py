"""Shared event terms for the manipulation tasks."""

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
"""Module-level default so the cfg is not constructed per call (mirrors
mjlab's own event-term convention)."""


def reset_joints_to_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    joint_pos: dict[str, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset an asset's joints to an absolute pose, at zero velocity.

    mjlab's ``reset_joints_by_offset`` resets to ``default_joint_pos`` plus an
    offset sampled from one range shared by every joint, so it cannot express
    "start somewhere else entirely". This writes per-joint absolute targets
    instead.

    The distinction matters because ``default_joint_pos`` is *also* the
    reference for the ``joint_pos_rel`` observation and the zero point of
    ``JointPositionActionCfg``. Moving the keyframe to change where episodes
    begin would silently reinterpret every recorded demonstration; moving the
    reset target alone does not.

    Joints absent from ``joint_pos`` keep their default position.
    """
    env_ids = resolve_env_ids(env, env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None

    target = default_joint_pos[env_ids].clone()
    names = asset.joint_names
    for name, value in joint_pos.items():
        target[:, names.index(name)] = value

    asset.write_joint_state_to_sim(target, torch.zeros_like(target), env_ids=env_ids)
