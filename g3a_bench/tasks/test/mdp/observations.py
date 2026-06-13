from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def root_pos_w(
    env: ManagerBasedRlEnv,
    entity_name: str,
) -> torch.Tensor:
    """Root body position of an entity in world frame.

    Shape: (num_envs, 3)
    """
    entity: Entity = env.scene[entity_name]
    return entity.data.root_link_pos_w


def root_quat_w(
    env: ManagerBasedRlEnv,
    entity_name: str,
) -> torch.Tensor:
    """Root body orientation quaternion of an entity in world frame.

    Shape: (num_envs, 4)
    """
    entity: Entity = env.scene[entity_name]
    return entity.data.root_link_quat_w


def site_pos_w(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """World position of a named site.

    Use this for end-effector, drawer handle, window handle, button top sites.

    Shape: (num_envs, 3)
    """
    entity: Entity = env.scene[asset_cfg.name]
    return entity.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)


def joint_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Joint position for selected joints.

    Shape: (num_envs, num_selected_joints)
    """
    entity: Entity = env.scene[asset_cfg.name]
    return entity.data.joint_pos[:, asset_cfg.joint_ids]


def joint_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Joint velocity for selected joints.

    Shape: (num_envs, num_selected_joints)
    """
    entity: Entity = env.scene[asset_cfg.name]
    return entity.data.joint_vel[:, asset_cfg.joint_ids]


def button_state_from_joint(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = -0.02,
) -> torch.Tensor:
    """Binary button pressed state from button joint position.

    Returns 1 if pressed, else 0.

    Shape: (num_envs, num_buttons)
    """
    entity: Entity = env.scene[asset_cfg.name]
    button_qpos = entity.data.joint_pos[:, asset_cfg.joint_ids]
    return (button_qpos <= threshold).float()


def logical_button_states(env) -> torch.Tensor:
    """Return persistent logical button states.

    Shape: (num_envs, 2)

    If not initialized yet, return zeros.
    """
    if not hasattr(env, "scene_button_states"):
        return torch.zeros((env.scene.num_envs, 2), device=env.device)

    return env.scene_button_states


def site_yaw_cos_sin(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return cos(yaw), sin(yaw) of a site orientation.

    Shape: (num_envs, 2)
    """
    entity: Entity = env.scene[asset_cfg.name]
    quat_w = entity.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
    yaw = yaw_from_quat_wxyz(quat_w)
    return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


def yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Compute yaw from quaternion in w, x, y, z format.

    Shape input: (num_envs, 4)
    Shape output: (num_envs,)
    """
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return torch.atan2(siny_cosp, cosy_cosp)


def root_yaw_cos_sin(
    env: ManagerBasedRlEnv,
    entity_name: str,
) -> torch.Tensor:
    """Return cos(yaw), sin(yaw) of root body orientation.

    Used for cube yaw.

    Shape: (num_envs, 2)
    """
    entity: Entity = env.scene[entity_name]
    yaw = yaw_from_quat_wxyz(entity.data.root_link_quat_w)
    return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


def gripper_opening(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Approximate gripper opening from selected finger joints.

    Shape: (num_envs, 1)

    You must pass the actual YAM gripper joint names.
    """
    entity: Entity = env.scene[asset_cfg.name]
    qpos = entity.data.joint_pos[:, asset_cfg.joint_ids]
    return qpos.sum(dim=-1, keepdim=True)


def gripper_contact_zero(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Placeholder gripper contact.

    OGBench includes gripper_contact, but for MJLab we should add a real
    contact sensor later. For now this returns 0.
    """
    num_envs = env.scene.num_envs
    return torch.zeros((num_envs, 1), device=env.device)
