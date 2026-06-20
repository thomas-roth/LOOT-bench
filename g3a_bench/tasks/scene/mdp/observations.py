from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from g3a_bench.tasks.scene.mdp.utils import get_scene_command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


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


def ee_pos_b(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EE site position in robot base frame. Shape: (num_envs, 3)"""
    entity: Entity = env.scene[asset_cfg.name]
    ee_pos_w = entity.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    robot_pos_w = entity.data.root_link_pos_w
    return quat_apply(quat_inv(entity.data.root_link_quat_w), ee_pos_w - robot_pos_w)


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
    """Return persistent logical button states from the command.

    Shape: (num_envs, 2)
    """
    return get_scene_command(env).scene_button_states


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


# ---- Relative vector observations ----


def ee_to_object_distance(
    env: ManagerBasedRlEnv,
    object_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """EE-to-object vector in robot base frame. Shape: (num_envs, 3)"""
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    obj_pos_w = env.scene[object_name].data.root_link_pos_w
    return quat_apply(quat_inv(robot.data.root_link_quat_w), obj_pos_w - ee_pos_w)


def object_to_goal_distance(
    env: ManagerBasedRlEnv,
    object_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Object-to-goal vector in robot base frame. Shape: (num_envs, 3)

    Goal is target_cube_pos from the scene_task command.
    """
    cmd = get_scene_command(env)
    robot: Entity = env.scene[asset_cfg.name]
    obj_pos_w = env.scene[object_name].data.root_link_pos_w
    return quat_apply(
        quat_inv(robot.data.root_link_quat_w), cmd.target_cube_pos - obj_pos_w
    )


def ee_to_site_distance(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg,
    site_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EE-to-named-site vector in robot base frame. Shape: (num_envs, 3)

    Use for drawer handle, window handle, button top.
    """
    robot: Entity = env.scene[robot_cfg.name]
    ee_pos_w = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
    target_entity: Entity = env.scene[site_cfg.name]
    site_pos_w = target_entity.data.site_pos_w[:, site_cfg.site_ids].squeeze(1)
    return quat_apply(quat_inv(robot.data.root_link_quat_w), site_pos_w - ee_pos_w)


def drawer_to_goal(env) -> torch.Tensor:
    """target_drawer_pos - current drawer_slide. Shape: (num_envs, 1)"""
    cmd = get_scene_command(env)
    return (
        cmd.target_drawer_pos - cmd._drawer.data.joint_pos[:, cmd._drawer_joint_slot[0]]
    ).unsqueeze(-1)


def window_to_goal(env) -> torch.Tensor:
    """target_window_pos - current window_slide. Shape: (num_envs, 1)"""
    cmd = get_scene_command(env)
    return (
        cmd.target_window_pos - cmd._window.data.joint_pos[:, cmd._window_joint_slot[0]]
    ).unsqueeze(-1)


def scene_target_button_states(env) -> torch.Tensor:
    """Goal logical button states for the current scene task. Shape: (num_envs, 2)"""
    return get_scene_command(env).target_button_states
