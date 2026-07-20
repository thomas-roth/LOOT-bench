"""Observation terms shared by every task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def ee_pose_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector site pose in the env-local world frame (E, 7).

    Position (xyz) then quaternion (wxyz). ``asset_cfg.site_names`` is set to
    the robot's grasp site by ``manipulator_wiring.wire_manipulator_into_env``.
    """
    robot: Entity = env.scene[asset_cfg.name]
    pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    quat = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
    pos = pos - env.scene.env_origins
    return torch.cat([pos, quat], dim=-1)


def flatten_pose(
    pos_w: torch.Tensor, quat_w: torch.Tensor, env_origins: torch.Tensor
) -> torch.Tensor:
    """A single object's pose in the env-local world frame (E, 7).

    Position (env-relative xyz) then quaternion (wxyz).
    """
    pos = pos_w - env_origins
    return torch.cat([pos, quat_w], dim=-1)


def flatten_object_poses(
    pos_w: torch.Tensor, quat_w: torch.Tensor, env_origins: torch.Tensor
) -> torch.Tensor:
    """Per-env object poses in the env-local world frame, flattened (E, N*7).

    Object-major: each object contributes (pos xyz, quat wxyz). Dropping the
    trailing objects' 7-dim slices adapts the observation to an easier task
    (demo cutoff).

    Args:
        pos_w: (E, N, 3) object positions.
        quat_w: (E, N, 4) object orientations.
        env_origins: (E, 3).
    """
    num_envs = pos_w.shape[0]
    pos = pos_w - env_origins.unsqueeze(1)
    return torch.cat([pos, quat_w], dim=-1).reshape(num_envs, -1)
