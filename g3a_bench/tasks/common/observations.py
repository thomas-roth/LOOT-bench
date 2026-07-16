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
