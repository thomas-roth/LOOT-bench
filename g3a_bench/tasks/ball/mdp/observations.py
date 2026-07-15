"""Observation terms for the ball task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g3a_bench.tasks.ball.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def ball_rel_grasp(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Ball position relative to the grasp site (E, 3)."""
    command = _command(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    return command.ball_pos_w - ee_pos


def ball_to_target(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Target position relative to the ball (E, 3)."""
    command = _command(env, command_name)
    return command.target_pos_w - command.ball_pos_w


def ball_velocity(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Ball linear velocity in the world frame (E, 3)."""
    return _command(env, command_name).ball_vel_w
