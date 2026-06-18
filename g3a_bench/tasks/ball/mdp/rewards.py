"""Reward terms for the ball task category.

Shared progress/success terms plus action-specific shaping:

- golf: reward imparting forward velocity to the ball (a strike) and lightly
  reward reaching it, but penalise *sustained* gripper-ball proximity so the
  policy hits once instead of leading the ball.
- push: a staged reach * (1 + bring) reward that keeps the gripper on the ball
  while the ball nears the target (leading is encouraged here).
- throw: reward reaching/raising the ball; landing is handled by the shared
  distance term.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g3a_bench.tasks.ball.mdp.commands import BallCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _command(env: ManagerBasedRlEnv, name: str) -> BallCommand:
    return cast(BallCommand, env.command_manager.get_term(name))


def _ee_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    return robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)


def ball_distance_reward(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Gaussian kernel on the ball-to-target horizontal distance."""
    command = _command(env, command_name)
    d = torch.sum(
        torch.square(command.ball_pos_w[:, :2] - command.target_pos_w[:, :2]), dim=-1
    )
    return torch.exp(-d / std**2)


def success_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Bonus while the ball is in the target at rest."""
    command = _command(env, command_name)
    return command.at_target


def strike_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Forward ball speed toward the target (rewards a golf strike)."""
    command = _command(env, command_name)
    vel_xy = command.ball_vel_w[:, :2]
    forward = torch.sum(vel_xy * command.dir_to_target(), dim=-1)
    return forward.clamp_min(0.0)


def gripper_ball_reach_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian kernel on grasp-site-to-ball distance."""
    command = _command(env, command_name)
    ee_pos = _ee_pos(env, asset_cfg)
    d = torch.sum(torch.square(ee_pos - command.ball_pos_w), dim=-1)
    return torch.exp(-d / std**2)


def contact_time_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    contact_dist: float = 0.06,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """1.0 while the gripper stays close to the ball (discourages leading)."""
    command = _command(env, command_name)
    ee_pos = _ee_pos(env, asset_cfg)
    d = torch.norm(ee_pos - command.ball_pos_w, dim=-1)
    return (d < contact_dist).float()


def push_guide_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    reaching_std: float,
    bringing_std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """reaching * (1 + bringing): keep the gripper on the ball as it nears goal."""
    command = _command(env, command_name)
    ee_pos = _ee_pos(env, asset_cfg)
    reach_err = torch.sum(torch.square(ee_pos - command.ball_pos_w), dim=-1)
    reaching = torch.exp(-reach_err / reaching_std**2)
    bring_err = torch.sum(
        torch.square(command.ball_pos_w[:, :2] - command.target_pos_w[:, :2]), dim=-1
    )
    bringing = torch.exp(-bring_err / bringing_std**2)
    return reaching * (1.0 + bringing)


def throw_height_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reward raising the ball above the surface (up to 0.3 m), to clear a wall."""
    command = _command(env, command_name)
    height = (command.ball_pos_w[:, 2] - command.surface_z).clamp(0.0, 0.3)
    return height / 0.3
