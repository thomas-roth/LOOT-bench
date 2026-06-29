"""Observation terms for the cube task category."""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g3a_bench.tasks.cube.mdp.commands import get_cube_command

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def pieces_rel_grasp(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """All piece positions relative to the grasp site, flattened (E, N*3)."""
    command = get_cube_command(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    piece_pos = torch.stack([p.data.root_link_pos_w for p in command.pieces], dim=1)
    rel = piece_pos - ee_pos.unsqueeze(1)
    return rel.reshape(env.num_envs, -1)


def slot_goal_deltas(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Goal minus assigned-piece position per slot, flattened (E, G*3)."""
    command = get_cube_command(env, command_name)
    delta = command.slot_goal_pos - command.slot_piece_pos()
    return delta.reshape(env.num_envs, -1)


def slot_placed_flags(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Per-slot placed flags (E, G)."""
    return get_cube_command(env, command_name).slot_at_goal
