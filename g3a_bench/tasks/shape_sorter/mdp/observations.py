"""Observation terms for the shape-sorter task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g3a_bench.tasks.shape_sorter.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def pegs_rel_grasp(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """All piece positions relative to the grasp site, flattened (E, N*3).

    Object-major (shape index order), so dropping the trailing pieces' dims
    adapts the observation to an easier task (demo cutoff).
    """
    command = _command(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    return (command.piece_pos_w - ee_pos.unsqueeze(1)).reshape(env.num_envs, -1)


def socket_deltas(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Hole mouth minus piece position per socket, flattened (E, N*3)."""
    command = _command(env, command_name)
    delta = command.socket_mouth_w - command.piece_pos_w
    return delta.reshape(env.num_envs, -1)


def socket_inserted(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Per-socket latched flags (E, N)."""
    return _command(env, command_name).latched.float()
