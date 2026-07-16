"""Observation terms for the stacking task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g3a_bench.tasks.stacking.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def blocks_rel_grasp(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """All block positions relative to the grasp site, flattened (E, N*3).

    Object-major (block index order), so dropping the trailing blocks' dims
    adapts the observation to an easier task (demo cutoff).
    """
    command = _command(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    block_pos = torch.stack([b.data.root_link_pos_w for b in command.blocks], dim=1)
    rel = block_pos - ee_pos.unsqueeze(1)
    return rel.reshape(env.num_envs, -1)
