"""Observation terms for the domino task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g3a_bench.tasks.domino.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def dominoes_rel_grasp(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """All domino positions relative to the grasp site, flattened (E, N*3).

    Object-major (domino index order), so dropping the trailing dominoes'
    dims adapts the observation to an easier task (demo cutoff).
    """
    command = _command(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    domino_pos = torch.stack([d.data.root_link_pos_w for d in command.dominoes], dim=1)
    rel = domino_pos - ee_pos.unsqueeze(1)
    return rel.reshape(env.num_envs, -1)


def slot_deltas(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Matched-domino position minus its slot goal, per slot (E, G*3)."""
    command = _command(env, command_name)
    delta = command.slot_piece_pos() - command.slot_goal_pos
    return delta.reshape(env.num_envs, -1)


def slot_placed(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Per-slot placed indicator, (E, G)."""
    return _command(env, command_name).slot_at_goal


def phase_obs(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(E, 1) build (0) / topple (1) phase, KnockOver variants only."""
    return _command(env, command_name).phase.unsqueeze(-1)
