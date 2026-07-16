"""Observation terms for the peg-insertion task category (archived, unregistered)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g3a_bench.tasks.peg_insertion.mdp.commands import get_command as _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def pegs_rel_grasp(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """All peg positions relative to the grasp site, flattened (E, N*3)."""
    command = _command(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    peg_pos = torch.stack([p.data.root_link_pos_w for p in command.pegs], dim=1)
    return (peg_pos - ee_pos.unsqueeze(1)).reshape(env.num_envs, -1)


def socket_deltas(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Bore mouth minus assigned-peg position per socket, flattened (E, S*3)."""
    command = _command(env, command_name)
    delta = command.socket_mouth_w - command.assigned_peg_pos()
    return delta.reshape(env.num_envs, -1)


def socket_inserted_flags(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Per-socket inserted flags (E, S)."""
    return _command(env, command_name).inserted


def sheet_rel_grasp(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Top-sheet position relative to grasp, and its offset from the bore (E, 6).

    Two-sheet variants only. Zeros if no sheet is present.
    """
    command = _command(env, command_name)
    if command.sheet is None:
        return torch.zeros(env.num_envs, 6, device=env.device)
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    sheet_pos = command.sheet.data.root_link_pos_w
    rel_grasp = sheet_pos - ee_pos
    rel_bore = command.socket_mouth_w[:, 0] - sheet_pos
    return torch.cat([rel_grasp, rel_bore], dim=-1)
