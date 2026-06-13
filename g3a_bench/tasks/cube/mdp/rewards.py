"""Reward terms for the cube task category."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g3a_bench.tasks.cube.mdp.commands import PieceArrangementCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _command(env: ManagerBasedRlEnv, name: str) -> PieceArrangementCommand:
    return cast(PieceArrangementCommand, env.command_manager.get_term(name))


def staged_next_slot_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    reaching_std: float,
    bringing_std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Staged shaping focused on the next unplaced slot.

    reaching * (1 + bringing), where reaching pulls the gripper to the piece
    assigned to the first unplaced slot and bringing pulls that piece to its
    goal. Same structure as mjlab's `staged_position_reward`.
    """
    command = _command(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)

    next_slot = command.next_slot()  # (E,)
    idx = next_slot.view(-1, 1, 1).expand(-1, 1, 3)
    piece_pos = torch.gather(command.slot_piece_pos(), 1, idx).squeeze(1)
    goal_pos = torch.gather(command.slot_goal_pos, 1, idx).squeeze(1)

    reach_err = torch.sum(torch.square(ee_pos - piece_pos), dim=-1)
    reaching = torch.exp(-reach_err / reaching_std**2)
    bring_err = torch.sum(torch.square(goal_pos - piece_pos), dim=-1)
    bringing = torch.exp(-bring_err / bringing_std**2)
    return reaching * (1.0 + bringing)


def placed_slots_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Fraction of placed slots; rewards progress and keeping pieces placed."""
    command = _command(env, command_name)
    return command.slot_at_goal.sum(dim=1) / command.num_slots


def success_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Sparse bonus while the full arrangement is complete."""
    return _command(env, command_name).slot_at_goal.all(dim=1).float()
