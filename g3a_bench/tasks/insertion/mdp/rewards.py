"""Reward terms for the object-insertion task category."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g3a_bench.tasks.insertion.mdp.commands import InsertionCommand
from g3a_bench.tasks.insertion.variants import PEG_HALF_LEN

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _command(env: ManagerBasedRlEnv, name: str) -> InsertionCommand:
    return cast(InsertionCommand, env.command_manager.get_term(name))


def _ee_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    return robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)


def staged_next_socket_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    reaching_std: float,
    bringing_std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """reaching * (1 + approaching) for the next un-inserted socket.

    reaching pulls the grasp site to the assigned peg; approaching pulls that
    peg toward a pre-insertion waypoint just outside the bore mouth.
    """
    command = _command(env, command_name)
    ee_pos = _ee_pos(env, asset_cfg)
    nxt = command.next_socket().view(-1, 1, 1).expand(-1, 1, 3)
    peg_pos = torch.gather(command.assigned_peg_pos(), 1, nxt).squeeze(1)
    waypoint = torch.gather(command.pre_insert_waypoint(), 1, nxt).squeeze(1)

    reach_err = torch.sum(torch.square(ee_pos - peg_pos), dim=-1)
    reaching = torch.exp(-reach_err / reaching_std**2)
    approach_err = torch.sum(torch.square(waypoint - peg_pos), dim=-1)
    approaching = torch.exp(-approach_err / bringing_std**2)
    return reaching * (1.0 + approaching)


def insertion_depth_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Mean per-socket insertion depth fraction, gated by lateral alignment.

    Only counts depth while the peg is roughly over the bore, so the policy is
    not rewarded for pushing a peg against the board surface off-axis.
    """
    command = _command(env, command_name)
    pos = command.assigned_peg_pos()
    axis = command.socket_axis_w
    rel = pos - command.socket_mouth_w
    axial = torch.sum(rel * axis, dim=-1)
    lateral = torch.norm(rel - axial.unsqueeze(-1) * axis, dim=-1)
    over_bore = (lateral < 2.0 * command.lateral_tol).float()
    depth_frac = (-axial + PEG_HALF_LEN).clamp(0.0, command.insert_depth) / (
        command.insert_depth
    )
    return (over_bore * depth_frac).mean(dim=1)


def inserted_count_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Fraction of pegs correctly inserted."""
    command = _command(env, command_name)
    return command.inserted.sum(dim=1) / command.num_sockets


def sheet_alignment_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    reaching_std: float,
    bringing_std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Two-sheet only: reach the top sheet and align its hole over the bore."""
    command = _command(env, command_name)
    if command.sheet is None:
        return torch.zeros(env.num_envs, device=env.device)
    ee_pos = _ee_pos(env, asset_cfg)
    sheet_pos = command.sheet.data.root_link_pos_w
    reach_err = torch.sum(torch.square(ee_pos - sheet_pos), dim=-1)
    reaching = torch.exp(-reach_err / reaching_std**2)
    align_err = torch.sum(
        torch.square(command.socket_mouth_w[:, 0, :2] - sheet_pos[:, :2]), dim=-1
    )
    aligning = torch.exp(-align_err / bringing_std**2)
    return reaching * (1.0 + aligning)


def success_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Sparse bonus while all pegs are inserted."""
    return _command(env, command_name).inserted.all(dim=1).float()
