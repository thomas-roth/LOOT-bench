from __future__ import annotations

import math

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .observations import (
    _EEF_BRC1_CFG,
    _EEF_BRC2_CFG,
    _EEF_CFG,
    _EEF_LEFT_CFG,
    _EEF_RIGHT_CFG,
    _EEF_TLC1_CFG,
    _EEF_TLC2_CFG,
    _PEG_GRASP_CFG,
    _PEG_HEAD_CFG,
    _site_pos,
    gripper_openness,
)

TARGET_RADIUS: float = 0.07
_GAUSSIAN_SCALE = math.sqrt(-2 * math.log(0.1))


def _gaussian_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
    if margin == 0:
        return (x == 0).float()
    return torch.exp(-0.5 * (x / margin * _GAUSSIAN_SCALE) ** 2)


def _long_tail_tolerance(x: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + (x / (margin + 1e-6)) ** 2)


def _hamacher_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    denom = a + b - a * b
    return torch.where(denom > 0, (a * b) / denom, torch.zeros_like(a))


def peg_insertion_reward(
    env,
    peg_cfg: SceneEntityCfg = _PEG_GRASP_CFG,
    peg_head_cfg: SceneEntityCfg = _PEG_HEAD_CFG,
    eef_cfg: SceneEntityCfg = _EEF_CFG,
) -> torch.Tensor:
    if not hasattr(env.unwrapped, "target_pos"):
        asset: Entity = env.scene["sawyer"]
        return torch.zeros(env.num_envs, device=asset.data.joint_pos.device)

    def sp(cfg: SceneEntityCfg) -> torch.Tensor:
        # Sites are spread across entities (peg / sawyer / box); resolve each
        # against its own entity.
        return _site_pos(env.scene[cfg.name], cfg)

    peg_pos = sp(peg_cfg)
    peg_head = sp(peg_head_cfg)
    right_finger = sp(_EEF_RIGHT_CFG)
    left_finger = sp(_EEF_LEFT_CFG)
    brc1 = sp(_EEF_BRC1_CFG)
    tlc1 = sp(_EEF_TLC1_CFG)
    brc2 = sp(_EEF_BRC2_CFG)
    tlc2 = sp(_EEF_TLC2_CFG)

    goal = env.unwrapped.target_pos
    tcp = (right_finger + left_finger) / 2.0
    tcp_open = gripper_openness(env, eef_cfg).squeeze(-1)

    scale = peg_pos.new_tensor([1.0, 2.0, 2.0])
    tcp_to_obj = torch.norm(peg_pos - tcp, dim=-1)
    obj_to_target = torch.norm((peg_head - goal) * scale, dim=-1)

    in_place = _long_tail_tolerance(obj_to_target, env.unwrapped.in_place_margin)

    def _rect_prism(curr, one, zero):
        lo = torch.min(one, zero)
        hi = torch.max(one, zero)
        return ((curr >= lo) & (curr <= hi)).all(dim=-1).float()

    collision_boxes = _hamacher_product(
        _rect_prism(peg_head, tlc1, brc1),
        _rect_prism(peg_head, tlc2, brc2),
    )
    in_place = _hamacher_product(in_place, collision_boxes)

    reach_xy = torch.norm(peg_pos[:, :2] - tcp[:, :2], dim=-1)
    reach_z = torch.abs(peg_pos[:, 2] - tcp[:, 2])
    object_grasped = _gaussian_tolerance(reach_xy, margin=0.005) * _gaussian_tolerance(
        reach_z, margin=0.08
    )

    lifted = (peg_pos[:, 2] - 0.01 > env.unwrapped.obj_init_pos[:, 2]).float()
    grasped_and_lifted = (tcp_to_obj < 0.08).float() * (tcp_open > 0).float() * lifted
    object_grasped = torch.where(
        grasped_and_lifted > 0,
        torch.ones_like(object_grasped),
        object_grasped,
    )

    reward = _hamacher_product(object_grasped, in_place)
    reward = reward + grasped_and_lifted * (1.0 + 5.0 * in_place)
    success = (obj_to_target <= TARGET_RADIUS).float()
    reward = torch.where(success > 0, torch.full_like(reward, 10.0), reward)

    return reward


__all__ = ["TARGET_RADIUS", "peg_insertion_reward"]
