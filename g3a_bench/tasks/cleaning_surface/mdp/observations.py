# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from .commands import CleaningSurfaceCommand


# ---------------------------------------------------------------------------
# Generic entity observation helpers (reusable; same pattern as scene task)
# ---------------------------------------------------------------------------


def root_pos_w(env: ManagerBasedRlEnv, entity_name: str) -> torch.Tensor:
    """Root body world position. Shape: (num_envs, 3)."""
    entity: Entity = env.scene[entity_name]
    return entity.data.root_link_pos_w


def root_quat_w(env: ManagerBasedRlEnv, entity_name: str) -> torch.Tensor:
    """Root body world orientation quaternion. Shape: (num_envs, 4)."""
    entity: Entity = env.scene[entity_name]
    return entity.data.root_link_quat_w


def site_pos_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """World position of a named site. Shape: (num_envs, 3)."""
    entity: Entity = env.scene[asset_cfg.name]
    return entity.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def site_yaw_cos_sin(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """cos/sin of site yaw. Shape: (num_envs, 2)."""
    entity: Entity = env.scene[asset_cfg.name]
    quat = entity.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
    yaw = _yaw_from_quat_wxyz(quat)
    return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


def gripper_opening(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Summed gripper finger joint positions. Shape: (num_envs, 1)."""
    entity: Entity = env.scene[asset_cfg.name]
    return entity.data.joint_pos[:, asset_cfg.joint_ids].sum(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Cleaning-surface-specific observations
# ---------------------------------------------------------------------------


def _cleaning_command(env: ManagerBasedRlEnv) -> CleaningSurfaceCommand:

    return env.command_manager.get_term("cleaning_task")  # type: ignore[return-value]


def dirt_cleaned_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Per-patch cleaned flags as floats. Shape: (num_envs, 4)."""
    if not hasattr(env, "dirt_cleaned_mask"):
        return torch.zeros((env.scene.num_envs, 4), device=env.device)
    return env.dirt_cleaned_mask.float()


def cleaned_ratio(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Fraction of dirt patches cleaned. Shape: (num_envs, 1)."""
    if not hasattr(env, "dirt_cleaned_mask"):
        return torch.zeros((env.scene.num_envs, 1), device=env.device)
    return env.dirt_cleaned_mask.float().mean(dim=-1, keepdim=True)


def bin_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """World position of the bin centre. Shape: (num_envs, 3)."""
    return _cleaning_command(env).bin_center_w


def task_variant_onehot(env: ManagerBasedRlEnv) -> torch.Tensor:
    """One-hot encoding of the current task variant. Shape: (num_envs, 3)."""
    cmd = _cleaning_command(env)
    E = env.scene.num_envs
    v_oh = torch.zeros(E, 3, device=env.device)
    v_oh.scatter_(1, cmd.task_variant.unsqueeze(-1), 1.0)
    return v_oh


def target_dirt_positions(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Dirt patch world positions, flattened. Shape: (num_envs, 12)."""
    cmd = _cleaning_command(env)
    return cmd.dirt_pos_w.reshape(env.scene.num_envs, -1)
