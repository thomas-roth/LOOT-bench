# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import VARIANT_DEBRIS_TO_BIN, VARIANT_FULL_CLEANING, VARIANT_WIPE_ONLY

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from .commands import CleaningSurfaceCommand


def _cleaning_command(env: ManagerBasedRlEnv) -> CleaningSurfaceCommand:
    from .commands import CleaningSurfaceCommand

    return cast(CleaningSurfaceCommand, env.command_manager.get_term("cleaning_task"))


def _debris_in_bin(env: ManagerBasedRlEnv, cmd: CleaningSurfaceCommand) -> torch.Tensor:
    """(E,) bool: all debris inside bin XY footprint and low enough."""
    bin_c = cmd.bin_center_w
    result = torch.ones(env.scene.num_envs, dtype=torch.bool, device=env.device)
    for i in range(2):
        deb_pos = env.scene[f"debris_{i}"].data.root_link_pos_w
        in_xy = (torch.abs(deb_pos[:, :2] - bin_c[:, :2]) < 0.08).all(dim=-1)
        in_z = deb_pos[:, 2] < 0.10
        result = result & in_xy & in_z
    return result


def cleaning_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Variant-aware composite reward. Range: [-1, 0].

    wipe_only:      cleaned_ratio - 1
    debris_to_bin:  debris_in_bin - 1
    full_cleaning:  0.5 * cleaned + 0.5 * debris - 1
    """
    if not hasattr(env, "dirt_cleaned_mask"):
        return torch.zeros(env.scene.num_envs, device=env.device)

    cmd = _cleaning_command(env)
    cleaned = env.dirt_cleaned_mask.float().mean(dim=-1)
    debris = _debris_in_bin(env, cmd).float()
    v = cmd.task_variant

    return torch.where(
        v == VARIANT_WIPE_ONLY,
        cleaned - 1.0,
        torch.where(
            v == VARIANT_DEBRIS_TO_BIN,
            debris - 1.0,
            0.5 * cleaned + 0.5 * debris - 1.0,
        ),
    )


def reach_sponge_reward(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Negative EE-to-sponge distance (potential-based shaping, no prev buffer needed)."""
    entity = env.scene[asset_cfg.name]
    ee_pos = entity.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    sponge_pos = env.scene["sponge"].data.root_link_pos_w
    return -torch.norm(ee_pos - sponge_pos, dim=-1)


def wipe_progress_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Delta increase in cleaned ratio per step (pre-computed in step event)."""
    if not hasattr(env, "delta_cleaned_ratio"):
        return torch.zeros(env.scene.num_envs, device=env.device)
    return env.delta_cleaned_ratio


def debris_approach_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Delta decrease in min debris-to-bin distance per step (pre-computed in step event)."""
    if not hasattr(env, "delta_debris_approach"):
        return torch.zeros(env.scene.num_envs, device=env.device)
    return env.delta_debris_approach


def cleaning_success(
    env: ManagerBasedRlEnv, command_name: str = "cleaning_task"
) -> torch.Tensor:
    """True when the variant-specific success condition is met."""
    if not hasattr(env, "dirt_cleaned_mask"):
        return torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)

    cmd = cast(
        "CleaningSurfaceCommand",
        env.command_manager.get_term(command_name),
    )
    cleaned_ratio = env.dirt_cleaned_mask.float().mean(dim=-1)
    in_bin = _debris_in_bin(env, cmd)
    v = cmd.task_variant
    return (
        ((v == VARIANT_WIPE_ONLY) & (cleaned_ratio >= 1.0))
        | ((v == VARIANT_DEBRIS_TO_BIN) & in_bin)
        | ((v == VARIANT_FULL_CLEANING) & (cleaned_ratio >= 1.0) & in_bin)
    )
