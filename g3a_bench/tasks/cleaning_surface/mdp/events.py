# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import torch

from .commands import CLEAN_RADIUS, CLEAN_Z_MAX


def update_cleaning_logic(env, env_ids=None) -> None:
    """Latch cleaned dirt patches and refresh delta-reward buffers each step."""
    if not hasattr(env, "dirt_cleaned_mask"):
        return

    from .commands import CleaningSurfaceCommand

    cmd: CleaningSurfaceCommand = env.command_manager.get_term("cleaning_task")  # type: ignore[assignment]

    sponge_pos = env.scene["sponge"].data.root_link_pos_w  # (E, 3)
    sponge_xy = sponge_pos[:, :2].unsqueeze(1)  # (E, 1, 2)
    sponge_z = sponge_pos[:, 2]  # (E,)

    dirt_xy = cmd.dirt_pos_w[:, :, :2]  # (E, 4, 2)
    dist = torch.norm(sponge_xy - dirt_xy, dim=-1)  # (E, 4)

    sponge_low = sponge_z < CLEAN_Z_MAX  # (E,)
    newly_cleaned = (dist < CLEAN_RADIUS) & sponge_low.unsqueeze(1)

    # Compute current cleaned ratio before latching (for delta)
    prev_ratio = env.prev_cleaned_ratio.clone()
    env.dirt_cleaned_mask = env.dirt_cleaned_mask | newly_cleaned
    cur_ratio = env.dirt_cleaned_mask.float().mean(dim=-1)
    env.delta_cleaned_ratio = (cur_ratio - prev_ratio).clamp(min=0.0)
    env.prev_cleaned_ratio = cur_ratio

    # Debris-to-bin distances
    bin_c = cmd.bin_center_w  # (E, 3)
    E = env.scene.num_envs
    cur_dist = torch.full((E,), float("inf"), device=env.device)
    for i in range(2):
        deb_pos = env.scene[f"debris_{i}"].data.root_link_pos_w
        d = torch.norm(deb_pos[:, :2] - bin_c[:, :2], dim=-1)
        cur_dist = torch.minimum(cur_dist, d)
    env.delta_debris_approach = (env.prev_debris_dist - cur_dist).clamp(min=0.0)
    env.prev_debris_dist = cur_dist
