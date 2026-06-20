from __future__ import annotations

import torch
from mjlab.envs.mdp.terminations import time_out  # noqa: F401


def cube_out_of_bounds(
    env, workspace_radius: float = 0.8, z_min: float = -0.05
) -> torch.Tensor:
    """True if cube left the reachable workspace (XY) or fell below the floor (Z)."""
    cube_pos_w = env.scene["cube"].data.root_link_pos_w
    local_pos = cube_pos_w - env.scene.env_origins
    xy_dist = torch.norm(local_pos[:, :2], dim=-1)
    return (xy_dist > workspace_radius) | (local_pos[:, 2] < z_min)
