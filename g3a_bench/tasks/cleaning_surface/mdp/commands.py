# pyright: reportAttributeAccessIssue=false
"""Cleaning-surface task command: multi-variant sampling, object placement, metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

N_PATCHES = 4
CLEAN_RADIUS = 0.06  # metres: sponge centre must be within this XY distance
CLEAN_Z_MAX = 0.08  # metres: sponge z must be below this to count as cleaning
WORK_X = (0.25, 0.45)
WORK_Y = (-0.15, 0.15)
BIN_LOCAL = (0.30, 0.25, 0.0)  # bin centre relative to env_origin

VARIANT_WIPE_ONLY = 0
VARIANT_DEBRIS_TO_BIN = 1
VARIANT_FULL_CLEANING = 2
N_VARIANTS = 3


@dataclass(kw_only=True)
class CleaningSurfaceCommandCfg(CommandTermCfg):
    def build(self, env: ManagerBasedRlEnv) -> CleaningSurfaceCommand:
        return CleaningSurfaceCommand(self, env)


class CleaningSurfaceCommand(CommandTerm):
    cfg: CleaningSurfaceCommandCfg

    def __init__(self, cfg: CleaningSurfaceCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        E = self.num_envs
        dev = self.device

        self._sponge = env.scene["sponge"]
        self._debris = [env.scene[f"debris_{i}"] for i in range(2)]

        # Per-env goal state
        self.dirt_pos_w = torch.zeros(E, N_PATCHES, 3, device=dev)
        self.task_variant = torch.zeros(E, dtype=torch.long, device=dev)
        self.bin_center_w = torch.zeros(E, 3, device=dev)

        # Latching state — stored on env so the step event can update it
        env.dirt_cleaned_mask = torch.zeros(E, N_PATCHES, dtype=torch.bool, device=dev)

        # Delta-reward prev-state buffers
        env.prev_cleaned_ratio = torch.zeros(E, device=dev)
        env.prev_debris_dist = torch.zeros(E, device=dev)

        self.episode_success = torch.zeros(E, device=dev)
        self.metrics["cleaned_ratio"] = torch.zeros(E, device=dev)
        self.metrics["debris_in_bin"] = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)

    @property
    def command(self) -> torch.Tensor:
        """Goal vector. Shape: (num_envs, 18) = one-hot(3) + dirt_pos(12) + bin_pos(3)."""
        E = self.num_envs
        dev = self.device
        v_oh = torch.zeros(E, N_VARIANTS, device=dev)
        v_oh.scatter_(1, self.task_variant.unsqueeze(-1), 1.0)
        dirt_flat = self.dirt_pos_w.reshape(E, N_PATCHES * 3)
        return torch.cat([v_oh, dirt_flat, self.bin_center_w], dim=-1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        dev = self.device
        env = self._env
        origins = env.scene.env_origins[env_ids]  # (n, 3)

        # --- Task variant ---
        self.task_variant[env_ids] = torch.randint(0, N_VARIANTS, (n,), device=dev)

        # --- Dirt patch positions (env-local XY within work zone, z at floor) ---
        dx = torch.zeros(n, N_PATCHES, device=dev).uniform_(*WORK_X)
        dy = torch.zeros(n, N_PATCHES, device=dev).uniform_(*WORK_Y)
        dz = torch.full((n, N_PATCHES), 0.005, device=dev)
        dirt_local = torch.stack([dx, dy, dz], dim=-1)  # (n, 4, 3)
        self.dirt_pos_w[env_ids] = dirt_local + origins.unsqueeze(1)

        # --- Bin centre ---
        bin_local = torch.tensor(BIN_LOCAL, dtype=torch.float32, device=dev)
        self.bin_center_w[env_ids] = bin_local.unsqueeze(0) + origins

        # --- Place sponge near workspace centre ---
        id_quat = self._identity_quat(n, dev)
        sx = torch.zeros(n, device=dev).uniform_(0.30, 0.38)
        sy = torch.zeros(n, device=dev).uniform_(-0.08, 0.08)
        sz = torch.full((n,), 0.01, device=dev)
        sponge_w = torch.stack([sx, sy, sz], dim=-1) + origins
        self._sponge.write_root_link_pose_to_sim(
            torch.cat([sponge_w, id_quat], dim=-1), env_ids=env_ids
        )
        self._sponge.write_root_link_velocity_to_sim(
            torch.zeros(n, 6, device=dev), env_ids=env_ids
        )

        # --- Place debris and capture world positions for init prev_debris_dist ---
        debris_world: list[torch.Tensor] = []
        for i, deb in enumerate(self._debris):
            y_lo = -0.10 + i * 0.20
            y_hi = y_lo + 0.10
            bx = torch.zeros(n, device=dev).uniform_(0.30, 0.40)
            by = torch.zeros(n, device=dev).uniform_(y_lo, y_hi)
            bz = torch.full((n,), 0.02, device=dev)
            deb_w = torch.stack([bx, by, bz], dim=-1) + origins
            debris_world.append(deb_w)
            deb.write_root_link_pose_to_sim(
                torch.cat([deb_w, id_quat], dim=-1), env_ids=env_ids
            )
            deb.write_root_link_velocity_to_sim(
                torch.zeros(n, 6, device=dev), env_ids=env_ids
            )

        # --- Reset latching mask ---
        env.dirt_cleaned_mask[env_ids] = False
        env.prev_cleaned_ratio[env_ids] = 0.0

        # Compute initial debris-to-bin distance for delta reward baseline
        bin_c = self.bin_center_w[env_ids]
        min_dist = torch.full((n,), float("inf"), device=dev)
        for deb_w in debris_world:
            d = torch.norm(deb_w[:, :2] - bin_c[:, :2], dim=-1)
            min_dist = torch.minimum(min_dist, d)
        env.prev_debris_dist[env_ids] = min_dist

        self.episode_success[env_ids] = 0.0

    def _update_metrics(self) -> None:
        env = self._env
        if not hasattr(env, "dirt_cleaned_mask"):
            return

        cleaned_ratio = env.dirt_cleaned_mask.float().mean(dim=-1)

        bin_c = self.bin_center_w
        in_bin = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(2):
            deb_pos = env.scene[f"debris_{i}"].data.root_link_pos_w
            in_xy = (torch.abs(deb_pos[:, :2] - bin_c[:, :2]) < 0.08).all(dim=-1)
            in_z = deb_pos[:, 2] < 0.10
            in_bin = in_bin & in_xy & in_z

        v = self.task_variant
        success = (
            ((v == VARIANT_WIPE_ONLY) & (cleaned_ratio >= 1.0))
            | ((v == VARIANT_DEBRIS_TO_BIN) & in_bin)
            | ((v == VARIANT_FULL_CLEANING) & (cleaned_ratio >= 1.0) & in_bin)
        )
        self.episode_success = torch.maximum(self.episode_success, success.float())

        self.metrics["cleaned_ratio"] = cleaned_ratio
        self.metrics["debris_in_bin"] = in_bin.float()
        self.metrics["episode_success"] = self.episode_success

    def _update_command(self) -> None:
        pass

    @staticmethod
    def _identity_quat(n: int, device: str) -> torch.Tensor:
        q = torch.zeros(n, 4, device=device)
        q[:, 0] = 1.0
        return q
