"""Ball command: target placement, ball scatter and at-target success.

Drives all ball variants (golf / push / throw x marked / U-pocket / hole). The
target is a fixed structure on the field; the command samples the ball spawn,
tracks the ball, and decides success (ball inside the target and at rest; for
hole variants the ball must also have dropped below the platform surface).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g3a_bench.tasks.ball.variants import (
    BALL_RADIUS,
    BALL_SPAWN_X,
    HOLE_RADIUS,
    BallVariantCfg,
)
from g3a_bench.tasks.common.debug_markers import marker_color

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class BallCommandCfg(CommandTermCfg):
    ball_name: str
    variant: BallVariantCfg
    spawn_x_range: tuple[float, float] = (BALL_SPAWN_X - 0.03, BALL_SPAWN_X + 0.03)
    spawn_y_range: tuple[float, float] = (-0.08, 0.08)
    grasp_cfg: SceneEntityCfg | None = None

    def build(self, env: ManagerBasedRlEnv) -> BallCommand:
        return BallCommand(self, env)


class BallCommand(CommandTerm):
    cfg: BallCommandCfg

    def __init__(self, cfg: BallCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.variant = cfg.variant
        self.ball: Entity = env.scene[cfg.ball_name]
        self.grasp_cfg = cfg.grasp_cfg or SceneEntityCfg("robot")
        # Resolve names -> indices against the scene. Unlike term ``asset_cfg``s
        # (which the managers resolve), this command-owned cfg is resolved here,
        # so ``grasp_cfg.site_ids`` picks out the single grasp site instead of
        # defaulting to all sites.
        self.grasp_cfg.resolve(env.scene)

        E, dev = self.num_envs, self.device
        self.surface_z = self.variant.surface_z
        self.target_radius = (
            HOLE_RADIUS if self.variant.target == "hole" else self.variant.target_radius
        )

        target_local = torch.tensor(
            [self.variant.target_x, 0.0, self.surface_z], device=dev
        )
        self.target_pos_w = self._env.scene.env_origins + target_local  # (E, 3)

        self.ball_pos_w = torch.zeros(E, 3, device=dev)
        self.ball_vel_w = torch.zeros(E, 3, device=dev)
        self.at_target = torch.zeros(E, device=dev)
        self.success = torch.zeros(E, device=dev)
        self.episode_success = torch.zeros(E, device=dev)

        self.metrics["target_distance"] = torch.zeros(E, device=dev)
        self.metrics["at_target"] = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return self.target_pos_w - self.ball_pos_w

    def is_ball_released(self) -> torch.Tensor:
        """Check if ball is released (not held by gripper)."""
        robot: Entity = self._env.scene[self.grasp_cfg.name]
        grasp_site_pos = robot.data.site_pos_w[:, self.grasp_cfg.site_ids].squeeze(1)
        gripper_distance = torch.norm(self.ball_pos_w - grasp_site_pos, dim=-1)
        return gripper_distance > 0.08

    def _update_metrics(self) -> None:
        self.ball_pos_w = self.ball.data.root_link_pos_w
        self.ball_vel_w = self.ball.data.root_link_lin_vel_w

        d_xy = torch.norm(self.ball_pos_w[:, :2] - self.target_pos_w[:, :2], dim=-1)
        speed = torch.norm(self.ball_vel_w, dim=-1)
        within = d_xy < self.target_radius
        if self.variant.target == "hole":
            dropped = self.ball_pos_w[:, 2] < self.surface_z
            at_target = within & dropped
        else:
            on_surface = self.ball_pos_w[:, 2] < self.surface_z + 3 * BALL_RADIUS
            at_target = within & on_surface
        at_rest = speed < self.variant.at_rest_speed
        is_released = self.is_ball_released()

        self.at_target = at_target.float()
        self.success = (at_target & at_rest & is_released).float()
        self.episode_success = torch.maximum(self.episode_success, self.success)

        self.metrics["target_distance"] = d_xy
        self.metrics["at_target"] = self.at_target
        self.metrics["episode_success"] = self.episode_success

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        dev = self.device
        origins = self._env.scene.env_origins[env_ids]

        self.success[env_ids] = 0.0
        self.episode_success[env_ids] = 0.0

        x0, x1 = self.cfg.spawn_x_range
        y0, y1 = self.cfg.spawn_y_range
        x = x0 + torch.rand(n, device=dev) * (x1 - x0)
        y = y0 + torch.rand(n, device=dev) * (y1 - y0)
        z = torch.full((n,), self.surface_z + BALL_RADIUS + 0.002, device=dev)
        pos = torch.stack([x, y, z], dim=-1) + origins
        quat = torch.zeros(n, 4, device=dev)
        quat[:, 0] = 1.0
        self.ball.write_root_link_pose_to_sim(
            torch.cat([pos, quat], dim=-1), env_ids=env_ids
        )
        self.ball.write_root_link_velocity_to_sim(
            torch.zeros(n, 6, device=dev), env_ids=env_ids
        )

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        for batch in env_indices:
            center = self.target_pos_w[batch].cpu().numpy()
            done = bool(self.at_target[batch] > 0.5)
            success = bool(self.success[batch] > 0.5)
            color = marker_color((0.9, 0.9, 0.2, 1.0), done, success)
            # Success is an xy-radius test on the surface: draw a flat disk of
            # the success radius sitting on it (lifted slightly to avoid
            # z-fighting), not a half-buried sphere.
            start = center.copy()
            start[2] += 0.0005
            end = center.copy()
            end[2] += 0.003
            visualizer.add_cylinder(
                start=start,
                end=end,
                radius=self.target_radius,
                color=color,
                label=f"target_{batch}",
            )
