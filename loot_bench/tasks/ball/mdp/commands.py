"""Ball command. Samples spawns, tracks the ball, tests at-target success."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from loot_bench.tasks.ball.mdp.success import ball_in_target_zone
from loot_bench.tasks.ball.variants import (
    BALL_RADIUS,
    BALL_SPAWN_X,
    HOLE_RADIUS,
    TARGET_X,
    WALL_X,
    BallVariantCfg,
)
from loot_bench.tasks.common.debug_markers import marker_color
from loot_bench.tasks.common.success import SuccessGate

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

# Grasp-site distance beyond which the ball counts as released (not held).
_RELEASE_DISTANCE = 0.08


@dataclass(kw_only=True)
class BallCommandCfg(CommandTermCfg):
    ball_name: str
    field_name: str
    variant: BallVariantCfg
    grasp_cfg: SceneEntityCfg
    """Robot grasp site for the release check; site set by manipulator wiring."""
    wall_name: str | None = None
    """Set only for ThrowWallVariation's independently-moving wall entity."""
    spawn_x_range: tuple[float, float] = (BALL_SPAWN_X - 0.03, BALL_SPAWN_X + 0.03)
    spawn_y_range: tuple[float, float] = (-0.08, 0.08)

    def build(self, env: ManagerBasedRlEnv) -> BallCommand:
        return BallCommand(self, env)


class BallCommand(CommandTerm):
    cfg: BallCommandCfg

    def __init__(self, cfg: BallCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.variant = cfg.variant
        self.ball: Entity = env.scene[cfg.ball_name]
        self.field: Entity = env.scene[cfg.field_name]
        self.wall: Entity | None = (
            env.scene[cfg.wall_name] if cfg.wall_name is not None else None
        )
        self.grasp_cfg = cfg.grasp_cfg
        # Resolve the grasp site here since this cfg is command owned.
        self.grasp_cfg.resolve(env.scene)

        E, dev = self.num_envs, self.device
        variant = self.variant
        self._is_hole = variant.task == "minigolf"
        self.spawn_z = variant.spawn_z
        self.target_z = variant.target_z
        self.target_radius = (
            HOLE_RADIUS if self._is_hole else variant.target_radius * variant.margin
        )

        target_y = variant.pit_y if self._is_hole else 0.0
        self._local_target = torch.tensor(
            [TARGET_X, target_y, self.target_z], device=dev
        )
        self._wall_local = torch.tensor([WALL_X, 0.0, 0.0], device=dev)

        self.field_pos_w = torch.zeros(E, 3, device=dev)
        self.field_quat_w = torch.zeros(E, 4, device=dev)
        self.wall_pos_w = torch.zeros(E, 3, device=dev)
        self.wall_quat_w = torch.zeros(E, 4, device=dev)
        self.target_pos_w = torch.zeros(E, 3, device=dev)
        self.ball_pos_w = torch.zeros(E, 3, device=dev)
        self.ball_vel_w = torch.zeros(E, 3, device=dev)
        self.at_target = torch.zeros(E, device=dev)
        self._gate = SuccessGate(E, device=dev, vel_threshold=variant.at_rest_speed)

        self.metrics["target_distance"] = torch.zeros(E, device=dev)
        self.metrics["at_target"] = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)

    @property
    def command(self) -> torch.Tensor:
        """(E, 3): env-local target position (was a ball-relative delta)."""
        return self.target_pos_w - self._env.scene.env_origins

    @property
    def success(self) -> torch.Tensor:
        """(E,) float: ball at target, at rest, and released."""
        return self._gate.success

    @property
    def episode_success(self) -> torch.Tensor:
        """(E,) float: success latched since the last reset."""
        return self._gate.episode_success

    def _is_ball_released(self) -> torch.Tensor:
        robot: Entity = self._env.scene[self.grasp_cfg.name]
        grasp_site_pos = robot.data.site_pos_w[:, self.grasp_cfg.site_ids].squeeze(1)
        gripper_distance = torch.norm(self.ball_pos_w - grasp_site_pos, dim=-1)
        return gripper_distance > _RELEASE_DISTANCE

    def _update_metrics(self) -> None:
        self.field_pos_w = self.field.data.root_link_pos_w
        self.field_quat_w = self.field.data.root_link_quat_w
        self.target_pos_w = self.field_pos_w + self._local_target
        if self.wall is not None:
            self.wall_pos_w = self.wall.data.root_link_pos_w
            self.wall_quat_w = self.wall.data.root_link_quat_w
        elif self.variant.has_wall:
            self.wall_pos_w = self.field_pos_w + self._wall_local
            self.wall_quat_w = self.field_quat_w

        self.ball_pos_w = self.ball.data.root_link_pos_w
        self.ball_vel_w = self.ball.data.root_link_lin_vel_w

        at_target, d_xy = ball_in_target_zone(
            self.ball_pos_w,
            self.target_pos_w,
            radius=self.target_radius,
            surface_z=self.target_z,
            is_hole=self._is_hole,
            ball_radius=BALL_RADIUS,
        )
        speed = torch.norm(self.ball_vel_w, dim=-1)

        self.at_target = at_target.float()
        self._gate.update(at_target, speed, self._is_ball_released())

        self.metrics["target_distance"] = d_xy
        self.metrics["at_target"] = self.at_target
        self.metrics["episode_success"] = self.episode_success

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        dev = self.device
        origins = self._env.scene.env_origins[env_ids]

        self._gate.reset(env_ids)

        x0, x1 = self.cfg.spawn_x_range
        y0, y1 = self.cfg.spawn_y_range
        x = x0 + torch.rand(n, device=dev) * (x1 - x0)
        y = y0 + torch.rand(n, device=dev) * (y1 - y0)
        z = torch.full((n,), self.spawn_z + BALL_RADIUS + 0.002, device=dev)
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
            # Flat disk marker on the surface, sized by the success radius.
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


def get_command(env: ManagerBasedRlEnv, name: str) -> BallCommand:
    return cast(BallCommand, env.command_manager.get_term(name))
