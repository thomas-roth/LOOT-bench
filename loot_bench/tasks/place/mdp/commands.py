"""Place command. Scatters the cube, draws a target, tests placement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor

from loot_bench.tasks.common.commands import (
    GRIP_CONTACT_SENSOR_NAME,
    ContactReleaseGate,
)
from loot_bench.tasks.common.debug_markers import marker_color
from loot_bench.tasks.common.scatter import ScatterGrid, write_upright_spawn
from loot_bench.tasks.common.success import OBJECT_VEL_THRESHOLD, SuccessGate
from loot_bench.tasks.place.mdp.success import place_progress, place_reached
from loot_bench.tasks.place.variants import CUBE_HALF_EXTENTS, PlaceVariantCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_TARGET_RGBA = (0.95, 0.95, 0.2, 1.0)

SPAWN_X_RANGE = (0.20, 0.28)
TARGET_X_RANGE = (0.36, 0.44)
WORKSPACE_Y = (-0.18, 0.18)


def _midpoint(r: tuple[float, float]) -> float:
    return 0.5 * (r[0] + r[1])


FIXED_SPAWN_XY = (_midpoint(SPAWN_X_RANGE), _midpoint(WORKSPACE_Y))
FIXED_TARGET_XY = (_midpoint(TARGET_X_RANGE), _midpoint(WORKSPACE_Y))
"""Poses the Fixed variants pin to. Derived from ranges, never drifts."""

FIXED_SPAWN_YAW = 0.0
"""A pinned spawn is fully deterministic: position and orientation."""

MAX_TRAVEL_DIST = math.hypot(
    TARGET_X_RANGE[1] - SPAWN_X_RANGE[0], WORKSPACE_Y[1] - WORKSPACE_Y[0]
)
"""Largest spawn to target distance. Normalizes progress to a constant."""


def get_command(env: ManagerBasedRlEnv, name: str) -> PlaceCommand:
    return cast(PlaceCommand, env.command_manager.get_term(name))


@dataclass(kw_only=True)
class PlaceCommandCfg(CommandTermCfg):
    cube_name: str
    variant: PlaceVariantCfg
    spawn_x_range: tuple[float, float] = SPAWN_X_RANGE
    spawn_y_range: tuple[float, float] = WORKSPACE_Y
    target_x_range: tuple[float, float] = TARGET_X_RANGE
    target_y_range: tuple[float, float] = WORKSPACE_Y
    pos_threshold: float = 0.02
    """Max 3D distance between the cube and the target to count as placed
    (x margin)."""
    cube_vel_threshold: float = OBJECT_VEL_THRESHOLD
    """Max linear speed (m/s) of the cube to count success."""

    def build(self, env: ManagerBasedRlEnv) -> PlaceCommand:
        return PlaceCommand(self, env)


class PlaceCommand(CommandTerm):
    cfg: PlaceCommandCfg

    def __init__(self, cfg: PlaceCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.variant = cfg.variant
        self.cube: Entity = env.scene[cfg.cube_name]
        E, dev = self.num_envs, self.device

        self._half_h = CUBE_HALF_EXTENTS[2]
        object_radius = math.hypot(CUBE_HALF_EXTENTS[0], CUBE_HALF_EXTENTS[1])
        self._scatter = ScatterGrid.tiling(
            cfg.spawn_x_range,
            cfg.spawn_y_range,
            object_radius=object_radius,
            device=dev,
        )

        self.target_pos_w = torch.zeros(E, 3, device=dev)
        self.cube_pos_w = torch.zeros(E, 3, device=dev)
        self.cube_quat_w = torch.zeros(E, 4, device=dev)
        self.dist_to_target = torch.zeros(E, device=dev)
        self.reached = torch.zeros(E, device=dev, dtype=torch.bool)
        self.progress = torch.zeros(E, device=dev)
        self._gate = SuccessGate(E, dev, vel_threshold=cfg.cube_vel_threshold)
        self._contact_gate = ContactReleaseGate()
        self._grip_sensor = cast(
            ContactSensor, env.scene.sensors[GRIP_CONTACT_SENSOR_NAME]
        )

        # Fixed poses / normalizer derived from cfg so overrides stay
        # consistent. Module constants remain as defaults for docs/imports.
        self._fixed_spawn_xy = (
            _midpoint(cfg.spawn_x_range),
            _midpoint(cfg.spawn_y_range),
        )
        self._fixed_target_xy = (
            _midpoint(cfg.target_x_range),
            _midpoint(cfg.target_y_range),
        )
        spawn_x0, spawn_x1 = cfg.spawn_x_range
        spawn_y0, spawn_y1 = cfg.spawn_y_range
        target_x0, target_x1 = cfg.target_x_range
        target_y0, target_y1 = cfg.target_y_range
        dx = max(abs(target_x1 - spawn_x0), abs(target_x0 - spawn_x1))
        dy = max(abs(target_y1 - spawn_y0), abs(target_y0 - spawn_y1))
        self._max_travel_dist = math.hypot(dx, dy)

        self.metrics["dist_to_target"] = torch.zeros(E, device=dev)
        self.metrics["reached"] = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)

    @property
    def command(self) -> torch.Tensor:
        """(E, 3): env-local target position."""
        return self.target_pos_w - self._env.scene.env_origins

    # Helpers used by observation/reward/termination terms. ----------------

    @property
    def success(self) -> torch.Tensor:
        """(E,) float: cube at target, at rest, gripper released."""
        return self._gate.success

    @property
    def episode_success(self) -> torch.Tensor:
        """(E,) float: success latched since the last reset."""
        return self._gate.episode_success

    # CommandTerm interface. ------------------------------------------------

    def _update_metrics(self) -> None:
        self.cube_pos_w = self.cube.data.root_link_pos_w
        self.cube_quat_w = self.cube.data.root_link_quat_w
        self.dist_to_target = torch.norm(self.cube_pos_w - self.target_pos_w, dim=-1)

        margin = self.variant.margin
        self.reached = place_reached(
            self.dist_to_target, self.cfg.pos_threshold * margin
        )
        self.progress = place_progress(self.dist_to_target, self._max_travel_dist)

        speed = self.cube.data.root_link_lin_vel_w.norm(dim=-1)
        released = self._contact_gate(self._grip_sensor)
        self._gate.update(self.reached, speed, released)

        # shaped_task_reward handles the exact zero on latched success.
        self.metrics["dist_to_target"] = self.dist_to_target
        self.metrics["reached"] = self.reached.float()
        self.metrics["episode_success"] = self.episode_success

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        env_ids_len = len(env_ids)
        if env_ids_len == 0:
            return
        dev = self.device
        origins = self._env.scene.env_origins[env_ids]

        self._gate.reset(env_ids)

        # Cube spawn: scatter draw, or the pinned pose if spawn is fixed.
        if self.variant.random_spawn:
            xy, spawn_yaw = self._scatter.sample_poses(env_ids_len, 1)
        else:
            xy = torch.tensor(self._fixed_spawn_xy, device=dev).expand(
                env_ids_len, 1, 2
            )
            spawn_yaw = torch.full((env_ids_len, 1), FIXED_SPAWN_YAW, device=dev)
        write_upright_spawn(
            [self.cube],
            xy,
            spawn_yaw,
            torch.full((1,), self._half_h, device=dev),
            origins,
            env_ids,
        )

        # Target: uniform draw in its region, or the pinned point.
        if self.variant.random_target:
            x0, x1 = self.cfg.target_x_range
            y0, y1 = self.cfg.target_y_range
            u = torch.rand(env_ids_len, 2, device=dev)
            x_tensor = x0 + u[:, 0] * (x1 - x0)
            y_tensor = y0 + u[:, 1] * (y1 - y0)
        else:
            x_tensor = torch.full((env_ids_len,), self._fixed_target_xy[0], device=dev)
            y_tensor = torch.full((env_ids_len,), self._fixed_target_xy[1], device=dev)
        z_tensor = torch.full((env_ids_len,), self._half_h, device=dev)
        self.target_pos_w[env_ids] = (
            torch.stack([x_tensor, y_tensor, z_tensor], dim=-1) + origins
        )

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        radius = self.cfg.pos_threshold * self.variant.margin
        for batch in env_indices:
            center = self.target_pos_w[batch].cpu().numpy()
            reached = bool(self.reached[batch])
            success = bool(self.success[batch] > 0.5)
            visualizer.add_sphere(
                center=center,
                radius=radius,
                color=marker_color(_TARGET_RGBA, reached, success),
                label=f"target_{batch}",
            )
