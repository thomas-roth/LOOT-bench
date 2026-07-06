"""Insertion command: goal bores, peg scatter and insertion success.

One command term drives all three insertion kinds (single peg, two-sheet, shape
sorter). It owns:

- the world pose of every bore mouth (from :func:`socket_layout`, so geometry
  and success metric stay in sync). Peg/two-sheet boards are fixed props, so
  their mouths are constant; the sorter cube is a *free body*, so its mouths
  and axes are re-derived from the cube's live root pose every step,
- the per-reset scatter of pegs, the offset of the movable top sheet, and (for
  the sorter) the cube's random flat orientation plus the commanded target
  piece, and
- the insertion test. Peg/two-sheet: instantaneous axial depth + lateral
  centring + tilt (+ sheet alignment). Sorter: a *latched* mouth-plane
  crossing -- a piece scores the moment it passes through its own hole into
  the cube, stays scored while it tumbles around inside, and no shape piece fits
  into a foreign hole.

Peg ``i`` is assigned to socket ``i``; extra pegs are distractors. Socket order
is the variant's socket order. For the sorter, success means the single
commanded target piece is latched in; the other pieces are clutter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
)

from g3a_bench.tasks.common.debug_markers import MARKER_INFLATE, marker_color
from g3a_bench.tasks.insertion.assets import (
    BOARD_POS,
    PALETTE,
    SHEET_TOP_ROOT_Z,
    SORTER_PIECE_CELLS,
    SORTER_SPAWN_JITTER,
    TRAY_POS,
    TRAY_SPAWN_HALF,
    frame_axes,
    load_sorter_layout,
    socket_layout,
)
from g3a_bench.tasks.insertion.variants import (
    PEG_HALF_LEN,
    SHAPE_GEOMS,
    InsertionVariantCfg,
    Shape,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class InsertionCommandCfg(CommandTermCfg):
    board_name: str
    peg_names: tuple[str, ...]
    """All peg entities; the first ``num_sockets`` are assigned to sockets."""
    variant: InsertionVariantCfg
    sheet_name: str | None = None
    """Movable top sheet entity (two-sheet variant only)."""
    sheet_align_tol: float = 0.010
    """Max top-sheet/bottom-hole offset to count as aligned."""
    peg_vel_threshold: float = 0.05
    """Max linear speed (m/s) of any assigned peg to count success."""
    gripper_open_threshold: float = -0.015
    """Min gripper_left joint position (m) to count success (> -0.015 = ≥2/3 open)."""

    def build(self, env: ManagerBasedRlEnv) -> InsertionCommand:
        return InsertionCommand(self, env)


class InsertionCommand(CommandTerm):
    cfg: InsertionCommandCfg

    def __init__(self, cfg: InsertionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        variant = cfg.variant
        self.variant = variant
        self.pegs: list[Entity] = [env.scene[n] for n in cfg.peg_names]
        self.sheet: Entity | None = (
            env.scene[cfg.sheet_name] if cfg.sheet_name is not None else None
        )
        self.num_pegs = len(self.pegs)
        self.num_sockets = variant.num_sockets
        assert self.num_pegs >= self.num_sockets

        E, S, N = self.num_envs, self.num_sockets, self.num_pegs
        dev = self.device
        self._is_sorter = variant.kind == "sorter"

        frames = socket_layout(variant)
        self._mouth_local = torch.tensor(
            np.stack([f.mouth for f in frames]), device=dev, dtype=torch.float32
        )
        self._axis_local = torch.tensor(
            np.stack([f.axis_out for f in frames]), device=dev, dtype=torch.float32
        )
        board_pos = torch.tensor(BOARD_POS, device=dev, dtype=torch.float32)
        origins = self._env.scene.env_origins  # (E, 3)
        if self._is_sorter:
            # The cube is a free body: mouth/axis buffers are refreshed from
            # its live root pose every step (see _update_metrics).
            self.board: Entity | None = env.scene[cfg.board_name]
            self.socket_mouth_w = torch.zeros(E, S, 3, device=dev)
            self.socket_axis_w = torch.zeros(E, S, 3, device=dev)
        else:
            # Fixed prop pinned to the env origin (identity orientation), so
            # world mouth poses are constant per env.
            self.board = None
            self.socket_mouth_w = (
                origins.unsqueeze(1) + board_pos + self._mouth_local.unsqueeze(0)
            )  # (E, S, 3)
            self.socket_axis_w = (
                self._axis_local.unsqueeze(0).expand(E, S, 3).contiguous()
            )

        # Per-socket marker data for debug vis: peg shape/color and the bore
        # cross-section basis (columns c0, c1, axis_out), matching the walls.
        self._socket_shapes: tuple[Shape, ...] = tuple(f.shape for f in frames)
        self._socket_colors = tuple(PALETTE[f.color] for f in frames)
        self._socket_mats = [np.stack(frame_axes(f.axis_out), axis=1) for f in frames]

        self.insert_depth = variant.insert_depth
        self.lateral_tol = variant.lateral_tol
        self.tilt_cos = math.cos(math.radians(variant.tilt_tol_deg))

        self.peg_pos_w = torch.zeros(E, N, 3, device=dev)
        self.inserted = torch.zeros(E, S, device=dev)
        self.success = torch.zeros(E, device=dev)
        self.episode_success = torch.zeros(E, device=dev)
        self._half_len = PEG_HALF_LEN

        if self._is_sorter:
            assert N == S, "sorter pieces map 1:1 onto holes (no distractor pegs)"
            layout = load_sorter_layout()
            self._cube_half = layout.half_size
            self._piece_half = torch.tensor(
                [layout.pieces[s.shape].half_len for s in variant.sockets],
                device=dev,
            )  # (S,)

            # Lateral gate of the insertion channel: hole outline + margin.
            self._hole_radius = (
                torch.tensor(
                    [layout.holes[s.shape].bounding_radius for s in variant.sockets],
                    device=dev,
                )
                + 0.005
            )

            # Crossing plane: once the piece centre passes s_cross (measured
            # from the mouth along axis_out, + = outside), its tail has
            # cleared the inner wall plane -- it is fully inside the cube.
            wall_t = torch.tensor(
                [layout.holes[s.shape].wall_thickness for s in variant.sockets],
                device=dev,
            )
            self._s_cross = -(wall_t + self._piece_half)  # (S,)
            self._sorter_tilt_cos = math.cos(math.radians(30.0))
            self.target_idx = torch.zeros(E, dtype=torch.long, device=dev)
            self.latched = torch.zeros(E, S, dtype=torch.bool, device=dev)
            self._prev_sv = torch.full((E, S), 1.0, device=dev)
            self._prev_in_channel = torch.zeros(E, S, dtype=torch.bool, device=dev)

            # The six flat resting orientations (each face down), composed
            # with a random yaw on reset.
            axes = torch.tensor(
                [
                    [1.0, 0.0, 0.0],  # identity ....... -z face down (circle)
                    [1.0, 0.0, 0.0],  # roll pi ........ +z face down (star)
                    [0.0, 1.0, 0.0],  # pitch +pi/2 .... +x face down (square)
                    [0.0, 1.0, 0.0],  # pitch -pi/2 .... -x face down (rect)
                    [1.0, 0.0, 0.0],  # roll -pi/2 ..... +y face down (hexagon)
                    [1.0, 0.0, 0.0],  # roll +pi/2 ..... -y face down (triangle)
                ],
                device=dev,
            )
            angles = torch.tensor(
                [0.0, math.pi, math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2],
                device=dev,
            )
            self._flat_quats = quat_from_angle_axis(angles, axes)  # (6, 4)

        # Resolved lazily: joint data isn't available at __init__ time.
        self._gripper_joint_idx: int = -1

        # Pegs/pieces are spawned upright into the parts tray, env-local (env
        # origins are added at spawn time). Pegs use a jittered grid; the bigger
        # sorter pieces use fixed 60 mm-pitch cells whose spacing rules out overlap
        # at any yaw.
        tx, ty, _ = TRAY_POS
        if self._is_sorter:
            self._cells = torch.tensor(
                [(tx + dx, ty + dy) for dx, dy in SORTER_PIECE_CELLS], device=dev
            )
            self._cell_jitter = torch.zeros(2, device=dev)
        else:
            hx, hy = TRAY_SPAWN_HALF
            x0, x1 = tx - hx, tx + hx
            y0, y1 = ty - hy, ty + hy
            nx, ny = 2, 4
            cx = torch.linspace(x0, x1, nx + 1, device=dev)
            cy = torch.linspace(y0, y1, ny + 1, device=dev)
            self._cells = torch.cartesian_prod(
                (cx[:-1] + cx[1:]) / 2, (cy[:-1] + cy[1:]) / 2
            )
            self._cell_jitter = 0.4 * torch.tensor(
                [(x1 - x0) / nx / 2, (y1 - y0) / ny / 2], device=dev
            )
        assert self._cells.shape[0] >= N

        self.metrics["lateral_error"] = torch.zeros(E, device=dev)
        self.metrics["num_inserted"] = torch.zeros(E, device=dev)
        self.metrics["all_inserted"] = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)
        if self._is_sorter:
            self.metrics["target_inserted"] = torch.zeros(E, device=dev)
        if self.sheet is not None:
            self.metrics["sheet_align_error"] = torch.zeros(E, device=dev)

    # Helpers. --------------------------------------------------------------

    def assigned_peg_pos(self) -> torch.Tensor:
        """Position of the peg assigned to each socket, (E, S, 3)."""
        return self.peg_pos_w[:, : self.num_sockets]

    def scored_fraction(self) -> torch.Tensor:
        """Task progress in [0, 1], (E,).

        Fixed kinds: fraction of sockets filled. Sorter: whether the single
        commanded target piece is latched in (the other pieces are clutter and
        do not score).
        """
        if self._is_sorter:
            tgt = self.target_idx.unsqueeze(1)
            return self.latched.gather(1, tgt).squeeze(1).float()
        return self.inserted.sum(dim=1) / self.num_sockets

    @property
    def command(self) -> torch.Tensor:
        mouth = (
            self.socket_mouth_w - self._env.scene.env_origins.unsqueeze(1)
        ).reshape(self.num_envs, -1)
        axis = self.socket_axis_w.reshape(self.num_envs, -1)
        if self._is_sorter:
            # One-hot selector of the commanded target piece/hole; the cube
            # pose itself is observable through the live mouths/axes above.
            one_hot = torch.nn.functional.one_hot(
                self.target_idx, self.num_sockets
            ).float()
            return torch.cat([mouth, axis, one_hot], dim=-1)
        return torch.cat([mouth, axis], dim=-1)

    # CommandTerm interface. ------------------------------------------------

    def _update_metrics(self) -> None:
        self.peg_pos_w = torch.stack([p.data.root_link_pos_w for p in self.pegs], dim=1)
        peg_quat = torch.stack([p.data.root_link_quat_w for p in self.pegs], dim=1)
        E, S = self.num_envs, self.num_sockets

        if self._is_sorter:
            # Re-derive the mouth/axis buffers from the cube's live root pose.
            # Everything downstream (socket_deltas obs, command tensor, debug
            # vis) reads these buffers, so they track a moving cube for free.
            assert self.board is not None
            cube_pos = self.board.data.root_link_pos_w  # (E, 3)
            cube_quat = self.board.data.root_link_quat_w  # (E, 4)
            q = cube_quat.unsqueeze(1).expand(E, S, 4)
            self.socket_mouth_w = cube_pos.unsqueeze(1) + quat_apply(
                q, self._mouth_local.unsqueeze(0).expand(E, S, 3)
            )
            self.socket_axis_w = quat_apply(
                q, self._axis_local.unsqueeze(0).expand(E, S, 3)
            )

        pos = self.assigned_peg_pos()  # (E, S, 3)
        quat = peg_quat[:, :S]
        axis = self.socket_axis_w
        rel = pos - self.socket_mouth_w
        axial = torch.sum(rel * axis, dim=-1)  # signed, + is outside
        lateral = torch.norm(rel - axial.unsqueeze(-1) * axis, dim=-1)

        w, x, y, z = quat.unbind(-1)
        peg_axis = torch.stack(
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1
        )
        cos_tilt = torch.sum(peg_axis * axis, dim=-1).abs()

        if self._is_sorter:
            # Latched crossing of the insertion channel: the piece was in the
            # channel above the crossing plane on the previous step and is at
            # or below it now. The event latches until reset, so it survives
            # the piece tumbling around inside the hollow cube.
            in_channel = (
                (lateral < self._hole_radius)
                & (axial > self._s_cross - 0.02)
                & (axial < self._piece_half + 0.01)
            )
            crossed = (
                self._prev_in_channel
                & in_channel
                & (self._prev_sv > self._s_cross)
                & (axial <= self._s_cross)
                & (cos_tilt > self._sorter_tilt_cos)
            )
            self.latched |= crossed
            self._prev_sv = axial.clone()
            self._prev_in_channel = in_channel
            inserted = self.latched
        else:
            center_depth = -axial
            tip_depth = center_depth + self._half_len * cos_tilt
            inserted = (
                (tip_depth >= self.insert_depth)
                & (lateral < self.lateral_tol)
                & (cos_tilt > self.tilt_cos)
            )
            if self.sheet is not None:
                sheet_xy = self.sheet.data.root_link_pos_w[:, :2]
                align_err = torch.norm(sheet_xy - self.socket_mouth_w[:, 0, :2], dim=-1)
                inserted = inserted & (align_err < self.cfg.sheet_align_tol).unsqueeze(
                    1
                )
                self.metrics["sheet_align_error"] = align_err

        self.inserted = inserted.float()
        all_in = inserted.all(dim=1)

        if self._gripper_joint_idx < 0:
            robot = self._env.scene["robot"]
            self._gripper_joint_idx = list(robot.joint_names).index("gripper_left")

        peg_lin_vel_w = torch.stack(
            [p.data.root_link_lin_vel_w for p in self.pegs], dim=1
        )  # (E, N, 3)
        assigned_speed = peg_lin_vel_w[:, :S].norm(dim=-1)  # (E, S)

        # Gripper-open gate.
        robot = self._env.scene["robot"]
        gripper_q = robot.data.joint_pos[:, self._gripper_joint_idx]
        gripper_open = gripper_q > self.cfg.gripper_open_threshold

        if self._is_sorter:
            tgt = self.target_idx.unsqueeze(1)
            target_in = self.latched.gather(1, tgt).squeeze(1)
            target_still = (
                assigned_speed.gather(1, tgt).squeeze(1) <= self.cfg.peg_vel_threshold
            )
            all_conditions = target_in & target_still & gripper_open
            self.metrics["target_inserted"] = target_in.float()
        else:
            pegs_still = assigned_speed.max(dim=-1).values <= self.cfg.peg_vel_threshold
            all_conditions = all_in & pegs_still & gripper_open

        self.success = all_conditions.float()
        self.episode_success = torch.maximum(self.episode_success, self.success)

        self.metrics["lateral_error"] = lateral.mean(dim=1)
        self.metrics["num_inserted"] = self.inserted.sum(dim=1)
        self.metrics["all_inserted"] = all_in.float()
        self.metrics["episode_success"] = self.episode_success

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        dev = self.device
        N = self.num_pegs
        origins = self._env.scene.env_origins[env_ids]

        self.success[env_ids] = 0.0
        self.episode_success[env_ids] = 0.0
        self.inserted[env_ids] = 0.0

        if self._is_sorter:
            assert self.board is not None
            self.latched[env_ids] = False
            self._prev_in_channel[env_ids] = False
            self._prev_sv[env_ids] = 1.0  # far outside: no spurious crossing
            self.target_idx[env_ids] = torch.randint(self.num_sockets, (n,), device=dev)
            # Drop the cube near the board spot at random yaw with any of its six faces down
            face = torch.randint(6, (n,), device=dev)
            yaw = quat_from_euler_xyz(
                torch.zeros(n, device=dev),
                torch.zeros(n, device=dev),
                torch.rand(n, device=dev) * 2 * math.pi,
            )
            cube_quat = quat_mul(yaw, self._flat_quats[face])
            jitter = (torch.rand(n, 2, device=dev) - 0.5) * 2.0
            jitter = jitter * torch.tensor(SORTER_SPAWN_JITTER, device=dev)
            cube_pos = origins + torch.tensor(
                BOARD_POS, device=dev, dtype=torch.float32
            )
            cube_pos = cube_pos + torch.cat(
                [
                    jitter,
                    torch.full((n, 1), self._cube_half + 0.002, device=dev),
                ],
                dim=-1,
            )
            self.board.write_root_link_pose_to_sim(
                torch.cat([cube_pos, cube_quat], dim=-1), env_ids=env_ids
            )
            self.board.write_root_link_velocity_to_sim(
                torch.zeros(n, 6, device=dev), env_ids=env_ids
            )

        # Scatter pegs upright on distinct jittered cells inside the parts tray.
        pick = torch.argsort(torch.rand(n, self._cells.shape[0], device=dev), dim=1)[
            :, :N
        ]
        xy = self._cells[pick] + (torch.rand(n, N, 2, device=dev) - 0.5) * 2.0 * (
            self._cell_jitter
        )
        spawn_yaw = torch.rand(n, N, device=dev) * 2 * math.pi
        zeros = torch.zeros(n, device=dev)
        for i, peg in enumerate(self.pegs):
            half = float(self._piece_half[i]) if self._is_sorter else self._half_len
            z = (half + 0.002 + origins[:, 2]).unsqueeze(-1)
            p = torch.cat([xy[:, i, :] + origins[:, :2], z], dim=-1)
            quat = quat_from_euler_xyz(zeros, zeros, spawn_yaw[:, i])
            peg.write_root_link_pose_to_sim(
                torch.cat([p, quat], dim=-1), env_ids=env_ids
            )
            peg.write_root_link_velocity_to_sim(
                torch.zeros(n, 6, device=dev), env_ids=env_ids
            )

        # Place the movable top sheet, hole offset from the bottom hole.
        if self.sheet is not None:
            bottom = self.socket_mouth_w[env_ids, 0]  # (n, 3)
            ang = torch.rand(n, device=dev) * 2 * math.pi
            off = 0.03 + torch.rand(n, device=dev) * 0.02
            sx = bottom[:, 0] + off * torch.cos(ang)
            sy = bottom[:, 1] + off * torch.sin(ang)
            sz = origins[:, 2] + SHEET_TOP_ROOT_Z
            pos = torch.stack([sx, sy, sz], dim=-1)
            quat = quat_from_euler_xyz(zeros, zeros, torch.zeros(n, device=dev))
            self.sheet.write_root_link_pose_to_sim(
                torch.cat([pos, quat], dim=-1), env_ids=env_ids
            )
            self.sheet.write_root_link_velocity_to_sim(
                torch.zeros(n, 6, device=dev), env_ids=env_ids
            )

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        for batch in env_indices:
            success = bool(self.success[batch] > 0.5)
            for s in range(self.num_sockets):
                mouth = self.socket_mouth_w[batch, s].cpu().numpy()
                axis = self.socket_axis_w[batch, s].cpu().numpy()
                done = bool(self.inserted[batch, s] > 0.5)
                color = marker_color(self._socket_colors[s], done, success)
                # Ghost peg at the minimum successful insertion: tip at
                # insert_depth below the mouth, extending one peg length out.
                tip = mouth - axis * self.insert_depth
                top = tip + axis * (2.0 * self._half_len)
                shape = self._socket_shapes[s]
                geom = SHAPE_GEOMS[shape]
                label = f"bore_{batch}_{s}"
                if shape in ("square", "rect"):
                    half_xy = geom.rect_half or (geom.inradius, geom.inradius)
                    visualizer.add_box(
                        center=(tip + top) / 2.0,
                        size=np.array([half_xy[0], half_xy[1], self._half_len])
                        + MARKER_INFLATE,
                        mat=self._socket_mats[s],
                        color=color,
                        label=label,
                    )
                else:
                    # Triangle/hexagon have no polygon-prism primitive; a
                    # cylinder of the circumradius is the closest hull.
                    visualizer.add_cylinder(
                        start=tip,
                        end=top,
                        radius=geom.circumradius + MARKER_INFLATE,
                        color=color,
                        label=label,
                    )
                is_target = self._is_sorter and int(self.target_idx[batch]) == s
                if done:
                    color = (0.2, 0.8, 0.2, 0.5)
                elif is_target:
                    color = (0.9, 0.2, 0.2, 0.7)  # commanded hole
                else:
                    color = (0.9, 0.9, 0.2, 0.4)
                visualizer.add_sphere(
                    center=mouth, radius=0.015, color=color, label=f"bore_{batch}_{s}"
                )


def get_command(env: ManagerBasedRlEnv, name: str) -> InsertionCommand:
    return cast(InsertionCommand, env.command_manager.get_term(name))
