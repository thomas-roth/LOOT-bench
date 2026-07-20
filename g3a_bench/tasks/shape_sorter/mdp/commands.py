"""Task command for the shape-sorter category.

Owns the world pose of every hole mouth (from the cube's live root pose,
re-derived every step -- the cube is a free body, so its holes move with it),
the per-reset cube drop and piece scatter, and the insertion test (a latched
mouth-plane crossing per piece, ``mdp/success.py``). Unlike the legacy
insertion sorter (one commanded target piece per episode), *every* piece must
latch into its own hole for success.

At every resample the command draws exactly one shared ``rand(n, 4)`` for the
cube (jitter x, jitter y, face pick, yaw -- the face/yaw columns are drawn but
unused when the variant's ``spawn`` mode doesn't need them, so the draw count
never depends on the spawn mode), then scatters all pieces via
``ScatterGrid.sample_poses`` in index order (seed-prefix safe across piece
counts).

Deviation from a literal "identity quat" for ``spawn="fixed"``: identity
puts the sorter's circle hole (needed by every variant) flat against the
table, which no free-body reorientation-free episode can insert into. Instead
"fixed" drops the cube on the one face that is never a required hole (the
unused star socket -- see ``variants.FACE_DOWN_SHAPES``), so every variant
using it stays solvable without the robot having to flip the cube first,
matching the design intent of "fixedOrientation" being the easy baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
)

from g3a_bench.tasks.common.colors import PALETTE
from g3a_bench.tasks.common.commands import (
    GRIP_CONTACT_SENSOR_NAME,
    ContactReleaseGate,
    GripperGateCommandCfg,
    GripperOpenGate,
)
from g3a_bench.tasks.common.debug_markers import marker_color
from g3a_bench.tasks.common.scatter import ScatterGrid, write_upright_spawn
from g3a_bench.tasks.common.success import SuccessGate
from g3a_bench.tasks.shape_sorter.assets import (
    BOARD_POS,
    SORTER_SPAWN_JITTER,
    load_sorter_layout,
)
from g3a_bench.tasks.shape_sorter.mdp.success import (
    SorterLatch,
    peg_axial_lateral,
    peg_axis_cos_tilt,
)
from g3a_bench.tasks.shape_sorter.variants import ShapeSorterVariantCfg, allowed_faces

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

# The six flat resting orientations of the sorter cube (each face down),
# composed with a random yaw on reset -- copied from the legacy insertion
# sorter's table (order matches ``variants.FACE_DOWN_SHAPES``).
_FLAT_AXES = torch.tensor(
    [
        [1.0, 0.0, 0.0],  # identity ....... -z face down (circle)
        [1.0, 0.0, 0.0],  # roll pi ........ +z face down (star)
        [0.0, 1.0, 0.0],  # pitch +pi/2 .... +x face down (square)
        [0.0, 1.0, 0.0],  # pitch -pi/2 .... -x face down (rect)
        [1.0, 0.0, 0.0],  # roll -pi/2 ..... +y face down (hexagon)
        [1.0, 0.0, 0.0],  # roll +pi/2 ..... -y face down (triangle)
    ]
)
_FLAT_ANGLES = torch.tensor(
    [0.0, math.pi, math.pi / 2, -math.pi / 2, -math.pi / 2, math.pi / 2]
)
# Index into the table above whose down face is the (never required) star
# socket -- the deterministic pose used by ``spawn="fixed"``.
_FIXED_FACE_INDEX = 1

_NORMAL_WORKSPACE_X = (0.13, 0.27)
_NORMAL_WORKSPACE_Y = (-0.16, 0.16)
_NORMAL_OBJECT_RADIUS = 0.03
# Explicit ring of piece-spawn cells around the cube (env-local xy), used by
# ``wide_positions`` variants instead of the near band above.
_WIDE_CELLS = (
    (0.20, -0.13),
    (0.20, 0.0),
    (0.20, 0.13),
    (0.34, -0.16),
    (0.34, 0.16),
    (0.46, -0.10),
    (0.46, 0.0),
    (0.46, 0.10),
)
_HOLE_SLACK = 0.005
_TILT_TOL_DEG = 30.0


def get_command(env: ManagerBasedRlEnv, name: str) -> ShapeSorterCommand:
    return cast(ShapeSorterCommand, env.command_manager.get_term(name))


@dataclass(kw_only=True)
class ShapeSorterCommandCfg(GripperGateCommandCfg):
    cube_name: str
    piece_names: tuple[str, ...]
    """One piece entity per shape, in the variant's (tail-extending) index
    order."""
    variant: ShapeSorterVariantCfg

    def build(self, env: ManagerBasedRlEnv) -> ShapeSorterCommand:
        return ShapeSorterCommand(self, env)


class ShapeSorterCommand(CommandTerm):
    cfg: ShapeSorterCommandCfg

    def __init__(self, cfg: ShapeSorterCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        variant = cfg.variant
        self.variant = variant
        self.cube: Entity = env.scene[cfg.cube_name]
        self.pieces: list[Entity] = [env.scene[n] for n in cfg.piece_names]
        self.num_pieces = N = len(self.pieces)

        E, dev = self.num_envs, self.device

        layout = load_sorter_layout()
        self._cube_half = layout.half_size
        holes = [layout.holes[s] for s in variant.shapes]
        self._mouth_local = torch.tensor(
            [h.mouth for h in holes], device=dev, dtype=torch.float32
        )
        self._axis_local = torch.tensor(
            [h.axis_out for h in holes], device=dev, dtype=torch.float32
        )
        piece_half = torch.tensor(
            [layout.pieces[s].half_len for s in variant.shapes], device=dev
        )
        hole_radius = (
            torch.tensor([h.bounding_radius for h in holes], device=dev)
            + _HOLE_SLACK * variant.margin
        )
        wall_t = torch.tensor([h.wall_thickness for h in holes], device=dev)
        self._latch = SorterLatch(
            E,
            N,
            dev,
            hole_radius=hole_radius,
            piece_half=piece_half,
            s_cross=-(wall_t + piece_half),
            tilt_cos=math.cos(math.radians(_TILT_TOL_DEG)),
        )

        self._flat_quats = quat_from_angle_axis(
            _FLAT_ANGLES.to(dev), _FLAT_AXES.to(dev)
        )  # (6, 4)
        self._allowed_faces = torch.tensor(
            allowed_faces(variant.shapes), device=dev, dtype=torch.long
        )
        self._all_faces = torch.arange(6, device=dev, dtype=torch.long)

        self._normal_scatter = ScatterGrid.tiling(
            _NORMAL_WORKSPACE_X,
            _NORMAL_WORKSPACE_Y,
            object_radius=_NORMAL_OBJECT_RADIUS,
            device=dev,
        )
        self._wide_scatter = ScatterGrid(
            cells=torch.tensor(_WIDE_CELLS, device=dev),
            jitter=torch.zeros(2, device=dev),
        )
        assert self._normal_scatter.num_cells >= N
        assert self._wide_scatter.num_cells >= N

        self.socket_mouth_w = torch.zeros(E, N, 3, device=dev)
        self.socket_axis_w = torch.zeros(E, N, 3, device=dev)
        self.cube_pos_w = torch.zeros(E, 3, device=dev)
        self.cube_quat_w = torch.zeros(E, 4, device=dev)
        self.piece_pos_w = torch.zeros(E, N, 3, device=dev)
        self.piece_quat_w = torch.zeros(E, N, 4, device=dev)
        self.progress = torch.zeros(E, device=dev)
        self._gate = SuccessGate(E, dev, vel_threshold=math.inf)
        self._gripper_gate = GripperOpenGate(cfg)
        self._contact_gate = ContactReleaseGate()
        self._grip_sensor = cast(
            ContactSensor, env.scene.sensors[GRIP_CONTACT_SENSOR_NAME]
        )

        self.metrics["placed_frac"] = torch.zeros(E, device=dev)
        self.metrics["episode_success"] = torch.zeros(E, device=dev)

    @property
    def command(self) -> torch.Tensor:
        """(E, N*6): per hole, in piece order: env-local mouth xyz, axis xyz.

        Not exposed as an observation (``command_obs=False`` -- see
        ``shape_sorter_env_cfg.py``): ``cube_pose`` already gives the policy
        everything needed to infer hole poses. Kept for internal use
        (debug-vis).
        """
        mouth = self.socket_mouth_w - self._env.scene.env_origins.unsqueeze(1)
        return torch.cat([mouth, self.socket_axis_w], dim=-1).reshape(self.num_envs, -1)

    # Helpers used by observation/reward/termination terms. ----------------

    @property
    def success(self) -> torch.Tensor:
        """(E,) float: every piece latched, gripper open."""
        return self._gate.success

    @property
    def episode_success(self) -> torch.Tensor:
        """(E,) float: success latched since the last reset."""
        return self._gate.episode_success

    @property
    def latched(self) -> torch.Tensor:
        """(E, N) bool: pieces latched inside the cube."""
        return self._latch.latched

    # CommandTerm interface. ------------------------------------------------

    def _update_metrics(self) -> None:
        E, N = self.num_envs, self.num_pieces
        self.cube_pos_w = self.cube.data.root_link_pos_w
        self.cube_quat_w = self.cube.data.root_link_quat_w
        q = self.cube_quat_w.unsqueeze(1).expand(E, N, 4)
        self.socket_mouth_w = self.cube_pos_w.unsqueeze(1) + quat_apply(
            q, self._mouth_local.unsqueeze(0).expand(E, N, 3)
        )
        self.socket_axis_w = quat_apply(
            q, self._axis_local.unsqueeze(0).expand(E, N, 3)
        )

        self.piece_pos_w = torch.stack(
            [p.data.root_link_pos_w for p in self.pieces], dim=1
        )
        self.piece_quat_w = torch.stack(
            [p.data.root_link_quat_w for p in self.pieces], dim=1
        )

        axial, lateral = peg_axial_lateral(
            self.piece_pos_w, self.socket_mouth_w, self.socket_axis_w
        )
        cos_tilt = peg_axis_cos_tilt(self.piece_quat_w, self.socket_axis_w)
        latched = self._latch.update(axial, lateral, cos_tilt)
        all_latched = latched.all(dim=1)
        self.progress = latched.float().mean(dim=1)

        # The at-rest gate is disabled (vel_threshold=inf): a latched piece
        # stays scored regardless of motion, so skip the speed computation.
        speed = torch.zeros(E, device=self.device)
        released = self._gripper_gate(self._env.scene["robot"]) & self._contact_gate(
            self._grip_sensor
        )
        self._gate.update(all_latched, speed, released)

        self.metrics["placed_frac"] = self.progress
        self.metrics["episode_success"] = self.episode_success

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        dev = self.device
        variant = self.variant
        N = self.num_pieces
        origins = self._env.scene.env_origins[env_ids]

        self._gate.reset(env_ids)
        self._latch.reset(env_ids)

        # --- Cube pose: one shared draw (jitter x, jitter y, face, yaw),
        # independent of spawn mode, so the draw count/mapping never shifts
        # across variants (seed-prefix rule). ---
        u = torch.rand(n, 4, device=dev)
        zeros = torch.zeros(n, device=dev)
        jitter = (u[:, :2] - 0.5) * 2.0 * torch.tensor(SORTER_SPAWN_JITTER, device=dev)
        cube_xy = origins[:, :2] + torch.tensor(BOARD_POS[:2], device=dev) + jitter
        cube_z = origins[:, 2] + BOARD_POS[2] + self._cube_half + 0.002
        cube_pos = torch.cat([cube_xy, cube_z.unsqueeze(-1)], dim=-1)

        if variant.spawn == "fixed":
            cube_quat = (
                self._flat_quats[_FIXED_FACE_INDEX]
                .unsqueeze(0)
                .expand(n, 4)
                .contiguous()
            )
        else:
            faces = (
                self._allowed_faces
                if variant.spawn == "no_rotation"
                else self._all_faces
            )
            num_faces = faces.shape[0]
            idx = (u[:, 2] * num_faces).long().clamp_max(num_faces - 1)
            face = faces[idx]
            yaw = quat_from_euler_xyz(zeros, zeros, u[:, 3] * 2 * math.pi)
            cube_quat = quat_mul(yaw, self._flat_quats[face])

        self.cube.write_root_link_pose_to_sim(
            torch.cat([cube_pos, cube_quat], dim=-1), env_ids=env_ids
        )
        self.cube.write_root_link_velocity_to_sim(
            torch.zeros(n, 6, device=dev), env_ids=env_ids
        )

        # --- Pieces: scatter upright on distinct jittered cells, per-piece
        # draws in index order via ``sample_poses``. ---
        scatter = self._wide_scatter if variant.wide_positions else self._normal_scatter
        xy, spawn_yaw = scatter.sample_poses(n, N)
        write_upright_spawn(
            self.pieces, xy, spawn_yaw, self._latch.piece_half, origins, env_ids
        )

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        for batch in env_indices:
            success = bool(self.success[batch] > 0.5)
            for i in range(self.num_pieces):
                mouth = self.socket_mouth_w[batch, i].cpu().numpy()
                done = bool(self.latched[batch, i])
                color = marker_color(PALETTE[self.variant.colors[i]], done, success)
                visualizer.add_sphere(
                    center=mouth,
                    radius=0.015,
                    color=color,
                    label=f"mouth_{batch}_{i}",
                )
