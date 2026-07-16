"""Teleport-based success-reward check for the four task categories.

See tests/tasks/README.md.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path
from typing import cast

import numpy as np
import torch
from _render import domain_of, save_env_render
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_from_euler_xyz, quat_mul

import g3a_bench  # noqa: F401  (registers all tasks)
from g3a_bench.tasks.ball.mdp.commands import BallCommand
from g3a_bench.tasks.ball.variants import (
    BALL_RADIUS,
    BALL_SPAWN_X,
    BALL_VARIANTS,
    TARGET_X,
)
from g3a_bench.tasks.domino.mdp.commands import DominoCommand
from g3a_bench.tasks.shape_sorter.assets import BOARD_POS, load_sorter_layout
from g3a_bench.tasks.shape_sorter.mdp.commands import ShapeSorterCommand
from g3a_bench.tasks.stacking.mdp.commands import StackingCommand
from g3a_bench.tasks.stacking.variants import FORM_HALF_EXTENTS

_NUM_ENVS = 8
_STEPS = 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(task_id: str, device: str, *, play: bool = True) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(task_id, play=play)
    # Play cfgs keep the "success" termination (live_bridge demo playback
    # terminates on success); pop it here, otherwise the env auto-resets the
    # instant our teleported state scores, wiping the state before we read
    # the fraction below.
    cfg.terminations.pop("success", None)
    cfg.scene.num_envs = _NUM_ENVS
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.reset()
    return env


def _zeros(env: ManagerBasedRlEnv, device: str) -> torch.Tensor:
    return torch.zeros((_NUM_ENVS, env.action_manager.action.shape[1]), device=device)


def _upright_quat(n: int, device: str) -> torch.Tensor:
    z = torch.zeros(n, device=device)
    return quat_from_euler_xyz(z, z, z)


def _render_goal(env: ManagerBasedRlEnv, task_id: str, render: bool) -> None:
    """Save a picture of the teleported goal state for *task_id*.

    Called at the scored instant (after the step loop that computes the success
    fraction) so the image shows the exact state the metric evaluated. A render
    failure is reported as a warning and never aborts the check.
    """
    if not render:
        return
    path = Path("out/render") / domain_of(task_id) / f"{task_id}_goal.png"
    warn = save_env_render(env, path)
    if warn is not None:
        print(f"        (render skipped {task_id}: {warn})")


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------


def _teleport_stack(
    cmd: StackingCommand,
    env: ManagerBasedRlEnv,
    device: str,
    *,
    xy: tuple[float, float],
) -> None:
    """Stack every block at one xy, resting directly on top of each other."""
    origins = env.scene.env_origins
    half_h = [FORM_HALF_EXTENTS[b.form][2] for b in cmd.variant.blocks]
    quat = _upright_quat(cmd.num_envs, device)
    xy_t = torch.tensor(xy, device=device)
    cum = 0.0
    for i, block in enumerate(cmd.blocks):
        h = half_h[i]
        z = origins[:, 2] + cum + h
        pos = torch.cat([origins[:, :2] + xy_t, z.unsqueeze(-1)], dim=-1)
        block.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
        block.write_root_link_velocity_to_sim(
            torch.zeros(cmd.num_envs, 6, device=device)
        )
        cum += 2 * h


def _teleport_two_stacks(
    cmd: StackingCommand, env: ManagerBasedRlEnv, device: str, *, separation: float
) -> None:
    """Two separate n-block stacks, ``separation`` m apart in y."""
    origins = env.scene.env_origins
    half_h = [FORM_HALF_EXTENTS[b.form][2] for b in cmd.variant.blocks]
    quat = _upright_quat(cmd.num_envs, device)
    n = cmd.variant.target_blocks
    xy_by_stack = (
        torch.tensor([0.30, -separation / 2], device=device),
        torch.tensor([0.30, separation / 2], device=device),
    )
    cum = [0.0, 0.0]
    for i, block in enumerate(cmd.blocks):
        stack = 0 if i < n else 1
        h = half_h[i]
        z = origins[:, 2] + cum[stack] + h
        pos = torch.cat([origins[:, :2] + xy_by_stack[stack], z.unsqueeze(-1)], dim=-1)
        block.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
        block.write_root_link_velocity_to_sim(
            torch.zeros(cmd.num_envs, 6, device=device)
        )
        cum[stack] += 2 * h


def _check_stacking_stack3(device: str, *, render: bool = False) -> str | None:
    task_id = "G3a-Stacking-Stack3-Dk1_uni"
    env = _make_env(task_id, device)
    cmd = cast(StackingCommand, env.command_manager.get_term("task"))
    zeros = _zeros(env, device)
    env.step(zeros)
    _teleport_stack(cmd, env, device, xy=(0.30, 0.0))
    for _ in range(_STEPS):
        env.step(zeros)
    frac = cmd.progress.mean().item()
    _render_goal(env, task_id, render)
    env.close()
    return None if frac > 0.9 else f"progress={frac:.3f} (expected >0.9)"


def _check_stacking_two_stacks(device: str, *, render: bool = False) -> str | None:
    task_id = "G3a-Stacking-Stack3TwoStacks-Dk1_uni"
    env = _make_env(task_id, device)
    cmd = cast(StackingCommand, env.command_manager.get_term("task"))
    zeros = _zeros(env, device)
    env.step(zeros)
    _teleport_two_stacks(cmd, env, device, separation=0.12)
    for _ in range(_STEPS):
        env.step(zeros)
    frac = cmd.progress.mean().item()
    _render_goal(env, task_id, render)
    env.close()
    return None if frac > 0.9 else f"progress={frac:.3f} (expected >0.9)"


# ---------------------------------------------------------------------------
# Domino
# ---------------------------------------------------------------------------


def _teleport_dominoes_upright(cmd: DominoCommand, device: str) -> None:
    """Place every domino at its own slot goal, upright and row-aligned."""
    zeros = torch.zeros(cmd.num_envs, device=device)
    quat = quat_from_euler_xyz(zeros, zeros, cmd.row_yaw)
    for i, domino in enumerate(cmd.dominoes):
        pos = cmd.slot_goal_pos[:, i]
        domino.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
        domino.write_root_link_velocity_to_sim(
            torch.zeros(cmd.num_envs, 6, device=device)
        )


def _lay_dominoes_flat(cmd: DominoCommand, env: ManagerBasedRlEnv, device: str) -> None:
    """Topple every domino sideways (about the row axis), in place.

    Rotating about the domino's own thin (row-aligned) body axis swings its
    height into the direction perpendicular to the row, so neighbouring slots
    (spaced along the row) never overlap.
    """
    n = cmd.num_envs
    origins = env.scene.env_origins
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).expand(n, 3)
    roll = quat_from_angle_axis(torch.full((n,), math.pi / 2, device=device), x_axis)
    zeros = torch.zeros(n, device=device)
    upright = quat_from_euler_xyz(zeros, zeros, cmd.row_yaw)
    flat = quat_mul(upright, roll)
    for i, domino in enumerate(cmd.dominoes):
        half_w = cmd.variant.half_extents[i][1]  # width becomes the new "up" extent
        pos = cmd.slot_goal_pos[:, i].clone()
        pos[:, 2] = origins[:, 2] + half_w + 0.003
        domino.write_root_link_pose_to_sim(torch.cat([pos, flat], dim=-1))
        domino.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=device))


def _check_domino_domino3(device: str, *, render: bool = False) -> str | None:
    task_id = "G3a-Domino-Domino3-Dk1_uni"
    env = _make_env(task_id, device)
    cmd = cast(DominoCommand, env.command_manager.get_term("task"))
    zeros = _zeros(env, device)
    env.step(zeros)
    _teleport_dominoes_upright(cmd, device)
    for _ in range(_STEPS):
        env.step(zeros)
    frac = cmd.progress.mean().item()
    _render_goal(env, task_id, render)
    env.close()
    return None if frac > 0.9 else f"placed_fraction={frac:.3f} (expected >0.9)"


def _check_domino_knockover(device: str, *, render: bool = False) -> str | None:
    task_id = "G3a-Domino-Domino3KnockOver-Dk1_uni"
    env = _make_env(task_id, device)
    cmd = cast(DominoCommand, env.command_manager.get_term("task"))
    zeros = _zeros(env, device)
    env.step(zeros)
    _teleport_dominoes_upright(cmd, device)
    env.step(zeros)  # all placed simultaneously -> phase latches to 1
    _lay_dominoes_flat(cmd, env, device)
    for _ in range(_STEPS):
        env.step(zeros)
    frac = cmd.progress.mean().item()
    _render_goal(env, task_id, render)
    env.close()
    return None if frac > 0.9 else f"progress={frac:.3f} (expected >0.9)"


# ---------------------------------------------------------------------------
# Shape sorter
# ---------------------------------------------------------------------------


def _vec_to_up_quat(ax: np.ndarray, device: str) -> torch.Tensor:
    """(4,) wxyz quaternion that rotates local vector *ax* onto world +z."""
    up = np.array([0.0, 0.0, 1.0])
    d = float(np.clip(np.dot(ax, up), -1.0, 1.0))
    if d > 0.9999:
        rot_axis, angle = np.array([1.0, 0.0, 0.0]), 0.0
    elif d < -0.9999:
        rot_axis, angle = np.array([1.0, 0.0, 0.0]), math.pi
    else:
        rot_axis = np.cross(ax, up)
        rot_axis /= np.linalg.norm(rot_axis)
        angle = math.acos(d)
    ca2, sa2 = math.cos(angle / 2), math.sin(angle / 2)
    return torch.tensor(
        [ca2, sa2 * rot_axis[0], sa2 * rot_axis[1], sa2 * rot_axis[2]],
        device=device,
        dtype=torch.float32,
    )


def _check_shape_sorter_full(device: str, *, render: bool = False) -> str | None:
    """Per socket: reorient the cube hole-up, two-station insert; all latch.

    The cube can only present one hole up at a time, so pieces are latched in
    sequentially -- the latch is monotonic (never clears except on reset), so
    an earlier piece stays scored while later sockets are reoriented to.
    """
    task_id = "G3a-ShapeSorter-ShapeSorter3Full-Dk1_uni"
    env = _make_env(task_id, device)
    cmd = cast(ShapeSorterCommand, env.command_manager.get_term("task"))
    zeros = _zeros(env, device)
    env.step(zeros)

    layout = load_sorter_layout()
    holes = [layout.holes[s] for s in cmd.variant.shapes]
    piece_half = torch.tensor(
        [layout.pieces[s].half_len for s in cmd.variant.shapes], device=device
    )
    wall_t = torch.tensor([h.wall_thickness for h in holes], device=device)
    s_cross = -(wall_t + piece_half)

    board_pos = torch.tensor(BOARD_POS, device=device, dtype=torch.float32)
    origins = env.scene.env_origins

    for s in range(cmd.num_pieces):
        axis_local = np.array(holes[s].axis_out)
        quat = _vec_to_up_quat(axis_local, device).unsqueeze(0).expand(_NUM_ENVS, -1)
        pos = origins[:, :2] + board_pos[:2]
        z = origins[:, 2] + board_pos[2] + layout.half_size + 0.001
        cube_pos = torch.cat([pos, z.unsqueeze(-1)], dim=-1)
        cmd.cube.write_root_link_pose_to_sim(torch.cat([cube_pos, quat], dim=-1))
        cmd.cube.write_root_link_velocity_to_sim(
            torch.zeros(_NUM_ENVS, 6, device=device)
        )
        env.step(zeros)  # refresh socket_mouth_w/socket_axis_w from the new pose

        mouth = cmd.socket_mouth_w[:, s]
        axis = cmd.socket_axis_w[:, s]
        upright = _upright_quat(_NUM_ENVS, device)

        station1 = mouth + axis * (piece_half[s] + 0.005)
        cmd.pieces[s].write_root_link_pose_to_sim(
            torch.cat([station1, upright], dim=-1)
        )
        cmd.pieces[s].write_root_link_velocity_to_sim(
            torch.zeros(_NUM_ENVS, 6, device=device)
        )
        env.step(zeros)

        station2 = mouth + axis * (s_cross[s] - 0.005)
        cmd.pieces[s].write_root_link_pose_to_sim(
            torch.cat([station2, upright], dim=-1)
        )
        cmd.pieces[s].write_root_link_velocity_to_sim(
            torch.zeros(_NUM_ENVS, 6, device=device)
        )
        env.step(zeros)

    for _ in range(_STEPS):
        env.step(zeros)
    frac = cmd.progress.mean().item()
    _render_goal(env, task_id, render)
    env.close()
    return None if frac > 0.9 else f"latched_fraction={frac:.3f} (expected >0.9)"


# ---------------------------------------------------------------------------
# Ball
# ---------------------------------------------------------------------------


def _teleport_ball(cmd: BallCommand, env: ManagerBasedRlEnv, device: str) -> None:
    """Place ball at target XY at its natural resting height."""
    ball_pos = cmd.target_pos_w.clone()
    if cmd.variant.task == "minigolf":
        # Ball resting on terrain inside the pit; avoid a large bounce.
        ball_pos[:, 2] = env.scene.env_origins[:, 2] + BALL_RADIUS + 0.001
    else:
        ball_pos[:, 2] = cmd.target_z + BALL_RADIUS
    quat = _upright_quat(_NUM_ENVS, device)
    cmd.ball.write_root_link_pose_to_sim(torch.cat([ball_pos, quat], dim=-1))
    cmd.ball.write_root_link_velocity_to_sim(torch.zeros(_NUM_ENVS, 6, device=device))


def _check_ball_at_target(
    device: str, task_id: str, *, render: bool = False
) -> str | None:
    env = _make_env(task_id, device)
    cmd = cast(BallCommand, env.command_manager.get_term("task"))
    zeros = _zeros(env, device)
    env.step(zeros)
    _teleport_ball(cmd, env, device)
    for _ in range(_STEPS):
        env.step(zeros)
    frac = cmd.at_target.mean().item()
    _render_goal(env, task_id, render)
    env.close()
    return None if frac > 0.9 else f"at_target_fraction={frac:.3f} (expected >0.9)"


def _segment_intersects_aabb(
    p0: tuple[float, float],
    p1: tuple[float, float],
    box_min: tuple[float, float],
    box_max: tuple[float, float],
) -> bool:
    """2D segment-vs-axis-aligned-box intersection (slab method)."""
    d = (p1[0] - p0[0], p1[1] - p0[1])
    tmin, tmax = 0.0, 1.0
    for axis in range(2):
        o, dd = p0[axis], d[axis]
        lo, hi = box_min[axis], box_max[axis]
        if abs(dd) < 1e-12:
            if o < lo or o > hi:
                return False
            continue
        t1, t2 = (lo - o) / dd, (hi - o) / dd
        if t1 > t2:
            t1, t2 = t2, t1
        tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmin > tmax:
            return False
    return True


def _check_minigolf_using_wall_blocks_direct_shots() -> str | None:
    """Pure-math check (no simulation): every straight line from a spawn
    corner to the pit crosses the blocker box, so a direct shot is never
    geometrically possible.

    Blocker geometry (must match ``ball/assets.py._add_using_wall_walls``):
    box centred at (0.58, 0.09), half extents (0.012, 0.13). Spawn rectangle
    matches ``BallCommandCfg``'s defaults (``BALL_SPAWN_X`` +/- 0.03 x,
    +/- 0.08 y).
    """
    variant = BALL_VARIANTS["MinigolfUsingWall"]
    pit = (TARGET_X, variant.pit_y)
    spawn_x = (BALL_SPAWN_X - 0.03, BALL_SPAWN_X + 0.03)
    spawn_y = (-0.08, 0.08)
    blocker_pos, blocker_half = (0.58, 0.09), (0.012, 0.13)
    box_min = (blocker_pos[0] - blocker_half[0], blocker_pos[1] - blocker_half[1])
    box_max = (blocker_pos[0] + blocker_half[0], blocker_pos[1] + blocker_half[1])

    misses = [
        (sx, sy)
        for sx in spawn_x
        for sy in spawn_y
        if not _segment_intersects_aabb((sx, sy), pit, box_min, box_max)
    ]
    if misses:
        return f"spawn corners with a clear line to the pit: {misses}"
    return None


def _check_ball_minigolf_using_wall(device: str, *, render: bool = False) -> str | None:
    task_id = "G3a-Ball-MinigolfUsingWall-Dk1_uni"
    err = _check_ball_at_target(device, task_id, render=render)
    if err is not None:
        return err
    return _check_minigolf_using_wall_blocks_direct_shots()


# ---------------------------------------------------------------------------
# Test table
# ---------------------------------------------------------------------------

_CHECKS: list[tuple[str, str]] = [
    ("Stacking  Stack3        ", "stacking_stack3"),
    ("Stacking  Stack3TwoStacks", "stacking_two_stacks"),
    ("Domino    Domino3       ", "domino_domino3"),
    ("Domino    Domino3KnockOver", "domino_knockover"),
    ("ShapeSorter ShapeSorter3Full", "shape_sorter_full"),
    ("Ball      MinigolfDirect", "ball_minigolf_direct"),
    ("Ball      MinigolfUsingWall", "ball_minigolf_using_wall"),
    ("Ball      ThrowU        ", "ball_throw_u"),
]


def _dispatch(key: str, device: str, *, render: bool = False) -> str | None:
    if key == "stacking_stack3":
        return _check_stacking_stack3(device, render=render)
    if key == "stacking_two_stacks":
        return _check_stacking_two_stacks(device, render=render)
    if key == "domino_domino3":
        return _check_domino_domino3(device, render=render)
    if key == "domino_knockover":
        return _check_domino_knockover(device, render=render)
    if key == "shape_sorter_full":
        return _check_shape_sorter_full(device, render=render)
    if key == "ball_minigolf_direct":
        return _check_ball_at_target(
            device, "G3a-Ball-MinigolfDirect-Dk1_uni", render=render
        )
    if key == "ball_minigolf_using_wall":
        return _check_ball_minigolf_using_wall(device, render=render)
    if key == "ball_throw_u":
        return _check_ball_at_target(device, "G3a-Ball-ThrowU-Dk1_uni", render=render)
    return f"Unknown check key: {key}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify success metrics fire for representative G3a tasks."
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU device.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip saving a goal-state picture per check to out/render/<domain>/.",
    )
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    render = not args.no_render
    print(f"Device: {device}   Checks: {len(_CHECKS)}")
    if render:
        print("Goal pictures: out/render/<domain>/")
    print()

    failures: dict[str, str] = {}
    for label, key in _CHECKS:
        try:
            err = _dispatch(key, device, render=render)
        except Exception as exc:
            err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if err is None:
            print(f"  PASS  {label}")
        else:
            print(f"  FAIL  {label}  ({err})")
            failures[label] = err

    passed = len(_CHECKS) - len(failures)
    print()
    print(f"Result: {passed}/{len(_CHECKS)} passed, {len(failures)} failed")

    if failures:
        print()
        for label, tb in failures.items():
            if "\n" in tb:
                print(f"--- {label.strip()} ---")
                print(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
