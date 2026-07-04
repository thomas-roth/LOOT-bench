"""Teleport-based success-reward check for the three task categories.

See tests/tasks/README.md.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from _render import domain_of, save_env_render
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

import g3a_bench  # noqa: F401  (registers all tasks)
from g3a_bench.tasks.ball.mdp.commands import BallCommand
from g3a_bench.tasks.ball.variants import BALL_RADIUS
from g3a_bench.tasks.cube.mdp.commands import PieceArrangementCommand
from g3a_bench.tasks.insertion.assets import SHEET_HALF_Z
from g3a_bench.tasks.insertion.mdp.commands import InsertionCommand

_NUM_ENVS = 8
_STEPS = 10  # cube uses play=True (no success termination); other categories still safe at 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(task_id: str, device: str, *, play: bool = False) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(task_id, play=play)
    cfg.scene.num_envs = _NUM_ENVS
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.reset()
    return env


def _zeros(env: ManagerBasedRlEnv, device: str) -> torch.Tensor:
    return torch.zeros((_NUM_ENVS, env.action_manager.action.shape[1]), device=device)


def _upright_quat(n: int, device: str) -> torch.Tensor:
    z = torch.zeros(n, device=device)
    return quat_from_euler_xyz(z, z, z)


def _axis_aligned_quat(axis_out: torch.Tensor, device: str) -> torch.Tensor:
    """(N, 4) wxyz quaternion that rotates world-z onto *axis_out* (N, 3)."""
    world_z = np.array([0.0, 0.0, 1.0])
    ax = axis_out[0].cpu().numpy()
    if abs(float(np.dot(world_z, ax))) > 0.9999:
        return _upright_quat(axis_out.shape[0], device)
    rot_axis = np.cross(world_z, ax)
    rot_axis /= np.linalg.norm(rot_axis) + 1e-9
    angle = math.acos(float(np.clip(np.dot(world_z, ax), -1.0, 1.0)))
    ca2, sa2 = math.cos(angle / 2), math.sin(angle / 2)
    qv = torch.tensor(
        [ca2, sa2 * rot_axis[0], sa2 * rot_axis[1], sa2 * rot_axis[2]],
        device=device,
        dtype=torch.float32,
    )
    return qv.unsqueeze(0).expand(axis_out.shape[0], -1).contiguous()


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
# Per-category teleport helpers
# ---------------------------------------------------------------------------


def _teleport_cube(cmd: PieceArrangementCommand, device: str) -> None:
    """Place every piece at its assigned slot goal, upright."""
    goals = cmd.slot_goal_pos.clone()  # (E, G, 3)
    assign = cmd.piece_for_slot.clone()  # (E, G)
    row_yaw = cmd.row_yaw.clone()  # (E,) – nonzero only for dominos
    is_domino = cmd.variant.arrangement == "domino"
    z = torch.zeros(1, device=device)
    for s in range(cmd.num_slots):
        for e in range(cmd.num_envs):
            p_idx = int(assign[e, s].item())
            piece = cmd.pieces[p_idx]
            pos = goals[e, s].unsqueeze(0)
            yaw = row_yaw[e : e + 1] if is_domino else z
            quat = quat_from_euler_xyz(z, z, yaw)
            env_id = torch.tensor([e], device=device)
            piece.write_root_link_pose_to_sim(
                torch.cat([pos, quat], dim=-1), env_ids=env_id
            )
            piece.write_root_link_velocity_to_sim(
                torch.zeros(1, 6, device=device), env_ids=env_id
            )


def _teleport_pegs(
    cmd: InsertionCommand, device: str, *, all_sockets: bool = False
) -> None:
    """Insert peg(s) 1 cm inside their bore; align quaternion to bore axis."""
    n = cmd.num_sockets if all_sockets else 1
    for s in range(n):
        axis_out = cmd.socket_axis_w[:, s]  # (E, 3)
        quat = _axis_aligned_quat(axis_out, device)
        peg_pos = cmd.socket_mouth_w[:, s] - axis_out * 0.01
        cmd.pegs[s].write_root_link_pose_to_sim(torch.cat([peg_pos, quat], dim=-1))
        cmd.pegs[s].write_root_link_velocity_to_sim(
            torch.zeros(_NUM_ENVS, 6, device=device)
        )


def _teleport_sheet(cmd: InsertionCommand, env: ManagerBasedRlEnv, device: str) -> None:
    """Snap the movable sheet to the bottom-bore XY."""
    assert cmd.sheet is not None
    mouth = cmd.socket_mouth_w[:, 0].clone()
    mouth[:, 2] = env.scene.env_origins[:, 2] + 2 * SHEET_HALF_Z + SHEET_HALF_Z + 0.001
    quat = _upright_quat(_NUM_ENVS, device)
    cmd.sheet.write_root_link_pose_to_sim(torch.cat([mouth, quat], dim=-1))
    cmd.sheet.write_root_link_velocity_to_sim(torch.zeros(_NUM_ENVS, 6, device=device))


def _teleport_ball(cmd: BallCommand, env: ManagerBasedRlEnv, device: str) -> None:
    """Place ball at target XY at its natural resting height."""
    ball_pos = cmd.target_pos_w.clone()
    if cmd.variant.target == "hole":
        # Ball resting on terrain inside the pit; avoid a large bounce.
        ball_pos[:, 2] = env.scene.env_origins[:, 2] + BALL_RADIUS + 0.001
    else:
        ball_pos[:, 2] = cmd.surface_z + BALL_RADIUS
    quat = _upright_quat(_NUM_ENVS, device)
    cmd.ball.write_root_link_pose_to_sim(torch.cat([ball_pos, quat], dim=-1))
    cmd.ball.write_root_link_velocity_to_sim(torch.zeros(_NUM_ENVS, 6, device=device))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_cube_stack(
    device: str, task_id: str = "G3a-Cube-Stack3-Dk1_uni", *, render: bool = False
) -> str | None:
    from typing import cast

    # play=True removes success termination; cube now terminates on step 1
    # which would reset env mid-measurement in the training config.
    env = _make_env(task_id, device, play=True)
    cmd = cast(PieceArrangementCommand, env.command_manager.get_term("arrangement"))
    zeros = _zeros(env, device)
    env.step(zeros)
    _teleport_cube(cmd, device)
    for _ in range(_STEPS):
        env.step(zeros)
    frac = cmd.slot_at_goal.mean().item()
    _render_goal(env, task_id, render)
    env.close()
    return None if frac > 0.9 else f"placed_fraction={frac:.3f} (expected >0.9)"


def _check_cube_domino(device: str, *, render: bool = False) -> str | None:
    return _check_cube_stack(device, "G3a-Cube-Domino5-Dk1_uni", render=render)


def _check_insertion_single(
    device: str, task_id: str, *, render: bool = False
) -> str | None:
    from typing import cast

    # play=True removes success termination; insertion terminates the instant
    # all pegs are inserted, which would reset the env mid-measurement in the
    # training config.
    env = _make_env(task_id, device, play=True)
    cmd = cast(InsertionCommand, env.command_manager.get_term("insertion"))
    zeros = _zeros(env, device)
    env.step(zeros)
    all_s = cmd.num_sockets > 1
    _teleport_pegs(cmd, device, all_sockets=all_s)
    for _ in range(_STEPS):
        env.step(zeros)
    frac = cmd.inserted.mean().item()
    _render_goal(env, task_id, render)
    env.close()
    return None if frac > 0.9 else f"inserted_fraction={frac:.3f} (expected >0.9)"


def _check_insertion_twosheet(device: str, *, render: bool = False) -> str | None:
    from typing import cast

    task_id = "G3a-Insertion-TwoSheet-Dk1_uni"
    env = _make_env(task_id, device, play=True)
    cmd = cast(InsertionCommand, env.command_manager.get_term("insertion"))
    zeros = _zeros(env, device)
    env.step(zeros)
    _teleport_pegs(cmd, device)
    _teleport_sheet(cmd, env, device)
    for _ in range(_STEPS):
        env.step(zeros)
    frac = cmd.inserted.mean().item()
    _render_goal(env, task_id, render)
    env.close()
    return None if frac > 0.9 else f"inserted_fraction={frac:.3f} (expected >0.9)"


def _check_ball(device: str, task_id: str, *, render: bool = False) -> str | None:
    from typing import cast

    # play=True removes the success termination (hold_time=0.0 means it fires
    # the instant the teleported ball registers at_target, auto-resetting the
    # env before the fraction below is measured) -- same reset-mid-measurement
    # issue play=True works around for the cube checks.
    env = _make_env(task_id, device, play=True)
    cmd = cast(BallCommand, env.command_manager.get_term("ball_target"))
    zeros = _zeros(env, device)
    env.step(zeros)
    _teleport_ball(cmd, env, device)
    for _ in range(_STEPS):
        env.step(zeros)
    frac = cmd.at_target.mean().item()
    _render_goal(env, task_id, render)
    env.close()
    return None if frac > 0.9 else f"at_target_fraction={frac:.3f} (expected >0.9)"


# ---------------------------------------------------------------------------
# Test table
# ---------------------------------------------------------------------------

_CHECKS: list[tuple[str, str]] = [
    # (label, task_id or special key)
    ("Cube  Stack3        ", "G3a-Cube-Stack3-Dk1_uni"),
    ("Cube  Domino5       ", "G3a-Cube-Domino5-Dk1_uni"),
    ("Ins   PegCylinder   ", "G3a-Insertion-PegCylinder-Dk1_uni"),
    ("Ins   PegCylinderH  ", "G3a-Insertion-PegCylinderH-Dk1_uni"),
    ("Ins   Sorter4       ", "G3a-Insertion-Sorter4-Dk1_uni"),
    ("Ins   TwoSheet      ", "G3a-Insertion-TwoSheet-Dk1_uni"),
    ("Ball  PushMarked    ", "G3a-Ball-PushMarked-Dk1_uni"),
    ("Ball  GolfHole      ", "G3a-Ball-GolfHole-Dk1_uni"),
    ("Ball  PushHole      ", "G3a-Ball-PushHole-Dk1_uni"),
]


def _dispatch(task_id: str, device: str, *, render: bool = False) -> str | None:
    cat = task_id.split("-")[1]
    if cat == "Cube":
        if "Domino" in task_id:
            return _check_cube_domino(device, render=render)
        return _check_cube_stack(device, task_id, render=render)
    if cat == "Insertion":
        if "TwoSheet" in task_id:
            return _check_insertion_twosheet(device, render=render)
        return _check_insertion_single(device, task_id, render=render)
    if cat == "Ball":
        return _check_ball(device, task_id, render=render)
    return f"Unknown category in task id: {task_id}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify success metrics fire for representative G3a tasks."
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU device.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip saving a goal-state picture per check to out/<cat>_check/.",
    )
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    render = not args.no_render
    print(f"Device: {device}   Checks: {len(_CHECKS)}")
    if render:
        print("Goal pictures: out/render/<domain>/")
    print()

    failures: dict[str, str] = {}
    for label, task_id in _CHECKS:
        try:
            err = _dispatch(task_id, device, render=render)
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
