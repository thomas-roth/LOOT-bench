"""IK reachability check over every spawn cell. See tests/tasks/README.md.

For every cell of the stack, domino and sorter spawn tables — plus the
sorter cube pin and Place's fixed spawn/target pins — at all four corners
of its jitter box (pins are exact), this solves the arm joints for a hover
above the corner with the gripper straight down (damped least squares on a
scratch model) and servos the environment's own joint-position action term
to that solution, reporting the worst position residual.

It runs as a script rather than a unit test because it needs the compiled
model. Only one task is built: the robot and the table are shared by every
category, so one compiled model covers all tables.
"""

from __future__ import annotations

import argparse
import math
import sys

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg

import loot_bench  # noqa: F401  (registers all tasks)
from loot_bench.robots.trlc_dk1 import get_unimanual_dk1_interface
from loot_bench.robots.trlc_dk1.dk1_actuators import ARM_JOINTS
from loot_bench.robots.trlc_dk1.dk1_constants import (
    START_JOINT_POS,
    get_unimanual_dk1_robot_cfg,
)
from loot_bench.tasks.common.actions import JOINT_POS_ACTION_NAME
from loot_bench.tasks.common.fixed_spawns import SPAWN_JITTER
from loot_bench.tasks.domino.mdp.commands import DOMINO_SPAWN_CELLS
from loot_bench.tasks.place.mdp.commands import FIXED_SPAWN_XY, FIXED_TARGET_XY
from loot_bench.tasks.shape_sorter.assets import BOARD_POS
from loot_bench.tasks.shape_sorter.mdp.commands import SORTER_SPAWN_CELLS
from loot_bench.tasks.stack.mdp.commands import PI_SPAWN_CELLS

_BUILD_TASK = "LOOT-Stack-3Pis-Dk1"
"""Task the check builds (robot and table are shared by every category)."""

_HOVER = 0.10
"""[m] grasp-site height above the table surface.

Above the tallest spawn: a pi block stands 0.065 m on its cell, and the
fingers reach ~0.035 m below the site, so anything lower would drive the
gripper into the very block sitting on the cell being probed. Still well
below the reach dome's ceiling (0.20 m even at the far cells), so a solve
here means the gripper can descend onto anything staged in the cell. If
this hover makes a cell unreachable, that is a real reach finding about the
geometry; report it rather than lowering the hover. Measured 2026-09-11 at
this hover: worst stack-cell residual 0.486 mm, worst overall 0.497 mm.
"""

_MAX_STEPS = 200
"""IK drive budget per corner."""

_SETTLE_RESIDUAL = 5e-4
"""[m] position residual that stops the drive early."""

_PASS_RESIDUAL = 5e-3
"""[m] worst residual the check accepts."""


def _targets() -> list[tuple[str, float, float]]:
    """``(label, x, y)`` over every cell at all four jitter corners."""
    out: list[tuple[str, float, float]] = []
    cube_pin: tuple[float, float] = (float(BOARD_POS[0]), float(BOARD_POS[1]))
    tables: list[tuple[str, tuple[tuple[float, float], ...]]] = [
        ("stack", PI_SPAWN_CELLS),
        ("domino", DOMINO_SPAWN_CELLS),
        ("sorter", SORTER_SPAWN_CELLS),
        ("sorter-cube", (cube_pin,)),
    ]
    for name, cells in tables:
        for i, (cx, cy) in enumerate(cells):
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    out.append(
                        (
                            f"{name}[{i}]({sx:+.0f},{sy:+.0f})",
                            cx + sx * SPAWN_JITTER,
                            cy + sy * SPAWN_JITTER,
                        )
                    )
    for name, pin in (
        ("place-spawn", FIXED_SPAWN_XY),
        ("place-target", FIXED_TARGET_XY),
    ):
        out.append((name, pin[0], pin[1]))
    return out


_ARM_JOINT_NAMES = list(ARM_JOINTS.keys())


def _quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[1:] = -out[1:]
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _solve_arm_ik(target_pos: np.ndarray, target_quat_wxyz: np.ndarray) -> np.ndarray:
    """Arm joints (6,) placing the grasp site at the target, DLS on scratch.

    Seeded at the episode start pose. Position and orientation share one
    damped solve (lambda 0.05, the IK damping the delta term used), so a
    solution here means the joint-controlled arm can hold the hover the
    check probes for.
    """
    model = Entity(get_unimanual_dk1_robot_cfg()).compile()
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, get_unimanual_dk1_interface().grasp_site
    )
    qpos_ids = np.array(
        [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in _ARM_JOINT_NAMES
        ]
    )
    dof_ids = np.array(
        [
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in _ARM_JOINT_NAMES
        ]
    )
    q = np.array([START_JOINT_POS[n] for n in _ARM_JOINT_NAMES], dtype=float)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(300):
        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)
        pos_err = target_pos - np.array(data.site_xpos[site_id])
        xmat = np.array(data.site_xmat[site_id]).reshape(3, 3)
        qw, qx, qy, qz = _mat_to_quat(xmat)
        cur = np.array([qw, qx, qy, qz])
        err_q = _quat_mul(target_quat_wxyz, _quat_conj(cur))
        if err_q[0] < 0:
            err_q = -err_q
        angle = 2.0 * math.acos(min(1.0, err_q[0]))
        if angle < 1e-9:
            ori_err = np.zeros(3)
        else:
            ori_err = err_q[1:] / math.sin(angle / 2.0) * angle
        err = np.concatenate([pos_err, 0.5 * ori_err])
        if float(np.linalg.norm(pos_err)) < 1e-4 and float(angle) < 1e-3:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])
        lam = 0.05
        dq = jac.T @ np.linalg.solve(jac @ jac.T + lam**2 * np.eye(6), err)
        q = q + np.clip(dq, -0.1, 0.1)
    return q.astype(np.float32)


def _mat_to_quat(m: np.ndarray) -> tuple[float, float, float, float]:
    t = float(np.trace(m))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        return (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return (
        (m[1, 0] - m[0, 1]) / s,
        (m[0, 2] + m[2, 0]) / s,
        (m[1, 2] + m[2, 1]) / s,
        0.25 * s,
    )


def _solve(
    env: ManagerBasedRlEnv,
    site_ids: list[int] | slice,
    home_quat: torch.Tensor,
    tx: float,
    ty: float,
    tz: float,
    device: str,
) -> tuple[float, int]:
    """Servo the grasp site to ``(tx, ty, tz)``; return best residual, steps."""
    env.reset()
    robot = env.scene["robot"]
    origins = env.scene.env_origins
    term = env.action_manager.get_term(JOINT_POS_ACTION_NAME)
    target_quat = home_quat[0].detach().cpu().numpy().astype(float)
    target_quat = target_quat / float(np.linalg.norm(target_quat))
    arm_joints = _solve_arm_ik(np.array([tx, ty, tz], dtype=float), target_quat)
    # Joint targets in the term's own channel order: the IK solution for the
    # arm, the upper limit (open) for the gripper.
    joint_targets = {
        name: float(v) for name, v in zip(_ARM_JOINT_NAMES, arm_joints, strict=True)
    }
    target = torch.tensor(
        [
            [
                joint_targets[name]
                if name in joint_targets
                else float(term.offset[0, i] + term.scale[0, i])
                for i, name in enumerate(term.target_names)
            ]
        ],
        device=device,
    )
    hold = ((target - term.offset) / term.scale).clamp(-1.0, 1.0)
    target_pos = torch.tensor([[tx, ty, tz]], device=device)
    best = math.inf
    steps = 0
    for step in range(_MAX_STEPS):
        env.step(hold)
        pos = robot.data.site_pos_w[:, site_ids].squeeze(1) - origins
        residual = float((pos - target_pos).norm())
        best = min(best, residual)
        steps = step + 1
        if residual <= _SETTLE_RESIDUAL:
            break
    return best, steps


def main() -> None:
    parser = argparse.ArgumentParser(description="IK reachability over spawn cells.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU device.")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_env_cfg(_BUILD_TASK)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.reset()

    grasp_site = get_unimanual_dk1_interface().grasp_site
    site_cfg = SceneEntityCfg("robot", site_names=(grasp_site,))
    site_cfg.resolve(env.scene)
    site_ids = site_cfg.site_ids
    robot = env.scene["robot"]
    home_quat = robot.data.site_quat_w[:, site_ids].squeeze(1).clone()
    table_z = float(env.scene.env_origins[0, 2])

    targets = _targets()
    print(f"Device: {device}   Task: {_BUILD_TASK}   Targets: {len(targets)}")
    print()

    worst: dict[str, tuple[float, str]] = {}
    overall = (0.0, "")
    for label, x, y in targets:
        residual, steps = _solve(
            env, site_ids, home_quat, x, y, table_z + _HOVER, device
        )
        group = label.split("[")[0]
        if residual > worst.get(group, (0.0, ""))[0]:
            worst[group] = (residual, label)
        if residual > overall[0]:
            overall = (residual, label)
        print(f"  {residual * 1000:7.3f} mm  {steps:3d} steps  {label}")

    env.close()
    print()
    for group in sorted(worst):
        residual, label = worst[group]
        print(f"  worst {group}: {residual * 1000:.3f} mm at {label}")
    print(f"  worst overall: {overall[0] * 1000:.3f} mm at {overall[1]}")
    print()
    if overall[0] > _PASS_RESIDUAL:
        print(
            f"Result: FAIL (worst {overall[0] * 1000:.3f} mm > {_PASS_RESIDUAL * 1000:.0f} mm)"
        )
        sys.exit(1)
    print(f"Result: PASS (worst {overall[0] * 1000:.3f} mm)")


if __name__ == "__main__":
    main()
