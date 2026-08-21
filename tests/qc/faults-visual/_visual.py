"""Shared helpers for the before/after visual checks. See tests/qc/faults-visual/README.md.

Every check reconstructs the *old* behaviour from constants that still exist in
the tree (the old residual action scale, the rest pose, mjlab's own quaternion
conversion), so "before" is recomputed on each run rather than quoted from a
document. Nothing here changes task code.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - headless: no display on the training boxes

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

import g3a_bench  # noqa: F401,E402  (registers all tasks)
from g3a_bench.robots.trlc_dk1.dk1_actuators import (  # noqa: E402
    GRIPPER_MECHANICAL_RANGE,
    get_arm_actuators,
)

# The offscreen renderer already used by the task verification scripts, loaded
# from its own folder rather than copied.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tasks"))
save_env_render = importlib.import_module("_render").save_env_render

TASK_ID = "G3A-Place-Easy-Dk1_uni"
OUT_DIR = Path("out/render/faults-visual")

# The two recorded sessions that bracket the rebuild. The first was recorded
# before the fixes and is kept exactly as it was; the second after them.
DATASET_BEFORE = Path(
    "dataset/demos/G3A-Place-Easy-Dk1_uni/G3A-Place-Easy-Dk1_uni_qc_physical.npz"
)
DATASET_AFTER = Path("out/qc/place-easy.npz")

BEFORE_COLOR = "#a53a24"
AFTER_COLOR = "#3f6b34"
DPI = 130


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cpu", action="store_true", help="Force CPU device.")
    parser.add_argument(
        "--out", type=Path, default=OUT_DIR, help=f"Output folder [{OUT_DIR}]."
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip the simulator pictures; write the plots only.",
    )
    return parser


def pick_device(args: argparse.Namespace) -> str:
    return "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")


def build_env(device: str, task_id: str = TASK_ID) -> ManagerBasedRlEnv:
    """One environment of *task_id*, reset once."""
    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.reset()
    return env


def action_joint_ids(env: ManagerBasedRlEnv) -> list[int]:
    """Entity joint indices the action term drives, in action order."""
    return [int(i) for i in env.action_manager.get_term("joint_pos").target_ids]


def old_residual_scale(joint_names: list[str]) -> np.ndarray:
    """The pre-rebuild residual action scale [rad], per named joint.

    Recomputed from the actuator configs exactly as the deleted
    ``_dk1_action_scale`` did (a quarter of the effort limit over the
    stiffness for the arm, half the rack stroke for the gripper), so this is
    the real old number and not a copy of it. Under mjlab's plain
    ``JointPositionAction`` a bounded policy could command only
    ``keyframe +/- scale``.
    """
    scale: dict[str, float] = {}
    for actuator in get_arm_actuators():
        assert actuator.effort_limit is not None
        for name in actuator.target_names_expr:
            scale[name] = 0.25 * actuator.effort_limit / actuator.stiffness
    scale["gripper_left"] = 0.5 * (
        GRIPPER_MECHANICAL_RANGE["open"] - GRIPPER_MECHANICAL_RANGE["closed"]
    )
    scale["gripper_right"] = scale["gripper_left"]
    return np.array([scale[name] for name in joint_names], dtype=np.float64)


def full_pose(
    env: ManagerBasedRlEnv, values: np.ndarray, ids: list[int]
) -> torch.Tensor:
    """A whole joint vector: the keyframe, with *values* written at *ids*.

    ``gripper_right`` follows ``gripper_left``: the two are one rack-and-pinion
    pair joined by an equality constraint, which ``mj_forward`` does not
    enforce on positions.
    """
    robot = env.scene["robot"]
    q = robot.data.default_joint_pos.clone()
    for slot, joint_id in enumerate(ids):
        q[0, joint_id] = float(values[slot])
    names = list(robot.joint_names)
    if "gripper_left" in names and "gripper_right" in names:
        q[0, names.index("gripper_right")] = q[0, names.index("gripper_left")]
    return q


def apply_pose(env: ManagerBasedRlEnv, q: torch.Tensor) -> None:
    """Teleport the robot to *q* (kinematics only, no physics step)."""
    robot = env.scene["robot"]
    robot.write_joint_position_to_sim(q)
    robot.write_joint_velocity_to_sim(torch.zeros_like(q))
    env.sim.forward()


def grasp_site_pos(env: ManagerBasedRlEnv, site: str = "grasp_site") -> np.ndarray:
    robot = env.scene["robot"]
    index = list(robot.site_names).index(site)
    pos = robot.data.site_pos_w[0, index] - env.scene.env_origins[0]
    return pos.detach().cpu().numpy()


def obs_slice(env: ManagerBasedRlEnv, term: str, group: str = "actor") -> slice:
    """Columns of *term* inside the flat observation vector.

    Read off the built environment rather than counted by hand, so a task that
    gains an observation term does not silently move the slice.
    """
    manager = env.observation_manager
    start = 0
    for name, dim in zip(
        manager.active_terms[group], manager.group_obs_term_dim[group], strict=True
    ):
        width = int(np.prod(dim))
        if name == term:
            return slice(start, start + width)
        start += width
    raise KeyError(f"{term} is not an observation term of group {group}")


def place_object(env: ManagerBasedRlEnv, name: str, pose: np.ndarray) -> None:
    """Teleport a free entity to an env-local pose (position then quaternion)."""
    entity = env.scene[name]
    state = torch.as_tensor(pose, dtype=torch.float32, device=env.device).unsqueeze(0)
    state = state.clone()
    state[:, :3] = state[:, :3] + env.scene.env_origins
    entity.write_root_link_pose_to_sim(state)
    env.sim.forward()


def render_pose(env: ManagerBasedRlEnv, q: torch.Tensor, path: Path) -> str | None:
    """Teleport to *q* and save a picture. Returns a warning, or ``None``."""
    apply_pose(env, q)
    return save_env_render(env, path, draw_goals=False)


def montage(images: list[Path], titles: list[str], out: Path, suptitle: str) -> None:
    """Lay rendered pictures out side by side, with a caption under each."""
    usable = [(p, t) for p, t in zip(images, titles, strict=True) if p.exists()]
    if not usable:
        return
    fig, axes = plt.subplots(1, len(usable), figsize=(7.5 * len(usable), 4.6))
    for ax, (path, title) in zip(np.atleast_1d(axes), usable, strict=True):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=13)
    save_fig(fig, out)


def save_fig(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {path}")


def write_result(
    out: Path, key: str, title: str, question: str, lines: list[str], images: list[Path]
) -> None:
    """Record one check's verdict for the index page, and print it."""
    print()
    print(f"  {title}")
    for line in lines:
        print(f"    {line}")
    payload = {
        "key": key,
        "title": title,
        "question": question,
        "lines": lines,
        "images": [str(p.name) for p in images if p.exists()],
    }
    path = out / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def first_frames(dataset: Path, width: int) -> np.ndarray:
    """First observation frame of every episode, joint block only (E, width).

    The joint block is ``joint_pos_rel``: the joint position minus the
    keyframe. Both recorded sessions put it first, so the same slice reads
    both, although the rest of the observation changed shape in between.
    """
    data = np.load(dataset, allow_pickle=True)
    obs = data["observations"]
    episodes = data["episode_index"]
    starts = [np.flatnonzero(episodes == e)[0] for e in np.unique(episodes)]
    return obs[starts, :width].astype(np.float64)
