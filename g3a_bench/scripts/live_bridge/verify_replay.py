# pyright: reportAttributeAccessIssue=false
"""Numerically verify that a recorded demo replays *exactly* — no viewer,
no eyeballing.

Re-runs a live_record.py episode headlessly through the sim (seeded reset +
recorded actions, exactly the way live_record stepped it) and compares the
resulting sim state against the ground truth stored per frame in the dataset
(``observation.qpos``), printing per-frame deviation stats, the first frame
that exceeds tolerance, the worst offending DOFs, and a hard PASS/FAIL
verdict (exit code 0/1).

What is compared per frame i (alignment matches recording_mixin._execute_step):
  - pre-step actor obs        vs  observation.state[i]
  - post-step full sim qpos   vs  observation.qpos[i]   (recorded post-step)
  - step reward               vs  next.reward[i]
Frames whose recorded ``next.done`` is True are excluded from the qpos check:
during recording the env auto-resets *inside* env.step() on done, so the
recorded qpos of that frame is the post-reset scatter, not the trajectory.

Known, expected divergence in the default mode: at recording time the arm
started from the operator's live pose (written back after the seeded reset),
which is not stored in the dataset — replay starts from the reset pose
instead. The robot DOFs therefore show an initial transient that the
position-controlled arm converges out of; the verdict allows it a settle
window (--settle-frames) while object DOFs must match from frame 0.
Use --sync-start to instead teleport the sim to the first recorded qpos
(velocities are not recorded and are zeroed — a documented approximation)
and require *every* compared frame to match.

Usage:
    # verify episode 0 of a recorded session
    uv run python g3a_bench/scripts/live_bridge/verify_replay.py \\
        --dataset-dir out/demos/success/2026-07-11_18-23-00

    # verify a specific episode, stricter tolerance, per-frame CSV
    uv run python g3a_bench/scripts/live_bridge/verify_replay.py \\
        --dataset-dir ... --episode-index 1 --tol 1e-6 --csv /tmp/drift.csv

    # offline: diff an existing record_replay.py output against the original
    uv run python g3a_bench/scripts/live_bridge/verify_replay.py \\
        --dataset-dir out/demos/success/<stamp> --episode-index 0 \\
        --sim-replay-dir out/demos/success/<stamp>/sim_replay/episode_0
"""

import os

# This script only ever reads *local* datasets, but lerobot silently falls
# back to pulling repo_id from the HuggingFace hub when the local metadata is
# incomplete — turning a local problem into a confusing 401. Stay offline.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Load lerobot/HF-datasets native libs before mjlab/mujoco-warp to avoid a
# native import-order segfault (HF `datasets` vs mujoco-warp). Must precede mjlab.
import lerobot.datasets.lerobot_dataset  # noqa: F401,E402,I001

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

import g3a_bench  # noqa: F401
from g3a_bench.scripts.live_bridge.live_record import (
    apply_teleop_contact_stability,
)
from g3a_bench.scripts.live_bridge.mapper import lerobot_to_mjlab_raw_action


@dataclass
class Episode:
    """One recorded episode, fully materialized as numpy arrays."""

    actions: np.ndarray  # (N, 7) raw leader actions as recorded
    state: np.ndarray  # (N, obs_dim) pre-step actor obs
    qpos: np.ndarray | None  # (N, nq) post-step sim qpos (ground truth)
    done: np.ndarray  # (N,) bool
    reward: np.ndarray  # (N,) float32
    seed: int | None
    seed_device: str | None


def _resolve_dataset_root(dataset_dir: str) -> Path:
    """Accept either a dataset root or a session dir containing success/."""
    root = Path(dataset_dir)
    success_sub = root / "success"
    if not (root / "meta").exists() and (success_sub / "meta").exists():
        return success_sub
    return root


def load_episode(dataset_dir: str, episode_index: int) -> Episode:
    root = _resolve_dataset_root(dataset_dir)
    if not (root / "meta" / "info.json").exists():
        raise SystemExit(
            f"[ERROR] {root} is not a LeRobot dataset (no meta/info.json)."
        )
    try:
        ds = LeRobotDataset("live_record", root=str(root))
    except (FileNotFoundError, NotADirectoryError, OSError) as e:
        raise SystemExit(
            f"[ERROR] Could not load dataset at {root}: {e}\n"
            "The dataset metadata looks incomplete (e.g. missing "
            "meta/episodes/) — was the recording session finalized "
            "(numpad 4 in live_record)?"
        ) from e
    hf = ds.hf_dataset
    ep_col = np.asarray([int(v) for v in hf["episode_index"]])
    idxs = np.nonzero(ep_col == episode_index)[0]
    if len(idxs) == 0:
        raise SystemExit(
            f"[ERROR] Episode {episode_index} not found in {root} "
            f"(episodes present: {sorted(set(ep_col.tolist()))})"
        )
    sub = hf.select(idxs)

    def col(name: str, dtype) -> np.ndarray | None:
        if name not in sub.column_names:
            return None
        return np.stack([np.asarray(v, dtype=dtype) for v in sub[name]])

    actions = col("action", np.float32)
    state = col("observation.state", np.float32)
    qpos = col("observation.qpos", np.float32)
    done = col("next.done", bool)
    reward = col("next.reward", np.float32)
    assert actions is not None and state is not None
    assert done is not None and reward is not None

    seed_val = ds.meta.info.get("episode_seeds", {}).get(str(episode_index))
    return Episode(
        actions=actions,
        state=state,
        qpos=qpos,
        done=done.reshape(len(actions)),
        reward=reward.reshape(len(actions)),
        seed=int(seed_val) if seed_val is not None else None,
        seed_device=ds.meta.info.get("seed_device"),
    )


def build_env_like_live_record(task: str, device: str) -> RslRlVecEnvWrapper:
    """Build the env with the exact same config mutations as live_record.main().

    Any difference here (contact solref, timestep, ...) changes the physics
    being compared, so this must stay in lockstep with live_record.py.
    """
    cfg = load_env_cfg(task, play=True)
    cfg.terminations["time_out"].time_out = False

    # Same finger-pad solref stabilization live_record applies before building
    # the env — different contact parameters would diverge at first grip.
    apply_teleop_contact_stability(cfg)

    # Unlike live_record, drop done-producing terminations: a done mid-replay
    # auto-resets the env inside step() and destroys the comparison.
    # Termination terms only produce `done`, never forces — physics identical.
    cfg.terminations.pop("ee_ground_collision", None)
    cfg.terminations.pop("success", None)

    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    return RslRlVecEnvWrapper(env)


def qpos_labels(mj_model: mujoco.MjModel) -> list[str]:
    """Human-readable name for every qpos index (free joints get 7 entries)."""
    labels = [f"qpos[{i}]" for i in range(mj_model.nq)]
    for j in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"jnt{j}"
        adr = int(mj_model.jnt_qposadr[j])
        jtype = int(mj_model.jnt_type[j])
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            comps = ["x", "y", "z", "qw", "qx", "qy", "qz"]
        elif jtype == mujoco.mjtJoint.mjJNT_BALL:
            comps = ["qw", "qx", "qy", "qz"]
        else:
            comps = [""]
        for k, c in enumerate(comps):
            labels[adr + k] = f"{name}[{c}]" if c else name
    return labels


@dataclass
class FrameStats:
    frame: int
    obs_dev: float
    robot_dev: float
    object_dev: float
    reward_dev: float
    worst_label: str
    qpos_compared: bool


def _summarize(name: str, devs: list[float], frames: list[int], tol: float) -> None:
    if not devs:
        print(f"  {name}: no frames compared")
        return
    arr = np.array(devs)
    n_exact = int(np.sum(arr == 0.0))
    n_over = int(np.sum(arr > tol))
    worst_i = int(np.argmax(arr))
    first_over = next((frames[i] for i in range(len(arr)) if arr[i] > tol), None)
    print(
        f"  {name}: {len(arr)} frames | bitwise-identical {n_exact} | "
        f"mean {arr.mean():.3e} | max {arr.max():.3e} @ frame {frames[worst_i]} | "
        f"{n_over} frames > tol"
        + (f" (first at frame {first_over})" if first_over is not None else "")
    )


def report(
    stats: list[FrameStats],
    labels: list[str],
    robot_mask: np.ndarray,
    per_dof_max: np.ndarray,
    tol: float,
    settle_frames: int,
    sync_start: bool,
    n_excluded_done: int,
) -> bool:
    print("\n=== Replay-vs-recording comparison ===")
    compared = [s for s in stats if s.qpos_compared]
    frames = [s.frame for s in compared]
    _summarize("object/scene qpos dev ", [s.object_dev for s in compared], frames, tol)
    _summarize("robot qpos dev        ", [s.robot_dev for s in compared], frames, tol)
    _summarize(
        "actor obs dev         ",
        [s.obs_dev for s in stats],
        [s.frame for s in stats],
        tol,
    )
    _summarize(
        "reward dev            ",
        [s.reward_dev for s in compared],
        frames,
        tol,
    )
    if n_excluded_done:
        print(
            f"  (excluded {n_excluded_done} done-frame(s) from the qpos check: "
            "recorded qpos there is the post-auto-reset scatter, not the "
            "trajectory — known live_record behavior)"
        )

    # Worst DOFs across the episode.
    order = np.argsort(per_dof_max)[::-1][:5]
    print("  worst DOFs over episode:")
    for di in order:
        kind = "robot " if robot_mask[di] else "object"
        print(f"    {per_dof_max[di]:.3e}  {kind}  {labels[di]}")

    # Verdict.
    obj_ok = all(s.object_dev <= tol for s in compared)
    if sync_start:
        rob_ok = all(s.robot_dev <= tol for s in compared)
        rob_rule = "all frames"
    else:
        rob_ok = all(s.robot_dev <= tol for s in compared if s.frame >= settle_frames)
        rob_rule = f"frames >= {settle_frames} (initial live-pose transient allowed)"
    print(f"\n  tolerance: {tol:g}")
    print(f"  object/scene DOFs within tol on all frames: {'YES' if obj_ok else 'NO'}")
    print(f"  robot DOFs within tol on {rob_rule}: {'YES' if rob_ok else 'NO'}")
    ok = obj_ok and rob_ok
    print(f"\nVERDICT: {'PASS — replay reproduces the recording' if ok else 'FAIL'}")
    if not ok:
        first_bad = next(
            (
                s
                for s in compared
                if s.object_dev > tol
                or (s.robot_dev > tol and (sync_start or s.frame >= settle_frames))
            ),
            None,
        )
        if first_bad is not None:
            print(
                f"  first offending frame: {first_bad.frame} "
                f"(robot {first_bad.robot_dev:.3e}, object "
                f"{first_bad.object_dev:.3e}, worst DOF {first_bad.worst_label})"
            )
    return ok


def write_csv(path: str, stats: list[FrameStats]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "frame",
                "obs_dev",
                "robot_qpos_dev",
                "object_qpos_dev",
                "reward_dev",
                "worst_dof",
                "qpos_compared",
            ]
        )
        for s in stats:
            w.writerow(
                [
                    s.frame,
                    f"{s.obs_dev:.9g}",
                    f"{s.robot_dev:.9g}",
                    f"{s.object_dev:.9g}",
                    f"{s.reward_dev:.9g}",
                    s.worst_label,
                    int(s.qpos_compared),
                ]
            )
    print(f"[INFO] Per-frame deviations written to {path}")


def resimulate_and_compare(args, ep: Episode) -> bool:
    if ep.seed is None:
        raise SystemExit(
            "[ERROR] Dataset has no recorded seed for this episode — the object "
            "scatter cannot be reproduced, so exact verification is impossible. "
            "Re-record with the current live_record.py."
        )
    if ep.qpos is None:
        raise SystemExit(
            "[ERROR] Dataset has no observation.qpos — ground-truth state was "
            "not recorded. Re-record with the current live_record.py."
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if ep.seed_device is not None and ep.seed_device != device.split(":")[0]:
        print(
            f"[WARN] Recorded on '{ep.seed_device}' but verifying on "
            f"'{device.split(':')[0]}' — torch RNG streams differ per device; "
            "the seeded scatter will NOT match and the check will fail."
        )
    print(f"[INFO] Using device: {device}")

    env = build_env_like_live_record(args.task, device)
    mj_model = env.env.sim.mj_model
    labels = qpos_labels(mj_model)
    robot_mask = np.array([lab.startswith("robot/") for lab in labels])

    nq = int(env.env.sim.data.qpos.shape[-1])
    if nq != ep.qpos.shape[1]:
        raise SystemExit(
            f"[ERROR] qpos size mismatch: env has nq={nq}, dataset recorded "
            f"{ep.qpos.shape[1]} — wrong --task for this recording?"
        )
    obs_dim = int(env.get_observations()["actor"].shape[-1])
    if obs_dim != ep.state.shape[1]:
        print(
            f"[WARN] actor obs size mismatch (env {obs_dim} vs recorded "
            f"{ep.state.shape[1]}) — obs drift will not be compared."
        )

    n = len(ep.actions)
    print(f"[INFO] Episode: {n} frames, seed {ep.seed}")
    env.env.reset(seed=ep.seed)

    # Seed-reproduction check: object DOFs of the freshly reset scene vs the
    # first recorded (post-step-0) qpos. Objects are at rest at episode start,
    # so one control step moves them ~0 — any real deviation here means the
    # seeded scatter itself did not reproduce.
    qpos_now = env.env.sim.data.qpos[0].cpu().numpy()
    scatter_dev = float(np.max(np.abs((qpos_now - ep.qpos[0])[~robot_mask])))
    print(
        f"[INFO] Seeded-scatter reproduction (object DOFs vs frame 0): "
        f"max dev {scatter_dev:.3e}"
    )

    start = 0
    if args.sync_start:
        # Teleport to the first recorded state. Velocities are not recorded —
        # zeroing them is the documented approximation of this mode.
        env.env.sim.data.qpos[0] = torch.from_numpy(ep.qpos[0]).to(device)
        env.env.sim.data.qvel[0] = 0.0
        env.env.sim.forward()
        start = 1
        print(
            "[INFO] --sync-start: sim state set to recorded frame-0 qpos "
            "(qvel zeroed, not recorded); comparing from frame 1"
        )

    stats: list[FrameStats] = []
    per_dof_max = np.zeros(nq)
    n_excluded_done = 0
    with torch.no_grad():
        for i in range(start, n):
            obs = env.get_observations()
            if obs_dim == ep.state.shape[1]:
                obs_sim = obs["actor"][0].cpu().numpy().ravel()
                obs_dev = float(np.max(np.abs(obs_sim - ep.state[i])))
            else:
                obs_dev = float("nan")

            raw = lerobot_to_mjlab_raw_action(ep.actions[i], gravity_compensate=True)
            _, rew, _dones, _ = env.step(torch.from_numpy(raw).unsqueeze(0).to(device))

            qpos_sim = env.env.sim.data.qpos[0].cpu().numpy()
            if bool(ep.done[i]):
                n_excluded_done += 1
                stats.append(
                    FrameStats(i, obs_dev, 0.0, 0.0, 0.0, "(done frame)", False)
                )
                continue
            d = np.abs(qpos_sim - ep.qpos[i])
            per_dof_max = np.maximum(per_dof_max, d)
            worst = int(np.argmax(d))
            stats.append(
                FrameStats(
                    frame=i,
                    obs_dev=obs_dev,
                    robot_dev=float(d[robot_mask].max()),
                    object_dev=float(d[~robot_mask].max()),
                    reward_dev=abs(float(rew[0]) - float(ep.reward[i])),
                    worst_label=labels[worst],
                    qpos_compared=True,
                )
            )
            if (i + 1) % 200 == 0:
                print(f"[INFO] Replayed {i + 1}/{n} frames")

    ok = report(
        stats,
        labels,
        robot_mask,
        per_dof_max,
        args.tol,
        args.settle_frames,
        args.sync_start,
        n_excluded_done,
    )
    if args.csv:
        write_csv(args.csv, stats)
    env.close()
    return ok


def diff_existing_replay(args, ep: Episode) -> bool:
    """Offline mode: diff a record_replay.py output dataset against the
    original recording. No sim is built.

    Both live_record and record_replay capture qpos post-step, so replay
    qpos[i] corresponds to original qpos[i] (1:1 alignment).
    """
    rep = load_episode(args.sim_replay_dir, 0)
    print(
        f"[INFO] Original: {len(ep.actions)} frames | "
        f"replay dataset: {len(rep.actions)} frames"
    )
    n = min(len(ep.actions), len(rep.actions))
    if len(ep.actions) != len(rep.actions):
        print("[WARN] Frame-count mismatch — comparing the overlapping prefix")

    if np.array_equal(ep.actions[:n], rep.actions[:n]):
        print(
            "[INFO] Actions: bitwise identical (replay fed back exactly "
            "what was recorded)"
        )
        actions_ok = True
    else:
        bad = np.max(np.abs(ep.actions[:n] - rep.actions[:n]))
        print(
            f"[ERROR] Actions differ (max abs {bad:.3e}) — the replay did "
            "not even feed the recorded actions back verbatim"
        )
        actions_ok = False

    obs_dev = np.max(np.abs(ep.state[:n] - rep.state[:n]), axis=1)
    rew_dev = np.abs(ep.reward[:n] - rep.reward[:n])

    stats: list[FrameStats] = []
    per_dof_max = None
    labels: list[str] = []
    robot_mask = np.array([])
    n_excluded_done = 0
    if ep.qpos is not None and rep.qpos is not None:
        # Need joint names for the breakdown → compile the model once, headless.
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        env = build_env_like_live_record(args.task, device)
        labels = qpos_labels(env.env.sim.mj_model)
        env.close()
        robot_mask = np.array([lab.startswith("robot/") for lab in labels])
        per_dof_max = np.zeros(len(labels))
        for i in range(n):
            # Terminal frame is post-auto-reset in both → qpos not comparable.
            if bool(ep.done[i]):
                n_excluded_done += 1
                stats.append(
                    FrameStats(
                        i, float(obs_dev[i]), 0.0, 0.0, 0.0, "(done frame)", False
                    )
                )
                continue
            d = np.abs(rep.qpos[i] - ep.qpos[i])  # both post-step → 1:1
            per_dof_max = np.maximum(per_dof_max, d)
            stats.append(
                FrameStats(
                    frame=i,
                    obs_dev=float(obs_dev[i]),
                    robot_dev=float(d[robot_mask].max()),
                    object_dev=float(d[~robot_mask].max()),
                    reward_dev=float(rew_dev[i]),
                    worst_label=labels[int(np.argmax(d))],
                    qpos_compared=True,
                )
            )
    else:
        print("[WARN] qpos missing in one of the datasets — comparing obs only")
        for i in range(n):
            stats.append(
                FrameStats(i, float(obs_dev[i]), 0.0, 0.0, float(rew_dev[i]), "", False)
            )

    ok = actions_ok and report(
        stats,
        labels,
        robot_mask,
        per_dof_max if per_dof_max is not None else np.zeros(1),
        args.tol,
        args.settle_frames,
        False,
        n_excluded_done,
    )
    if args.csv:
        write_csv(args.csv, stats)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a recorded demo replays exactly (numeric comparison)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="live_record session/dataset dir (e.g. out/demos/success/<stamp>)",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--task",
        type=str,
        default="G3a-Cube-Stack3-Dk1_uni",
        help="MJLab task the demo was recorded on",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Max allowed per-DOF |deviation| (rad / m / quat components). "
        "Bitwise-identical frames are reported separately.",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=50,
        help="Default mode only: frames the robot DOFs get to converge from the "
        "reset pose to the (unrecorded) live start pose before they must "
        "be within --tol (50 = 1 s at 50 fps).",
    )
    parser.add_argument(
        "--sync-start",
        action="store_true",
        help="Teleport the sim to the first recorded qpos instead of only the "
        "seeded reset (qvel is not recorded and is zeroed); every compared "
        "frame must then be within --tol.",
    )
    parser.add_argument(
        "--sim-replay-dir",
        type=str,
        default=None,
        help="Offline mode: instead of re-simulating, diff this existing "
        "record_replay.py output (…/sim_replay/episode_N) against the "
        "original recording.",
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Write per-frame deviations to CSV."
    )
    args = parser.parse_args()

    ep = load_episode(args.dataset_dir, args.episode_index)
    if args.sim_replay_dir:
        ok = diff_existing_replay(args, ep)
    else:
        ok = resimulate_and_compare(args, ep)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
