# pyright: reportAttributeAccessIssue=false
"""Exact 1-1 state-set replay in the MuJoCo viewer.

Instead of re-simulating (feeding actions back and letting physics integrate),
this script reads qpos+qvel per frame from the dataset and *sets* them
directly into the simulator, then calls mj_forward() (kinematics only, no
physics integration).  The viewer renders from the set state — drift-free,
bitwise-exact reproduction of the recorded trajectory.

Usage:
    uv run python g3a_bench/scripts/live_bridge/state_replay.py \
        --dataset-dir out/demos/success/G3A-Stack3-Easy-Dk1_uni/2026-07-11_18-23-00

    # replay a specific episode
    uv run python g3a_bench/scripts/live_bridge/state_replay.py \
        --dataset-dir ... --episode-index 2

The native MuJoCo window opens and replays the dataset by teleporting state.
After the episode finishes, the viewer keeps running (hold last frame) until
you close the window or press Ctrl+C.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import lerobot.datasets.lerobot_dataset  # noqa: F401,E402,I001

import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import VerbosityLevel

import g3a_bench  # noqa: F401
from g3a_bench.scripts.live_bridge.dataset_extra_info import load_extra_info


# ── Episode loading ───────────────────────────────────────────────────────────


@dataclass
class Episode:
    """One recorded episode, fully materialized as numpy arrays."""

    qpos: np.ndarray  # (N, nq) per-frame sim positions
    qvel: np.ndarray  # (N, nv) per-frame sim velocities
    done: np.ndarray  # (N,) bool
    seed: int | None
    seed_device: str | None


def _resolve_dataset_root(dataset_dir: str) -> Path:
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

    qpos = col("observation.qpos", np.float32)
    qvel = col("observation.qvel", np.float32)
    done = col("next.done", bool)

    if qpos is None:
        raise SystemExit(
            "[ERROR] Dataset has no observation.qpos — state-set replay "
            "requires ground-truth sim state. Re-record with the updated "
            "live_record.py."
        )
    if qvel is None:
        raise SystemExit(
            "[ERROR] Dataset has no observation.qvel — state-set replay "
            "requires velocities. Re-record with the updated live_record.py."
        )

    extra_info = load_extra_info(ds.meta.root)
    seed_val = extra_info.get("episode_seeds", {}).get(str(episode_index))
    return Episode(
        qpos=qpos,
        qvel=qvel,
        done=done.reshape(len(qpos)),
        seed=int(seed_val) if seed_val is not None else None,
        seed_device=extra_info.get("seed_device"),
    )


# ── Env builder ───────────────────────────────────────────────────────────────


def build_env(task: str, device: str) -> RslRlVecEnvWrapper:
    cfg = load_env_cfg(task, play=True)
    cfg.terminations["time_out"].time_out = False
    cfg.terminations.pop("ee_ground_collision", None)
    cfg.terminations.pop("success", None)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    return RslRlVecEnvWrapper(env)


# ── State-set replay viewer ───────────────────────────────────────────────────


class _StateReplayViewer(NativeMujocoViewer):
    """Viewer that replays by setting qpos+qvel each frame (no physics)."""

    def __init__(self, env, episode: Episode, device: str, **kwargs):
        super().__init__(
            env,
            _NullPolicy(device),
            verbosity=VerbosityLevel.DEBUG,
            **kwargs,
        )
        self._ep = episode
        self._device = device
        self._frame = 0
        self._finished = False
        self._n = len(episode.qpos)

    def _execute_step(self) -> bool:
        try:
            if self._finished:
                return True  # hold last frame

            if self._frame >= self._n:
                print(f"[INFO] State-set replay complete — {self._n} frames played")
                self._finished = True
                return True

            qpos_np = self._ep.qpos[self._frame]
            qvel_np = self._ep.qvel[self._frame]

            # Teleport sim state — no mj_step(), no physics integration.
            self.env.env.sim.data.qpos[0] = torch.from_numpy(qpos_np).to(self._device)
            self.env.env.sim.data.qvel[0] = torch.from_numpy(qvel_np).to(self._device)
            self.env.env.sim.forward()

            # Refresh scene entities and command metrics so the debug
            # visualizer reflects success status (green target spheres).
            self.env.env.scene.update(dt=0.0)
            self.env.env.command_manager.compute(dt=0.0)

            self._frame += 1
            if self._frame % 100 == 0:
                print(f"[INFO] Frame {self._frame}/{self._n}")

            return True
        except Exception:
            self._last_error = traceback.format_exc()
            self.log(
                f"[ERROR] Exception during state-set step:\n{self._last_error}",
                VerbosityLevel.SILENT,
            )
            self.pause()
            return False


class _NullPolicy:
    """Dummy policy — never called, but the viewer expects one."""

    def __init__(self, device: str):
        self._device = device

    def __call__(self, obs):
        return torch.zeros(1, 7, device=self._device)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Exact 1-1 state-set replay in the MuJoCo viewer"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="live_record session/dataset dir (e.g. out/demos/success/<task>/<stamp>)",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--task",
        type=str,
        default="G3A-Stack3-Easy-Dk1_uni",
        help="MJLab task name to load.",
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    ep = load_episode(args.dataset_dir, args.episode_index)
    print(f"[INFO] Episode loaded: {len(ep.qpos)} frames")
    if ep.seed is not None:
        print(f"[INFO] Seed: {ep.seed} (device: {ep.seed_device})")

    env = build_env(args.task, device)

    # Seed-reproduce the object layout
    if ep.seed is not None:
        print(f"[INFO] Restoring episode seed {ep.seed}")
        env.env.reset(seed=ep.seed)

    # Teleport to the first recorded state
    env.env.sim.data.qpos[0] = torch.from_numpy(ep.qpos[0]).to(device)
    env.env.sim.data.qvel[0] = torch.from_numpy(ep.qvel[0]).to(device)
    env.env.sim.forward()
    print("[INFO] Sim state set to first recorded frame")

    viewer = _StateReplayViewer(env=env, episode=ep, device=device)
    print("[INFO] Starting state-set replay (no physics integration)")
    print("[INFO] Close the viewer window or press Ctrl+C to stop")
    viewer.run()

    env.close()
    print("[INFO] Done")


if __name__ == "__main__":
    main()
