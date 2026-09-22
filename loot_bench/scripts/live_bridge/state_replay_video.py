"""Headless state-set replay to MP4 (mirrors state_replay.py).

Instead of opening the MuJoCo viewer, this renders offscreen
(render_mode="rgb_array") and pipes frames to ffmpeg (libx264/yuv420p).

Usage:
    uv run python loot_bench/scripts/live_bridge/state_replay_video.py \\
        --dataset-dir dataset/demos/LOOT-Domino-4Pieces-KnockOver-Dk1/2026-09-21_09-36-57 \\
        --task LOOT-Domino-4Pieces-KnockOver-Dk1 \\
        --out out/videos/domino4_knockover_ep0.mp4
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import lerobot.datasets.lerobot_dataset  # noqa: F401,E402,I001

import argparse
import contextlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

import loot_bench  # noqa: F401  (registers LOOT-* tasks)
from loot_bench.scripts.live_bridge.state_replay import load_episode


def build_render_env(task: str, device: str, width: int, height: int):
    cfg = load_env_cfg(task, play=True)
    cfg.terminations["time_out"].time_out = False
    cfg.terminations.pop("ee_ground_collision", None)
    cfg.terminations.pop("success", None)
    cfg.viewer.width = width
    cfg.viewer.height = height
    inner = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
    return RslRlVecEnvWrapper(inner), inner


def main():
    p = argparse.ArgumentParser(description="Headless state-set replay to MP4")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--task", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--crf", type=int, default=18)
    a = p.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("[ERROR] ffmpeg not found on PATH")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    ep = load_episode(a.dataset_dir, a.episode_index)
    n = len(ep.qpos)
    print(f"[INFO] Episode loaded: {n} frames")

    env, inner = build_render_env(a.task, device, a.width, a.height)
    if ep.seed is not None:
        print(f"[INFO] Restoring episode seed {ep.seed}")
        inner.reset(seed=ep.seed)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{a.width}x{a.height}",
        "-r",
        str(a.fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(a.crf),
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
        str(out),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    assert proc.stdin is not None
    try:
        for i in range(n):
            inner.sim.data.qpos[0] = torch.from_numpy(ep.qpos[i]).to(device)
            inner.sim.data.qvel[0] = torch.from_numpy(ep.qvel[i]).to(device)
            inner.sim.forward()
            inner.scene.update(dt=0.0)
            inner.command_manager.compute(dt=0.0)
            frame = inner.render()
            if frame is None:
                raise RuntimeError(
                    "env.render() returned None (render_mode != rgb_array?)"
                )
            frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if frame.shape[:2] != (a.height, a.width):
                raise RuntimeError(
                    f"frame shape {frame.shape} != {(a.height, a.width, 3)}"
                )
            proc.stdin.write(frame.tobytes())
            if (i + 1) % 100 == 0:
                print(f"[INFO] Frame {i + 1}/{n}")
    finally:
        with contextlib.suppress(BrokenPipeError):
            proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"[ERROR] ffmpeg exited with code {rc}")
    env.close()
    size_mb = out.stat().st_size / 1e6
    print(f"[INFO] Done: {out} ({n} frames @ {a.fps}fps, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
