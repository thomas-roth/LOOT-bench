"""Measure the native viewer's real FPS (needs a display; not for CI).

Launches the native MuJoCo viewer on a 1-env zero-policy task and reports the
smoothed FPS samples the viewer computes every 0.5 s. Defaults to 1/32x
slow-motion so the render loop is not physics-bound even on CPU-only machines
— the point is to measure the *renderer* cap, not the sim throughput.

Usage (from repo root):
  uv run python tests/viewer/measure_viewer_fps.py                  # expect >> 60
  uv run python tests/viewer/measure_viewer_fps.py --viewer stock   # shows 60 cap
  uv run python tests/viewer/measure_viewer_fps.py --plots          # figures path
  uv run python tests/viewer/measure_viewer_fps.py --frame-rate 20 --min-fps 0
"""

import argparse
import sys

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import NativeMujocoViewer

import g3a_bench  # noqa: F401  (registers tasks)
from g3a_bench.scripts.viewer_wrapper import NativeViewerWrapper

DEFAULT_TASK = "G3a-Cube-Stack3-Dk1_uni"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument(
        "--viewer",
        choices=["stock", "throttled"],
        default="throttled",
        help="'throttled' is what g3a-play uses; 'stock' is mjlab's viewer",
    )
    ap.add_argument("--frame-rate", type=float, default=240.0)
    ap.add_argument("--steps", type=int, default=25, help="env steps before exit")
    ap.add_argument(
        "--speed-index",
        type=int,
        default=0,
        help="index into BaseViewer.SPEED_MULTIPLIERS (0 = 1/32x slow-mo)",
    )
    ap.add_argument(
        "--plots",
        action="store_true",
        help="show reward plots (exercises the set_figures path)",
    )
    ap.add_argument(
        "--min-fps",
        type=float,
        default=None,
        help="exit non-zero if the median FPS is below this",
    )
    ap.add_argument("--cpu", action="store_true", help="force CPU execution")
    args = ap.parse_args()

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(args.task)

    device = "cuda:0" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    action_shape = env.unwrapped.action_space.shape

    def policy(obs):
        del obs
        return torch.zeros(action_shape, device=env.unwrapped.device)

    base_cls = NativeViewerWrapper if args.viewer == "throttled" else NativeMujocoViewer

    samples: list[float] = []

    class MeteredViewer(base_cls):  # pyright: ignore[reportGeneralTypeIssues]
        def _update_stats(self):
            before = self._stats_last_time
            super()._update_stats()
            if self._stats_last_time != before:  # a 0.5 s window just closed
                samples.append(self._fps)
                print(f"  fps={self._fps:.1f} sps={self._sps:.1f}")

    viewer = MeteredViewer(env, policy, frame_rate=args.frame_rate)
    viewer._speed_index = args.speed_index
    viewer._time_multiplier = viewer.SPEED_MULTIPLIERS[args.speed_index]
    viewer._show_plots = args.plots
    viewer.run(num_steps=args.steps)
    env.close()

    steady = samples[2:]  # drop warmup windows
    if not steady:
        print("RESULT: no samples (run too short?)")
        return 1
    median = sorted(steady)[len(steady) // 2]
    print(
        f"RESULT: viewer={args.viewer} frame_rate={args.frame_rate:g} "
        f"plots={args.plots} median_fps={median:.1f} (n={len(steady)})"
    )
    if args.min_fps is not None and median < args.min_fps:
        print(f"FAIL: median {median:.1f} < required {args.min_fps:g}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
