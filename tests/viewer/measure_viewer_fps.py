"""Measure the native viewer's real FPS.

See tests/viewer/README.md.
"""

import argparse
import sys

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import NativeMujocoViewer

import g3a_bench  # noqa: F401  (registers tasks)
from g3a_bench.scripts.utils.viewer_wrapper import NativeViewerWrapper

DEFAULT_TASK = "G3A-Stack-Low-Dk1_uni"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument(
        "--viewer",
        choices=["stock", "wrapper"],
        default="wrapper",
        help="'stock' is MuJoCo's native viewer; 'wrapper' is G3A's wrapper of MuJoCo's native viewer",
    )
    ap.add_argument("--frame-rate", type=float, default=240.0)
    ap.add_argument("--steps", type=int, default=25, help="env steps before exit")
    ap.add_argument(
        "--reward-plots",
        action="store_true",
        help="show reward plots (exercises the set_figures path)",
    )
    ap.add_argument("--cpu", action="store_true", help="force CPU execution")
    args = ap.parse_args()

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(args.task)

    device = "cuda:0" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    BaseViewer = NativeViewerWrapper if args.viewer == "wrapper" else NativeMujocoViewer
    samples: list[float] = []

    class MeteredViewer(BaseViewer):  # pyright: ignore[reportGeneralTypeIssues]
        def _update_stats(self):
            before = self._stats_last_time
            super()._update_stats()
            if self._stats_last_time != before:  # a 0.5 s window just closed
                samples.append(self._fps)
                print(f"  fps={self._fps:.1f} sps={self._sps:.1f}")

    def policy(obs):
        del obs
        return torch.zeros(
            env.unwrapped.action_space.shape, device=env.unwrapped.device
        )

    viewer = MeteredViewer(env, policy, frame_rate=args.frame_rate)
    viewer._speed_index = 0  # 0 = 1/32x slo-mo
    viewer._time_multiplier = viewer.SPEED_MULTIPLIERS[viewer._speed_index]
    viewer._show_plots = args.reward_plots
    viewer.run(num_steps=args.steps)
    env.close()

    steady = samples[2:]  # drop warmup windows
    if not steady:
        print("RESULT: no samples (run too short?)")
        return 1
    median = sorted(steady)[len(steady) // 2]
    print(
        f"RESULT: viewer={args.viewer} frame_rate={args.frame_rate:g} "
        f"plots={args.reward_plots} median_fps={median:.1f} (n={len(steady)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
