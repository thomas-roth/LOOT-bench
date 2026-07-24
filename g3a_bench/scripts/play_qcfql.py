"""Evaluate a QC-FQL checkpoint in the G3A MJLab sim.

Run from the QC repo root with g3a-bench on PYTHONPATH:

    cd ../qc
    PYTHONPATH=../g3a-bench:$PYTHONPATH python ../g3a-bench/g3a_bench/scripts/play_qcfql.py \\
        --checkpoint exp/qc/G3A-Low_Cube_stacking/G3A-Stack-Low-Dk1_unisd04220260721_180047/params_2000000.pkl \\
        --task G3A-Stack-Low-Dk1_uni \\
        --episodes 10

To load the .pkl.zip checkpoints from dataset/models/:
    --checkpoint ../g3a-bench/dataset/models/G3A-Stack-Low-Dk1_uni/params_2000000.pkl.zip

Flags:
    --save_video out/eval.mp4   saves an mp4 (requires imageio[ffmpeg])
"""

from __future__ import annotations

import os

# JAX defaults to preallocating ~75% of GPU memory as one arena on first use
# (agent.create() below). mjlab/Warp shares the same GPU and CUDA context; if
# JAX grabs its arena after Warp has already captured a reset/step CUDA graph,
# Warp can be left with nothing for that graph's *launch* (not its capture),
# failing with "Warp CUDA error 2: out of memory" on a later env.reset()/step()
# even though the first reset succeeded. Must be set before `import jax`.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import pickle
import zipfile

import jax  # pyright: ignore[reportMissingImports]
import numpy as np


def _load_checkpoint(path: str) -> dict:
    """Load a .pkl or .pkl.zip checkpoint produced by QC's save_agent."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as f:
                data = f.read()
        return pickle.loads(data)
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a QC-FQL checkpoint in G3A.")
    parser.add_argument(
        "--checkpoint", required=True, help=".pkl or .pkl.zip checkpoint path"
    )
    parser.add_argument("--task", default="G3A-Stack-Low-Dk1_uni", help="G3A task id")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--horizon_length", type=int, default=5, help="must match training"
    )
    parser.add_argument(
        "--alpha", type=float, default=300.0, help="must match training"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save_video",
        default=None,
        help="save mp4 to this path (requires imageio[ffmpeg])",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    render_mode = "rgb_array" if args.save_video else None

    # --- imports from QC (must be on PYTHONPATH or cwd) ---
    import flax  # pyright: ignore[reportMissingImports]
    from agents.acfql import (  # pyright: ignore[reportMissingImports]
        ACFQLAgent,
        get_config,
    )

    # --- G3A env ---
    import g3a_bench  # noqa: F401 — registers tasks
    from g3a_bench.qc import G3AMjlabEnv

    env = G3AMjlabEnv(args.task, device=args.device, render_mode=render_mode)
    obs, _ = env.reset()
    action_dim = env.action_space.shape[0]
    dummy_action = np.zeros(action_dim, dtype=np.float32)

    # --- build agent skeleton with same config as training ---
    config = get_config()
    config.alpha = args.alpha
    config.horizon_length = args.horizon_length
    agent = ACFQLAgent.create(args.seed, obs, dummy_action, config)

    # --- restore checkpoint ---
    load_dict = _load_checkpoint(args.checkpoint)
    agent = flax.serialization.from_state_dict(agent, load_dict["agent"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- eval loop ---
    rng = jax.random.PRNGKey(args.seed)
    successes: list[float] = []
    returns: list[float] = []
    all_frames: list[np.ndarray] = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_success = 0.0
        action_queue: list[np.ndarray] = []
        ep_frames: list[np.ndarray] = []

        while not done:
            if not action_queue:
                rng, key = jax.random.split(rng)
                action_chunk = agent.sample_actions(observations=obs, rng=key)
                for a in np.array(action_chunk).reshape(-1, action_dim):
                    action_queue.append(a)

            act = action_queue.pop(0)
            obs, reward, terminated, truncated, info = env.step(np.clip(act, -1.0, 1.0))
            done = terminated or truncated
            ep_return += reward

            if "success" in info:
                ep_success = float(info["success"])

            if render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    ep_frames.append(frame)

        successes.append(ep_success)
        returns.append(ep_return)
        all_frames.extend(ep_frames)
        print(f"Episode {ep + 1:2d}: return={ep_return:7.3f}  success={ep_success:.1f}")

    print(f"\nResults over {args.episodes} episodes:")
    print(f"  Success rate : {np.mean(successes) * 100:.1f}%")
    print(f"  Mean return  : {np.mean(returns):.3f}")

    if args.save_video and all_frames:
        import imageio

        imageio.mimwrite(args.save_video, all_frames, fps=30)  # pyright: ignore[reportCallIssue, reportArgumentType]
        print(f"Saved video: {args.save_video}")

    env.close()


if __name__ == "__main__":
    main()
