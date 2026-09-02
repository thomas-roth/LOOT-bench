"""Write a synthetic dataset that the training path will accept.

**This is not a recording.** The rows are made up: random observations, random
actions, and a reward ramp that satisfies the convention in
``docs/adr/006-one-task-reward-in-minus-one-to-zero.md``. Nothing here says
anything about whether a policy can learn the task, and no number read off a
run against it means anything.

What it is for is the wiring. Steps 6 to 8 of the rebuild can be checked end to
end before a single demonstration exists: the launcher, the ``G3A-`` branch in
qc's ``make_env_and_datasets``, the dataset check, and the shapes qc's replay
buffer expects. The widths and the action stamp are read off the real
environment, so a dataset from here fails exactly where a stale recording
would.

    uv run python tests/qc/make_smoke_dataset.py --cpu \\
        --task G3A-Stack-Low-Dk1_uni --output out/qc/smoke.npz

Then, with the qc checkout in place:

    RUN_DIR=/var/tmp/$USER/qc-runs WANDB_MODE=offline G3A_ALLOW_NO_TMUX=1 \\
    ./train_qc.sh G3A-Stack-Low-Dk1_uni out/qc/smoke.npz \\
        --offline_steps 2 --online_steps 0 --eval_episodes 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gymnasium.spaces import Space

from g3a_bench.qc import G3aEnv
from g3a_bench.qc.transitions import MIN_REWARD, assemble_transitions
from g3a_bench.tasks.common.rewards import MIN_SHORTFALL


def _width(space: Space) -> int:
    """The single dimension of a flat ``Box``. The adapter emits nothing else."""
    shape = space.shape
    if shape is None or len(shape) != 1:
        raise TypeError(f"expected a flat space, got {space}")
    return int(shape[0])


def build(
    obs_dim: int,
    action_dim: int,
    episodes: int,
    length: int,
    seed: int,
    action_scale: np.ndarray,
    action_offset: np.ndarray,
    start_pose_noise: float = 0.0,
) -> dict[str, np.ndarray]:
    """Synthetic rows, shaped like a recording of ``episodes`` attempts.

    The first episode succeeds, so the dataset carries one absorbing row; the
    rest run out of time, so their unusable final row is dropped. That is the
    mix a real session has, and the two paths through
    :mod:`g3a_bench.qc.transitions` are the ones worth exercising.
    """
    rng = np.random.default_rng(seed)
    episode_index = np.repeat(np.arange(episodes), length)
    observations = rng.normal(size=(episodes * length, obs_dim)).astype(np.float32)
    actions = rng.uniform(-1.0, 1.0, size=(episodes * length, action_dim)).astype(
        np.float32
    )

    # A ramp from the floor up to just short of zero: the shaped reward of a
    # task that is making progress and has not scored.
    ramp = np.linspace(MIN_REWARD, -MIN_SHORTFALL, length, dtype=np.float32)
    rewards = np.tile(ramp, episodes)
    rewards[length - 1] = 0.0  # episode 0 reaches the goal.

    return {
        **assemble_transitions(
            episode_index=episode_index,
            observations=observations,
            actions=actions,
            rewards=rewards,
        ),
        "action_scale": np.asarray(action_scale, dtype=np.float32).reshape(-1),
        "action_offset": np.asarray(action_offset, dtype=np.float32).reshape(-1),
        "start_pose_noise": np.array(float(start_pose_noise), dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="G3A-Stack-Low-Dk1_uni")
    parser.add_argument("--output", default="out/qc/smoke.npz")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--length", type=int, default=60, help="rows per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="build the env on CPU")
    args = parser.parse_args()

    # The environment is built only to read the widths and the conventions
    # off it. A stamp copied from anywhere else would defeat the check it
    # exists to feed.
    env = G3aEnv(args.task, device="cpu" if args.cpu else "cuda")
    try:
        obs_dim = _width(env.observation_space)
        action_dim = _width(env.action_space)
        start_pose_noise = env.start_pose_noise()
        term = env.action_term()
        env_scale = term.scale
        env_offset = term.offset
        if hasattr(env_scale, "detach"):
            env_scale = env_scale.detach().cpu().numpy()  # type: ignore[union-attr]
        if hasattr(env_offset, "detach"):
            env_offset = env_offset.detach().cpu().numpy()  # type: ignore[union-attr]
        env_scale = np.asarray(env_scale, dtype=np.float32).reshape(-1)
        env_offset = np.asarray(env_offset, dtype=np.float32).reshape(-1)
    finally:
        env.close()

    data = build(
        obs_dim,
        action_dim,
        args.episodes,
        args.length,
        args.seed,
        action_scale=env_scale,
        action_offset=env_offset,
        start_pose_noise=start_pose_noise,
    )
    # Stamp the task id too, so the smoke run also exercises the task
    # identity check in check_dataset_matches_env.
    data["task"] = np.array(args.task, dtype=np.str_)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **data)  # type: ignore[arg-type]
    print(f"[INFO] wrote {output}  (synthetic, for wiring checks only)")
    print(f"       {len(data['rewards'])} transitions from {args.episodes} episodes")
    print(f"       observations {data['observations'].shape}")
    print(f"       actions      {data['actions'].shape}")
    print(f"       absorbing    {int((data['masks'] == 0.0).sum())}")


if __name__ == "__main__":
    main()
