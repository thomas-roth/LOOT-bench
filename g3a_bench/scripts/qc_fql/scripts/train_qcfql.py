"""QC-FQL training entry point: offline pretraining, then online fine-tuning.

Port of the ACFQL-relevant parts of `qc/main.py` -- and specifically
`main.py`, not `main_online.py`: the reference repo wires `main_online.py`
to a different agent (ACRLPD) by default, and its own README reproduces all
QC-FQL results through `main.py`. The structure is therefore two phases:

1. Offline: sample chunk windows from the demo dataset, `agent.update` each.
2. Online: seed a ReplayBuffer with the full dataset, then alternate
   env-stepping (with the chunk queue) and `agent.batch_update` on windows
   sampled from the growing buffer.

Installed as the `g3a-train-qcfql` console script (see pyproject.toml),
sibling to `g3a-train`/`g3a-play`. The CLI (`main()`) runs the offline phase
and optionally the online phase (`--online-steps N --task TASK_ID`).
Requires `--task` for online fine-tuning or eval; raises ValueError if
the task id is missing when needed.

Dropped from the reference (all specific to its d4rl/ogbench/robomimic
benchmark suite, none applicable here): dataset-proportion subsampling,
robomimic/antmaze reward shifts, sparse-reward rewriting, huge-dataset
rotation, wandb/CSV logging (plain prints instead), and the multi-agent
registry (this script trains exactly one agent).
"""

import argparse

import numpy as np
import torch

from g3a_bench.scripts.qc_fql.agent.config import QCFQLConfig
from g3a_bench.scripts.qc_fql.agent.qcfql_agent import QCFQLAgent
from g3a_bench.scripts.qc_fql.data.dataset import Dataset
from g3a_bench.scripts.qc_fql.data.replay_buffer import ReplayBuffer
from g3a_bench.scripts.qc_fql.scripts.evaluate import evaluate

# The six arrays lerobot_to_qc.py writes for training. Its .npz also carries
# metadata keys (action_min/action_max/episode_index/...) that must NOT
# reach Dataset.create: the ReplayBuffer inherits the offline key set, and
# online transitions only produce these six.
_DATASET_KEYS = [
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "terminals",
    "masks",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="g3a-train-qcfql",
        description=(
            "Train QC-FQL (flow Q-learning with action chunking) offline on a "
            "recorded demonstration dataset (.npz from lerobot_to_qc.py). "
            "Defaults mirror the reference implementation."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="demo dataset .npz produced by live_bridge/lerobot_to_qc.py",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--offline-steps", type=int, default=1_000_000, help="offline gradient steps"
    )
    parser.add_argument(
        "--online-steps",
        type=int,
        default=0,
        help="online fine-tuning env steps (requires --task)",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=2_000_000, help="online replay capacity"
    )
    parser.add_argument("--log-interval", type=int, default=5000)
    parser.add_argument(
        "--eval-interval", type=int, default=100_000, help="0 disables eval"
    )
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument(
        "--start-training",
        type=int,
        default=5000,
        help="online env steps collected before updates begin",
    )
    parser.add_argument(
        "--utd-ratio", type=int, default=1, help="gradient updates per online env step"
    )
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument(
        "--horizon-length", type=int, default=5, help="action-chunk length H"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--alpha", type=float, default=100.0, help="BC/distillation coefficient"
    )
    parser.add_argument(
        "--actor-type",
        type=str,
        default="distill-ddpg",
        choices=["distill-ddpg", "best-of-n"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="mjlab task id for online fine-tuning (e.g. G3a-Cube-Stack3-Dk1_uni)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device for the mjlab env (cpu or cuda)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="out/agent.pt",
        help="where to torch.save the trained agent's state_dict",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "path to agent.pt checkpoint — skip offline training and resume"
            " from this policy (requires --online-steps and --task)"
        ),
    )
    return parser


def load_dataset(dataset_path: str) -> Dataset:
    raw = np.load(dataset_path)
    return Dataset.create(**{k: raw[k] for k in _DATASET_KEYS})


def _batch_to_torch(
    batch: dict[str, np.ndarray], device: str = "cpu"
) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).float().to(device) for k, v in batch.items()}


def train_offline(
    agent: QCFQLAgent,
    dataset: Dataset,
    args: argparse.Namespace,
    eval_env=None,
    action_dim: int | None = None,
) -> None:
    """Phase 1: gradient steps on demo chunk windows; optional periodic eval."""
    for step in range(1, args.offline_steps + 1):
        batch = dataset.sample_sequence(
            args.batch_size, args.horizon_length, args.discount
        )
        info = agent.update(_batch_to_torch(batch, getattr(args, "device", "cpu")))

        if step % args.log_interval == 0:
            print(
                f"[offline {step}] " + " ".join(f"{k}={v:.4f}" for k, v in info.items())
            )

        if (
            eval_env is not None
            and args.eval_interval
            and (step == args.offline_steps - 1 or step % args.eval_interval == 0)
        ):
            assert action_dim is not None, "action_dim is required when eval_env is set"
            eval_info = evaluate(
                agent,
                eval_env,
                action_dim=action_dim,
                num_eval_episodes=args.eval_episodes,
            )
            print(
                f"[eval (offline) {step}] "
                + " ".join(f"{k}={v:.4f}" for k, v in eval_info.items())
            )


def train_online(
    agent: QCFQLAgent,
    replay_buffer: ReplayBuffer,
    env,
    args: argparse.Namespace,
    action_dim: int,
    eval_env=None,
) -> None:
    """Phase 2: interleave env stepping and replay-buffer updates.

    Env contract is the same gym-style one as `evaluate` (see
    `scripts/mjlab_env.py` for the real adapter). Chunks are executed to
    completion, and the queue is dropped on episode end -- a fresh episode
    must not start mid-chunk.
    """
    observation, _info = env.reset()
    action_queue: list[np.ndarray] = []

    for step in range(1, args.online_steps + 1):
        if len(action_queue) == 0:
            obs_t = (
                torch.from_numpy(np.asarray(observation))
                .float()
                .to(args.device)
                .unsqueeze(0)
            )
            with torch.no_grad():
                action_chunk = agent.sample_actions(obs_t).squeeze(0).cpu().numpy()
            action_queue.extend(action_chunk.reshape(-1, action_dim))
        action = action_queue.pop(0)

        next_observation, reward, terminated, truncated, _info = env.step(
            np.clip(action, -1.0, 1.0)
        )
        done = terminated or truncated

        transition = {
            "observations": observation,
            "actions": action,
            "rewards": reward,
            "terminals": float(done),
            # 1 - terminated, NOT 1 - done: a truncated episode (time limit)
            # still bootstraps -- the state wasn't absorbing, the clock just
            # ran out. Matches the reference exactly.
            "masks": 1.0 - float(terminated),
            "next_observations": next_observation,
        }
        replay_buffer.add_transition(transition)

        if done:
            observation, _info = env.reset()
            action_queue = []
        else:
            observation = next_observation

        if step >= args.start_training:
            # One oversized sample reshaped to (utd_ratio, batch, ...):
            # batch_update consumes the leading axis as separate updates,
            # reproducing the reference's scan-over-minibatches.
            batch = replay_buffer.sample_sequence(
                args.batch_size * args.utd_ratio, args.horizon_length, args.discount
            )
            batch = {
                k: torch.from_numpy(v)
                .float()
                .to(args.device)
                .reshape((args.utd_ratio, args.batch_size) + v.shape[1:])
                for k, v in batch.items()
            }
            info = agent.batch_update(batch)

            if step % args.log_interval == 0:
                print(
                    f"[online {step}] "
                    + " ".join(f"{k}={v:.4f}" for k, v in info.items())
                )

        if (
            eval_env is not None
            and args.eval_interval
            and (step == args.online_steps - 1 or step % args.eval_interval == 0)
        ):
            eval_info = evaluate(
                agent,
                eval_env,
                action_dim=action_dim,
                num_eval_episodes=args.eval_episodes,
            )
            print(
                f"[eval (online) {step}] "
                + " ".join(f"{k}={v:.4f}" for k, v in eval_info.items())
            )


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = load_dataset(args.dataset_path)
    # Unbatched examples on purpose -- see QCFQLAgent.create.
    ex_obs = dataset.data["observations"][0]
    ex_act = dataset.data["actions"][0]

    config = QCFQLConfig(
        horizon_length=args.horizon_length,
        batch_size=args.batch_size,
        lr=args.lr,
        discount=args.discount,
        alpha=args.alpha,
        actor_type=args.actor_type,
    )
    agent = QCFQLAgent.create(config, ex_obs, ex_act)

    if args.resume:
        agent.load_state_dict(torch.load(args.resume, map_location=args.device))
        print(
            f"[INFO] Loaded checkpoint from {args.resume} — skipping offline training"
        )

    agent = agent.to(args.device)

    action_dim = dataset.data["actions"].shape[-1]

    if args.online_steps > 0 or args.eval_interval:
        # Online phase needs the env, and offline eval also needs it.
        if args.task is None:
            raise ValueError("--task is required for online fine-tuning or eval")
        import g3a_bench  # noqa: F401  (registers tasks)
        from g3a_bench.scripts.qc_fql.scripts.mjlab_env import MjlabSingleEnv

        eval_env = (
            MjlabSingleEnv(args.task, device=args.device, play=False)
            if args.eval_interval
            else None
        )
    else:
        eval_env = None

    if not args.resume:
        train_offline(agent, dataset, args, eval_env=eval_env, action_dim=action_dim)

    if args.checkpoint_path and not args.resume:
        torch.save(agent.state_dict(), args.checkpoint_path)
        print(f"Saved checkpoint to {args.checkpoint_path}")

    if args.online_steps > 0:
        if args.task is None:
            raise ValueError("--task is required for online fine-tuning")
        from g3a_bench.scripts.qc_fql.scripts.mjlab_env import MjlabSingleEnv

        print(f"Seeding replay buffer with {dataset.size} offline transitions...")
        replay_buffer = ReplayBuffer.create_from_initial_dataset(
            dict(dataset.data), size=max(args.buffer_size, dataset.size + 1)
        )

        env = MjlabSingleEnv(args.task, device=args.device, play=False)
        train_online(
            agent, replay_buffer, env, args, action_dim=action_dim, eval_env=eval_env
        )
        env.close()

        if args.checkpoint_path:
            torch.save(agent.state_dict(), args.checkpoint_path)
            print(f"Saved online checkpoint to {args.checkpoint_path}")

    if eval_env is not None:
        eval_env.close()


if __name__ == "__main__":
    main()
