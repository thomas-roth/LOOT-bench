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
sibling to `g3a-play`/`g3a-play-qcfql`. The CLI (`main()`) runs the offline
phase and optionally the online phase (`--online-steps N --task TASK_ID`).
Requires `--task` for online fine-tuning or eval; raises ValueError if
the task id is missing when needed.

Dropped from the reference (all specific to its d4rl/ogbench/robomimic
benchmark suite, none applicable here): dataset-proportion subsampling,
robomimic/antmaze reward shifts, sparse-reward rewriting, huge-dataset
rotation, wandb/CSV logging (plain prints instead), and the multi-agent
registry (this script trains exactly one agent).
"""

import argparse
import json
import os
import time
from dataclasses import fields

import numpy as np
import torch

from g3a_bench.scripts.qc_fql.agent.chunked_policy import ChunkedPolicy
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
            "[DEPRECATED: superseded by the upstream-qc integration, see "
            "g3a_bench/scripts/qc_fql/README.md and "
            "docs/QC-FQL-G3A-Backend.md; use qc/main.py instead.] "
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
        help="mjlab task id for online fine-tuning (e.g. G3A-Stack-Low-Dk1_uni)",
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
        "--save-interval",
        type=int,
        default=1,
        help="number of evenly-spaced checkpoints to save during each phase (0 disables)",
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
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="enable wandb logging (entity: uesma-karlsruhe-institute-of-technology, project: G3a)",
    )
    return parser


def load_dataset(dataset_path: str) -> Dataset:
    raw = np.load(dataset_path)
    return Dataset.create(**{k: raw[k] for k in _DATASET_KEYS})


def save_checkpoint(agent: QCFQLAgent, config: QCFQLConfig, path: str) -> None:
    """Save weights + a config sidecar so play can rebuild the exact architecture."""
    torch.save(agent.state_dict(), path)
    # ob_dims/action_dim are re-inferred from data by create() → only init fields.
    init = {f.name: getattr(config, f.name) for f in fields(config) if f.init}
    with open(path + ".json", "w") as f:
        json.dump(init, f, indent=2)


def load_checkpoint_config(path: str) -> QCFQLConfig | None:
    """Reconstruct the training config from the sidecar; None if it is absent."""
    sidecar = path + ".json"
    if not os.path.exists(sidecar):
        return None
    with open(sidecar) as f:
        raw = json.load(f)
    init_names = {f.name for f in fields(QCFQLConfig) if f.init}
    # JSON arrays → tuples for the tuple-typed fields (actor_hidden_dims, ...).
    kw = {
        k: tuple(v) if isinstance(v, list) else v
        for k, v in raw.items()
        if k in init_names
    }
    return QCFQLConfig(**kw)  # pyright: ignore[reportArgumentType]  (json → typed)


def _batch_to_torch(
    batch: dict[str, np.ndarray], device: str = "cpu"
) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).float().to(device) for k, v in batch.items()}


def train_offline(
    agent: QCFQLAgent,
    dataset: Dataset,
    args: argparse.Namespace,
    config: QCFQLConfig,
    eval_env=None,
    action_dim: int | None = None,
    save_dir: str = "",
) -> str | None:
    """Phase 1: gradient steps on demo chunk windows; optional periodic eval.

    Returns the path to the best checkpoint (highest eval return), or the
    last checkpoint if no eval was performed.
    """
    save_interval = getattr(args, "save_interval", 3)
    best_return = -float("inf")
    best_ckpt: str | None = None

    for step in range(1, args.offline_steps + 1):
        batch = dataset.sample_sequence(
            args.batch_size, args.horizon_length, args.discount
        )
        info = agent.update(_batch_to_torch(batch, getattr(args, "device", "cpu")))

        if step % args.log_interval == 0:
            print(
                f"[offline {step}] " + " ".join(f"{k}={v:.4f}" for k, v in info.items())
            )
            if getattr(args, "wandb", False):
                import wandb

                wandb.log({"offline/" + k: v for k, v in info.items()}, step=step)

        # --- periodic checkpoint saving ---
        if save_dir and save_interval > 0:
            save_every = max(args.offline_steps // save_interval, 1)
            if step % save_every == 0 or step == args.offline_steps:
                ckpt_idx = min(step // save_every, save_interval) - 1
                if step == args.offline_steps:
                    ckpt_idx = save_interval - 1
                ckpt_path = os.path.join(save_dir, f"offline_{ckpt_idx}.pt")
                save_checkpoint(agent, config, ckpt_path)
                print(f"[checkpoint] Saved offline_{ckpt_idx}.pt at step {step}")

        # --- eval ---
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
            if getattr(args, "wandb", False):
                import wandb

                wandb.log(
                    {"offline_eval/" + k: v for k, v in eval_info.items()}, step=step
                )
            if eval_info["return"] > best_return:
                best_return = eval_info["return"]
                best_ckpt = os.path.join(save_dir, "offline_best.pt")
                save_checkpoint(agent, config, best_ckpt)
                print(f"[checkpoint] New best offline (return={best_return:.2f})")

    # If no eval happened, best is the last regular checkpoint
    if best_ckpt is None and save_dir and save_interval > 0:
        best_ckpt = os.path.join(save_dir, f"offline_{save_interval - 1}.pt")

    return best_ckpt


def train_online(
    agent: QCFQLAgent,
    replay_buffer: ReplayBuffer,
    env,
    args: argparse.Namespace,
    config: QCFQLConfig,
    action_dim: int,
    eval_env=None,
    save_dir: str = "",
) -> str | None:
    """Phase 2: interleave env stepping and replay-buffer updates.

    Env contract is the same gym-style one as `evaluate` (see
    `scripts/mjlab_env.py` for the real adapter). Chunks are executed to
    completion, and the queue is dropped on episode end -- a fresh episode
    must not start mid-chunk.

    Returns the path to the best checkpoint (highest eval return), or the
    last checkpoint if no eval was performed.
    """
    observation, _info = env.reset()
    policy = ChunkedPolicy(agent, action_dim, args.device)

    save_interval = getattr(args, "save_interval", 3)
    best_return = -float("inf")
    best_ckpt: str | None = None

    for step in range(1, args.online_steps + 1):
        action = policy.next_action(observation)

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
            policy.reset()
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
                if getattr(args, "wandb", False):
                    import wandb

                    wandb.log({"online/" + k: v for k, v in info.items()}, step=step)

        # --- periodic checkpoint saving ---
        if save_dir and save_interval > 0:
            save_every = max(args.online_steps // save_interval, 1)
            if step % save_every == 0 or step == args.online_steps:
                ckpt_idx = min(step // save_every, save_interval) - 1
                if step == args.online_steps:
                    ckpt_idx = save_interval - 1
                ckpt_path = os.path.join(save_dir, f"online_{ckpt_idx}.pt")
                save_checkpoint(agent, config, ckpt_path)
                print(f"[checkpoint] Saved online_{ckpt_idx}.pt at step {step}")

        # --- eval ---
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
            if getattr(args, "wandb", False):
                import wandb

                wandb.log(
                    {"online_eval/" + k: v for k, v in eval_info.items()}, step=step
                )
            if eval_info["return"] > best_return:
                best_return = eval_info["return"]
                best_ckpt = os.path.join(save_dir, "online_best.pt")
                save_checkpoint(agent, config, best_ckpt)
                print(f"[checkpoint] New best online (return={best_return:.2f})")

    # If no eval happened, best is the last regular checkpoint
    if best_ckpt is None and save_dir and save_interval > 0:
        best_ckpt = os.path.join(save_dir, f"online_{save_interval - 1}.pt")

    return best_ckpt


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Create checkpoint directory
    save_dir = os.path.join("out", "agents", timestamp)
    if args.save_interval > 0:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Checkpoints will be saved to {save_dir}/")
    else:
        save_dir = ""

    # wandb
    if args.wandb:
        import wandb

        wandb.init(
            entity="uesma-karlsruhe-institute-of-technology",
            project="G3a",
            name=timestamp,
            config=vars(args),
        )

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
        best_ckpt = train_offline(
            agent,
            dataset,
            args,
            config,
            eval_env=eval_env,
            action_dim=action_dim,
            save_dir=save_dir,
        )
        if best_ckpt:
            print(f"Loading best offline checkpoint: {best_ckpt}")
            agent.load_state_dict(torch.load(best_ckpt, map_location=args.device))

    if args.checkpoint_path and not args.resume:
        save_checkpoint(agent, config, args.checkpoint_path)
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
        best_ckpt = train_online(
            agent,
            replay_buffer,
            env,
            args,
            config,
            action_dim=action_dim,
            eval_env=eval_env,
            save_dir=save_dir,
        )
        env.close()

        if best_ckpt:
            print(f"Loading best online checkpoint: {best_ckpt}")
            agent.load_state_dict(torch.load(best_ckpt, map_location=args.device))

        if args.checkpoint_path:
            save_checkpoint(agent, config, args.checkpoint_path)
            print(f"Saved online checkpoint to {args.checkpoint_path}")

    if eval_env is not None:
        eval_env.close()

    if args.wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
