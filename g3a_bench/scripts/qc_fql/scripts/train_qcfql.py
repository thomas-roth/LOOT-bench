"""QC-FQL training script (offline pretrain, then online fine-tune).

Port of the ACFQLAgent-specific parts of `qc/main.py` (NOT `main_online.py` --
confirmed against the qc repo's own README and `config_flags.DEFINE_config_file`
defaults that QC-FQL uses `main.py`; `main_online.py` defaults to a different
agent entirely, `ACRLPDAgent`). See the README's step 10 section for the
full copied/dropped/added breakdown.

The online phase needs a gym-style env (`obs, info = env.reset()`; `next_obs,
reward, terminated, truncated, info = env.step(action)`) -- mjlab integration
is step 11, not this one. `train_online()` is fully implemented and tested
against a synthetic env; `main()` runs the offline phase for real (needs no
env) and raises clearly if online training is requested without one.
"""

import argparse

import numpy as np
import torch
from agent.config import QCFQLConfig
from agent.qcfql_agent import QCFQLAgent
from data.dataset import Dataset
from data.replay_buffer import ReplayBuffer
from scripts.evaluate import evaluate

_DATASET_KEYS = [
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "terminals",
    "masks",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline-steps", type=int, default=1_000_000)
    parser.add_argument("--online-steps", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=2_000_000)
    parser.add_argument("--log-interval", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--start-training", type=int, default=5000)
    parser.add_argument("--utd-ratio", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--horizon-length", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--actor-type", type=str, default="distill-ddpg")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    return parser


def load_dataset(dataset_path: str) -> Dataset:
    raw = np.load(dataset_path)
    return Dataset.create(**{k: raw[k] for k in _DATASET_KEYS})


def _batch_to_torch(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).float() for k, v in batch.items()}


def train_offline(
    agent: QCFQLAgent,
    dataset: Dataset,
    args: argparse.Namespace,
    eval_env=None,
    action_dim: int | None = None,
) -> None:
    for step in range(1, args.offline_steps + 1):
        batch = dataset.sample_sequence(
            args.batch_size, args.horizon_length, args.discount
        )
        info = agent.update(_batch_to_torch(batch))

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
    observation, _info = env.reset()
    action_queue: list[np.ndarray] = []

    for step in range(1, args.online_steps + 1):
        if len(action_queue) == 0:
            obs_t = torch.from_numpy(np.asarray(observation)).float().unsqueeze(0)
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
            # Bootstrap disabled only on true termination, not truncation --
            # matches `qc/main.py` exactly (`masks=1.0-terminated`, not `1.0-done`).
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
            batch = replay_buffer.sample_sequence(
                args.batch_size * args.utd_ratio, args.horizon_length, args.discount
            )
            batch = {
                k: torch.from_numpy(v)
                .float()
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

    train_offline(agent, dataset, args)

    if args.checkpoint_path:
        torch.save(agent.state_dict(), args.checkpoint_path)
        print(f"Saved checkpoint to {args.checkpoint_path}")

    if args.online_steps > 0:
        raise NotImplementedError(
            "Online fine-tuning needs a gym-style mjlab env, which is step 11's "
            "job, not this script's -- train_online() is implemented and "
            "tested against a synthetic env, ready to use once that lands. "
            "Call train_online() directly with a real env instead of going "
            "through main()."
        )


if __name__ == "__main__":
    main()
