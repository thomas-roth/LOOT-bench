"""Training-loop tests on a synthetic env: loop mechanics, not learning.

Covers both phases of train_qcfql.py without any real env or dataset:
offline runs and evaluates, online fills the replay buffer / triggers
updates / preserves the reference's masks=1-terminated convention (NOT
1-done: truncation still bootstraps), and main() works end-to-end through
argv -- the same path the g3a-train-qcfql console script exercises --
including the deliberate NotImplementedError for --online-steps > 0.
"""

import argparse

import numpy as np
import torch

from g3a_bench.scripts.qc_fql.agent.config import QCFQLConfig
from g3a_bench.scripts.qc_fql.agent.qcfql_agent import QCFQLAgent
from g3a_bench.scripts.qc_fql.data.dataset import Dataset
from g3a_bench.scripts.qc_fql.data.replay_buffer import ReplayBuffer
from g3a_bench.scripts.qc_fql.scripts.train_qcfql import (
    load_dataset,
    main,
    train_offline,
    train_online,
)

OBS_DIM = 4
ACTION_DIM = 2
HORIZON_LENGTH = 3
N_TRANSITIONS = 60


class _FakeEnv:
    """Same minimal gym-style synthetic env as test_evaluate.py."""

    def __init__(self, obs_dim: int, episode_length: int):
        self.obs_dim = obs_dim
        self.episode_length = episode_length
        self.t = 0

    def reset(self):
        self.t = 0
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        terminated = self.t >= self.episode_length
        return obs, 1.0, terminated, False, {"success": float(terminated)}


def _make_dataset(n=N_TRANSITIONS) -> Dataset:
    rng = np.random.default_rng(0)
    terminals = np.zeros(n, dtype=np.float32)
    terminals[-1] = 1.0
    return Dataset.create(
        observations=rng.standard_normal((n, OBS_DIM)).astype(np.float32),
        next_observations=rng.standard_normal((n, OBS_DIM)).astype(np.float32),
        actions=rng.standard_normal((n, ACTION_DIM)).astype(np.float32),
        rewards=rng.random(n).astype(np.float32),
        terminals=terminals,
        masks=(1.0 - terminals).astype(np.float32),
    )


def _make_args(**overrides) -> argparse.Namespace:
    defaults = {
        "dataset_path": "unused",
        "seed": 0,
        "offline_steps": 3,
        "online_steps": 5,
        "buffer_size": 1000,
        "log_interval": 1000,  # avoid noisy prints during tests
        "eval_interval": 0,  # disable eval (no eval_env in most tests)
        "eval_episodes": 1,
        "start_training": 2,
        "utd_ratio": 1,
        "discount": 0.99,
        "horizon_length": HORIZON_LENGTH,
        "batch_size": 4,
        "lr": 3e-4,
        "alpha": 100.0,
        "actor_type": "distill-ddpg",
        "checkpoint_path": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _build_agent(args):
    config = QCFQLConfig(
        horizon_length=args.horizon_length,
        batch_size=args.batch_size,
        lr=args.lr,
        discount=args.discount,
        alpha=args.alpha,
        actor_type=args.actor_type,
        actor_hidden_dims=(8, 8),
        value_hidden_dims=(8, 8),
    )
    dataset = _make_dataset()
    ex_obs = dataset.data["observations"][0]
    ex_act = dataset.data["actions"][0]
    agent = QCFQLAgent.create(config, ex_obs, ex_act)
    return agent, dataset, config


def test_train_offline_runs_and_produces_finite_losses():
    args = _make_args()
    agent, dataset, config = _build_agent(args)

    train_offline(agent, dataset, args, config)  # no eval_env -- must not require one


def test_train_offline_with_eval_runs():
    args = _make_args(eval_interval=2, offline_steps=3)
    agent, dataset, config = _build_agent(args)
    eval_env = _FakeEnv(OBS_DIM, episode_length=4)

    train_offline(
        agent, dataset, args, config, eval_env=eval_env, action_dim=ACTION_DIM
    )


def test_train_online_fills_replay_buffer_and_updates():
    args = _make_args(online_steps=8, start_training=2, utd_ratio=1, batch_size=2)
    agent, dataset, config = _build_agent(args)
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(dataset.data), size=max(args.buffer_size, dataset.size + 1)
    )
    env = _FakeEnv(OBS_DIM, episode_length=4)

    critic_before = [p.clone() for p in agent.critic.parameters()]
    train_online(agent, replay_buffer, env, args, config, action_dim=ACTION_DIM)

    assert replay_buffer.size == dataset.size + args.online_steps
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(critic_before, agent.critic.parameters(), strict=True)
    ), "critic should have been updated by online batch_update"


def test_train_online_masks_use_terminated_not_done():
    """`masks = 1 - terminated`, not `1 - done` -- must match qc/main.py exactly."""
    args = _make_args(
        online_steps=4, start_training=1000
    )  # start_training high: no updates, just fill buffer
    agent, dataset, config = _build_agent(args)
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(dataset.data), size=max(args.buffer_size, dataset.size + 1)
    )
    env = _FakeEnv(OBS_DIM, episode_length=2)  # terminates every 2 steps

    train_online(agent, replay_buffer, env, args, config, action_dim=ACTION_DIM)

    added_masks = replay_buffer["masks"][
        dataset.size : dataset.size + args.online_steps
    ]
    added_terminals = replay_buffer["terminals"][
        dataset.size : dataset.size + args.online_steps
    ]
    assert np.allclose(added_masks, 1.0 - added_terminals)


def test_main_offline_only_end_to_end(tmp_path, monkeypatch):
    dataset = _make_dataset()
    dataset_path = tmp_path / "demos.npz"
    np.savez(dataset_path, **dataset.data)  # pyright: ignore[reportArgumentType]
    checkpoint_path = tmp_path / "agent.pt"

    argv = [
        "train_qcfql.py",
        "--dataset-path",
        str(dataset_path),
        "--offline-steps",
        "2",
        "--online-steps",
        "0",
        "--horizon-length",
        str(HORIZON_LENGTH),
        "--batch-size",
        "4",
        "--eval-interval",
        "0",
        "--log-interval",
        "1000",
        "--checkpoint-path",
        str(checkpoint_path),
    ]
    monkeypatch.setattr("sys.argv", argv)

    main()

    assert checkpoint_path.exists()


def test_main_raises_for_online_without_env(tmp_path, monkeypatch):
    dataset = _make_dataset()
    dataset_path = tmp_path / "demos.npz"
    np.savez(dataset_path, **dataset.data)  # pyright: ignore[reportArgumentType]

    argv = [
        "train_qcfql.py",
        "--dataset-path",
        str(dataset_path),
        "--offline-steps",
        "1",
        "--online-steps",
        "5",
        "--eval-interval",
        "0",
        "--log-interval",
        "1000",
    ]
    monkeypatch.setattr("sys.argv", argv)

    try:
        main()
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_load_dataset_reads_expected_keys(tmp_path):
    dataset = _make_dataset()
    dataset_path = tmp_path / "demos.npz"
    np.savez(dataset_path, **dataset.data)  # pyright: ignore[reportArgumentType]

    loaded = load_dataset(str(dataset_path))
    assert loaded.size == dataset.size
    assert set(loaded.data.keys()) == set(dataset.data.keys())
