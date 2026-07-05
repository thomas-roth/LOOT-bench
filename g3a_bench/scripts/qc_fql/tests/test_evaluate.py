import math

import numpy as np
from agent.config import QCFQLConfig
from agent.qcfql_agent import QCFQLAgent
from scripts.evaluate import evaluate

OBS_DIM = 4
ACTION_DIM = 2
HORIZON_LENGTH = 3
EPISODE_LENGTH = 10


class _FakeEnv:
    """Minimal gym-style env: fixed-length episodes, reward=1 per step,
    reports `success` in the terminal info dict."""

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
        truncated = False
        info = {"success": float(terminated)}
        return obs, 1.0, terminated, truncated, info


def _build_agent():
    config = QCFQLConfig(
        horizon_length=HORIZON_LENGTH,
        action_chunking=True,
        actor_type="distill-ddpg",
        actor_hidden_dims=(8, 8),
        value_hidden_dims=(8, 8),
    )
    ex_obs = np.zeros(OBS_DIM, dtype=np.float32)
    ex_act = np.zeros(ACTION_DIM, dtype=np.float32)
    return QCFQLAgent.create(config, ex_obs, ex_act)


def test_evaluate_aggregates_return_length_and_info():
    agent = _build_agent()
    env = _FakeEnv(OBS_DIM, EPISODE_LENGTH)

    stats = evaluate(agent, env, action_dim=ACTION_DIM, num_eval_episodes=4)

    assert stats["length"] == EPISODE_LENGTH
    assert stats["return"] == EPISODE_LENGTH * 1.0
    assert stats["success"] == 1.0


def test_evaluate_samples_actions_once_per_horizon_length():
    agent = _build_agent()
    env = _FakeEnv(OBS_DIM, EPISODE_LENGTH)

    call_count = 0
    real_sample_actions = agent.sample_actions

    def counting_sample_actions(observations, noise=None):
        nonlocal call_count
        call_count += 1
        return real_sample_actions(observations, noise=noise)

    agent.sample_actions = counting_sample_actions

    num_episodes = 3
    evaluate(agent, env, action_dim=ACTION_DIM, num_eval_episodes=num_episodes)

    expected_calls_per_episode = math.ceil(EPISODE_LENGTH / HORIZON_LENGTH)
    assert call_count == expected_calls_per_episode * num_episodes
