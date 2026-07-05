import numpy as np
import torch
from agent.config import QCFQLConfig
from agent.qcfql_agent import QCFQLAgent
from tests.conftest import load_jax_mlp_weights, load_jax_value_weights

_GOLD = "g3a_bench/scripts/qc_fql/tests/gold_data/agent_loss.npz"

_BATCH_KEYS = [
    "observations",
    "actions",
    "rewards",
    "masks",
    "valid",
    "next_observations",
]


def _build_agent(gold):
    config = QCFQLConfig(
        horizon_length=int(gold["horizon_length"]),
        action_chunking=True,
        actor_type="distill-ddpg",
        actor_hidden_dims=tuple(gold["actor_hidden_dims"].tolist()),
        value_hidden_dims=tuple(gold["value_hidden_dims"].tolist()),
    )
    obs_dim, action_dim = int(gold["obs_dim"]), int(gold["action_dim"])
    ex_obs = np.zeros(obs_dim, dtype=np.float32)
    ex_act = np.zeros(action_dim, dtype=np.float32)
    agent = QCFQLAgent.create(config, ex_obs, ex_act)

    load_jax_mlp_weights(
        agent.actor_bc_flow.mlp, gold, prefix="modules_actor_bc_flow/mlp/"
    )
    load_jax_mlp_weights(
        agent.actor_onestep_flow.mlp, gold, prefix="modules_actor_onestep_flow/mlp/"
    )
    load_jax_value_weights(agent.critic, gold, prefix="modules_critic/value_net/")
    load_jax_value_weights(
        agent.critic_target, gold, prefix="modules_target_critic/value_net/"
    )
    return agent


def _build_batch(gold):
    return {k: torch.from_numpy(gold[f"batch_{k}"]) for k in _BATCH_KEYS}


def test_critic_loss_matches_qc():
    gold = np.load(_GOLD)
    agent = _build_agent(gold)
    batch = _build_batch(gold)

    next_action_noise = torch.from_numpy(gold["next_action_noise"])
    loss, info = agent.critic_loss(batch, next_action_noise=next_action_noise)

    assert abs(loss.item() - float(gold["critic_loss"])) < 1e-4, (
        f"critic_loss mismatch: {loss.item()} vs {float(gold['critic_loss'])}"
    )
    assert abs(info["q_mean"] - float(gold["q_mean"])) < 1e-4
    assert abs(info["q_max"] - float(gold["q_max"])) < 1e-4
    assert abs(info["q_min"] - float(gold["q_min"])) < 1e-4


def test_actor_loss_matches_qc():
    gold = np.load(_GOLD)
    agent = _build_agent(gold)
    batch = _build_batch(gold)

    x0 = torch.from_numpy(gold["x_0"])
    t = torch.from_numpy(gold["t"])
    distill_noise = torch.from_numpy(gold["distill_noise"])
    loss, info = agent.actor_loss(batch, x0=x0, t=t, distill_noise=distill_noise)

    assert abs(loss.item() - float(gold["actor_loss"])) < 1e-4, (
        f"actor_loss mismatch: {loss.item()} vs {float(gold['actor_loss'])}"
    )
    assert abs(info["bc_flow_loss"] - float(gold["bc_flow_loss"])) < 1e-4
    assert abs(info["distill_loss"] - float(gold["distill_loss"])) < 1e-4


def test_critic_only_gets_gradient_from_its_own_loss():
    """The freeze/unfreeze around actor_loss's critic call must leave the
    critic's own weights with zero gradient contribution from q_loss --
    only actor_onestep_flow should be trained by it."""
    gold = np.load(_GOLD)
    agent = _build_agent(gold)
    batch = _build_batch(gold)

    distill_noise = torch.from_numpy(gold["distill_noise"])
    actor_loss, _ = agent.actor_loss(batch, distill_noise=distill_noise)

    agent.zero_grad()
    actor_loss.backward()

    for p in agent.critic.parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            "actor_loss leaked gradient into the critic's own parameters"
        )
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in agent.actor_onestep_flow.parameters()
    ), "actor_loss produced no gradient into actor_onestep_flow"


def test_update_runs_and_changes_params():
    """Not gold-compared (no reference `update()` trace was captured) --
    smoke test that a single combined step runs end-to-end and actually
    moves the critic target towards the critic (Polyak update)."""
    gold = np.load(_GOLD)
    agent = _build_agent(gold)
    batch = _build_batch(gold)

    target_before = [p.clone() for p in agent.critic_target.parameters()]
    info = agent.update(batch)

    assert set(info.keys()) >= {
        "critic_loss",
        "actor_loss",
        "bc_flow_loss",
        "distill_loss",
    }
    assert all(np.isfinite(v) for v in info.values())
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(
            target_before, agent.critic_target.parameters(), strict=True
        )
    ), "critic_target should move after a Polyak update"


def test_batch_update_runs_utd_ratio_updates():
    gold = np.load(_GOLD)
    agent = _build_agent(gold)
    batch = _build_batch(gold)

    utd_ratio = 2
    stacked_batch = {k: torch.stack([v, v], dim=0) for k, v in batch.items()}
    assert stacked_batch["observations"].shape[0] == utd_ratio

    info = agent.batch_update(stacked_batch)
    assert set(info.keys()) >= {"critic_loss", "actor_loss"}
    assert all(np.isfinite(v) for v in info.values())


def test_sample_actions_distill_ddpg_shape_and_bounds():
    gold = np.load(_GOLD)
    agent = _build_agent(gold)
    batch = _build_batch(gold)

    actions = agent.sample_actions(batch["observations"])
    full_action_dim = int(gold["action_dim"]) * int(gold["horizon_length"])
    assert actions.shape == (batch["observations"].shape[0], full_action_dim)
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)


def test_sample_actions_best_of_n_shape_and_bounds():
    gold = np.load(_GOLD)
    agent = _build_agent(gold)
    agent.config.actor_type = "best-of-n"
    agent.config.actor_num_samples = 4
    batch = _build_batch(gold)

    actions = agent.sample_actions(batch["observations"])
    full_action_dim = int(gold["action_dim"]) * int(gold["horizon_length"])
    assert actions.shape == (batch["observations"].shape[0], full_action_dim)
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)


def test_compute_flow_actions_runs_without_grad():
    gold = np.load(_GOLD)
    agent = _build_agent(gold)
    batch = _build_batch(gold)

    full_action_dim = int(gold["action_dim"]) * int(gold["horizon_length"])
    noises = torch.randn(batch["observations"].shape[0], full_action_dim)
    actions = agent.compute_flow_actions(batch["observations"], noises)

    assert actions.shape == noises.shape
    assert not actions.requires_grad
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)
