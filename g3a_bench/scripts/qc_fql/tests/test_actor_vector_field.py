import numpy as np
import torch
from networks.actor_vector_field import ActorVectorField
from tests.conftest import load_jax_mlp_weights


def _load_case(name):
    return np.load(
        f"g3a_bench/scripts/qc_fql/tests/gold_data/actor_vector_field_{name}.npz"
    )


def _build_and_load(gold):
    avf = ActorVectorField(
        obs_dim=int(gold["obs_dim"]),
        action_dim=int(gold["action_dim"]),
        hidden_dims=tuple(gold["hidden_dims"].tolist()),
        use_time=bool(gold["use_time"]),
        use_fourier_features=bool(gold["use_fourier_features"]),
        fourier_feature_dim=int(gold["fourier_feature_dim"]),
    )
    load_jax_mlp_weights(avf.mlp, gold, prefix="mlp/")
    return avf


def _run_and_compare(name):
    gold = _load_case(name)
    avf = _build_and_load(gold)

    obs = torch.from_numpy(gold["input_observations"])
    act = torch.from_numpy(gold["input_actions"])
    times = torch.from_numpy(gold["input_times"]) if "input_times" in gold else None

    out = avf(obs, act, times)
    expected = torch.from_numpy(gold["output"])

    max_diff = (out - expected).abs().max().item()
    assert torch.allclose(out, expected, atol=1e-5), (
        f"ActorVectorField ({name}) output mismatch, max diff: {max_diff}"
    )


def test_actor_vector_field_plain_matches_jax():
    _run_and_compare("plain")


def test_actor_vector_field_fourier_matches_jax():
    _run_and_compare("fourier")


def test_actor_vector_field_no_time_matches_jax():
    _run_and_compare("no_time")
