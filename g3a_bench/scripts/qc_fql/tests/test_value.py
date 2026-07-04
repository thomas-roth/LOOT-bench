import numpy as np
import torch
from networks.value import Value
from tests.conftest import load_jax_value_weights


def _run_and_compare(name):
    gold = np.load(f"g3a_bench/scripts/qc_fql/tests/gold_data/value_{name}.npz")

    value = Value(
        in_features=int(gold["obs_dim"]) + int(gold["action_dim"]),
        hidden_dims=tuple(gold["hidden_dims"].tolist()),
        layer_norm=bool(gold["layer_norm"]),
        num_ensembles=int(gold["num_ensembles"]),
    )
    load_jax_value_weights(value, gold)

    obs = torch.from_numpy(gold["input_observations"])
    act = torch.from_numpy(gold["input_actions"])

    out = value(obs, act)
    expected = torch.from_numpy(gold["output"])

    max_diff = (out - expected).abs().max().item()
    assert torch.allclose(out, expected, atol=1e-5), (
        f"Value ({name}) output mismatch, max diff: {max_diff}"
    )


def test_value_ensemble_matches_jax():
    _run_and_compare("ensemble")


def test_value_single_matches_jax():
    _run_and_compare("single")
