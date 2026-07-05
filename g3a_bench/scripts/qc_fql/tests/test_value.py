"""Gold tests: torch Value ensemble == real qc Value, both structural paths.

"ensemble" covers the vmap path (num_ensembles=2, layer_norm=True -- the
critic's real configuration); "single" covers the reference's no-vmap
num_ensembles=1 fork with its unstacked (B,) output.
"""

import numpy as np
import torch

from g3a_bench.scripts.qc_fql.networks.value import Value
from g3a_bench.scripts.qc_fql.tests.conftest import load_jax_value_weights


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
