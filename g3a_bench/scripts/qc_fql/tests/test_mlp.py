import numpy as np
import torch
from networks.mlp import MLP
from tests.conftest import load_jax_mlp_weights


def test_mlp_matches_jax():
    gold = np.load("g3a_bench/scripts/qc_fql/tests/gold_data/mlp.npz")

    mlp = MLP(in_features=16, hidden_dims=[64, 32])
    load_jax_mlp_weights(mlp, gold, prefix="")

    x = torch.from_numpy(gold["input"])
    out = mlp(x)
    expected = torch.from_numpy(gold["output"])

    max_diff = (out - expected).abs().max().item()
    assert torch.allclose(out, expected, atol=1e-5), (
        f"MLP output mismatch, max diff: {max_diff}"
    )
