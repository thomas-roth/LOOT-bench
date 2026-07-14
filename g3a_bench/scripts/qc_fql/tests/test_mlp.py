"""Gold tests: torch MLP == real qc MLP, with and without LayerNorm.

The layer_norm case exists for a reason: it pins the flax-vs-torch LayerNorm
epsilon difference (1e-6 vs 1e-5), a real divergence that once escaped
detection precisely because only the plain case was tested. Its gold data
carries randomized norm params so the loader path is discriminating (both
frameworks default LayerNorm to scale=1/bias=0, which would mask a loader
that skips them).
"""

import numpy as np
import torch

from g3a_bench.scripts.qc_fql.networks.mlp import MLP
from g3a_bench.scripts.qc_fql.tests.conftest import load_jax_mlp_weights


def _run_and_compare(name):
    gold = np.load(f"g3a_bench/scripts/qc_fql/tests/gold_data/mlp_{name}.npz")

    mlp = MLP(
        in_features=int(gold["in_features"]),
        hidden_dims=tuple(gold["hidden_dims"].tolist()),
        layer_norm=bool(gold["layer_norm"]),
    )
    load_jax_mlp_weights(mlp, gold, prefix="")

    x = torch.from_numpy(gold["input"])
    out = mlp(x)
    expected = torch.from_numpy(gold["output"])

    max_diff = (out - expected).abs().max().item()
    assert torch.allclose(out, expected, atol=1e-5), (
        f"MLP ({name}) output mismatch, max diff: {max_diff}"
    )


def test_mlp_plain_matches_jax():
    _run_and_compare("plain")


def test_mlp_layer_norm_matches_jax():
    _run_and_compare("layer_norm")
