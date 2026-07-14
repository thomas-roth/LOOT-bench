"""Gold test: torch FourierFeatures output == real qc FourierFeatures output.

Parameter-free module (non-learnable mode), so this is a pure input/output
comparison -- no weight loading. Gold data from gen_gold_fourier_features.py.
"""

import numpy as np
import torch

from g3a_bench.scripts.qc_fql.networks.fourier_features import FourierFeatures


def test_fourier_features_matches_jax():
    gold = np.load("g3a_bench/scripts/qc_fql/tests/gold_data/fourier_features.npz")

    ff = FourierFeatures(output_size=int(gold["output_size"]))

    x = torch.from_numpy(gold["input"])
    out = ff(x)
    expected = torch.from_numpy(gold["output"])

    max_diff = (out - expected).abs().max().item()
    assert torch.allclose(out, expected, atol=1e-5), (
        f"FourierFeatures output mismatch, max diff: {max_diff}"
    )
