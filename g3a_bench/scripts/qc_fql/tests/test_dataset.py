"""Gold test: port's sample_sequence == real qc Dataset's, on identical draws.

Reconstructs the dataset from the gold file's saved inputs, re-seeds the
global NumPy RNG with the generator's seed (sample_sequence's only
randomness is one randint draw, so both sides pick identical windows), and
compares every output key exactly -- shapes and values. The critical shape
contract guarded here: `observations` is (B, obs_dim) window-start
observations, while `full_observations` is the (B, T, obs_dim) sequence.
"""

import numpy as np

from g3a_bench.scripts.qc_fql.data.dataset import Dataset
from g3a_bench.scripts.qc_fql.tests.gen_gold_dataset import (
    BATCH_SIZE,
    DISCOUNT,
    SAMPLE_SEED,
    SEQUENCE_LENGTH,
)


def test_sample_sequence_matches_qc():
    gold = np.load(
        "g3a_bench/scripts/qc_fql/tests/gold_data/dataset_sample_sequence.npz"
    )

    ds = Dataset.create(
        observations=gold["in_observations"].copy(),
        next_observations=gold["in_next_observations"].copy(),
        actions=gold["in_actions"].copy(),
        rewards=gold["in_rewards"].copy(),
        terminals=gold["in_terminals"].copy(),
        masks=gold["in_masks"].copy(),
    )

    np.random.seed(SAMPLE_SEED)
    result = ds.sample_sequence(BATCH_SIZE, SEQUENCE_LENGTH, DISCOUNT)

    for key in [
        "observations",
        "full_observations",
        "actions",
        "masks",
        "rewards",
        "terminals",
        "valid",
        "next_observations",
        "next_actions",
    ]:
        expected = gold[f"out_{key}"]
        actual = result[key]
        assert actual.shape == expected.shape, (
            f"{key}: shape {actual.shape} != {expected.shape}"
        )
        assert np.allclose(actual, expected, atol=1e-6), (
            f"{key}: max diff {np.abs(actual - expected).max()}"
        )
