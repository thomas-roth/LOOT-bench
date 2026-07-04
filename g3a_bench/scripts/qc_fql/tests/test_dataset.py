import numpy as np
from data.dataset import Dataset
from tests.gen_gold_dataset import BATCH_SIZE, DISCOUNT, SAMPLE_SEED, SEQUENCE_LENGTH


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
