import numpy as np
from data.replay_buffer import ReplayBuffer
from tests.gen_gold_replay_buffer import (
    BUFFER_SIZE,
    KEYS,
    N_ADD,
    SAMPLE_BATCH,
    SAMPLE_DISCOUNT,
    SAMPLE_SEED,
    SAMPLE_SEQ_LEN,
)

_GOLD = "g3a_bench/scripts/qc_fql/tests/gold_data/replay_buffer.npz"


def test_create_and_wraparound_add_matches_qc():
    gold = np.load(_GOLD)
    trans = {k: gold[f"trans_{k}"] for k in KEYS}

    example = {k: trans[k][0] for k in KEYS}
    rb = ReplayBuffer.create(example, size=BUFFER_SIZE)
    for t in range(N_ADD):
        rb.add_transition({k: trans[k][t] for k in KEYS})

    assert rb.pointer == int(gold["add_pointer"])
    assert rb.size == int(gold["add_size"])
    assert rb.max_size == int(gold["add_max_size"])
    for k in KEYS:
        assert np.array_equal(rb[k], gold[f"add_buffer_{k}"]), f"buffer mismatch: {k}"


def test_create_from_initial_dataset_matches_qc():
    gold = np.load(_GOLD)
    init = {k: gold[f"init_{k}"] for k in KEYS}

    rb = ReplayBuffer.create_from_initial_dataset(dict(init), size=BUFFER_SIZE)

    assert rb.pointer == int(gold["cfid_pointer"])
    assert rb.size == int(gold["cfid_size"])
    for k in KEYS:
        assert np.array_equal(rb[k], gold[f"cfid_buffer_{k}"]), f"buffer mismatch: {k}"


def test_inherited_sample_sequence_matches_qc():
    gold = np.load(_GOLD)
    init = {k: gold[f"init_{k}"] for k in KEYS}

    rb = ReplayBuffer.create_from_initial_dataset(dict(init), size=BUFFER_SIZE)
    np.random.seed(SAMPLE_SEED)
    result = rb.sample_sequence(SAMPLE_BATCH, SAMPLE_SEQ_LEN, SAMPLE_DISCOUNT)

    for key, actual in result.items():
        expected = gold[f"seq_{key}"]
        assert actual.shape == expected.shape, f"{key}: shape mismatch"
        assert np.allclose(actual, expected, atol=1e-6), (
            f"{key}: max diff {np.abs(actual - expected).max()}"
        )


def test_clear_resets_pointer_and_size():
    gold = np.load(_GOLD)
    init = {k: gold[f"init_{k}"] for k in KEYS}

    rb = ReplayBuffer.create_from_initial_dataset(dict(init), size=BUFFER_SIZE)
    rb.clear()
    assert rb.size == 0
    assert rb.pointer == 0
