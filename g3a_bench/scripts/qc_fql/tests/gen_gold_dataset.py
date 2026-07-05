# pyright: reportMissingImports=false
"""Dump one reference sample_sequence batch from the REAL qc Dataset for test_dataset.

Determinism trick: the reference's `sample_sequence` is pure NumPy whose only
randomness is one `np.random.randint` draw at the top -- so seeding the
global NumPy RNG identically right before the call makes the reference and
the port pick the same window indices, and everything downstream (n-step
returns, masks, valid flags) must then match exactly. The input arrays
include one mid-buffer terminal so the episode-boundary latching logic is
actually exercised, not just the happy path.

Run from the repo root:
    uv run python g3a_bench/scripts/qc_fql/tests/gen_gold_dataset.py
"""

import os
import sys

import numpy as np

_QC_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "qc"
    )
)
if _QC_DIR not in sys.path:
    sys.path.insert(0, _QC_DIR)

from utils.datasets import Dataset  # noqa: E402

BATCH_SIZE = 8
SEQUENCE_LENGTH = 5
DISCOUNT = 0.99
SAMPLE_SEED = 0


def main():
    rng = np.random.default_rng(42)
    n, obs_dim, action_dim = 100, 16, 7

    observations = rng.standard_normal((n, obs_dim)).astype(np.float32)
    next_observations = rng.standard_normal((n, obs_dim)).astype(np.float32)
    actions = rng.standard_normal((n, action_dim)).astype(np.float32)
    rewards = rng.random(n).astype(np.float32)
    terminals = np.zeros(n, dtype=np.float32)
    terminals[49] = 1.0  # one terminal mid-buffer to exercise valid/terminal logic
    masks = (1.0 - terminals).astype(np.float32)

    inputs = {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "masks": masks,
    }

    ds = Dataset.create(**{k: v.copy() for k, v in inputs.items()})

    np.random.seed(SAMPLE_SEED)
    result = ds.sample_sequence(BATCH_SIZE, SEQUENCE_LENGTH, DISCOUNT)

    out_dir = os.path.join(os.path.dirname(__file__), "gold_data")
    os.makedirs(out_dir, exist_ok=True)

    saved = {f"in_{k}": v for k, v in inputs.items()}
    saved.update({f"out_{k}": np.asarray(v) for k, v in result.items()})
    np.savez(os.path.join(out_dir, "dataset_sample_sequence.npz"), **saved)

    print(f"Saved gold data to {out_dir}/dataset_sample_sequence.npz")
    for k, v in result.items():
        print(f"  out_{k}: {np.asarray(v).shape}")


if __name__ == "__main__":
    main()
