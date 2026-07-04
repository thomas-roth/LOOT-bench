# pyright: reportMissingImports=false
"""Generate reference ReplayBuffer states from the real qc code.

Covers: `create` + wraparound `add_transition`s, `create_from_initial_dataset`,
and the inherited `sample_sequence` (seeded for determinism, same trick as the
Dataset gold test). The port replays the identical saved inputs and must match
the final buffer arrays / pointer / size and the sampled batch.

Run from the repo root with:
    uv run python g3a_bench/scripts/qc_fql/tests/gen_gold_replay_buffer.py
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

from utils.datasets import ReplayBuffer  # noqa: E402

OBS_DIM = 4
ACTION_DIM = 3
BUFFER_SIZE = 10
N_ADD = 13  # > BUFFER_SIZE so the circular wraparound is exercised
INIT_SIZE = 6
KEYS = ["observations", "next_observations", "actions", "rewards", "terminals", "masks"]

# sample_sequence params for the determinism check
SAMPLE_SEED = 0
SAMPLE_BATCH = 4
SAMPLE_SEQ_LEN = 3
SAMPLE_DISCOUNT = 0.99


def _make_transitions(rng, n):
    return {
        "observations": rng.standard_normal((n, OBS_DIM)).astype(np.float32),
        "next_observations": rng.standard_normal((n, OBS_DIM)).astype(np.float32),
        "actions": rng.standard_normal((n, ACTION_DIM)).astype(np.float32),
        "rewards": rng.random(n).astype(np.float32),
        "terminals": (rng.random(n) < 0.2).astype(np.float32),
        "masks": rng.random(n).astype(np.float32),
    }


def main():
    rng = np.random.default_rng(42)
    saved = {}

    # --- Scenario 1: create() + wraparound add_transition() ---
    trans = _make_transitions(rng, N_ADD)
    for k, v in trans.items():
        saved[f"trans_{k}"] = v

    example = {k: trans[k][0] for k in KEYS}
    rb = ReplayBuffer.create(example, size=BUFFER_SIZE)
    for t in range(N_ADD):
        rb.add_transition({k: trans[k][t] for k in KEYS})

    for k in KEYS:
        saved[f"add_buffer_{k}"] = np.asarray(rb[k])
    saved["add_pointer"] = rb.pointer
    saved["add_size"] = rb.size
    saved["add_max_size"] = rb.max_size

    # --- Scenario 2: create_from_initial_dataset() ---
    init = _make_transitions(rng, INIT_SIZE)
    for k, v in init.items():
        saved[f"init_{k}"] = v

    rb2 = ReplayBuffer.create_from_initial_dataset(dict(init), size=BUFFER_SIZE)
    for k in KEYS:
        saved[f"cfid_buffer_{k}"] = np.asarray(rb2[k])
    saved["cfid_pointer"] = rb2.pointer
    saved["cfid_size"] = rb2.size

    # --- Scenario 3: inherited sample_sequence on the preloaded buffer ---
    np.random.seed(SAMPLE_SEED)
    result = rb2.sample_sequence(SAMPLE_BATCH, SAMPLE_SEQ_LEN, SAMPLE_DISCOUNT)
    for k, v in result.items():
        saved[f"seq_{k}"] = np.asarray(v)

    out_dir = os.path.join(os.path.dirname(__file__), "gold_data")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "replay_buffer.npz"), **saved)

    print(f"Saved gold data to {out_dir}/replay_buffer.npz")
    print(
        f"  after {N_ADD} adds: pointer={rb.pointer}, size={rb.size}, max={rb.max_size}"
    )
    print(f"  create_from_initial_dataset: pointer={rb2.pointer}, size={rb2.size}")


if __name__ == "__main__":
    main()
