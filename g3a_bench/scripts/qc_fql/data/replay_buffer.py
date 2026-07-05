"""Circular transition buffer for the online fine-tuning phase.

After offline pretraining, QC-FQL keeps learning from its own environment
interaction. The reference (`qc/main.py`) seeds a fixed-size ring buffer
with the entire offline dataset, then appends each new env transition and
samples chunk windows from it exactly as from the offline dataset. Port of
`qc/utils/datasets.py::ReplayBuffer`.

Subclasses `Dataset` purely to inherit `sample_sequence` -- the window/
n-step logic must be identical for offline and online batches or the agent
would be trained on two different reward conventions. `self.size` doubles
as the "filled up to here" watermark, which is what keeps inherited
sampling away from the untouched zero region of the preallocated arrays.

Reference quirk preserved on purpose: `size = max(pointer, size)` means
that once the buffer wraps, `size` stays at `max_size - 1` forever (the
pointer resets to 0 before ever reaching max_size), so one slot is written
but never sampled. Off-by-one in the original; kept so the gold state test
matches the reference bit-for-bit.

The reference's `jax.tree_util.tree_map` over transitions is replaced with
plain per-key loops -- transitions here are flat fixed-key dicts, which is
all tree_map was doing for it.
"""

import numpy as np

from g3a_bench.scripts.qc_fql.data.dataset import Dataset


class ReplayBuffer(Dataset):
    @classmethod
    def create(cls, transition: dict, size: int) -> "ReplayBuffer":
        """Preallocate from one example transition (shapes/dtypes template)."""

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = {k: create_buffer(v) for k, v in transition.items()}
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(
        cls, init_dataset: dict, size: int
    ) -> "ReplayBuffer":
        """Preallocate and copy the offline dataset into the front; new
        transitions append after it (pointer starts at the dataset's end)."""

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = {k: create_buffer(v) for k, v in init_dataset.items()}
        buffer = cls(buffer_dict)
        buffer.size = buffer.pointer = len(init_dataset["observations"])
        return buffer

    def __init__(self, data: dict[str, np.ndarray]):
        # Deliberately NOT Dataset.create: buffers must stay writable
        # (create() flips numpy's write-protect flag on).
        super().__init__(data)
        self.max_size = len(data["observations"])
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition: dict):
        for k in self.data:
            self.data[k][self.pointer] = transition[k]
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        self.size = self.pointer = 0
