import numpy as np
from data.dataset import Dataset


class ReplayBuffer(Dataset):
    """Circular replay buffer for online fine-tuning.

    Port of `qc/utils/datasets.py::ReplayBuffer`. Subclasses `Dataset` and
    inherits `sample_sequence` unchanged (it samples within `self.size`, the
    current fill count). JAX's `jax.tree_util.tree_map` over the flat
    transition/buffer dicts is replaced by plain per-key loops -- equivalent
    for a flat, fixed-key dict.
    """

    @classmethod
    def create(cls, transition: dict, size: int) -> "ReplayBuffer":
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = {k: create_buffer(v) for k, v in transition.items()}
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(
        cls, init_dataset: dict, size: int
    ) -> "ReplayBuffer":
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = {k: create_buffer(v) for k, v in init_dataset.items()}
        buffer = cls(buffer_dict)
        buffer.size = buffer.pointer = len(init_dataset["observations"])
        return buffer

    def __init__(self, data: dict[str, np.ndarray]):
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
