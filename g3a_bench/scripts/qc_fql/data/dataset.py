import numpy as np


class Dataset:
    """State-based offline dataset with n-step / action-chunk sequence sampling.

    Port of `qc/utils/datasets.py::Dataset`. The JAX original subclasses
    `flax.core.FrozenDict` (an immutable, jit/pytree-compatible dict); nothing
    in the PyTorch path traces or jits a dataset, so this wraps a plain dict
    instead. The frame-stacking / image-augmentation / `add_history`
    machinery in the original is dead code for ACFQL (grep-confirmed: never
    referenced by `agents/acfql.py`, `main.py`, or `main_online.py`) and is
    dropped -- state-based observations only.
    """

    def __init__(self, data: dict[str, np.ndarray]):
        assert "observations" in data
        self.data = data
        self.size = len(data["observations"])

    @classmethod
    def create(cls, freeze: bool = True, **fields: np.ndarray) -> "Dataset":
        assert "observations" in fields
        if freeze:
            for arr in fields.values():
                arr.setflags(write=False)
        return cls(fields)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def items(self):
        return self.data.items()

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idxs = np.random.randint(self.size, size=batch_size)
        return {k: v[idxs] for k, v in self.data.items()}

    def sample_sequence(
        self, batch_size: int, sequence_length: int, discount: float
    ) -> dict[str, np.ndarray]:
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)

        data = {k: v[idxs] for k, v in self.items()}

        # Pre-compute all required indices
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]
        all_idxs = all_idxs.flatten()

        # Batch fetch data to avoid loops
        batch_observations = self["observations"][all_idxs].reshape(
            batch_size, sequence_length, *self["observations"].shape[1:]
        )
        batch_next_observations = self["next_observations"][all_idxs].reshape(
            batch_size, sequence_length, *self["next_observations"].shape[1:]
        )
        batch_actions = self["actions"][all_idxs].reshape(
            batch_size, sequence_length, *self["actions"].shape[1:]
        )
        batch_rewards = self["rewards"][all_idxs].reshape(
            batch_size, sequence_length, *self["rewards"].shape[1:]
        )
        batch_masks = self["masks"][all_idxs].reshape(
            batch_size, sequence_length, *self["masks"].shape[1:]
        )
        batch_terminals = self["terminals"][all_idxs].reshape(
            batch_size, sequence_length, *self["terminals"].shape[1:]
        )

        # Calculate next_actions
        next_action_idxs = np.minimum(all_idxs + 1, self.size - 1)
        batch_next_actions = self["actions"][next_action_idxs].reshape(
            batch_size, sequence_length, *self["actions"].shape[1:]
        )

        # Use vectorized operations to calculate cumulative rewards and masks
        rewards = np.zeros((batch_size, sequence_length), dtype=float)
        masks = np.ones((batch_size, sequence_length), dtype=float)
        terminals = np.zeros((batch_size, sequence_length), dtype=float)
        valid = np.ones((batch_size, sequence_length), dtype=float)

        rewards[:, 0] = batch_rewards[:, 0].squeeze()
        masks[:, 0] = batch_masks[:, 0].squeeze()
        terminals[:, 0] = batch_terminals[:, 0].squeeze()

        discount_powers = discount ** np.arange(sequence_length)
        for i in range(1, sequence_length):
            rewards[:, i] = (
                rewards[:, i - 1] + batch_rewards[:, i].squeeze() * discount_powers[i]
            )
            masks[:, i] = np.minimum(masks[:, i - 1], batch_masks[:, i].squeeze())
            terminals[:, i] = np.maximum(
                terminals[:, i - 1], batch_terminals[:, i].squeeze()
            )
            valid[:, i] = 1.0 - terminals[:, i - 1]

        # State data: maintain (batch, seq, state_dim) shape (the original also
        # has a 5-D visual branch here, dropped -- state obs only).
        observations = batch_observations
        next_observations = batch_next_observations
        actions = batch_actions
        next_actions = batch_next_actions

        return {
            "observations": data["observations"].copy(),
            "full_observations": observations,
            "actions": actions,
            "masks": masks,
            "rewards": rewards,
            "terminals": terminals,
            "valid": valid,
            "next_observations": next_observations,
            "next_actions": next_actions,
        }
