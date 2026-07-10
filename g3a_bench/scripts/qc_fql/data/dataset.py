"""Offline demonstration dataset with action-chunk sequence sampling.

This is where QC-FQL's "action chunking" actually happens on the data side:
`sample_sequence` turns flat per-step transitions into length-`horizon`
windows whose per-step rewards are pre-collapsed into an n-step return, so
the agent can treat a whole chunk as one macro-action with one TD backup.
Port of `qc/utils/datasets.py::Dataset` -- the reference method is already
pure NumPy, so the body below is a near-verbatim copy; only the container
changed (plain dict instead of `flax.core.FrozenDict`, whose immutability
exists for jit/pytree reasons torch never hits).

Deliberately not ported (all grep-confirmed unused by `acfql.py`/`main.py`):
frame stacking, image augmentation, `add_history`, and `sample_sequence`'s
5-D visual-observation branch -- this project is state-only.

Consumers: `scripts/train_qcfql.py` (offline phase), `data/replay_buffer.py`
(subclasses this and inherits `sample_sequence` for the online phase).
"""

import numpy as np


class Dataset:
    def __init__(self, data: dict[str, np.ndarray]):
        assert "observations" in data
        self.data = data
        self.size = len(data["observations"])

    @classmethod
    def create(cls, freeze: bool = True, **fields: np.ndarray) -> "Dataset":
        assert "observations" in fields
        if freeze:
            # numpy-level write protection on the raw arrays; ReplayBuffer
            # bypasses `create` (goes through __init__) because it must write.
            for arr in fields.values():
                arr.setflags(write=False)
        return cls(fields)

    # dict-style access so `sample_sequence` below can stay a verbatim copy
    # of the reference (which subclasses FrozenDict and uses self[...]).
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
        """Sample chunk windows with n-step returns, matching the reference exactly.

        Key output-shape contract (easy to get wrong): `observations` is the
        window's FIRST observation, shape (B, obs_dim) -- the losses condition
        on where the chunk started. The full (B, T, obs_dim) sequence is under
        `full_observations`. `rewards[:, i]` is the discounted sum of steps
        0..i, so the critic reads the whole chunk's return at `[:, -1]`.
        """
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)

        data = {k: v[idxs] for k, v in self.items()}

        # (B, T) window indices, flattened for one gather per field.
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]
        all_idxs = all_idxs.flatten()

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

        # Clamp keeps the +1 lookahead in-bounds at the buffer's last row
        # (the reference accepts the resulting slightly-wrong next_action
        # there; masks/valid make it irrelevant for training).
        next_action_idxs = np.minimum(all_idxs + 1, self.size - 1)
        batch_next_actions = self["actions"][next_action_idxs].reshape(
            batch_size, sequence_length, *self["actions"].shape[1:]
        )

        rewards = np.zeros((batch_size, sequence_length), dtype=float)
        masks = np.ones((batch_size, sequence_length), dtype=float)
        terminals = np.zeros((batch_size, sequence_length), dtype=float)
        valid = np.ones((batch_size, sequence_length), dtype=float)

        rewards[:, 0] = batch_rewards[:, 0].squeeze()
        masks[:, 0] = batch_masks[:, 0].squeeze()
        terminals[:, 0] = batch_terminals[:, 0].squeeze()

        # Running cumulants over the window: rewards accumulate a discounted
        # sum; masks/terminals latch once an episode boundary is crossed
        # (min/max make them sticky); valid[i] marks steps still inside the
        # episode the window started in -- the losses zero out everything
        # after a boundary, so windows straddling episodes are safe to sample.
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

        return {
            "observations": data["observations"].copy(),
            "full_observations": batch_observations,
            "actions": batch_actions,
            "masks": masks,
            "rewards": rewards,
            "terminals": terminals,
            "valid": valid,
            "next_observations": batch_next_observations,
            "next_actions": batch_next_actions,
        }
