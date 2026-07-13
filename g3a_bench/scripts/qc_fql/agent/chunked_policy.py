"""Action-chunk rollout wrapper shared by evaluate / train_online / play.

The agent emits H actions per query; this hands out one action per step and
resamples only when the chunk is exhausted, so the agent runs once per H env
steps. Previously copy-pasted into evaluate.py, train_qcfql.py and play_qcfql.py.
"""

import numpy as np
import torch


class ChunkedPolicy:
    def __init__(self, agent, action_dim: int, device: str | torch.device):
        self.agent = agent
        self.action_dim = action_dim
        self.device = device
        self._queue: list[np.ndarray] = []

    def reset(self) -> None:
        """Drop the current chunk — a fresh episode must not start mid-chunk."""
        self._queue = []

    def next_action(self, observation) -> np.ndarray:
        """Next [-1, 1] action; resamples a chunk when the queue is empty."""
        if not self._queue:
            obs_t = (
                torch.from_numpy(np.asarray(observation))
                .float()
                .to(self.device)
                .unsqueeze(0)
            )
            with torch.no_grad():
                chunk = self.agent.sample_actions(obs_t).squeeze(0).cpu().numpy()
            self._queue.extend(chunk.reshape(-1, self.action_dim))
        return self._queue.pop(0)
