from unittest.mock import MagicMock, patch

import numpy as np
import torch

from g3a_bench.scripts.live_bridge.recorder import Recorder


class TestRecorder:
    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_create_dataset(self, mock_dataset_cls):
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        mock_dataset_cls.create.assert_called_once()
        assert rec.dataset is mock_ds

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_add_frame(self, mock_dataset_cls):
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        action = np.zeros(7, dtype=np.float32)
        rec.add_frame(
            obs_buf={"actor": torch.zeros(1, 10)},
            action=action,
            reward=0.5,
            done=False,
            task="cube",
        )
        mock_ds.add_frame.assert_called_once()
        call_frame = mock_ds.add_frame.call_args[0][0]
        assert call_frame["observation.state"].shape == (10,)
        assert call_frame["action"].shape == (7,)
        assert call_frame["next.reward"] == np.array([0.5], dtype=np.float32)

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_end_episode_and_close(self, mock_dataset_cls):
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        rec.end_episode()
        mock_ds.save_episode.assert_called_once()
        rec.close()
        mock_ds.finalize.assert_called_once()
