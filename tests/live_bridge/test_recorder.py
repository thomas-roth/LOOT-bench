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
        rec.add_frame(
            obs_buf={"actor": torch.zeros(1, 10)},
            action=np.zeros(7, dtype=np.float32),
            reward=0.0,
            done=False,
        )
        rec.end_episode()
        mock_ds.save_episode.assert_called_once()
        rec.close()
        mock_ds.finalize.assert_called_once()

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_end_episode_skips_when_empty(self, mock_dataset_cls):
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        rec.end_episode()  # nothing added
        mock_ds.save_episode.assert_not_called()


def test_recorder_real_roundtrip(tmp_path):
    """Un-mocked round-trip against the installed lerobot.

    Uses the SAME add_frame signature the live/replay scripts use --
    ``task`` omitted, so it defaults to "" -- then reads the dataset back
    the way offline training would, so a future lerobot bump can't silently
    break the schema or the empty-task path.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "ds"
    rec = Recorder(repo_id="test_roundtrip", fps=50, obs_dim=53, root=str(root))
    for i in range(2):
        rec.add_frame(
            {"actor": torch.zeros(1, 53)},
            np.zeros(7, dtype=np.float32),
            float(i),
            bool(i),
            robot_obs=np.zeros(7, dtype=np.float32),
        )
    rec.end_episode()
    rec.close()

    ds = LeRobotDataset("test_roundtrip", root=str(root))
    assert ds.num_frames == 2
    assert ds.num_episodes == 1
    frame = ds[0]
    assert np.asarray(frame["observation.state"]).shape == (53,)
    assert np.asarray(frame["observation.robot_state"]).shape == (7,)
    assert np.asarray(frame["action"]).shape == (7,)
