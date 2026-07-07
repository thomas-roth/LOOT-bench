from unittest.mock import MagicMock, patch

import numpy as np
import torch

from g3a_bench.scripts.live_bridge.recorder import Recorder


class TestRecorder:
    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_create_dataset_lazily(self, mock_dataset_cls):
        """No dataset is created on init — only on the first save_episode(),
        so a session never leaves empty success/failure datasets on disk."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        mock_dataset_cls.create.assert_not_called()
        rec.add_frame(
            {"actor": torch.zeros(1, 10)}, np.zeros(7, np.float32), 0.0, False
        )
        rec.save_episode(success=True)
        mock_dataset_cls.create.assert_called_once()
        assert rec.dataset_success is mock_ds
        assert rec.dataset_failure is None

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_add_frame_buffers_until_save(self, mock_dataset_cls):
        """add_frame must NOT write to the dataset yet -- frames are held in a
        buffer so the whole episode can be labeled success/failure at the end."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        rec.add_frame(
            obs_buf={"actor": torch.zeros(1, 10)},
            action=np.zeros(7, dtype=np.float32),
            reward=0.5,
            done=False,
            task="cube",
        )
        mock_ds.add_frame.assert_not_called()
        assert rec.num_frames_in_episode == 1
        frame = rec._pending[0]
        assert frame["observation.state"].shape == (10,)
        assert frame["action"].shape == (7,)
        assert frame["next.reward"] == np.array([0.5], dtype=np.float32)

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_save_episode_labels_frames_success(self, mock_dataset_cls):
        """save_episode(True) flushes every buffered frame stamped success=True,
        then saves -- that label is what training filters on."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        rec.add_frame(
            {"actor": torch.zeros(1, 10)}, np.zeros(7, np.float32), 0.0, False
        )
        rec.save_episode(success=True)
        assert mock_ds.add_frame.call_count == 1
        flushed = mock_ds.add_frame.call_args[0][0]
        assert flushed["success"].dtype == bool
        assert bool(flushed["success"][0]) is True
        mock_ds.save_episode.assert_called_once()
        assert rec.num_frames_in_episode == 0

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_save_episode_labels_frames_failure(self, mock_dataset_cls):
        """save_episode(False) stamps success=False so failures are kept separate."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        rec.add_frame(
            {"actor": torch.zeros(1, 10)}, np.zeros(7, np.float32), 0.0, False
        )
        rec.save_episode(success=False)
        flushed = mock_ds.add_frame.call_args[0][0]
        assert bool(flushed["success"][0]) is False

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_save_episode_flushes_all_frames(self, mock_dataset_cls):
        """Every buffered frame is flushed on save, not just the first."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        for _ in range(3):
            rec.add_frame(
                {"actor": torch.zeros(1, 10)}, np.zeros(7, np.float32), 0.0, False
            )
        rec.save_episode(success=True)
        assert mock_ds.add_frame.call_count == 3

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_discard_episode_drops_buffer(self, mock_dataset_cls):
        """discard_episode() throws away the current episode without saving --
        the operator must be able to reject a botched demo."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        rec.add_frame(
            {"actor": torch.zeros(1, 10)}, np.zeros(7, np.float32), 0.0, False
        )
        rec.discard_episode()
        assert rec.num_frames_in_episode == 0
        mock_ds.add_frame.assert_not_called()
        mock_ds.save_episode.assert_not_called()

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_save_episode_empty_is_noop(self, mock_dataset_cls):
        """Saving with nothing buffered must not write an empty episode —
        and must not even create a dataset directory."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        rec.save_episode(success=True)
        mock_dataset_cls.create.assert_not_called()
        mock_ds.save_episode.assert_not_called()

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_close_finalizes_dataset(self, mock_dataset_cls):
        """close() calls finalize() on every dataset that was created."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        rec.add_frame(
            {"actor": torch.zeros(1, 10)}, np.zeros(7, np.float32), 0.0, False
        )
        rec.save_episode(success=True)
        rec.close()
        mock_ds.finalize.assert_called_once()

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_discard_then_save_new_episode(self, mock_dataset_cls):
        """After discard, a fresh episode can be buffered and saved normally."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        rec.add_frame(
            {"actor": torch.zeros(1, 10)}, np.zeros(7, np.float32), 0.0, False
        )
        rec.discard_episode()
        rec.add_frame({"actor": torch.zeros(1, 10)}, np.zeros(7, np.float32), 1.0, True)
        rec.save_episode(success=True)
        assert mock_ds.add_frame.call_count == 1
        mock_ds.save_episode.assert_called_once()


def test_recorder_real_roundtrip(tmp_path):
    """Un-mocked round-trip against the installed lerobot.

    Verifies the buffering+label path end-to-end: add frames, save with
    success=True, read back and confirm the success column persists — and
    that no failure dataset was created for a success-only session.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    success_root = tmp_path / "success" / "stamp"
    failure_root = tmp_path / "failure" / "stamp"
    rec = Recorder(
        repo_id="test_roundtrip",
        fps=50,
        obs_dim=53,
        success_root=str(success_root),
        failure_root=str(failure_root),
    )
    for i in range(2):
        rec.add_frame(
            {"actor": torch.zeros(1, 53)},
            np.zeros(7, dtype=np.float32),
            float(i),
            bool(i),
            robot_obs=np.zeros(7, dtype=np.float32),
        )
    rec.save_episode(success=True)
    rec.close()

    assert not failure_root.exists()
    ds = LeRobotDataset("test_roundtrip", root=str(success_root))
    assert ds.num_frames == 2
    assert ds.num_episodes == 1
    frame = ds[0]
    assert np.asarray(frame["observation.state"]).shape == (53,)
    assert np.asarray(frame["observation.robot_state"]).shape == (7,)
    assert np.asarray(frame["action"]).shape == (7,)
    assert bool(np.asarray(frame["success"]).flatten()[0]) is True


def test_close_stores_operator_id(tmp_path):
    """operator_id passed to close() must appear in dataset info after finalize."""
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from g3a_bench.scripts.live_bridge.recorder import Recorder

    root = tmp_path / "ds_op" / "success"
    rec = Recorder(repo_id="test_op", fps=50, obs_dim=53, success_root=str(root))
    rec.add_frame(
        {"actor": torch.zeros(1, 53)},
        np.zeros(7, dtype=np.float32),
        0.0,
        False,
        robot_obs=np.zeros(7, dtype=np.float32),
    )
    rec.save_episode(success=True)
    rec.close(operator_id=3)

    ds = LeRobotDataset("test_op", root=str(root))
    assert ds.meta.info.get("operator_id") == 3


def test_close_without_operator_id_still_finalizes(tmp_path):
    """close() with no operator_id must not crash (backwards compat)."""
    from g3a_bench.scripts.live_bridge.recorder import Recorder

    root = tmp_path / "ds_noop" / "success"
    rec = Recorder(repo_id="test_noop", fps=50, obs_dim=53, success_root=str(root))
    rec.close()  # no argument, nothing recorded — must not raise
