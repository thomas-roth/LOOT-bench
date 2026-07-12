from unittest.mock import MagicMock, patch

import numpy as np
import torch

from g3a_bench.scripts.live_bridge.recorder import Recorder


def _add_dummy_frame(rec: Recorder, reward: float = 0.0, done: bool = False) -> None:
    rec.add_frame({"actor": torch.zeros(1, 10)}, np.zeros(7, np.float32), reward, done)


class TestRecorder:
    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_create_dataset_lazily(self, mock_dataset_cls):
        """No dataset is created on init or on staging — only when a staged
        episode is committed, so an all-undone session leaves nothing on disk."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        mock_dataset_cls.create.assert_not_called()
        _add_dummy_frame(rec)
        rec.save_episode(success=True)
        mock_dataset_cls.create.assert_not_called()
        rec.close()
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
    def test_save_episode_stages_without_writing(self, mock_dataset_cls):
        """save_episode() only stages the episode — nothing hits the dataset
        until the next episode ends or the session finalizes, enabling undo."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        _add_dummy_frame(rec)
        rec.save_episode(success=True)
        mock_ds.add_frame.assert_not_called()
        mock_ds.save_episode.assert_not_called()
        assert rec.has_staged
        assert rec.num_frames_in_episode == 0

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_next_save_commits_staged(self, mock_dataset_cls):
        """Staging episode 2 commits episode 1 with its original label."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        _add_dummy_frame(rec)
        rec.save_episode(success=True)
        _add_dummy_frame(rec)
        rec.save_episode(success=False)
        # Episode 1 (success) committed; episode 2 (failure) still staged.
        assert mock_ds.add_frame.call_count == 1
        flushed = mock_ds.add_frame.call_args[0][0]
        assert flushed["success"].dtype == bool
        assert bool(flushed["success"][0]) is True
        mock_ds.save_episode.assert_called_once()
        assert rec.has_staged

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_close_commits_staged(self, mock_dataset_cls):
        """close() commits the still-staged last episode before finalizing."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        _add_dummy_frame(rec)
        rec.save_episode(success=False)
        rec.close()
        assert mock_ds.add_frame.call_count == 1
        flushed = mock_ds.add_frame.call_args[0][0]
        assert bool(flushed["success"][0]) is False
        mock_ds.save_episode.assert_called_once()
        mock_ds.finalize.assert_called_once()

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_undo_drops_staged_episode(self, mock_dataset_cls):
        """undo_last_save() drops the staged episode and reports its label —
        an undone episode must never reach the dataset."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        _add_dummy_frame(rec)
        rec.save_episode(success=True)
        assert rec.undo_last_save() is True
        assert not rec.has_staged
        rec.close()
        mock_dataset_cls.create.assert_not_called()
        mock_ds.add_frame.assert_not_called()
        mock_ds.save_episode.assert_not_called()

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_undo_without_staged_returns_none(self, mock_dataset_cls):
        """Single-level undo: a second undo has nothing to drop."""
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        assert rec.undo_last_save() is None
        _add_dummy_frame(rec)
        rec.save_episode(success=False)
        assert rec.undo_last_save() is False
        assert rec.undo_last_save() is None

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_undo_is_single_level(self, mock_dataset_cls):
        """Only the last save is undoable: staging episode 2 committed episode 1,
        so undo drops only episode 2."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        _add_dummy_frame(rec)
        rec.save_episode(success=True)
        _add_dummy_frame(rec)
        rec.save_episode(success=True)
        assert rec.undo_last_save() is True
        rec.close()
        # Episode 1 was committed and stays; episode 2 was dropped.
        assert mock_ds.save_episode.call_count == 1

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_save_episode_flushes_all_frames(self, mock_dataset_cls):
        """Every buffered frame of a committed episode is flushed, not just
        the first."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        for _ in range(3):
            _add_dummy_frame(rec)
        rec.save_episode(success=True)
        rec.close()
        assert mock_ds.add_frame.call_count == 3

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_discard_episode_drops_buffer_not_staged(self, mock_dataset_cls):
        """discard_episode() throws away the in-progress episode only — a
        previously staged episode survives and commits at close."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        _add_dummy_frame(rec)
        rec.save_episode(success=True)
        _add_dummy_frame(rec)
        rec.discard_episode()
        assert rec.num_frames_in_episode == 0
        assert rec.has_staged
        rec.close()
        assert mock_ds.save_episode.call_count == 1

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_save_episode_empty_is_noop(self, mock_dataset_cls):
        """Saving with nothing buffered must not stage an empty episode —
        and must not even create a dataset directory."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        rec.save_episode(success=True)
        assert not rec.has_staged
        rec.close()
        mock_dataset_cls.create.assert_not_called()
        mock_ds.save_episode.assert_not_called()

    @patch("g3a_bench.scripts.live_bridge.recorder.LeRobotDataset")
    def test_discard_then_save_new_episode(self, mock_dataset_cls):
        """After discard, a fresh episode can be buffered and saved normally."""
        mock_ds = MagicMock()
        mock_dataset_cls.create.return_value = mock_ds
        rec = Recorder(repo_id="test/ds", fps=30, obs_dim=10)
        _add_dummy_frame(rec)
        rec.discard_episode()
        _add_dummy_frame(rec, reward=1.0, done=True)
        rec.save_episode(success=True)
        rec.close()
        assert mock_ds.add_frame.call_count == 1
        mock_ds.save_episode.assert_called_once()


def test_recorder_real_roundtrip(tmp_path):
    """Un-mocked round-trip against the installed lerobot.

    Verifies the staging+label path end-to-end: add frames, save with
    success=True, close (commits the staged episode), read back and confirm
    the success column persists — and that no failure dataset was created
    for a success-only session.
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


def test_recorder_real_undo_roundtrip(tmp_path):
    """Un-mocked: save two episodes, undo the second — only the first lands
    on disk."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    success_root = tmp_path / "success" / "stamp"
    rec = Recorder(
        repo_id="test_undo",
        fps=50,
        obs_dim=53,
        success_root=str(success_root),
    )
    for _ in range(2):
        rec.add_frame(
            {"actor": torch.zeros(1, 53)},
            np.zeros(7, dtype=np.float32),
            0.0,
            False,
            robot_obs=np.zeros(7, dtype=np.float32),
        )
        rec.save_episode(success=True)
    assert rec.undo_last_save() is True
    rec.close()

    ds = LeRobotDataset("test_undo", root=str(success_root))
    assert ds.num_episodes == 1
    assert ds.num_frames == 1


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
