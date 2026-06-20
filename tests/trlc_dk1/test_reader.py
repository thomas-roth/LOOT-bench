import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from g3a_bench.scripts.live_bridge.reader import ReaderFrame


class TestReaderFrame:
    def test_creation_with_all_fields(self):
        frame = ReaderFrame(
            joint_pos=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            leader_action=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            cameras={"wrist": np.zeros((480, 640, 3), dtype=np.uint8)},
            timestamp=12345.0,
        )
        assert frame.joint_pos.shape == (7,)
        assert frame.leader_action.shape == (7,)
        assert "wrist" in frame.cameras
        assert frame.cameras["wrist"].shape == (480, 640, 3)
        assert frame.timestamp == 12345.0

    def test_default_fields(self):
        """Only joint_pos is required — everything else has defaults."""
        frame = ReaderFrame(joint_pos=np.zeros(7))
        assert frame.leader_action is None
        assert frame.cameras == {}
        assert frame.timestamp == 0.0


class TestDatasetReader:
    """DatasetReader wraps a LeRobotDataset to produce ReaderFrame objects."""

    @staticmethod
    def _make_mock_dataset(num_frames: int):
        """Build a mock LeRobotDataset with num_frames frames in episode 0."""
        import numpy as np

        mock_ds = MagicMock()
        # hf_dataset must expose "episode_index" as something array-like
        mock_hf = MagicMock()
        mock_hf.__getitem__ = MagicMock()
        ep_indices = np.zeros(num_frames, dtype=np.int64)
        mock_hf.__getitem__.side_effect = lambda k: {
            "episode_index": ep_indices,
        }.get(k, MagicMock())
        mock_ds.hf_dataset = mock_hf

        # ds[i] returns frame dicts
        def make_frame(i):
            return {
                "observation.state": np.full(7, 0.1 * (i + 1), dtype=np.float32),
                "action": np.full(7, 0.01 * (i + 1), dtype=np.float32),
                "timestamp": np.float32(i * 0.033),
            }

        mock_ds.__getitem__ = MagicMock(side_effect=make_frame)
        return mock_ds

    @patch("g3a_bench.scripts.live_bridge.reader.LeRobotDataset")
    def test_read_returns_readerframe(self, mock_dataset_cls):
        from g3a_bench.scripts.live_bridge.reader import DatasetReader

        mock_dataset_cls.return_value = self._make_mock_dataset(2)
        reader = DatasetReader("some/repo-id")

        frame = reader.read()
        assert isinstance(frame, ReaderFrame)
        assert frame.joint_pos.shape == (7,)
        assert np.allclose(frame.joint_pos, 0.1)
        assert frame.leader_action is not None
        assert np.allclose(frame.leader_action, 0.01)

        frame2 = reader.read()
        assert np.allclose(frame2.joint_pos, 0.2)
        reader.close()

    @patch("g3a_bench.scripts.live_bridge.reader.LeRobotDataset")
    def test_read_raises_stopiteration_when_exhausted(self, mock_dataset_cls):
        from g3a_bench.scripts.live_bridge.reader import DatasetReader

        mock_dataset_cls.return_value = self._make_mock_dataset(1)
        reader = DatasetReader("some/repo-id")
        reader.read()
        with pytest.raises(StopIteration):
            reader.read()
        reader.close()


@pytest.mark.skipif(
    not os.environ.get("HARDWARE_TEST"),
    reason="Requires network: set HARDWARE_TEST=1 to download FabianKerj/corkinbox100-fullrel",
)
class TestDatasetReaderIntegration:
    """End-to-end test with a real LeRobot dataset from HuggingFace."""

    def test_read_first_frame(self):
        from g3a_bench.scripts.live_bridge.reader import DatasetReader

        reader = DatasetReader("FabianKerj/corkinbox100-fullrel")
        frame = reader.read()
        assert frame.joint_pos.shape == (7,)
        assert frame.leader_action is not None
        assert frame.leader_action.shape == (7,)
        # Joint positions should be within sensible range for DK1
        assert np.all(np.abs(frame.joint_pos[:6]) <= np.pi)  # joints in rad
        assert 0.0 <= frame.joint_pos[6] <= 1.0  # gripper in [0, 1]
        assert frame.timestamp >= 0.0
        reader.close()

    def test_read_multiple_frames(self):
        from g3a_bench.scripts.live_bridge.reader import DatasetReader

        reader = DatasetReader("FabianKerj/corkinbox100-fullrel")
        frames = [reader.read() for _ in range(10)]
        assert len(frames) == 10
        # Timestamps should increase
        timestamps = [f.timestamp for f in frames]
        assert all(t2 >= t1 for t1, t2 in zip(timestamps, timestamps[1:], strict=True))
        reader.close()


class TestLiveReader:
    """LiveReader wraps DK1Follower + DK1Leader for real-time data."""

    @patch("g3a_bench.scripts.live_bridge.reader.DK1Follower")
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Leader")
    def test_lazy_init(self, mock_leader_cls, mock_follower_cls):
        """Can instantiate without connecting (connect_on_init=False)."""
        from g3a_bench.scripts.live_bridge.reader import LiveReader

        reader = LiveReader(
            follower_port="/dev/null",
            leader_port="/dev/null",
            connect_on_init=False,
        )
        assert reader.follower_port == "/dev/null"
        assert reader.leader_port == "/dev/null"
        # Should NOT have called connect() on either
        mock_follower_cls.assert_not_called()
        mock_leader_cls.assert_not_called()
        reader.close()

    @patch("g3a_bench.scripts.live_bridge.reader.DK1Follower")
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Leader")
    def test_connect_on_init(self, mock_leader_cls, mock_follower_cls):
        from g3a_bench.scripts.live_bridge.reader import LiveReader

        reader = LiveReader(
            follower_port="/dev/ttyACM_test",
            leader_port="/dev/ttyUSB_test",
            connect_on_init=True,
        )
        mock_follower_cls.assert_called_once()
        mock_follower_cls.return_value.connect.assert_called_once()
        mock_leader_cls.assert_called_once()
        mock_leader_cls.return_value.connect.assert_called_once()
        reader.close()

    @patch("g3a_bench.scripts.live_bridge.reader.DK1Follower")
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Leader")
    def test_read_returns_readerframe(self, mock_leader_cls, mock_follower_cls):
        from g3a_bench.scripts.live_bridge.reader import LiveReader

        # Mock observation from follower
        mock_follower = MagicMock()
        mock_follower.get_observation.return_value = {
            "joint_1.pos": 0.1,
            "joint_2.pos": 0.2,
            "joint_3.pos": 0.3,
            "joint_4.pos": 0.4,
            "joint_5.pos": 0.5,
            "joint_6.pos": 0.6,
            "gripper.pos": 0.5,
            "context": np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        mock_follower_cls.return_value = mock_follower

        # Mock leader action
        mock_leader = MagicMock()
        mock_leader.get_action.return_value = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        mock_leader_cls.return_value = mock_leader

        reader = LiveReader(
            follower_port="/dev/ttyACM_test",
            leader_port="/dev/ttyUSB_test",
            connect_on_init=True,
        )
        frame = reader.read()
        assert isinstance(frame, ReaderFrame)
        assert frame.joint_pos.shape == (7,)
        assert np.allclose(frame.joint_pos, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5])
        assert frame.leader_action is not None
        assert "context" in frame.cameras
        assert "wrist" in frame.cameras
        reader.close()

    @patch("g3a_bench.scripts.live_bridge.reader.DK1Follower")
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Leader")
    def test_close_disconnects(self, mock_leader_cls, mock_follower_cls):
        from g3a_bench.scripts.live_bridge.reader import LiveReader

        mock_follower = MagicMock()
        mock_leader = MagicMock()
        mock_follower_cls.return_value = mock_follower
        mock_leader_cls.return_value = mock_leader

        reader = LiveReader(
            follower_port="/dev/ttyACM_test",
            leader_port="/dev/ttyUSB_test",
            connect_on_init=True,
        )
        reader.close()
        mock_follower.disconnect.assert_called_once()
        mock_leader.disconnect.assert_called_once()
