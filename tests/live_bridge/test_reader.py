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
        assert all(
            t2 >= t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:], strict=True)
        )
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

    @patch(
        "g3a_bench.scripts.live_bridge.reader.DK1FollowerConfig",
        return_value=MagicMock(),
    )
    @patch(
        "g3a_bench.scripts.live_bridge.reader.DK1LeaderConfig", return_value=MagicMock()
    )
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Follower")
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Leader")
    def test_connect_on_init(
        self,
        mock_leader_cls,
        mock_follower_cls,
        mock_leader_cfg_cls,
        mock_follower_cfg_cls,
    ):
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

    @patch(
        "g3a_bench.scripts.live_bridge.reader.DK1FollowerConfig",
        return_value=MagicMock(),
    )
    @patch(
        "g3a_bench.scripts.live_bridge.reader.DK1LeaderConfig", return_value=MagicMock()
    )
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Follower")
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Leader")
    def test_read_returns_readerframe(
        self,
        mock_leader_cls,
        mock_follower_cls,
        mock_leader_cfg_cls,
        mock_follower_cfg_cls,
    ):
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

        # Mock leader action — real contract: get_action() returns a dict.
        mock_leader = MagicMock()
        mock_leader.get_action.return_value = {
            "joint_1.pos": 0.0,
            "joint_2.pos": 0.0,
            "joint_3.pos": 0.0,
            "joint_4.pos": 0.0,
            "joint_5.pos": 0.0,
            "joint_6.pos": 0.0,
            "gripper.pos": 0.0,
        }
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

    @patch(
        "g3a_bench.scripts.live_bridge.reader.DK1FollowerConfig",
        return_value=MagicMock(),
    )
    @patch(
        "g3a_bench.scripts.live_bridge.reader.DK1LeaderConfig", return_value=MagicMock()
    )
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Follower")
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Leader")
    def test_close_disconnects(
        self,
        mock_leader_cls,
        mock_follower_cls,
        mock_leader_cfg_cls,
        mock_follower_cfg_cls,
    ):
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

    @patch(
        "g3a_bench.scripts.live_bridge.reader.DK1FollowerConfig",
        return_value=MagicMock(),
    )
    @patch(
        "g3a_bench.scripts.live_bridge.reader.DK1LeaderConfig", return_value=MagicMock()
    )
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Follower")
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Leader")
    def test_send_action(
        self,
        mock_leader_cls,
        mock_follower_cls,
        mock_leader_cfg_cls,
        mock_follower_cfg_cls,
    ):
        from g3a_bench.scripts.live_bridge.reader import LiveReader

        mock_follower = MagicMock()
        mock_follower_cls.return_value = mock_follower

        reader = LiveReader(
            follower_port="/dev/ttyACM_test",
            leader_port="/dev/ttyUSB_test",
            connect_on_init=True,
        )

        action_np = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
        result = reader.send_action(action_np)

        mock_follower.send_action.assert_called_once()
        sent = mock_follower.send_action.call_args[0][0]
        assert sent == {
            "joint_1.pos": float(action_np[0]),
            "joint_2.pos": float(action_np[1]),
            "joint_3.pos": float(action_np[2]),
            "joint_4.pos": float(action_np[3]),
            "joint_5.pos": float(action_np[4]),
            "joint_6.pos": float(action_np[5]),
            "gripper.pos": float(action_np[6]),
        }
        assert result == mock_follower.send_action.return_value
        reader.close()

    @patch(
        "g3a_bench.scripts.live_bridge.reader.DK1FollowerConfig",
        return_value=MagicMock(),
    )
    @patch(
        "g3a_bench.scripts.live_bridge.reader.DK1LeaderConfig", return_value=MagicMock()
    )
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Follower")
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Leader")
    def test_read_leader_action_dict_is_unpacked_in_order(
        self, mock_leader_cls, mock_follower_cls, mock_leader_cfg, mock_follower_cfg
    ):
        from g3a_bench.scripts.live_bridge.reader import LiveReader

        mock_follower = MagicMock()
        mock_follower.get_observation.return_value = {
            f"joint_{i}.pos": 0.0 for i in range(1, 7)
        } | {"gripper.pos": 0.0}
        mock_follower_cls.return_value = mock_follower

        mock_leader = MagicMock()
        # REAL contract: get_action() returns a dict, not an array.
        mock_leader.get_action.return_value = {
            "joint_1.pos": 0.1,
            "joint_2.pos": 0.2,
            "joint_3.pos": 0.3,
            "joint_4.pos": 0.4,
            "joint_5.pos": 0.5,
            "joint_6.pos": 0.6,
            "gripper.pos": 0.7,
        }
        mock_leader_cls.return_value = mock_leader

        reader = LiveReader("/dev/ttyACM_x", "/dev/ttyUSB_x", connect_on_init=True)
        frame = reader.read()
        assert frame.leader_action is not None
        assert frame.leader_action.shape == (7,)
        assert frame.leader_action.dtype == np.float32
        np.testing.assert_allclose(
            frame.leader_action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], rtol=1e-6
        )
        reader.close()

    @patch(
        "g3a_bench.scripts.live_bridge.reader.DK1LeaderConfig", return_value=MagicMock()
    )
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Follower")
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Leader")
    def test_use_follower_false_never_touches_follower(
        self, mock_leader_cls, mock_follower_cls, mock_leader_cfg_cls
    ):
        """Leader-only mode: no follower connect, joint_pos mirrors leader_action."""
        from g3a_bench.scripts.live_bridge.reader import LiveReader

        mock_leader = MagicMock()
        mock_leader.get_action.return_value = {
            "joint_1.pos": 0.1,
            "joint_2.pos": 0.2,
            "joint_3.pos": 0.3,
            "joint_4.pos": 0.4,
            "joint_5.pos": 0.5,
            "joint_6.pos": 0.6,
            "gripper.pos": 0.7,
        }
        mock_leader_cls.return_value = mock_leader

        reader = LiveReader(leader_port="/dev/ttyACM_test", use_follower=False)
        mock_follower_cls.assert_not_called()

        frame = reader.read()
        np.testing.assert_allclose(
            frame.joint_pos, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], rtol=1e-6
        )
        assert frame.leader_action is not None
        np.testing.assert_allclose(frame.leader_action, frame.joint_pos, rtol=1e-6)
        assert frame.cameras == {}

        with pytest.raises(RuntimeError, match="not connected"):
            reader.send_action(np.zeros(7))

        reader.close()
        mock_leader.disconnect.assert_called_once()

    @patch(
        "g3a_bench.scripts.live_bridge.reader.DK1FollowerConfig",
        return_value=MagicMock(),
    )
    @patch(
        "g3a_bench.scripts.live_bridge.reader.DK1LeaderConfig", return_value=MagicMock()
    )
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Follower")
    @patch("g3a_bench.scripts.live_bridge.reader.DK1Leader")
    def test_send_action_raises_without_connect(
        self,
        mock_leader_cls,
        mock_follower_cls,
        mock_leader_cfg_cls,
        mock_follower_cfg_cls,
    ):
        from g3a_bench.scripts.live_bridge.reader import LiveReader

        reader = LiveReader(
            follower_port="/dev/ttyACM_test",
            leader_port="/dev/ttyUSB_test",
            connect_on_init=False,
        )
        with pytest.raises(RuntimeError, match="not connected"):
            reader.send_action(np.zeros(7))


class _FakePort:
    """Minimal stand-in for a pyserial ListPortInfo (device/description/vid)."""

    def __init__(self, device, desc="", vid=None):
        self.device = device
        self.description = desc
        self.vid = vid


def test_find_ports_uses_pyserial(monkeypatch):
    """_find_ports() classifies leader/follower from pyserial's enumeration
    (cross-platform), not from Linux-only /dev globbing."""
    import g3a_bench.scripts.live_bridge.reader as reader_mod

    fake = [
        _FakePort("COM3", "USB Serial (GELLO)"),
        _FakePort("COM4", "DM CAN follower"),
    ]
    monkeypatch.setattr(reader_mod, "_list_comports", lambda: fake, raising=False)

    leader, follower = reader_mod._find_ports()
    assert leader == "COM3"
    assert follower == "COM4"


def test_find_ports_prefers_device_path(monkeypatch):
    """Linux tty paths are unambiguous and must win over description text, so
    role assignment never depends on enumeration order."""
    import g3a_bench.scripts.live_bridge.reader as reader_mod

    # Deliberately list the follower first to prove ordering doesn't matter.
    fake = [
        _FakePort("/dev/ttyACM0", "some adapter"),
        _FakePort("/dev/ttyUSB0", "some adapter"),
    ]
    monkeypatch.setattr(reader_mod, "_list_comports", lambda: fake, raising=False)

    leader, follower = reader_mod._find_ports()
    assert leader == "/dev/ttyUSB0"
    assert follower == "/dev/ttyACM0"


def test_find_ports_ftdi_vid_identifies_leader(monkeypatch):
    """The Windows-swap fix: when both ports report ambiguous "USB Serial
    Device" text, the FTDI vendor id pins the leader and the other port
    becomes the follower by elimination (instead of a silent swap)."""
    import g3a_bench.scripts.live_bridge.reader as reader_mod

    # Follower listed first, both with leader-colliding description text.
    fake = [
        _FakePort("COM7", "USB Serial Device", vid=0x0483),  # STM CDC follower
        _FakePort("COM5", "USB Serial Device", vid=reader_mod._FTDI_VID),  # leader
    ]
    monkeypatch.setattr(reader_mod, "_list_comports", lambda: fake, raising=False)

    leader, follower = reader_mod._find_ports()
    assert leader == "COM5"
    assert follower == "COM7"
