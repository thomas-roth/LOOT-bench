import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Hardware-specific imports — only needed for LiveReader (real robot).
# Protected by try/except so the module can be imported on machines
# without the trlc-dk1 hardware libraries.
try:
    from lerobot_robot_trlc_dk1.follower import (  # type: ignore[import]
        DK1Follower,
        DK1FollowerConfig,
    )
    from lerobot_robot_trlc_dk1.leader import (  # type: ignore[import]
        DK1Leader,
        DK1LeaderConfig,
    )
except ImportError:
    DK1Follower: Any = None
    DK1FollowerConfig: Any = None
    DK1Leader: Any = None
    DK1LeaderConfig: Any = None


@dataclass
class ReaderFrame:
    """Uniform data envelope for one timestep from any source.

    Every Reader subclass returns this same shape regardless of whether
    the source is a live robot or a saved dataset.  Downstream code
    (Mapper, Simulator, Recorder) never needs to know where data came from.
    """

    # (7,) [joint_1..joint_6, gripper] in real-robot format (radians, gripper 0=open..1=closed)
    joint_pos: np.ndarray
    # (7,) same format as joint_pos — what the human commanded via leader arm
    leader_action: np.ndarray | None = None
    # dict of camera_name → raw image as (H, W, 3) uint8 numpy array
    cameras: dict[str, np.ndarray] = field(default_factory=dict)
    # seconds since epoch
    timestamp: float = 0.0


class Reader(ABC):
    """Abstract base class for all data sources.

    Concrete subclasses: LiveReader (real robot), DatasetReader (file).
    The rest of the bridge calls .read() and never queries the type.
    """

    @abstractmethod
    def read(self) -> ReaderFrame:
        """Return the next frame, or raise StopIteration when exhausted."""

    @abstractmethod
    def close(self):
        """Release any resources (serial ports, file handles)."""


class DatasetReader(Reader):
    """Reads frames from a saved LeRobotDataset (offline replay mode).

    Useful for testing the bridge without hardware — feed it a dataset
    and the rest of the pipeline processes it identically.
    """

    def __init__(self, repo_id: str, episode_index: int = 0, root: str | None = None):
        # LeRobotDataset wraps a HuggingFace dataset under the hood.
        # Episodes are contiguous blocks of rows; we find boundaries once.
        self._ds = LeRobotDataset(repo_id, root=root)
        self._episode_index = episode_index
        self._frame_index = 0
        self._load_episode()

    def _load_episode(self):
        """Find the start and end indices for the target episode.

        LeRobot datasets store all episodes contiguously in a single
        HuggingFace dataset table, with an episode_index column marking
        which episode each row belongs to.
        """
        import numpy as np

        ep_idxs = np.array(self._ds.hf_dataset["episode_index"])
        # Find boundaries by detecting where episode_index changes
        boundaries = np.where(ep_idxs[:-1] != ep_idxs[1:])[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(ep_idxs)]])
        self._start = int(starts[self._episode_index])
        self._end = int(ends[self._episode_index])
        self._frame_index = 0

    def read(self) -> ReaderFrame:
        if self._frame_index >= (self._end - self._start):
            raise StopIteration
        idx = self._start + self._frame_index
        self._frame_index += 1
        frame = self._ds[idx]
        return ReaderFrame(
            joint_pos=np.asarray(frame["observation.state"], dtype=np.float32),
            leader_action=np.asarray(frame["action"], dtype=np.float32),
            timestamp=float(frame["timestamp"]),
        )

    def close(self):
        pass


class LiveReader(Reader):
    """Reads frames from a real DK1 robot in real-time.

    Connects to DK1Follower (the physical arm, for observations)
    and optionally DK1Leader (the GELLO controller, for commanded actions).
    Ports can be specified explicitly or auto-detected.
    """

    def __init__(
        self,
        follower_port: str | None = None,
        leader_port: str | None = None,
        connect_on_init: bool = True,
        use_follower: bool = False,
    ):
        if DK1Leader is None:
            raise ImportError(
                "lerobot_robot_trlc_dk1 not installed — cannot connect to real robot. "
                "Run: uv pip install -e /path/to/trlc-dk1"
            )
        import sys

        self.use_follower = use_follower
        if use_follower:
            assert DK1Follower is not None
            # Only provide Linux defaults; require explicit ports elsewhere.
            if follower_port is None or leader_port is None:
                if sys.platform.startswith("linux"):
                    leader_port = leader_port or "/dev/ttyACM0"
                    follower_port = follower_port or "/dev/ttyACM1"
                else:
                    raise ValueError(
                        "leader_port/follower_port must be provided when not on Linux"
                    )
        else:
            if leader_port is None:
                if sys.platform.startswith("linux"):
                    leader_port = "/dev/ttyACM0"
                else:
                    raise ValueError("leader_port must be provided when not on Linux")
            follower_port = None
        self.follower_port = follower_port
        self.leader_port = leader_port
        self._follower: DK1Follower | None = None  # type: ignore
        self._leader: DK1Leader | None = None  # type: ignore
        if connect_on_init:
            self.connect()

    def connect(self):
        """Open serial connection(s) to the leader, and the follower if enabled."""
        if self.use_follower:
            self._follower = DK1Follower(DK1FollowerConfig(port=self.follower_port))
            self._follower.connect()
        if self.leader_port:
            self._leader = DK1Leader(DK1LeaderConfig(port=self.leader_port))
            self._leader.connect()

    def read(self) -> ReaderFrame:
        if self.use_follower and self._follower is None:
            raise RuntimeError("Follower not connected. Call .connect() first.")
        if self._leader is None:
            raise RuntimeError("Leader not connected. Call .connect() first.")

        cameras: dict[str, np.ndarray] = {}
        joint_pos: np.ndarray | None = None
        if self._follower is not None:
            obs = self._follower.get_observation()
            # Build joint_pos array in standard real-robot order
            joint_pos = np.array(
                [
                    obs["joint_1.pos"],
                    obs["joint_2.pos"],
                    obs["joint_3.pos"],
                    obs["joint_4.pos"],
                    obs["joint_5.pos"],
                    obs["joint_6.pos"],
                    obs["gripper.pos"],
                ],
                dtype=np.float32,
            )
            # Extract camera images if available
            for cam_name in ("context", "wrist"):
                if cam_name in obs:
                    cameras[cam_name] = obs[cam_name]

        # get_action() returns a dict {joint_1.pos..joint_6.pos, gripper.pos};
        # unpack in fixed order.
        la = self._leader.get_action()
        leader_action = np.array(
            [
                la["joint_1.pos"],
                la["joint_2.pos"],
                la["joint_3.pos"],
                la["joint_4.pos"],
                la["joint_5.pos"],
                la["joint_6.pos"],
                la["gripper.pos"],
            ],
            dtype=np.float32,
        )

        if joint_pos is None:
            # No follower connected — the leader's own reading is the only
            # physical arm state available, so it doubles as joint_pos.
            joint_pos = leader_action.copy()

        return ReaderFrame(
            joint_pos=joint_pos,
            leader_action=leader_action,
            cameras=cameras,
            timestamp=time.time(),
        )

    def send_action(self, action_np: np.ndarray) -> dict[str, Any]:
        """Send a (7,) numpy action to the real follower arm.

        Converts the standard bridge format (numpy array of
        [joint_1..joint_6, gripper]) to the dict format expected by
        DK1Follower.send_action() and sends it.

        Args:
            action_np: (7,) array — joints in radians, gripper in [0, 1].
        Returns:
            The action dict that was sent to the follower.
        """
        if self._follower is None:
            raise RuntimeError("Follower not connected. Call .connect() first.")
        action_dict = {
            "joint_1.pos": float(action_np[0]),
            "joint_2.pos": float(action_np[1]),
            "joint_3.pos": float(action_np[2]),
            "joint_4.pos": float(action_np[3]),
            "joint_5.pos": float(action_np[4]),
            "joint_6.pos": float(action_np[5]),
            "gripper.pos": float(action_np[6]),
        }
        return self._follower.send_action(action_dict)

    def close(self):
        if self._follower is not None:
            self._follower.disconnect()
        if self._leader is not None:
            self._leader.disconnect()
