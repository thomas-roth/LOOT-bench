import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
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

    def __init__(self, repo_id: str, episode_index: int = 0):
        # LeRobotDataset wraps a HuggingFace dataset under the hood.
        # Episodes are contiguous blocks of rows; we find boundaries once.
        self._ds = LeRobotDataset(repo_id)
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


# Substring hints (lowercase) used to classify enumerated serial ports.
# Leader = GELLO / Dynamixel USB-serial adapter (FTDI/U2D2 -> ttyUSB on Linux,
# usbserial on macOS). Follower = DM-motor CAN/CDC-ACM adapter (ttyACM on
# Linux, usbmodem on macOS).
_LEADER_PORT_HINTS = (
    "gello",
    "ftdi",
    "u2d2",
    "dynamixel",
    "usb-serial",
    "usb serial",
    "usbserial",
    "ttyusb",
)
_FOLLOWER_PORT_HINTS = ("can", "cdc", "ttyacm", "usbmodem")


def _list_comports() -> list:
    """Return available serial ports via pyserial (cross-platform).

    Wrapped in its own function so it can be monkeypatched in tests and so a
    missing/broken pyserial degrades to an empty list rather than raising.
    """
    try:
        from serial.tools import list_ports

        return list(list_ports.comports())
    except Exception:
        return []


def _port_text(port) -> str:
    parts = [
        getattr(port, "device", "") or "",
        getattr(port, "description", "") or "",
        getattr(port, "manufacturer", "") or "",
    ]
    return " ".join(parts).lower()


# USB vendor id that strongly identifies the leader regardless of OS-supplied
# description text. The GELLO/Dynamixel U2D2 adapter is an FTDI device; the
# DM-motor follower adapter is not, so an FTDI port is the leader.
_FTDI_VID = 0x0403


def _role_from_device_path(device: str) -> str | None:
    """Linux /dev names are unambiguous: ttyUSB*=leader, ttyACM*=follower."""
    d = device.lower()
    if "ttyusb" in d:
        return "leader"
    if "ttyacm" in d:
        return "follower"
    return None


def _find_ports() -> tuple[str | None, str | None]:
    """Auto-detect (leader, follower) serial ports, cross-platform.

    Classification is layered strongest-first so the result can't silently
    swap roles:

      1. Linux device path (``ttyUSB``=leader, ``ttyACM``=follower) — unambiguous.
      2. FTDI USB vendor id → leader (the U2D2/GELLO adapter is FTDI; the DM-motor
         follower adapter is not). This resolves Windows ``COMx`` ports where the
         driver description is ambiguous.
      3. Description/manufacturer substring hints (weak — a CDC follower often
         reports "USB Serial Device", which collides with the leader hints).
      4. Two-port elimination: if exactly one role is known and one port is left,
         the leftover fills the other role.

    When a role is resolved only by the weak steps (3/4), a warning is printed
    because that path can mis-assign on Windows — pass explicit ports to be sure.
    Falls back to Linux ``/dev`` globbing when enumeration yields nothing.
    """
    ports = _list_comports()
    if not ports:
        # Fallback: Linux /dev globbing (original behaviour).
        tty_usb = sorted(Path("/dev").glob("ttyUSB*"))
        tty_acm = sorted(Path("/dev").glob("ttyACM*"))
        leader = str(tty_usb[0]) if tty_usb else None
        follower = str(tty_acm[0]) if tty_acm else None
        return leader, follower

    leader: str | None = None
    follower: str | None = None
    unresolved: list = []
    used_weak = False

    # Pass 1 — strong signals: Linux device path, then FTDI vendor id (leader).
    for p in ports:
        device = getattr(p, "device", None)
        role = _role_from_device_path(device or "")
        if role is None and getattr(p, "vid", None) == _FTDI_VID:
            role = "leader"
        if role == "leader" and leader is None:
            leader = device
        elif role == "follower" and follower is None:
            follower = device
        else:
            unresolved.append(p)

    # Pass 2 — weak signals: description/manufacturer substring hints.
    leftover: list[str | None] = []
    for p in unresolved:
        device = getattr(p, "device", None)
        text = _port_text(p)
        if leader is None and any(h in text for h in _LEADER_PORT_HINTS):
            leader = device
            used_weak = True
        elif follower is None and any(h in text for h in _FOLLOWER_PORT_HINTS):
            follower = device
            used_weak = True
        else:
            leftover.append(device)

    # Pass 3 — two-port elimination for a single remaining unclassified port.
    if leader is None and follower is not None and len(leftover) == 1:
        leader = leftover[0]
        used_weak = True
    elif follower is None and leader is not None and len(leftover) == 1:
        follower = leftover[0]
        used_weak = True

    if used_weak:
        print(
            "[WARN] Serial ports auto-detected from ambiguous USB description "
            "text (no Linux tty path or FTDI vendor id). On Windows this can "
            "swap leader/follower — verify, or pass --follower-port / "
            "--leader-port explicitly."
        )
    return leader, follower


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
    ):
        if DK1Follower is None:
            raise ImportError(
                "lerobot_robot_trlc_dk1 not installed — cannot connect to real robot. "
                "Run: uv pip install -e /path/to/trlc-dk1"
            )
        assert DK1Follower is not None
        # Auto-detect any unspecified ports
        if follower_port is None or leader_port is None:
            auto_leader, auto_follower = _find_ports()
            leader_port = leader_port or auto_leader
            follower_port = follower_port or auto_follower
        self.follower_port = follower_port
        self.leader_port = leader_port
        self._follower: DK1Follower | None = None  # type: ignore
        self._leader: DK1Leader | None = None  # type: ignore
        if connect_on_init:
            self.connect()

    def connect(self):
        """Open serial connections to both leader and follower."""
        self._follower = DK1Follower(DK1FollowerConfig(port=self.follower_port))
        self._follower.connect()
        if self.leader_port:
            self._leader = DK1Leader(DK1LeaderConfig(port=self.leader_port))
            self._leader.connect()

    def read(self) -> ReaderFrame:
        if self._follower is None:
            raise RuntimeError("Follower not connected. Call .connect() first.")
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
        cameras = {}
        for cam_name in ("context", "wrist"):
            if cam_name in obs:
                cameras[cam_name] = obs[cam_name]
        # Get leader action if connected. get_action() returns a dict
        # {joint_1.pos..joint_6.pos, gripper.pos}; unpack in fixed order.
        leader_action = None
        if self._leader is not None:
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
