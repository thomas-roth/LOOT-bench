import numpy as np
from lerobot.datasets.dataset_metadata import write_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class Recorder:
    def __init__(
        self,
        repo_id: str,
        fps: int,
        obs_dim: int,
        root: str | None = None,
        qpos_dim: int | None = None,
        device: str | None = None,
    ):
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (obs_dim,),
                "names": None,
            },
            "observation.robot_state": {
                "dtype": "float32",
                "shape": (7,),
                "names": None,
            },
            "action": {"dtype": "float32", "shape": (7,), "names": None},
            "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
            "next.done": {"dtype": "bool", "shape": (1,), "names": None},
            "success": {"dtype": "bool", "shape": (1,), "names": None},
        }
        if qpos_dim is not None:
            # Full sim qpos per frame — ground-truth state, enables exact
            # replay verification / state resync without re-recording.
            features["observation.qpos"] = {
                "dtype": "float32",
                "shape": (qpos_dim,),
                "names": None,
            }
        self.dataset = LeRobotDataset.create(
            repo_id,
            fps=fps,
            features=features,
            root=root,
        )
        self._pending: list[dict] = []
        self._qpos_dim = qpos_dim
        self._device = device
        self._episode_seeds: dict[str, int] = {}
        self._episodes_saved = 0

    def add_frame(
        self,
        obs_buf: dict,
        action: np.ndarray,
        reward: float,
        done: bool,
        task: str = "",
        robot_obs: np.ndarray | None = None,
        qpos: np.ndarray | None = None,
    ) -> None:
        # Buffer frames in memory until save_episode() is called with a label.
        # Nothing is written to the dataset until then, enabling discard/redo.
        if robot_obs is None:
            robot_obs = np.zeros(7, dtype=np.float32)
        frame = {
            "observation.state": obs_buf["actor"][0]
            .cpu()
            .numpy()
            .flatten()
            .astype(np.float32),
            "observation.robot_state": np.asarray(robot_obs, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "next.reward": np.array([reward], dtype=np.float32),
            "next.done": np.array([done], dtype=bool),
            "task": task,
        }
        if self._qpos_dim is not None:
            if qpos is None:
                qpos = np.zeros(self._qpos_dim, dtype=np.float32)
            frame["observation.qpos"] = np.asarray(qpos, dtype=np.float32)
        self._pending.append(frame)

    @property
    def num_frames_in_episode(self) -> int:
        return len(self._pending)

    def save_episode(self, success: bool, seed: int | None = None) -> None:
        """Flush buffered frames stamped with the operator's label, then persist.

        No-op if nothing was buffered (avoids empty/garbage episodes).
        ``seed`` is the RNG seed the env was reset with at this episode's start;
        it is persisted to info.json immediately (not at close) so seeds survive
        a crash, letting record_replay.py restore the exact object layout.
        """
        if not self._pending:
            return
        label = np.array([success], dtype=bool)
        for frame in self._pending:
            frame["success"] = label
            self.dataset.add_frame(frame)
        self.dataset.save_episode()
        self._pending = []
        ep_index = self._episodes_saved
        self._episodes_saved += 1
        if seed is not None:
            self._episode_seeds[str(ep_index)] = int(seed)
            self.dataset.meta.info["episode_seeds"] = self._episode_seeds
            if self._device is not None:
                # RNG streams differ between cuda and cpu — replay warns on mismatch.
                self.dataset.meta.info["seed_device"] = self._device.split(":")[0]
            write_info(self.dataset.meta.info, self.dataset.meta.root)

    def discard_episode(self) -> None:
        """Drop the current episode without saving (operator rejected the demo)."""
        self._pending = []

    def close(self, operator_id: int | None = None) -> None:
        if operator_id is not None:
            self.dataset.meta.info["operator_id"] = operator_id
            write_info(self.dataset.meta.info, self.dataset.meta.root)
        self.dataset.finalize()
