import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class Recorder:
    def __init__(
        self,
        repo_id: str,
        fps: int,
        obs_dim: int,
        root: str | None = None,
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
        self.dataset = LeRobotDataset.create(
            repo_id,
            fps=fps,
            features=features,
            root=root,
        )
        self._pending: list[dict] = []

    def add_frame(
        self,
        obs_buf: dict,
        action: np.ndarray,
        reward: float,
        done: bool,
        task: str = "",
        robot_obs: np.ndarray | None = None,
    ) -> None:
        # Buffer frames in memory until save_episode() is called with a label.
        # Nothing is written to the dataset until then, enabling discard/redo.
        if robot_obs is None:
            robot_obs = np.zeros(7, dtype=np.float32)
        self._pending.append(
            {
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
        )

    @property
    def num_frames_in_episode(self) -> int:
        return len(self._pending)

    def save_episode(self, success: bool) -> None:
        """Flush buffered frames stamped with the operator's label, then persist.

        No-op if nothing was buffered (avoids empty/garbage episodes).
        """
        if not self._pending:
            return
        label = np.array([success], dtype=bool)
        for frame in self._pending:
            frame["success"] = label
            self.dataset.add_frame(frame)
        self.dataset.save_episode()
        self._pending = []

    def discard_episode(self) -> None:
        """Drop the current episode without saving (operator rejected the demo)."""
        self._pending = []

    def close(self) -> None:
        self.dataset.finalize()
