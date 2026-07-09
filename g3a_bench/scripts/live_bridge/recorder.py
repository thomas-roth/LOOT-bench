import numpy as np
from lerobot.datasets.dataset_metadata import write_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class Recorder:
    def __init__(
        self,
        repo_id: str,
        fps: int,
        obs_dim: int,
        success_root: str | None = None,
        failure_root: str | None = None,
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
        self._repo_id = repo_id
        self._fps = fps
        self._features = features
        self._success_root = success_root
        self._failure_root = failure_root

        # Datasets are created lazily on the first save_episode() of each kind,
        # so a session with only successes never leaves an empty failure dataset
        # on disk (and vice versa).
        self.dataset_success: LeRobotDataset | None = None
        self.dataset_failure: LeRobotDataset | None = None
        self._pending: list[dict] = []

    def _get_dataset(self, success: bool) -> LeRobotDataset:
        if success:
            if self.dataset_success is None:
                self.dataset_success = LeRobotDataset.create(
                    self._repo_id,
                    fps=self._fps,
                    features=self._features,
                    root=self._success_root,
                )
            return self.dataset_success
        if self.dataset_failure is None:
            self.dataset_failure = LeRobotDataset.create(
                self._repo_id,
                fps=self._fps,
                features=self._features,
                root=self._failure_root,
            )
        return self.dataset_failure

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

        Success episodes go to the success dataset, failures to the failure
        dataset. No-op if nothing was buffered (avoids empty/garbage episodes).
        """
        if not self._pending:
            return
        label = np.array([success], dtype=bool)
        dataset = self._get_dataset(success)
        for frame in self._pending:
            frame["success"] = label
            dataset.add_frame(frame)
        dataset.save_episode()
        self._pending = []

    def discard_episode(self) -> None:
        """Drop the current episode without saving (operator rejected the demo)."""
        self._pending = []

    def close(self, operator_id: int | None = None) -> None:
        for dataset in (self.dataset_success, self.dataset_failure):
            if dataset is None:
                continue
            if operator_id is not None:
                dataset.meta.info["operator_id"] = operator_id
                write_info(dataset.meta.info, dataset.meta.root)
            dataset.finalize()
