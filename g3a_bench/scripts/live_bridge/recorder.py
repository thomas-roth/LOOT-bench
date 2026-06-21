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
        }
        self.dataset = LeRobotDataset.create(
            repo_id,
            fps=fps,
            features=features,
            root=root,
        )

    def add_frame(
        self,
        obs_buf: dict,
        action: np.ndarray,
        reward: float,
        done: bool,
        task: str = "",
        robot_obs: np.ndarray | None = None,
    ) -> None:
        if robot_obs is None:
            robot_obs = np.zeros(7, dtype=np.float32)
        self.dataset.add_frame(
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

    def end_episode(self) -> None:
        self.dataset.save_episode()

    def close(self) -> None:
        self.dataset.finalize()
