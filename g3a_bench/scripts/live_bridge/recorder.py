import numpy as np
import torch
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
        self._repo_id = repo_id
        self._fps = fps
        if qpos_dim is not None:
            # Full sim qpos per frame — ground-truth state, enables exact
            # replay verification / state resync without re-recording.
            features["observation.qpos"] = {
                "dtype": "float32",
                "shape": (qpos_dim,),
                "names": None,
            }
        self._features = features
        self._success_root = success_root
        self._failure_root = failure_root

        # Datasets are created lazily on the first save_episode() of each kind,
        # so a session with only successes never leaves an empty failure dataset
        # on disk (and vice versa).
        self.dataset_success: LeRobotDataset | None = None
        self.dataset_failure: LeRobotDataset | None = None

        self._pending: list[dict] = []
        self._qpos_dim = qpos_dim
        self._device = device
        # Seeds and save-counts are tracked per dataset (success vs failure).
        # The success and failure datasets are indexed independently starting at
        # 0, and replay looks a seed up by that per-dataset --episode-index, so a
        # single shared counter mis-mapped seeds whenever the two were interleaved.
        self._episode_seeds: dict[bool, dict[str, int]] = {True: {}, False: {}}
        self._episodes_saved: dict[bool, int] = {True: 0, False: 0}

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
        reward: float | torch.Tensor,
        done: bool | torch.Tensor,
        task: str = "",
        robot_obs: np.ndarray | None = None,
        qpos: np.ndarray | torch.Tensor | None = None,
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
            "next.reward": reward,  # deferred — may be GPU tensor
            "next.done": done,  # deferred — may be GPU tensor
            "task": task,
        }
        if self._qpos_dim is not None:
            if qpos is None:
                qpos = np.zeros(self._qpos_dim, dtype=np.float32)
            # Store as-is: qpos may be an on-device torch tensor. The GPU→CPU
            # copy is deferred to save_episode() so the per-frame recording loop
            # pays no CUDA sync (that stall dropped live fps from 60 to 45).
            frame["observation.qpos"] = qpos
        self._pending.append(frame)

    @property
    def num_frames_in_episode(self) -> int:
        return len(self._pending)

    def save_episode(self, success: bool, seed: int | None = None) -> None:
        """Flush buffered frames stamped with the operator's label, then persist.

        Success episodes go to the success dataset, failures to the failure
        dataset. No-op if nothing was buffered (avoids empty/garbage episodes).
        ``seed`` is the RNG seed the env was reset with at this episode's start;
        it is persisted to info.json immediately (not at close) so seeds survive
        a crash, letting record_replay.py restore the exact object layout.
        """
        if not self._pending:
            return
        label = np.array([success], dtype=bool)
        dataset = self._get_dataset(success)
        for frame in self._pending:
            frame["success"] = label
            # Deferred host copy: convert GPU tensors to numpy here, off the hot
            # recording loop. This mirrors the pattern already used for qpos.
            for key in ("next.reward", "next.done", "observation.qpos"):
                val = frame[key]
                if hasattr(val, "detach"):
                    val = val.detach().cpu().numpy()
                if key == "next.reward":
                    val = np.array([float(val)], dtype=np.float32)
                elif key == "next.done":
                    val = np.array([bool(val)], dtype=bool)
                else:
                    val = np.asarray(val, dtype=np.float32)
                frame[key] = val
            dataset.add_frame(frame)
        dataset.save_episode()
        self._pending = []
        ep_index = self._episodes_saved[success]
        self._episodes_saved[success] += 1
        if seed is not None:
            seeds = self._episode_seeds[success]
            seeds[str(ep_index)] = int(seed)
            dataset.meta.info["episode_seeds"] = seeds
            if self._device is not None:
                # RNG streams differ between cuda and cpu — replay warns on mismatch.
                dataset.meta.info["seed_device"] = self._device.split(":")[0]
            write_info(dataset.meta.info, dataset.meta.root)

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
