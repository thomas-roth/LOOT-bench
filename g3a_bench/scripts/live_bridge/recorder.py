import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from g3a_bench.scripts.live_bridge.dataset_extra_info import write_extra_info


class Recorder:
    def __init__(
        self,
        repo_id: str,
        fps: int,
        obs_dim: int,
        success_root: str | None = None,
        failure_root: str | None = None,
        qpos_dim: int | None = None,
        qvel_dim: int | None = None,
        device: str | None = None,
        gravity_compensate: bool = True,
        start_pose_noise: float | None = None,
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
            # Full sim qpos per frame for exact replay and resync.
            features["observation.qpos"] = {
                "dtype": "float32",
                "shape": (qpos_dim,),
                "names": None,
            }
            if qvel_dim is None:
                qvel_dim = qpos_dim
            # Full sim qvel per frame for state-set replay (no drift).
            features["observation.qvel"] = {
                "dtype": "float32",
                "shape": (qvel_dim,),
                "names": None,
            }
        self._features = features
        self._success_root = success_root
        self._failure_root = failure_root

        # Datasets are created lazily, so empty kinds leave no files.
        self.dataset_success: LeRobotDataset | None = None
        self.dataset_failure: LeRobotDataset | None = None

        self._pending: list[dict] = []
        # Last saved episode, staged for single-level undo.
        self._staged: tuple[list[dict], bool, int | None] | None = None
        self._qpos_dim = qpos_dim
        self._qvel_dim = qvel_dim
        self._device = device
        self._gravity_compensate = gravity_compensate
        # Start-pose half-width [rad] used for recording (checked on load).
        self._start_pose_noise = start_pose_noise
        # Seeds tracked per dataset (success/failure indexed from 0).
        self._episode_seeds: dict[bool, dict[str, int]] = {True: {}, False: {}}
        self._episodes_saved: dict[bool, int] = {True: 0, False: 0}
        # Extra metadata kept in a sidecar (outside lerobot schema).
        self._extra_info: dict[bool, dict] = {True: {}, False: {}}

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
        qvel: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        # Buffer frames in memory until save_episode() is called with a label.
        # Nothing is written to the dataset until then, enabling discard/redo.
        if robot_obs is None:
            robot_obs = np.zeros(7, dtype=np.float32)
        # Clone deferred GPU tensors now; buffers are reused next step.
        if hasattr(reward, "detach"):
            reward = reward.detach().clone()
        if hasattr(done, "detach"):
            done = done.detach().clone()
        frame = {
            "observation.state": obs_buf["actor"][0]
            .cpu()
            .numpy()
            .flatten()
            .astype(np.float32),
            "observation.robot_state": np.asarray(robot_obs, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "next.reward": reward,  # deferred, may be GPU tensor
            "next.done": done,  # deferred, may be GPU tensor
            "task": task,
        }
        if self._qpos_dim is not None:
            if qpos is None:
                qpos = np.zeros(self._qpos_dim, dtype=np.float32)
            # Clone qpos; sim buffer is reused next step.
            qpos = qpos.detach().clone() if hasattr(qpos, "detach") else qpos.copy()
            frame["observation.qpos"] = qpos
        if self._qvel_dim is not None:
            if qvel is None:
                qvel = np.zeros(self._qvel_dim, dtype=np.float32)
            qvel = qvel.detach().clone() if hasattr(qvel, "detach") else qvel.copy()
            frame["observation.qvel"] = qvel
        self._pending.append(frame)

    @property
    def num_frames_in_episode(self) -> int:
        return len(self._pending)

    def save_episode(self, success: bool, seed: int | None = None) -> None:
        """Flush buffered frames with label; seed restores replay layout."""
        if not self._pending:
            return
        # Take ownership of pending frames before committing staged one.
        frames = self._pending
        self._pending = []
        self._commit_staged()
        self._staged = (frames, success, seed)

    @property
    def has_staged(self) -> bool:
        return self._staged is not None

    def undo_last_save(self) -> bool | None:
        """Drop staged episode; returns its label or None."""
        if self._staged is None:
            return None
        _, success, _ = self._staged
        self._staged = None
        return success

    def _commit_staged(self) -> None:
        """Write the staged episode to its dataset (point of no return)."""
        if self._staged is None:
            return
        frames, success, seed = self._staged
        self._staged = None
        label = np.array([success], dtype=bool)
        dataset = self._get_dataset(success)
        for frame in frames:
            frame["success"] = label
            # Deferred host copy: convert GPU tensors to numpy here, off the hot
            # recording loop. This mirrors the pattern already used for qpos.
            for key in (
                "next.reward",
                "next.done",
                "observation.qpos",
                "observation.qvel",
            ):
                if key not in frame:
                    continue
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
        ep_index = self._episodes_saved[success]
        self._episodes_saved[success] += 1
        extra = self._extra_info[success]
        extra["gravity_compensate"] = self._gravity_compensate
        if self._start_pose_noise is not None:
            extra["start_pose_noise"] = self._start_pose_noise
        if seed is not None:
            seeds = self._episode_seeds[success]
            seeds[str(ep_index)] = int(seed)
            extra["episode_seeds"] = seeds
            if self._device is not None:
                # RNG streams differ between cuda and cpu; replay warns on mismatch.
                extra["seed_device"] = self._device.split(":")[0]
        write_extra_info(dataset.meta.root, extra)

    def discard_episode(self) -> None:
        """Drop the current episode without saving (operator rejected the demo)."""
        self._pending = []

    def close(self, operator_id: int | None = None) -> None:
        self._commit_staged()
        for success, dataset in (
            (True, self.dataset_success),
            (False, self.dataset_failure),
        ):
            if dataset is None:
                continue
            if operator_id is not None:
                extra = self._extra_info[success]
                extra["operator_id"] = operator_id
                write_extra_info(dataset.meta.root, extra)
            dataset.finalize()
