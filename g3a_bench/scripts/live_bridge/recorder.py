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
        qvel_dim: int | None = None,
        device: str | None = None,
        gravity_compensate: bool = True,
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
            if qvel_dim is None:
                qvel_dim = qpos_dim
            # Full sim qvel per frame — needed for state-set replay (set qpos+
            # qvel each frame, call mj_forward, render — no physics integration,
            # no drift). Note: qvel may differ from qpos when the model has
            # free/ball joints (quaternions in qpos vs angular velocities in
            # qvel), so caller must provide the actual dim.
            features["observation.qvel"] = {
                "dtype": "float32",
                "shape": (qvel_dim,),
                "names": None,
            }
        self._features = features
        self._success_root = success_root
        self._failure_root = failure_root

        # Datasets are created lazily on the first committed episode of each
        # kind, so a session with only successes never leaves an empty failure
        # dataset on disk (and vice versa).
        self.dataset_success: LeRobotDataset | None = None
        self.dataset_failure: LeRobotDataset | None = None

        self._pending: list[dict] = []
        # Last saved episode (frames, label, seed), held back for single-level
        # undo. Committed to the dataset when the next episode is saved or at
        # close(). The seed travels with the frames so it is written under the
        # correct episode index at commit time.
        self._staged: tuple[list[dict], bool, int | None] | None = None
        self._qpos_dim = qpos_dim
        self._qvel_dim = qvel_dim
        self._device = device
        self._gravity_compensate = gravity_compensate
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
        qvel: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        # Buffer frames in memory until save_episode() is called with a label.
        # Nothing is written to the dataset until then, enabling discard/redo.
        if robot_obs is None:
            robot_obs = np.zeros(7, dtype=np.float32)
        # Deferred GPU tensors must be cloned NOW: mjlab reuses its
        # reward/done buffers every step, so storing the view would make every
        # buffered frame read the episode's *final* value at save time (this
        # silently flattened next.reward to a constant per episode). clone()
        # stays on-device — still no per-step CUDA sync.
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
            "next.reward": reward,  # deferred — may be GPU tensor
            "next.done": done,  # deferred — may be GPU tensor
            "task": task,
        }
        if self._qpos_dim is not None:
            if qpos is None:
                qpos = np.zeros(self._qpos_dim, dtype=np.float32)
            # Clone to avoid aliasing: MuJoCo/DM control reuse their internal
            # qpos buffer across steps, so storing the view would make every
            # buffered frame read the episode's *final* qpos at save time
            # (same class of bug that flattened next.reward to a constant).
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
        """Flush buffered frames stamped with the operator's label, then persist.

        Success episodes go to the success dataset, failures to the failure
        dataset. No-op if nothing was buffered (avoids empty/garbage episodes).
        ``seed`` is the RNG seed the env was reset with at this episode's start;
        it is staged with the frames and written to info.json when the episode
        is committed (next save or close), letting record_replay.py restore the
        exact object layout.
        """
        if not self._pending:
            return
        # Take ownership of the pending frames BEFORE committing: commit code
        # must never touch self._pending, and staging after grabbing the list
        # keeps the new episode safe even if commit misbehaves.
        frames = self._pending
        self._pending = []
        self._commit_staged()
        self._staged = (frames, success, seed)

    @property
    def has_staged(self) -> bool:
        return self._staged is not None

    def undo_last_save(self) -> bool | None:
        """Drop the staged (last saved, not yet committed) episode.

        Returns the dropped episode's label, or None if nothing was staged
        (undo is single-level: only the most recent save can be taken back).
        """
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
        dataset.meta.info["gravity_compensate"] = self._gravity_compensate
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
        self._commit_staged()
        for dataset in (self.dataset_success, self.dataset_failure):
            if dataset is None:
                continue
            if operator_id is not None:
                dataset.meta.info["operator_id"] = operator_id
                write_info(dataset.meta.info, dataset.meta.root)
            dataset.finalize()
