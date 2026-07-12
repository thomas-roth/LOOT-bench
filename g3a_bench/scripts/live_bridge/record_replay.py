"""Replay LeRobot dataset through simulation, show in native viewer,
and save a recorded LeRobot dataset to logs/.

Usage:
    # replay a locally recorded session (from live_record.py)
    uv run python g3a_bench/scripts/live_bridge/record_replay.py \\
        --dataset-dir out/demos/success/2026-06-29_10-00-00

    # replay a specific episode
    uv run python g3a_bench/scripts/live_bridge/record_replay.py \\
        --dataset-dir ... --episode-index 2

    # replay the default HuggingFace dataset (no --dataset-dir)
    uv run python g3a_bench/scripts/live_bridge/record_replay.py

The native MuJoCo window opens and replays the dataset through the sim.
Output is saved to <dataset-dir>/sim_replay/episode_<N>/success/ (local) or
logs/sim_replay_cube_stack3/episode_<N>/success/ (HuggingFace).
After the episode finishes, the viewer keeps running (hold-last-action mode)
until you close the window or press Ctrl+C.
"""

# Load lerobot/HF-datasets native libs before mjlab/mujoco-warp to avoid a
# native import-order segfault (HF `datasets` vs mujoco-warp). Must precede mjlab.
import lerobot.datasets.lerobot_dataset  # noqa: F401,E402,I001

import argparse
import shutil
import traceback
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import VerbosityLevel

import g3a_bench  # noqa: F401
from g3a_bench.scripts.live_bridge.live_record import apply_teleop_contact_stability
from g3a_bench.scripts.live_bridge.mapper import lerobot_to_mjlab_raw_action
from g3a_bench.scripts.live_bridge.reader import DatasetReader, ReaderFrame
from g3a_bench.scripts.live_bridge.recorder import Recorder

LOGS_DIR = Path(__file__).resolve().parent / "logs"


def main():
    parser = argparse.ArgumentParser(
        description="Replay a LeRobot dataset through the sim and record results"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help=(
            "Path to a local live_record dataset directory "
            "(e.g. out/demos/success/2026-06-29_10-00-00). "
            "When omitted, replays the default HuggingFace dataset."
        ),
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Which episode to replay (default: 0).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="G3a-Cube-Stack3-Dk1_uni",
        help="MJLab task name to load.",
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1. Load task config (play mode: infinite, no noise).
    cfg = load_env_cfg(args.task, play=True)
    cfg.terminations["time_out"].time_out = False
    # Drop ee_ground_collision: it survives the play-mode termination pop, and a
    # single >10 N gripper-vs-ground contact in the replayed trajectory would
    # auto-reset the env mid-episode (auto_reset=True) and corrupt the recording.
    cfg.terminations.pop("ee_ground_collision", None)
    cfg.terminations.pop(
        "success", None
    )  # replay runs through the full episode uninterrupted
    # Same finger-pad solref stabilization live_record applies — without it the
    # replayed contact physics differs from the recording at first grip.
    apply_teleop_contact_stability(cfg)

    # 2. Create environment.
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env = RslRlVecEnvWrapper(env)
    print("[INFO] Environment created")

    # 3. Open dataset to replay.
    if args.dataset_dir:
        # Local session recorded by live_record.py (repo_id is always "live_record").
        # Sessions store the actual LeRobot datasets under success/ and failure/.
        repo_id_in = "live_record"
        root_path = Path(args.dataset_dir)
        success_root = root_path / "success"
        root_in = str(success_root) if success_root.exists() else args.dataset_dir
        repo_id_out = "sim_replay"
        out_root = str(root_path / "sim_replay")
    else:
        repo_id_in = "FabianKerj/corkinbox100-fullrel"
        root_in = None
        repo_id_out = "sim_replay_cube_stack3"
        out_root = str(LOGS_DIR / repo_id_out)

    # Isolate each episode's replay output. Without this, replaying a second
    # episode (or re-running one) collides: LeRobotDataset.create does
    # mkdir(exist_ok=False) and raises FileExistsError on the existing root.
    # sim_replay is regenerated output, so a stale run is safe to overwrite.
    out_root = str(Path(out_root) / f"episode_{args.episode_index}")
    if Path(out_root).exists():
        print(f"[WARN] Overwriting existing replay output at {out_root}")
        shutil.rmtree(out_root)

    reader = DatasetReader(repo_id_in, root=root_in, episode_index=args.episode_index)
    total_frames = reader._end - reader._start
    print(f"[INFO] Dataset opened — episode has {total_frames} frames")

    # 3.5 Restore the per-episode seed saved by live_record.py: a seeded reset
    #     re-runs the same RNG draws, so the object scatter and the sampled
    #     goal come out exactly as they were during recording — for any task.
    episode_seed: int | None = None
    seed_val = reader.meta_info.get("episode_seeds", {}).get(str(args.episode_index))
    if seed_val is not None:
        episode_seed = int(seed_val)
        rec_device = reader.meta_info.get("seed_device")
        if rec_device is not None and rec_device != device.split(":")[0]:
            print(
                f"[WARN] Dataset was recorded on '{rec_device}' but replaying on "
                f"'{device.split(':')[0]}' — torch RNG streams differ per device, "
                "object positions may not match the recording."
            )
        print(f"[INFO] Restoring recorded episode seed {episode_seed}")
        env.env.reset(seed=episode_seed)
    else:
        print(
            "[WARN] Dataset has no recorded seed for this episode — object "
            "positions will NOT match the recording. Re-record with the "
            "updated live_record.py to get reproducible replays."
        )

    # 4. Create recorder.
    # Derive the sim observation width from the env rather than hardcoding it,
    # so a different task (with a different actor-obs size) still records a
    # correctly-shaped dataset.
    obs_dim = int(env.get_observations()["actor"].shape[-1])
    qpos_dim = int(env.env.sim.data.qpos.shape[-1])
    recorder = Recorder(
        repo_id=repo_id_out,
        fps=50,
        obs_dim=obs_dim,
        success_root=str(Path(out_root) / "success"),
        failure_root=str(Path(out_root) / "failure"),
        qpos_dim=qpos_dim,
        device=device,
    )
    print(f"[INFO] Recording to {out_root}")

    # 5. Policy that reads dataset + records results.
    replay = _ReplayAndRecord(
        reader, recorder, total_frames, device, task=args.task, seed=episode_seed
    )

    # 6. Launch native viewer with recording.
    print("[INFO] Starting Native MuJoCo viewer (recording enabled)")
    print("[INFO] Close the viewer window or press Ctrl+C to stop")
    viewer = _RecordingViewer(env=env, policy=replay)
    viewer.run()

    # 7. Cleanup.
    replay.finalize()
    env.close()
    reader.close()
    print(f"[INFO] Done. Dataset saved to {out_root}")


class _ReplayAndRecord:
    """Acts as the viewer's policy AND manages recording.

    Flow per step:
      1. __call__(obs) is invoked by the viewer with current obs
         → reads next dataset frame → maps to raw_action → returns tensor
      2. viewer calls env.step(raw_action) internally
      3. record_step_result(rew, done) is called with step results
    """

    def __init__(
        self,
        reader: DatasetReader,
        recorder: Recorder,
        total_frames: int,
        device: str,
        task: str = "",
        seed: int | None = None,
    ):
        self.reader = reader
        self.recorder = recorder
        self.total_frames = total_frames
        self.device = device
        self.task = task
        self.seed = seed
        self.frames_recorded = 0
        self.recording_finished = False

        self._last_obs = None
        self._last_action_np = None
        self._last_robot_obs = None
        self._last_state = None
        self._fallback_action_np = None

        # Drift vs the original recording: for locally recorded sessions the
        # dataset's observation.state is the sim actor obs, so per frame we can
        # compare it with the replayed obs. Quantifies replay faithfulness
        # (sim nondeterminism / open-loop divergence) as a printed number.
        self._drift_sum = 0.0
        self._drift_max = 0.0
        self._drift_frames = 0

    def __call__(self, obs):
        self._last_obs = obs

        # Read next dataset frame.
        try:
            frame = self.reader.read()
        except StopIteration:
            if not self.recording_finished:
                print(
                    f"[INFO] Dataset replay complete after "
                    f"{self.frames_recorded} frames — viewer keeps running"
                )
                self._save_episode()
                self.recording_finished = True
            # Keep the viewer alive by repeating the last action.
            frame = ReaderFrame(
                joint_pos=np.zeros(7, dtype=np.float32),
                leader_action=(
                    self._fallback_action_np
                    if self._fallback_action_np is not None
                    else np.zeros(7, dtype=np.float32)
                ),
                timestamp=0.0,
            )

        if frame.leader_action is None:
            raise RuntimeError("DatasetReader frame is missing an action")
        self._last_action_np = frame.leader_action
        self._last_robot_obs = frame.joint_pos
        self._last_state = frame.state
        if self._fallback_action_np is None:
            self._fallback_action_np = frame.leader_action.copy()

        raw_action = lerobot_to_mjlab_raw_action(frame.leader_action, True)
        return torch.from_numpy(raw_action).unsqueeze(0).to(self.device)

    def record_step_result(
        self, rew: float, done: bool, qpos: np.ndarray | None = None
    ):
        """Called by the custom viewer after env.step()."""
        if self.recording_finished:
            return
        if self._last_obs is None or self._last_action_np is None:
            return

        replayed_obs = self._last_obs["actor"][0].cpu().numpy().flatten()
        recorded_obs = self._last_state
        if recorded_obs is not None and recorded_obs.shape == replayed_obs.shape:
            drift = float(np.max(np.abs(replayed_obs - recorded_obs)))
            self._drift_sum += drift
            self._drift_max = max(self._drift_max, drift)
            self._drift_frames += 1

        self.recorder.add_frame(
            {"actor": self._last_obs["actor"]},
            self._last_action_np,
            rew,
            done,
            task=self.task,
            robot_obs=self._last_robot_obs,
            qpos=qpos,
        )
        self.frames_recorded += 1
        if self.frames_recorded % 100 == 0:
            print(f"[INFO] Recorded {self.frames_recorded}/{self.total_frames} frames")

        if self.frames_recorded >= self.total_frames and not self.recording_finished:
            print(
                f"[INFO] All {self.total_frames} frames recorded — finalizing dataset"
            )
            self._save_episode()
            self.recording_finished = True

    def _save_episode(self):
        self.recorder.save_episode(success=True, seed=self.seed)
        self._report_drift()

    def _report_drift(self):
        if self._drift_frames == 0:
            return
        print(
            f"[INFO] Replay drift vs recording (max-abs obs deviation per frame, "
            f"{self._drift_frames} frames): "
            f"mean {self._drift_sum / self._drift_frames:.5f}, "
            f"max {self._drift_max:.5f}"
        )

    def finalize(self):
        if not self.recording_finished:
            self._save_episode()
        self.recorder.close()


class _RecordingViewer(NativeMujocoViewer):
    """NativeMujocoViewer that records obs/action/reward/done per step."""

    def __init__(self, env, policy, **kwargs):
        super().__init__(
            env,
            policy,
            verbosity=VerbosityLevel.DEBUG,
            **kwargs,
        )

    def _execute_step(self) -> bool:
        """Override base to capture step results for recording.

        Base viewer calls: obs → policy(obs) → env.step(actions)
        without capturing return values. We need reward and done.
        """
        try:
            with torch.no_grad():
                obs = self.env.get_observations()
                actions = self.policy(obs)
                _obs_next, rew, dones, _extras = self.env.step(actions)
                # Capture qpos AFTER step so it aligns with the recorded reward
                # and done (same timing as live_record.py's _LiveRecordingViewer).
                qpos_np = self.env.env.sim.data.qpos[0].cpu().numpy().copy()

                # Notify the recording policy with step results.
                record_fn = getattr(self.policy, "record_step_result", None)
                if record_fn is not None:
                    record_fn(float(rew[0]), bool(dones[0]), qpos_np)

                self._step_count += 1
                self._stats_steps += 1
                return True
        except Exception:
            self._last_error = traceback.format_exc()
            self.log(
                f"[ERROR] Exception during step:\n{self._last_error}",
                VerbosityLevel.SILENT,
            )
            self.pause()
            return False


if __name__ == "__main__":
    main()
