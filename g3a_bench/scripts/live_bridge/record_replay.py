"""Replay LeRobot dataset through simulation, show in native viewer,
and save a recorded LeRobot dataset to logs/.

Usage:
    cd /home/minhtrung/Documents/UNI/G3a/g3a-bench
    uv run python g3a_bench/scripts/live_bridge/record_replay.py

The native MuJoCo window opens and replays the dataset through the sim.
All obs/action/reward/done are recorded to logs/sim_replay_cube_stack3/.
After the episode finishes, the viewer keeps running (hold-last-action mode)
until you close the window or press Ctrl+C.
"""

# Load lerobot/HF-datasets native libs before mjlab/mujoco-warp to avoid a
# native import-order segfault (HF `datasets` vs mujoco-warp). Must precede mjlab.
import lerobot.datasets.lerobot_dataset  # noqa: F401,E402,I001

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
from g3a_bench.scripts.live_bridge.mapper import lerobot_to_mjlab_raw_action
from g3a_bench.scripts.live_bridge.reader import DatasetReader, ReaderFrame
from g3a_bench.scripts.live_bridge.recorder import Recorder

LOGS_DIR = Path(__file__).resolve().parent / "logs"


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1. Load cube task config (play mode: infinite, no noise).
    cfg = load_env_cfg("G3a-Cube-Stack3-Dk1_uni", play=True)
    cfg.terminations["time_out"].time_out = False
    # Drop ee_ground_collision: it survives the play-mode termination pop, and a
    # single >10 N gripper-vs-ground contact in the replayed trajectory would
    # auto-reset the env mid-episode (auto_reset=True) and corrupt the recording.
    cfg.terminations.pop("ee_ground_collision", None)
    cfg.terminations.pop(
        "success", None
    )  # replay runs through the full episode uninterrupted

    # 2. Create environment.
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env = RslRlVecEnvWrapper(env)
    print("[INFO] Environment created")

    # 3. Open dataset to replay.
    reader = DatasetReader("FabianKerj/corkinbox100-fullrel")
    total_frames = reader._end - reader._start
    print(f"[INFO] Dataset opened — episode has {total_frames} frames")

    # 4. Create recorder.
    logs_dir = LOGS_DIR
    repo_id = "sim_replay_cube_stack3"
    dataset_root = logs_dir / repo_id
    # Derive the sim observation width from the env rather than hardcoding it,
    # so a different task (with a different actor-obs size) still records a
    # correctly-shaped dataset.
    obs_dim = int(env.get_observations()["actor"].shape[-1])
    recorder = Recorder(
        repo_id=repo_id, fps=50, obs_dim=obs_dim, root=str(dataset_root)
    )
    save_path = logs_dir / repo_id
    print(f"[INFO] Recording to {save_path}")

    # 5. Policy that reads dataset + records results.
    replay = _ReplayAndRecord(
        reader, recorder, total_frames, device, task="G3a-Cube-Stack3-Dk1_uni"
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
    print(f"[INFO] Done. Dataset saved to {save_path}")


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
    ):
        self.reader = reader
        self.recorder = recorder
        self.total_frames = total_frames
        self.device = device
        self.task = task
        self.frames_recorded = 0
        self.recording_finished = False

        self._last_obs = None
        self._last_action_np = None
        self._last_robot_obs = None
        self._fallback_action_np = None

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
                self.recorder.save_episode(success=True)
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
        if self._fallback_action_np is None:
            self._fallback_action_np = frame.leader_action.copy()

        raw_action = lerobot_to_mjlab_raw_action(frame.leader_action)
        return torch.from_numpy(raw_action).unsqueeze(0).to(self.device)

    def record_step_result(self, rew: float, done: bool):
        """Called by the custom viewer after env.step()."""
        if self.recording_finished:
            return
        if self._last_obs is None or self._last_action_np is None:
            return

        self.recorder.add_frame(
            {"actor": self._last_obs["actor"]},
            self._last_action_np,
            rew,
            done,
            task=self.task,
            robot_obs=self._last_robot_obs,
        )
        self.frames_recorded += 1
        if self.frames_recorded % 100 == 0:
            print(f"[INFO] Recorded {self.frames_recorded}/{self.total_frames} frames")

        if self.frames_recorded >= self.total_frames and not self.recording_finished:
            print(
                f"[INFO] All {self.total_frames} frames recorded — finalizing dataset"
            )
            self.recorder.save_episode(success=True)
            self.recording_finished = True

    def finalize(self):
        if not self.recording_finished:
            self.recorder.save_episode(success=True)
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

                # Notify the recording policy with step results.
                record_fn = getattr(self.policy, "record_step_result", None)
                if record_fn is not None:
                    record_fn(float(rew[0]), bool(dones[0]))

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
