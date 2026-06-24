"""Live teleop + sim mirroring + recording.

Connects to real DK1 hardware (leader + follower), mirrors movements
in MJLab simulation, records sim and real observations to a LeRobot dataset,
and shows the sim in a native MuJoCo viewer window.

Architecture (decoupled threads):

  Hardware thread (~200 Hz):
      leader.get_action() → follower.send_action()
      → push leader action to shared list (last-writer-wins)

  Main thread (LiveReaderViewer, sim-paced):
      pop latest action → mapper → env.step() → recorder.add_frame()

State machine (per episode):
  RECORDING      normal step loop; done=True → _handle_done()
  AWAITING_LABEL ground collision: sim frozen, waiting for F or D key
  RESET_PHASE    ghost arm visible at home; sim live (not recorded); arm moves to ghost

Key bindings:
    S  save episode as SUCCESS (also fires automatically on task completion)
    F  save episode as FAILURE  (required when collision prompt appears)
    D  discard episode
    Q  stop session and finalize dataset

Usage:
    uv run --extra cpu python g3a_bench/scripts/live_bridge/live_record.py

    Optionally specify ports:
    uv run --extra cpu python g3a_bench/scripts/live_bridge/live_record.py \\
        --follower-port /dev/ttyACM0 --leader-port /dev/ttyUSB0
"""

# Load lerobot/HF-datasets native libs before mjlab/mujoco-warp to avoid a
# native import-order segfault (HF `datasets` vs mujoco-warp). Must precede mjlab.
import lerobot.datasets.lerobot_dataset  # noqa: F401,E402,I001

import argparse
import datetime
import threading
import time
import traceback
from pathlib import Path

import mujoco as _mujoco
import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import VerbosityLevel
from mjlab.viewer.native.keys import KEY_D, KEY_F, KEY_Q, KEY_S

import g3a_bench  # noqa: F401
from g3a_bench.robots.trlc_dk1.dk1_constants import get_ghost_spec
from g3a_bench.scripts.live_bridge.mapper import (
    HOME_JOINT_POS,
    lerobot_to_mjlab_raw_action,
)
from g3a_bench.scripts.live_bridge.recorder import Recorder

LOGS_DIR = Path(__file__).resolve().parent / "logs"

USE_FOLLOWER_STATE_FOR_SIM = (
    False  # True → follower joint obs drives sim instead of leader action
)
RESET_THRESHOLD_RAD = 0.2  # all 6 arm joints must be within this many rad of home


def ask_operator() -> int:
    """Block until the user enters a valid operator number (1–4)."""
    while True:
        raw = input("Operator (1 / 2 / 3 / 4): ").strip()
        if raw in ("1", "2", "3", "4"):
            return int(raw)
        print(f"  Invalid: '{raw}'. Enter 1, 2, 3, or 4.")


def _arm_at_home(obs: np.ndarray | None, threshold: float) -> bool:
    """True iff all 6 arm joints are within threshold rad of HOME_JOINT_POS."""
    if obs is None:
        return False
    return bool(np.all(np.abs(obs[:6] - HOME_JOINT_POS[:6]) < threshold))


def _hw_loop(
    action_list: list,
    robot_obs_list: list,
    shutdown_flag: list,
    hw_failed: list,
    follower_port: str | None,
    leader_port: str | None,
):
    """Read leader at ~200 Hz → send to follower → push to shared lists.

    Runs in a background daemon thread.  Uses single-element lists as
    GIL-safe last-writer-wins storage for both the latest action and
    the latest robot observation.

    Args:
        action_list: Single-element list updated with latest (7,) action for sim.
        robot_obs_list: Single-element list updated with latest (7,) robot obs.
        shutdown_flag: Single-element list; set [True] to stop the loop.
        hw_failed: Single-element list; set [True] if the loop crashes so the
            viewer can abort instead of recording frozen frames.
        follower_port: Serial port for follower arm.
        leader_port: Serial port for leader (GELLO) controller.
    """
    from g3a_bench.scripts.live_bridge.reader import LiveReader

    reader = LiveReader(
        follower_port=follower_port,
        leader_port=leader_port,
        connect_on_init=True,
    )
    freq = 200
    period = 1.0 / freq

    try:
        while not shutdown_flag[0]:
            t0 = time.perf_counter()
            frame = reader.read()
            robot_obs_list[0] = frame.joint_pos
            if frame.leader_action is not None:
                reader.send_action(frame.leader_action)
                action_list[0] = (
                    frame.joint_pos.copy()
                    if USE_FOLLOWER_STATE_FOR_SIM
                    else frame.leader_action.copy()
                )
            elapsed = time.perf_counter() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    except Exception:
        # A serial hiccup / unplug must not leave the sim silently recording a
        # frozen stale action. Flag it so the viewer aborts the session cleanly.
        hw_failed[0] = True
        shutdown_flag[0] = True
        print("[HW] ERROR — hardware thread crashed:")
        traceback.print_exc()
    finally:
        reader.close()
        print("[HW] Thread stopped")


class _LivePolicy:
    """Viewer policy: returns the latest leader action from the shared list.

    Each time the viewer calls ``__call__(obs)``, the policy reads the
    most recent action that the HW thread pushed into ``action_list``.
    """

    def __init__(self, action_list: list, device: str):
        self.action_list = action_list
        self._device = device
        self._last_raw_action = torch.zeros(1, 7, device=device)
        self.latest_leader_action: np.ndarray | None = None

    def __call__(self, obs):
        leader_action = self.action_list[0]
        if leader_action is not None:
            self.latest_leader_action = leader_action
            raw_action = lerobot_to_mjlab_raw_action(leader_action)
            self._last_raw_action = (
                torch.from_numpy(raw_action).unsqueeze(0).to(self._device)
            )
        return self._last_raw_action


class _LiveRecordingViewer(NativeMujocoViewer):
    """NativeMujocoViewer with multi-episode recording and ghost-arm reset phase.

    State machine:
      RECORDING        → normal step loop
      AWAITING_LABEL   → collision fired; sim frozen; waiting for F/D key
      RESET_PHASE      → episode ended; ghost visible; waiting for arm at home
    """

    def __init__(
        self,
        env,
        policy,
        recorder: Recorder,
        robot_obs_list: list,
        hw_failed: list,
        mj_model: "_mujoco.MjModel | None",
        ghost_geom_ids: list,
        task: str = "",
        num_episodes: int = 0,
        **kwargs,
    ):
        self._recorder = recorder
        self._robot_obs_list = robot_obs_list
        self._hw_failed = hw_failed
        self._mj_model = mj_model
        self._ghost_geom_ids = ghost_geom_ids
        self._task = task
        self._num_episodes = num_episodes
        self.episodes_saved = 0
        self._in_reset_phase = False
        self._awaiting_collision_label = False
        super().__init__(
            env,
            policy,
            verbosity=VerbosityLevel.DEBUG,
            key_callback=self._on_key,
            **kwargs,
        )

    # ── Ghost arm helpers ────────────────────────────────────────────────────

    def _set_ghost_alpha(self, alpha: float) -> None:
        if self._mj_model is None:
            return
        for gid in self._ghost_geom_ids:
            self._mj_model.geom_rgba[gid, 3] = alpha

    def _enter_reset_phase(self) -> None:
        self._in_reset_phase = True
        self._set_ghost_alpha(0.3)
        print(
            f"[INFO] Reset phase — move the arm in the sim to the ghost position "
            f"(within {RESET_THRESHOLD_RAD:.2f} rad per joint) to start next episode"
        )

    def _exit_reset_phase(self) -> None:
        self._set_ghost_alpha(0.0)
        self._in_reset_phase = False
        # env already auto-reset when done=True fired; no second reset needed

    # ── Key handling ─────────────────────────────────────────────────────────

    def _on_key(self, key: int) -> None:
        # Runs on the viewer thread → only enqueue; actions are processed on
        # the main thread in _handle_custom_action to avoid racing the step loop.
        if key == KEY_S:
            self.request_action("CUSTOM", "save_success")
        elif key == KEY_F:
            self.request_action("CUSTOM", "save_failure")
        elif key == KEY_D:
            self.request_action("CUSTOM", "discard")
        elif key == KEY_Q:
            self._interrupted = True

    def _handle_custom_action(self, action, payload) -> bool:  # type: ignore[override]
        if payload == "save_success":
            self._awaiting_collision_label = False
            self._end_episode(success=True)
        elif payload == "save_failure":
            self._awaiting_collision_label = False
            self._end_episode(success=False)
        elif payload == "discard":
            frames = self._recorder.num_frames_in_episode
            self._recorder.discard_episode()
            self._awaiting_collision_label = False
            print(f"[INFO] Episode discarded ({frames} frames dropped)")
            self._enter_reset_phase()
        else:
            return False
        return True

    def _end_episode(self, success: bool) -> None:
        frames = self._recorder.num_frames_in_episode
        if frames == 0:
            print("[INFO] Nothing to save (no frames recorded yet)")
            return
        self._recorder.save_episode(success=success)
        self.episodes_saved += 1
        label = "SUCCESS" if success else "FAILURE"
        print(
            f"[INFO] Episode {self.episodes_saved} saved as {label} ({frames} frames)"
        )
        if self._num_episodes and self.episodes_saved >= self._num_episodes:
            print(f"[INFO] Reached {self._num_episodes} episodes — stopping")
            self._interrupted = True
        else:
            self._enter_reset_phase()

    def _handle_done(self) -> None:
        """Called when env signals done=True. Distinguishes success from collision."""
        try:
            cmd = self.env.env.command_manager.get_term("arrangement")
            is_success = bool(cmd.slot_at_goal.all(dim=1)[0].item())
        except Exception:
            # If the command term is unavailable, fall back to treating as collision.
            is_success = False

        if is_success:
            print("[INFO] Task succeeded — auto-saving as success")
            self._end_episode(success=True)
        else:
            print("[WARN] Ground collision!  Press  F = save as failure    D = discard")
            self._awaiting_collision_label = True

    # ── Step loop ────────────────────────────────────────────────────────────

    def _execute_step(self) -> bool:
        if self._hw_failed[0]:
            self.log(
                "[ERROR] Hardware thread failed — aborting session.",
                VerbosityLevel.SILENT,
            )
            self._interrupted = True
            return False

        # ── Reset phase: ghost visible, sim live, not recorded ──
        # Operator moves the arm in the sim toward the ghost (home pose).
        # joint_pos_rel is first 6 entries of actor obs — it is q − HOME, so
        # ≈ 0 means the sim arm matches the ghost.  Nothing is added to the
        # recorder during this phase.
        if self._in_reset_phase:
            try:
                with torch.no_grad():
                    obs = self.env.get_observations()
                    actions = self.policy(obs)
                    _obs_next, _rew, _dones, _extras = self.env.step(actions)
                    self._step_count += 1
                    self._stats_steps += 1
                    joint_pos_rel = obs["actor"][0, :6].cpu().numpy()
                    if np.all(np.abs(joint_pos_rel) < RESET_THRESHOLD_RAD):
                        print("[INFO] Sim arm at home — starting next episode")
                        self._exit_reset_phase()
            except Exception:
                self._last_error = traceback.format_exc()
                self.log(
                    f"[ERROR] Exception during reset step:\n{self._last_error}",
                    VerbosityLevel.SILENT,
                )
                self.pause()
                return False
            return True

        # ── Collision label: sim frozen, waiting for F or D key ──
        if self._awaiting_collision_label:
            return True

        # ── Normal recording step ──
        try:
            with torch.no_grad():
                obs = self.env.get_observations()
                actions = self.policy(obs)
                _obs_next, rew, dones, _extras = self.env.step(actions)

                leader_action = getattr(self.policy, "latest_leader_action", None)
                robot_obs = self._robot_obs_list[0]
                if leader_action is not None and robot_obs is not None:
                    self._recorder.add_frame(
                        {"actor": obs["actor"]},
                        leader_action,
                        float(rew[0]),
                        bool(dones[0]),
                        task=self._task,
                        robot_obs=robot_obs,
                    )

                self._step_count += 1
                self._stats_steps += 1

                if dones[0]:
                    self._handle_done()

                return True
        except Exception:
            self._last_error = traceback.format_exc()
            self.log(
                f"[ERROR] Exception during step:\n{self._last_error}",
                VerbosityLevel.SILENT,
            )
            self.pause()
            return False


def _get_mj_model(wrapped_env) -> "_mujoco.MjModel | None":
    """Find the compiled MjModel inside the mjlab env wrapper.

    Tries common mjlab/Isaac-Lab attribute paths in order.
    Returns None if none succeed (ghost toggle is silently disabled).
    """
    candidates = [
        ("env", "scene", "physics", "model"),
        ("env", "sim", "model"),
        ("env", "physics", "model"),
        ("env", "scene", "sim", "physics", "model"),
    ]
    for path in candidates:
        obj = wrapped_env
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "ngeom"):
            return obj
    print("[WARN] Cannot locate mujoco.MjModel — ghost arm alpha toggle disabled")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Live teleop + sim mirroring + recording"
    )
    parser.add_argument(
        "--follower-port",
        type=str,
        default=None,
        help="Serial port for follower (e.g. /dev/ttyACM0). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--leader-port",
        type=str,
        default=None,
        help="Serial port for leader (e.g. /dev/ttyUSB0). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="G3a-Cube-Stack3-Dk1_uni",
        help="MJLab task name",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help=(
            "Output directory. Defaults to logs/live_record/<timestamp>/. "
            "Errors if the directory already exists (data is never overwritten)."
        ),
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="Stop after N saved episodes (0 = record until Q is pressed).",
    )
    args = parser.parse_args()

    # 0. Identify operator for session metadata.
    operator_id = ask_operator()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1. Load task config (play mode: infinite, no noise).
    #    Keep both terminations active:
    #      ee_ground_collision → prompts fail/discard labeling
    #      success             → auto-saves as success
    cfg = load_env_cfg(args.task, play=True)
    cfg.terminations["time_out"].time_out = False

    # Ghost arm: visual-only DK1 at home pose, starts invisible (alpha=0).
    # Springs in the spec hold all joints at home; no actuators needed.
    cfg.scene.entities["robot_ghost"] = EntityCfg(spec_fn=get_ghost_spec)

    # 2. Create environment.
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env = RslRlVecEnvWrapper(env)
    print("[INFO] Environment created")

    # 3. Locate ghost geom IDs for alpha toggling (group 4 = ghost geoms).
    mj_model = _get_mj_model(env)
    ghost_geom_ids: list[int] = []
    if mj_model is not None:
        ghost_geom_ids = [
            i for i in range(mj_model.ngeom) if mj_model.geom_group[i] == 4
        ]
        print(f"[INFO] Ghost arm: {len(ghost_geom_ids)} geoms found (group 4)")

    # 4. Resolve dataset output directory — never wipe existing data.
    if args.dataset_dir:
        dataset_root = Path(args.dataset_dir)
        if dataset_root.exists():
            raise SystemExit(
                f"[ERROR] {dataset_root} already exists — choose a new --dataset-dir "
                "(refusing to overwrite recorded data)."
            )
    else:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dataset_root = LOGS_DIR / "live_record" / stamp

    # 5. Set up recorder.
    obs_dim = int(env.get_observations()["actor"].shape[-1])
    recorder = Recorder(
        repo_id="live_record", fps=50, obs_dim=obs_dim, root=str(dataset_root)
    )
    print(f"[INFO] Recording to {dataset_root} (obs_dim={obs_dim})")

    # 6. Shared single-element lists (GIL-safe last-writer-wins).
    action_list: list = [None]
    robot_obs_list: list = [None]
    shutdown_flag: list = [False]
    hw_failed: list = [False]

    # 7. Start HW thread (reads leader → sends to follower → fills lists).
    hw_thread = threading.Thread(
        target=_hw_loop,
        args=(
            action_list,
            robot_obs_list,
            shutdown_flag,
            hw_failed,
            args.follower_port,
            args.leader_port,
        ),
        daemon=True,
    )
    hw_thread.start()
    print("[INFO] Hardware thread started (reading leader at ~200 Hz)")

    # 8. Create live policy and viewer.
    live_policy = _LivePolicy(action_list, device)

    print("[INFO] Starting native MuJoCo viewer")
    print("[INFO] Move the leader arm to control the sim and follower")
    print("[INFO] Keys:  S=save success   F=save failure   D=discard/redo   Q=stop")

    viewer: _LiveRecordingViewer | None = None
    try:
        viewer = _LiveRecordingViewer(
            env=env,
            policy=live_policy,
            recorder=recorder,
            robot_obs_list=robot_obs_list,
            hw_failed=hw_failed,
            mj_model=mj_model,
            ghost_geom_ids=ghost_geom_ids,
            task=args.task,
            num_episodes=args.num_episodes,
        )
        viewer.run()
    finally:
        shutdown_flag[0] = True
        hw_thread.join(timeout=2.0)
        recorder.discard_episode()  # drop any partial unsaved episode
        recorder.close(operator_id=operator_id)
        env.close()
        saved = viewer.episodes_saved if viewer is not None else 0
        print(f"[INFO] Done. Saved {saved} episode(s) to {dataset_root}")
        if hw_failed[0]:
            print(
                "[WARN] Session ended due to a hardware-thread failure "
                "(see traceback above) — the saved episodes may be incomplete."
            )


if __name__ == "__main__":
    main()
