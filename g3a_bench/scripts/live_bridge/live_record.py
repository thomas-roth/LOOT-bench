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
  RECORDING            normal step loop; done=True → _handle_done()
  PROVISIONAL_FAILURE  ground collision: stepping paused, auto-saves as
                       FAILURE after a 2s grace period unless numpad 2
                       (save now) or numpad 3 (discard) is pressed first
  RESET_PHASE          ghost arm visible at home; sim live (not recorded); arm moves to ghost

Key bindings:
    Numpad 1  save episode as SUCCESS (also fires automatically on task completion)
    Numpad 2  save episode as FAILURE  (also fires automatically ~2s after a collision)
    Numpad 3  discard episode
    Numpad 4  stop session and finalize dataset
    W         toggle viewer between orbit camera and the robot's wrist camera
              (clearer gripper view while teleoperating demos)

    Main-row digit keys are NOT usable here — MuJoCo's native viewer swallows
    them internally to toggle render-group visibility before they ever reach
    this script's key handler.

Usage:
    uv run python g3a_bench/scripts/live_bridge/live_record.py

    With real follower arm:
    uv run python g3a_bench/scripts/live_bridge/live_record.py --follower

    With explicit ports:
    uv run python g3a_bench/scripts/live_bridge/live_record.py --follower \\
        --follower-port /dev/ttyACM1 --leader-port /dev/ttyACM0
"""

# Load lerobot/HF-datasets native libs before mjlab/mujoco-warp to avoid a
# native import-order segfault (HF `datasets` vs mujoco-warp). Must precede mjlab.
import lerobot.datasets.lerobot_dataset  # noqa: F401,E402,I001

import argparse
import datetime
import gc
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco as _mujoco

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import NativeMujocoViewer, ViewerConfig
from mjlab.viewer.base import VerbosityLevel
from mjlab.viewer.native.keys import KEY_KP_1, KEY_KP_2, KEY_KP_3, KEY_KP_4, KEY_W
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

import g3a_bench  # noqa: F401
from g3a_bench.robots.trlc_dk1.dk1_constants import _HOME_JOINT_POS
from g3a_bench.scripts.live_bridge.mapper import (
    lerobot_to_mjlab_raw_action,
)
from g3a_bench.scripts.live_bridge.recorder import Recorder
from g3a_bench.scripts.live_bridge.recording_mixin import RecordingViewerMixin


def _get_pip_app() -> "QApplication":
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])  # type: ignore[reportReturnType]


def _get_qimage_and_qpixmap():
    from PyQt5.QtGui import QImage, QPixmap

    return QImage, QPixmap


# Repo-root out/ directory (consistent with the out/<cat>_check render outputs).
DEMOS_DIR = Path(__file__).resolve().parents[3] / "out" / "demos"

USE_FOLLOWER_STATE_FOR_SIM = (
    False  # True → follower joint obs drives sim instead of leader action
)
# Ghost-arm colors, feedback timing, the rest-pose threshold and the recording
# state machine all live in recording_mixin (shared with the Viser viewer).

LIVE_REST_POSE: dict[str, float] = {
    "joint1": +0.1565,
    "joint2": +0.0015,
    "joint3": +0.0015,
    "joint4": -0.0153,
    "joint5": +0.0153,
    "joint6": +0.0276,
}
# Same rest pose mapped into the sim frame — lives in the same frame as
# the teleop'd arm and the action mapper.
_SIM_REST_POSE: dict[str, float] = {
    **LIVE_REST_POSE,
}
_LIVE_REST_POSE_REL = np.array(
    [_SIM_REST_POSE[f"joint{i}"] - _HOME_JOINT_POS[f"joint{i}"] for i in range(1, 7)],
    dtype=np.float32,
)


KEY_SAVE_SUCCESS = KEY_KP_1
KEY_SAVE_FAILURE = KEY_KP_2
KEY_DISCARD = KEY_KP_3
KEY_QUIT = KEY_KP_4
KEY_WRIST_CAM = KEY_W

# Name of the wrist camera in the DK1 MJCF. The compiled model prefixes entity
# cameras (e.g. "robot/wrist_camera"), so it is matched by suffix — see
# _find_camera_id.
WRIST_CAMERA_NAME = "wrist_camera"


# Default orbit/free-camera framing for the teleop viewer (the view you start in,
# before pressing W for the wrist cam). Positioned above and behind the robot
ORBIT_CAM_LOOKAT = (0.20, 0.0, 0.15)  # world point the camera aims at (≈ workspace)
ORBIT_CAM_DISTANCE = 1.3
ORBIT_CAM_AZIMUTH = 200.0
ORBIT_CAM_ELEVATION = -35.0


def apply_teleop_contact_stability(cfg) -> None:
    """Keep the finger-pad contact solref stable for the play/teleop timestep.

    Raises any robot collision solref timeconst below 2x the sim timestep (the
    stability floor) to stop grip jitter. Shared with record_replay.py and
    verify_replay.py — replayed physics must run with the exact same contact
    parameters as the recording, or trajectories diverge at first contact.
    """
    min_solref_timeconst = 2.0 * cfg.sim.mujoco.timestep
    for _coll in cfg.scene.entities["robot"].collisions:
        solref = getattr(_coll, "solref", None) or {}
        for _key, _ref in list(solref.items()):
            _timeconst, _dampratio = _ref
            if _timeconst < min_solref_timeconst:
                solref[_key] = (min_solref_timeconst, _dampratio)
                print(
                    f"[INFO] contact-stability: raised solref timeconst for "
                    f"'{_key}' {_timeconst:g}->{min_solref_timeconst:g} s "
                    f"(>=2x timestep {cfg.sim.mujoco.timestep:g}) to stop grip jitter"
                )


def ask_operator() -> int:
    """Block until the user enters a valid operator number (1–4)."""
    while True:
        raw = input("Operator ( 1-Minh / 2-Darius / 3-Thomas / 4-Till ): ").strip()
        if raw in ("1", "2", "3", "4"):
            return int(raw)
        print(f"  Invalid: '{raw}'. Enter 1, 2, 3, or 4.")


def _hw_loop(
    action_list: list,
    robot_obs_list: list,
    shutdown_flag: list,
    hw_failed: list,
    follower_port: str | None,
    leader_port: str | None,
    use_follower: bool = False,
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
        use_follower: If False, never connects to or sends actions to the
            real follower — only the leader is read, and its own action
            doubles as the recorded robot state.
    """
    from g3a_bench.scripts.live_bridge.reader import LiveReader

    reader = LiveReader(
        follower_port=follower_port,
        leader_port=leader_port,
        connect_on_init=True,
        use_follower=use_follower,
    )
    freq = 200
    period = 1.0 / freq

    try:
        while not shutdown_flag[0]:
            t0 = time.perf_counter()
            frame = reader.read()
            robot_obs_list[0] = frame.joint_pos
            if frame.leader_action is not None:
                if use_follower:
                    reader.send_action(frame.leader_action)
                action_list[0] = frame.leader_action.copy()

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
            raw_action = lerobot_to_mjlab_raw_action(
                leader_action,
                gravity_compensate=True,
            )
            self._last_raw_action = (
                torch.from_numpy(raw_action).unsqueeze(0).to(self._device)
            )
        return self._last_raw_action


class _LiveRecordingViewer(RecordingViewerMixin, NativeMujocoViewer):
    """NativeMujocoViewer with multi-episode recording and ghost-arm reset phase.

    The per-episode recording state machine (record → reset phase → collision
    label, plus ghost-color feedback) lives in RecordingViewerMixin. This
    subclass only adds the native-specific rendering of the ghost arm and the wrist-camera toggle.

    State machine:
      RECORDING            → normal step loop
      PROVISIONAL_FAILURE  → collision fired; stepping paused; auto-saves as
                             failure after a 2s grace period unless numpad 2/3
      RESET_PHASE          → episode ended; ghost visible; waiting for arm at home
    """

    def __init__(
        self,
        env,
        policy,
        recorder: Recorder,
        robot_obs_list: list,
        hw_failed: list,
        mj_model: "_mujoco.MjModel | None",
        task: str = "",
        num_episodes: int = 0,
        debug_timing: bool = False,
        wrist_renderer: "_mujoco.Renderer | None" = None,
        wrist_cam_id_pip: int = -1,
        pip_app: "QApplication | None" = None,
        pip_window: "QMainWindow | None" = None,
        pip_label: "QLabel | None" = None,
        **kwargs,
    ):
        self._init_recording(
            recorder=recorder,
            robot_obs_list=robot_obs_list,
            hw_failed=hw_failed,
            mj_model=mj_model,
            live_rest_pose_rel=_LIVE_REST_POSE_REL,
            task=task,
            num_episodes=num_episodes,
            debug_timing=debug_timing,
            sim_rest_pose=_SIM_REST_POSE,
            home_joint_pos=_HOME_JOINT_POS,
        )
        # Wrist-cam toggle (W): resolved lazily once the viewer/model exist.
        self._wrist_cam_on = False
        self._wrist_cam_id = -1
        self._saved_cam: tuple | None = None
        # Snapshot of the intended MjvOption visualization flags
        self._baseline_visflags: np.ndarray | None = None
        # Cached debug visualizers — constructed once in setup() to avoid
        # per-frame MjData allocations that caused gradual FPS decay.
        self._vis_task: MujocoNativeDebugVisualizer | None = None
        self._vis_ghost: MujocoNativeDebugVisualizer | None = None
        self._vis_wrist: MujocoNativeDebugVisualizer | None = None
        # Wrist-camera offscreen renderer (PiP window). Built by the caller
        # (main(), before the hw thread starts — see _build_wrist_pip) rather
        # than lazily here in setup()/__init__: constructing mujoco.Renderer's
        # own GLFW context and the QApplication/QMainWindow are one-time
        # operations that can block for a long stretch (GLFW context creation
        # racing the native viewer's own render thread), which was starving
        # the already-running ~200 Hz hw thread and causing spurious
        # "no status packet" serial timeouts the instant the session started.
        self._wrist_renderer = wrist_renderer
        self._wrist_cam_id_pip = wrist_cam_id_pip
        self._pip_app = pip_app
        self._pip_window = pip_window
        self._pip_label = pip_label
        # Throttle: the offscreen render + Qt image conversion + processEvents()
        # are GIL-holding C calls. Running them at full render frame_rate (up to
        # 120 Hz) starved the hw thread's ~200 Hz / tight-timeout serial reads of
        # scheduling time and caused spurious "no status packet" ConnectionErrors.
        # A human only needs ~15 Hz to judge gripper alignment.
        self._wrist_pip_last_update: float = 0.0
        self._WRIST_PIP_INTERVAL: float = 1.0 / 15.0
        super().__init__(
            env,
            policy,
            verbosity=VerbosityLevel.DEBUG,
            key_callback=self._on_key,
            **kwargs,
        )

    def setup(self) -> None:
        """Capture the intended visualization-flag baseline after base setup."""
        super().setup()
        assert self.viewer is not None
        self._baseline_visflags = self.viewer.opt.flags.copy()
        assert self.mjm is not None
        # Pre-construct both debug visualizers once, reusing the same MjData
        # throughout the session instead of allocating a new one every frame.
        self._vis_task = MujocoNativeDebugVisualizer(
            self.viewer.user_scn, self.mjm, self.env_idx, self._show_all_envs
        )
        self._vis_ghost = MujocoNativeDebugVisualizer(
            self.viewer.user_scn, self.mjm, self.env_idx, self._show_all_envs
        )
        # The wrist-cam PiP is a *separate* offscreen mujoco.Renderer with its
        # own internal MjvScene — task-command debug_vis geoms (target marks)
        # drawn into viewer.user_scn above never reach it. Give it its own
        # cached visualizer bound to that renderer's scene, re-populated each
        # PiP frame in _update_debug_visualizers (see there for why cached).
        if self._wrist_renderer is not None and self._wrist_cam_id_pip >= 0:
            self._vis_wrist = MujocoNativeDebugVisualizer(
                self._wrist_renderer.scene,
                self.mjm,
                self.env_idx,
                self._show_all_envs,
            )

        # ── Ghost arm rendering (native-specific) ────────────────────────────────

    def _update_debug_visualizers(self, viewer: "_mujoco.viewer.Handle") -> None:  # type: ignore[name-defined]
        """Draw the ghost-at-home overlay every frame, on top of whatever the
        base class / task command terms already draw (e.g. cube task target
        marks). Uses the same DebugVisualizer.add_ghost_mesh technique as
        mjlab's own tracking task — see recording_mixin.build_ghost_model_and_qpos.
        """
        viewer.user_scn.ngeom = 0
        # Reuse the cached task visualizer instead of constructing a new one
        # each frame — constructing MjData is expensive (~21,600 alloc/free
        # over a 3 min session caused gradual FPS decay from 60 to 46).
        if self._show_debug_vis and hasattr(self.env.unwrapped, "update_visualizers"):
            assert self._vis_task is not None
            self.env.unwrapped.update_visualizers(self._vis_task)
        # Re-assert the visualization-flag baseline every frame so the native
        # viewer's built-in single-key flag toggles can't alter the scene
        if viewer is not None and self._baseline_visflags is not None:
            viewer.opt.flags[:] = self._baseline_visflags
        if self._ghost_model is None or self._ghost_qpos is None:
            return
        assert self.mjm is not None
        # Use feedback color (success/failure flash) if active, otherwise normal
        # ghost. The recolored model comes from the mixin's per-color cache — no
        # per-frame deepcopy (that cost ~5 FPS before).
        current_color = self._get_ghost_color()
        ghost_model = self._get_ghost_model_with_color(current_color)
        self._vis_ghost.add_ghost_mesh(
            self._ghost_qpos, model=ghost_model, label="ghost"
        )

        # PiP: render wrist camera offscreen → show in PyQt window. Throttled
        # (see _WRIST_PIP_INTERVAL) — this must never run at full frame_rate.
        now = time.perf_counter()
        if (
            self._wrist_renderer is not None
            and self._wrist_cam_id_pip >= 0
            and self._pip_label is not None
            and now - self._wrist_pip_last_update >= self._WRIST_PIP_INTERVAL
        ):
            self._wrist_pip_last_update = now
            self._wrist_renderer.update_scene(
                self.mjd,
                camera=self._wrist_cam_id_pip,
            )
            # update_scene() rebuilds the renderer's scene from the model each
            # call, so re-add the task's target markers (same debug_vis geoms
            # drawn into the main viewer above) on top, every PiP frame.
            if (
                self._show_debug_vis
                and self._vis_wrist is not None
                and hasattr(self.env.unwrapped, "update_visualizers")
            ):
                self.env.unwrapped.update_visualizers(self._vis_wrist)
            img = self._wrist_renderer.render()
            QImage, QPixmap = _get_qimage_and_qpixmap()
            qimg = QImage(img.tobytes(), 320, 240, QImage.Format_RGB888).rgbSwapped()
            self._pip_label.setPixmap(QPixmap.fromImage(qimg))
            self._pip_app.processEvents()

    def _exit_reset_phase(self) -> None:
        super()._exit_reset_phase()
        self.start_episode_seeded_reset()

    def start_episode_seeded_reset(self) -> None:
        """Reseed all RNGs and reset the env so this episode's object scatter
        and sampled goal can be reproduced at replay time from the saved seed."""

        robot = self.env.env.scene["robot"]
        live_joint_pos = robot.data.joint_pos.clone()

        self._current_ep_seed = int(np.random.randint(0, 2**31 - 1))
        self.env.env.reset(seed=self._current_ep_seed)

        robot.write_joint_position_to_sim(live_joint_pos)
        self.env.env.sim.forward()

        # self._current_ep_seed = int(np.random.randint(0, 2**31 - 1))
        # self.env.env.reset(seed=self._current_ep_seed)

    # ── Key handling ─────────────────────────────────────────────────────────

    def _on_key(self, key: int) -> None:
        # Runs on the viewer thread → only enqueue; actions are processed on
        # the main thread in _handle_custom_action to avoid racing the step loop.
        if key == KEY_SAVE_SUCCESS:
            self.request_action("CUSTOM", "save_success")
        elif key == KEY_SAVE_FAILURE:
            self.request_action("CUSTOM", "save_failure")
        elif key == KEY_DISCARD:
            self.request_action("CUSTOM", "discard")
        elif key == KEY_WRIST_CAM:
            # Only mutates the viewer's own camera (never env/sim), so it is
            # safe to handle directly on the viewer thread.
            self._toggle_wrist_cam()
        elif key == KEY_QUIT:
            self._interrupted = True

    def _toggle_wrist_cam(self) -> None:
        """Toggle the viewer between the orbit/free camera and the robot's
        wrist camera. Seeing the live wrist view makes gripper alignment much
        easier to judge while teleoperating demos.

        Toggling back restores the exact orbit framing the operator had, so W
        is non-destructive to whatever view they set up.
        """
        v = self.viewer
        if v is None or self.mjm is None:
            return
        if self._wrist_cam_id < 0:
            self._wrist_cam_id = _find_camera_id(self.mjm, WRIST_CAMERA_NAME)
            if self._wrist_cam_id < 0:
                print(
                    f"[WARN] No '{WRIST_CAMERA_NAME}' camera in model — "
                    "W toggle disabled"
                )
                return
        if not self._wrist_cam_on:
            self._saved_cam = (
                int(v.cam.type),
                int(v.cam.fixedcamid),
                int(v.cam.trackbodyid),
                tuple(v.cam.lookat),
                float(v.cam.distance),
                float(v.cam.azimuth),
                float(v.cam.elevation),
            )
            v.cam.type = _mujoco.mjtCamera.mjCAMERA_FIXED.value
            v.cam.fixedcamid = self._wrist_cam_id
            self._wrist_cam_on = True
            print("[INFO] Wrist camera ON (press W to return to orbit view)")
        else:
            if self._saved_cam is not None:
                ctype, fcam, tbody, lookat, dist, azi, elev = self._saved_cam
                v.cam.type = ctype
                v.cam.fixedcamid = fcam
                v.cam.trackbodyid = tbody
                v.cam.lookat = lookat
                v.cam.distance = dist
                v.cam.azimuth = azi
                v.cam.elevation = elev
            else:
                v.cam.type = _mujoco.mjtCamera.mjCAMERA_FREE.value
            self._wrist_cam_on = False
            print("[INFO] Wrist camera OFF (orbit view restored)")

    # Recording state machine (_handle_custom_action, _end_episode, _handle_done,
    # _execute_step, ghost-color helpers) is inherited from RecordingViewerMixin.


def _find_camera_id(mj_model: "_mujoco.MjModel", suffix: str) -> int:
    """Return the id of the first camera whose name is, or ends with, ``suffix``.

    The compiled model prefixes entity cameras (e.g. "robot/wrist_camera"), so a
    plain mj_name2id on the bare name would miss it — match by suffix instead.
    Returns -1 if no camera matches.
    """
    for ci in range(mj_model.ncam):
        name = _mujoco.mj_id2name(mj_model, _mujoco.mjtObj.mjOBJ_CAMERA, ci) or ""
        if name == suffix or name.endswith("/" + suffix):
            return ci
    return -1


def _build_wrist_pip(mj_model: "_mujoco.MjModel | None"):
    """One-time construction of the wrist-camera PiP window.

    Must be called before the hw thread starts polling the real arm.
    mujoco.Renderer opens its own GLFW context and QApplication/QMainWindow
    init talks to the display server — both are one-time operations that can
    block for a long stretch. Doing that after the ~200 Hz hw thread is
    already running starved it of scheduling time and tripped its tight
    serial-reply timeout the instant the session started.
    """
    if mj_model is None:
        return None, -1, None, None, None
    cam_id = _find_camera_id(mj_model, WRIST_CAMERA_NAME)
    if cam_id < 0:
        return None, -1, None, None, None

    from PyQt5.QtWidgets import QLabel, QMainWindow

    renderer = _mujoco.Renderer(mj_model, height=240, width=320)
    app = _get_pip_app()
    window = QMainWindow()
    label = QLabel()
    window.setCentralWidget(label)
    window.setWindowTitle("wrist")
    window.resize(320, 240)
    window.show()
    return renderer, cam_id, app, window, label


def _get_mj_model(wrapped_env) -> "_mujoco.MjModel | None":
    """Find the compiled MjModel inside the mjlab env wrapper.

    Tries common mjlab/Isaac-Lab attribute paths in order.
    Returns None if none succeed (ghost toggle is silently disabled).
    """
    candidates = [
        ("env", "sim", "mj_model"),
        ("env", "scene", "physics", "model"),
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
        "--follower",
        action="store_true",
        help=(
            "Connect to the real follower robot. By default only the leader is "
            "read and its action also drives observation.robot_state."
        ),
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
            "Output base directory. Defaults to out/demos/ at the repo root. "
            "Episodes are saved to <base>/success/<timestamp>/ and "
            "<base>/failure/<timestamp>/ (data is never overwritten)."
        ),
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="Stop after N saved episodes (0 = record until numpad 4 is pressed).",
    )
    parser.add_argument(
        "--debug-timing",
        action="store_true",
        help=(
            "Log a per-stage breakdown for any recording step over budget "
            "(prints [STEP-SPIKE] lines) to diagnose intermittent FPS drops."
        ),
    )
    args = parser.parse_args()

    # 0. Identify operator for session metadata.
    operator_id = ask_operator()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1. Load task config (play mode: infinite, no noise).
    #    success termination stays active (auto-saves as success).
    cfg = load_env_cfg(args.task, play=True)
    cfg.terminations["time_out"].time_out = False

    apply_teleop_contact_stability(cfg)
    # 2. Create environment.
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env = RslRlVecEnvWrapper(env)
    print("[INFO] Environment created")

    # Override the task's default (body-tracking) viewer camera with a static
    # world-frame orbit view tuned for teleop recording
    vcfg = env.cfg.viewer
    vcfg.origin_type = ViewerConfig.OriginType.WORLD
    vcfg.entity_name = None
    vcfg.body_name = None
    vcfg.lookat = ORBIT_CAM_LOOKAT
    vcfg.distance = ORBIT_CAM_DISTANCE
    vcfg.azimuth = ORBIT_CAM_AZIMUTH
    vcfg.elevation = ORBIT_CAM_ELEVATION

    # 3. Build the ghost-at-home overlay (see build_ghost_model_and_qpos).
    mj_model = _get_mj_model(env)
    if mj_model is None:
        print("[WARN] Ghost-at-home overlay disabled (could not locate MjModel)")

    # 4. Resolve dataset output directories — never wipe existing data.
    #    Layout: <base>/success/<stamp>/ and <base>/failure/<stamp>/; each
    #    dataset is only created once an episode of that kind is saved.
    base_dir = Path(args.dataset_dir) if args.dataset_dir else DEMOS_DIR
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    success_root = base_dir / "success" / stamp
    failure_root = base_dir / "failure" / stamp
    for _path in (success_root, failure_root):
        if _path.exists():
            raise SystemExit(
                f"[ERROR] {_path} already exists (refusing to overwrite recorded data)."
            )

    # 5. Set up recorder.
    obs_dim = int(env.get_observations()["actor"].shape[-1])
    qpos_dim = int(env.env.sim.data.qpos.shape[-1])
    recorder = Recorder(
        repo_id="live_record",
        fps=50,
        obs_dim=obs_dim,
        success_root=str(success_root),
        failure_root=str(failure_root),
        qpos_dim=qpos_dim,
        device=device,
    )
    print(
        f"[INFO] Recording to {base_dir}/{{success,failure}}/{stamp} "
        f"(obs_dim={obs_dim}, qpos_dim={qpos_dim})"
    )

    # 6. Shared single-element lists (GIL-safe last-writer-wins).
    action_list: list = [None]
    robot_obs_list: list = [None]
    shutdown_flag: list = [False]
    hw_failed: list = [False]

    # 6b. Build the wrist-cam PiP window now, before the hw thread starts —
    # its one-time GLFW/Qt init can block for a long stretch, and doing that
    # concurrently with the ~200 Hz hw thread starved it and blew its serial
    # reply timeout. See _build_wrist_pip.
    wrist_renderer, wrist_cam_id_pip, pip_app, pip_window, pip_label = _build_wrist_pip(
        mj_model
    )
    if wrist_cam_id_pip < 0:
        print(f"[WARN] No '{WRIST_CAMERA_NAME}' camera in model — PiP window disabled")

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
            args.follower,
        ),
        daemon=True,
    )
    hw_thread.start()
    print("[INFO] Hardware thread started (reading leader at ~200 Hz)")
    if not args.follower:
        print(
            "[INFO] Running leader-only (no follower connected). Pass --follower to enable."
        )

    # 8. Create live policy and viewer.
    live_policy = _LivePolicy(action_list, device)

    print("[INFO] Starting native MuJoCo viewer")
    print("[INFO] Move the leader arm to control the sim and follower")
    print("[INFO] Keys (numpad):")
    print("[INFO]   Numpad 1  → save current episode as SUCCESS")
    print("[INFO]   Numpad 2  → save current episode as FAILURE")
    print(
        "[INFO]   Numpad 3  → discard/redo current episode "
        "(also discards a collision within 2s)"
    )
    print("[INFO]   Numpad 4  → stop recording session")
    print("[INFO]   W         → toggle wrist camera <-> orbit view")
    print("[INFO] === Episode 1 started — recording resumed ===")

    viewer: _LiveRecordingViewer | None = None
    try:
        viewer = _LiveRecordingViewer(
            env=env,
            policy=live_policy,
            recorder=recorder,
            robot_obs_list=robot_obs_list,
            hw_failed=hw_failed,
            mj_model=mj_model,
            task=args.task,
            num_episodes=args.num_episodes,
            debug_timing=args.debug_timing,
            wrist_renderer=wrist_renderer,
            wrist_cam_id_pip=wrist_cam_id_pip,
            pip_app=pip_app,
            pip_window=pip_window,
            pip_label=pip_label,
            frame_rate=120,
        )
        # Take the big static object graph (mujoco model, warp kernels, torch
        # tensors) out of the GC's scan set
        gc.collect()
        gc.freeze()
        # Episode 1 also starts from a seeded reset so it is replayable.
        viewer.start_episode_seeded_reset()
        viewer.run()
    finally:
        shutdown_flag[0] = True
        hw_thread.join(timeout=2.0)
        recorder.discard_episode()  # drop any partial unsaved episode
        recorder.close(operator_id=operator_id)
        if viewer is not None:
            if viewer._wrist_renderer is not None:
                viewer._wrist_renderer.close()
            if viewer._pip_window is not None:
                viewer._pip_window.close()
        env.close()
        saved = viewer.episodes_saved if viewer is not None else 0
        print(f"[INFO] Done. Saved {saved} episode(s)")
        print(f"[INFO]   Success episodes: {success_root}/")
        print(f"[INFO]   Failure episodes: {failure_root}/")
        print("[INFO]   (a directory only exists if such episodes were saved)")
        if hw_failed[0]:
            print(
                "[WARN] Session ended due to a hardware-thread failure "
                "(see traceback above) — the saved episodes may be incomplete."
            )


if __name__ == "__main__":
    main()
