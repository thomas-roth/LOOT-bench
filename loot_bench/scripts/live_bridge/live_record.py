"""Live teleop + sim mirroring + recording for DK1.

HW thread (~200 Hz) feeds leader actions; main thread steps sim and records.
Non-success done auto-saves as failure after 2 s grace.

State machine (per episode):
  RECORDING            normal step loop; done=True → _handle_done()
  PROVISIONAL_FAILURE  non-success termination (out of bounds, collision,
                       timeout): stepping paused, auto-saves as FAILURE after a
                       2s grace period unless F (save now as failure),
                       D (discard) or V (override to success) is pressed first
  RESET_PHASE          ghost arm visible at home; sim live (not recorded)

Key bindings:
    V  save episode as SUCCESS (also fires automatically on task completion)
    F  save episode as FAILURE  (also fires automatically ~2s after a collision)
    D  discard episode
    U  undo the last saved episode
    Q  stop session and finalize dataset
    C         cycle the viewer camera: front → wrist → orbit → front. Wrist is
              the clearest gripper view; orbit is an over-the-shoulder framing.

    Main-row digit keys are NOT usable here — MuJoCo's native viewer swallows
    them internally to toggle render-group visibility before they ever reach
    this script's key handler. An on-screen HUD (bottom-left) shows the state,
    saved counts and the currently valid keys.

Usage:
    uv run loot-record-demos

    With real follower arm:
    uv run loot-record-demos --follower

    With explicit ports:
    uv run loot-record-demos --follower \\
        --follower-port /dev/ttyACM1 --leader-port /dev/ttyACM0

    The file is still runnable directly, which is the fallback if the console
    script ever segfaults on import (see the import note below):
    uv run python loot_bench/scripts/live_bridge/live_record.py
"""

# Load lerobot/HF-datasets native libs before mjlab/mujoco-warp to avoid a
# native import-order segfault (HF `datasets` vs mujoco-warp). Must precede mjlab.
#
# The ``loot-record-demos`` console script cannot honour that order: importing
# this module imports loot_bench/__init__.py first, which registers the tasks and
# therefore loads mjlab; the same limit loot_bench.qc.convert documents. The
# reversed order raised nothing when measured, so the entry point exists; run
# the file directly if a segfault ever appears.
import lerobot.datasets.lerobot_dataset  # noqa: F401,E402,I001

import argparse
import contextlib
import datetime
import gc
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING
from types import SimpleNamespace

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
from mjlab.viewer.native.keys import KEY_C, KEY_D, KEY_F, KEY_Q, KEY_U, KEY_V
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

import loot_bench  # noqa: F401
from loot_bench.robots.trlc_dk1.dk1_constants import HOME_JOINT_POS, START_JOINT_POS
from loot_bench.scripts.live_bridge.mapper import (
    leader_to_delta_action,
)
from loot_bench.scripts.live_bridge.recorder import Recorder
from loot_bench.scripts.live_bridge.recording_mixin import RecordingViewerMixin
from loot_bench.scripts.utils.viewer_wrapper import NativeViewerWrapper
from loot_bench.tasks.common.actions import EE_DELTA_ACTION_NAME


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
# Ghost-arm colors, feedback timing and the recording state machine all live
# in recording_mixin. The episode-start gate lives in mapper.leader_is_aligned:
# it asks whether one unclamped action reaches the leader from the reset pose,
# which is a question about the action box, not about joint angles.

# Sourced from the robot constant so the pose the operator aligns to and the
# pose the env resets to cannot drift apart.
LIVE_START_POSE: dict[str, float] = dict(START_JOINT_POS)
# Sim-frame copy of the start pose (same frame as arm and mapper).
_SIM_START_POSE: dict[str, float] = {
    **LIVE_START_POSE,
}


KEY_SAVE_SUCCESS = KEY_V
KEY_SAVE_FAILURE = KEY_F
KEY_DISCARD = KEY_D
KEY_UNDO = KEY_U
KEY_QUIT = KEY_Q
KEY_CYCLE_VIEW = KEY_C

# Wrist camera name; compiled model prefixes it, so match by suffix.
WRIST_CAMERA_NAME = "wrist_camera"


FRONT_CAMERA_NAME = "front_cam"


# Orbit view: over-the-shoulder free camera, also the fallback when a fixed
# camera is missing from the model.
ORBIT_CAM_LOOKAT = (0.22, 0.0, 0.12)  # world point the camera aims at (≈ workspace)
ORBIT_CAM_DISTANCE = 1.1
ORBIT_CAM_AZIMUTH = 30.0  # off-axis; near 0 = straight behind the base
ORBIT_CAM_ELEVATION = -22.0

# Views the C key steps through, in order; front is the startup view.
VIEW_FRONT = "front"
VIEW_WRIST = "wrist"
VIEW_ORBIT = "orbit"
VIEW_CYCLE = (VIEW_FRONT, VIEW_WRIST, VIEW_ORBIT)

GRASP_SITE_NAME = "grasp_site"
# Robot body prefix; laser ray skips these geoms, stops at first other hit.
ROBOT_ENTITY_PREFIX = "robot/"

# Bright magenta
GRIPPER_LASER_COLOR = (1.0, 0.0, 1.0, 1.0)
GRIPPER_LASER_ROD_RADIUS = 0.0025  # m
GRIPPER_LASER_DOT_RADIUS = 0.015  # m
GRIPPER_LASER_DOT_LIFT = 0.001  # m above the hit surface, avoids z-fighting
GRIPPER_LASER_DOT_THICKNESS = 0.002  # m
# Generous upper bound on robot geoms (fingers, wrist, forearm...) the ray
# may pierce before reaching a non-robot surface, plus the extra jumps spent
# exiting a pierced held block (one 5 mm jump per 5 mm of block thickness).
_GRIPPER_RAY_MAX_BOUNCES = 25
_GRIPPER_RAY_NUDGE = 1e-4  # m, push past a pierced surface before recasting

# Max distance (m) from the grasp site for a body to count as held. Mirrors
# GRIP_REACH in tasks/stack/mdp/commands.py (same 0.08 m the success gate
# uses). Proximity fallback only; pad contacts find the held body first.
_HELD_PROXIMITY = 0.08
# A pad contact counts as touching at or below this gap (m). Penetration is
# negative; 1 mm of tolerance covers a lightly resting grip.
_CONTACT_TOUCH_DIST = 1e-3
# A graspable object geom: largest half extent (m) below this. Keeps the
# table/board (and the floor plane) from ever counting as held.
_SMALL_GEOM_MAX_HALF = 0.05
# Finger-pad geom name suffixes (compiled model prefixes them with robot/).
_PAD_GEOM_SUFFIXES = ("finger_left_pad_collision", "finger_right_pad_collision")
# Step (m) past a zero-distance hit. mj_ray reports ~0 when the origin sits
# inside a geom (projection xz buried in the held block); the normal nudge
# would need hundreds of bounces to exit, so jump instead.
_GRIPPER_RAY_EXIT_STEP = 0.005
# Ghost footprint preview (KISS alternative to a full block ghost): one
# translucent axis-aligned box at the laser hit, sized from the held geom.
_FOOTPRINT_COLOR = (1.0, 0.0, 1.0, 0.35)
_FOOTPRINT_INFLATE = 0.0015  # m, same anti-z-fight margin as MARKER_INFLATE
_FOOTPRINT_FALLBACK_HALF = (0.010, 0.010, 0.010)  # m, if held geom isn't a box

# Wrist PiP render size. Fixed 640x480 (2x the original 320x240): sharp
# enough when enlarged, no per-resize renderer recreation.
_PIP_WIDTH = 640
_PIP_HEIGHT = 480


def ask_operator() -> int:
    """Block until the user enters a valid operator number (1-4)."""
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
    """Read leader at ~200 Hz, forward to follower, push latest to lists."""
    from loot_bench.scripts.live_bridge.reader import LiveReader

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
    """Viewer policy: the delta action chasing the latest leader pose.

    The delta is built from where the sim arm is *now*, so the recorded action
    is the one env.step() takes and one a bounded policy could emit. The
    current pose comes off EndEffectorDeltaAction.frame_pose — the buffer the
    IK solver reads — not the ee_pose observation, which is another frame and
    a lossy 6-D rotation.
    """

    def __init__(self, env, action_list: list, device: str):
        self._env = env
        self.action_list = action_list
        self._device = device
        self._last_raw_action = torch.zeros(1, 7, device=device)
        self.latest_leader_action: np.ndarray | None = None
        self.latest_raw_action: np.ndarray | None = None
        """The action last returned, as the recorder stores it. Published so
        the dataset holds exactly what ``env.step()`` was given rather than a
        second, re-derived copy of it (ADR-011)."""

    def _current_grasp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """The sim grasp site's pose in the robot base frame."""
        inner = self._env.env
        term = inner.action_manager.get_term(EE_DELTA_ACTION_NAME)
        pos_w, quat_w = term.frame_pose
        # frame_pose is world-frame; the leader's FK is base-frame. One
        # env_origin apart, and silently wrong if not subtracted.
        pos = (pos_w - inner.scene.env_origins)[0].detach().cpu().numpy()
        return pos, quat_w[0].detach().cpu().numpy()

    def __call__(self, obs):
        leader_action = self.action_list[0]
        if leader_action is not None:
            self.latest_leader_action = leader_action
            current_pos, current_quat = self._current_grasp_pose()
            raw_action = leader_to_delta_action(
                leader_action, current_pos, current_quat
            )
            self.latest_raw_action = raw_action
            self._last_raw_action = (
                torch.from_numpy(raw_action).unsqueeze(0).to(self._device)
            )
        return self._last_raw_action


class _LiveRecordingViewer(RecordingViewerMixin, NativeViewerWrapper):
    """Native viewer with recording; state machine lives in the mixin."""

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
        debug_obs_append: bool = False,
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
            task=task,
            num_episodes=num_episodes,
            debug_timing=debug_timing,
            sim_start_pose=_SIM_START_POSE,
            home_joint_pos=HOME_JOINT_POS,
            env=env,
            debug_obs_append=debug_obs_append,
        )
        # View cycle (C); camera ids resolved lazily, see _apply_view.
        self._view_idx = 0
        self._front_cam_id = -1
        self._wrist_cam_id = -1
        # Wrist PiP renderer is built by the caller before HW start.
        # Cached visualizers avoid per-frame MjData allocs.
        self._baseline_visflags: np.ndarray | None = None
        self._vis_task: MujocoNativeDebugVisualizer | None = None
        self._vis_ghost: MujocoNativeDebugVisualizer | None = None
        self._vis_wrist: MujocoNativeDebugVisualizer | None = None
        self._vis_gripper_laser: MujocoNativeDebugVisualizer | None = None
        self._grasp_site_id: int = -1
        # Finger-pad geoms for the laser's held-body contact check; resolved
        # lazily on first laser frame, then cached for the session.
        self._pad_geom_ids: list[int] | None = None
        self._wrist_renderer = wrist_renderer
        self._wrist_cam_id_pip = wrist_cam_id_pip
        self._pip_app = pip_app
        self._pip_window = pip_window
        self._pip_label = pip_label
        # PiP updates throttled to ~15 Hz to protect the HW thread.
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

        self._front_cam_id = _find_camera_id(self.mjm, FRONT_CAMERA_NAME)
        if self._front_cam_id < 0:
            print(
                f"[WARN] No '{FRONT_CAMERA_NAME}' camera in model — "
                "front view falls back to the orbit free-cam"
            )
        self._view_idx = 0
        self._apply_view(VIEW_CYCLE[self._view_idx])

        # Pre-construct both debug visualizers once, reusing the same MjData
        # throughout the session instead of allocating a new one every frame.
        self._vis_task = MujocoNativeDebugVisualizer(
            self.viewer.user_scn, self.mjm, self.env_idx, self._show_all_envs
        )
        self._vis_ghost = MujocoNativeDebugVisualizer(
            self.viewer.user_scn, self.mjm, self.env_idx, self._show_all_envs
        )
        self._vis_gripper_laser = MujocoNativeDebugVisualizer(
            self.viewer.user_scn, self.mjm, self.env_idx, self._show_all_envs
        )
        self._grasp_site_id = _find_site_id(self.mjm, GRASP_SITE_NAME)
        if self._grasp_site_id < 0:
            print(
                f"[WARN] No '{GRASP_SITE_NAME}' site in model — gripper laser disabled"
            )
        # PiP renderer has its own scene; needs its own cached visualizer.
        if self._wrist_renderer is not None and self._wrist_cam_id_pip >= 0:
            self._vis_wrist = MujocoNativeDebugVisualizer(
                self._wrist_renderer.scene,
                self.mjm,
                self.env_idx,
                self._show_all_envs,
            )

    # ── Ghost arm rendering (native-specific) ────────────────────────────────

    def _reset_visflags(self) -> None:
        """Restore baseline vis flags (recording keys overlap viewer toggles)."""
        if self.viewer is not None and self._baseline_visflags is not None:
            self.viewer.opt.flags[:] = self._baseline_visflags

    def _update_debug_visualizers(self, viewer: "_mujoco.viewer.Handle") -> None:  # type: ignore[name-defined]
        """Draw ghost-at-home overlay on top of task debug geoms each frame."""
        viewer.user_scn.ngeom = 0
        # Reuse cached visualizer; per-frame MjData allocs caused FPS decay.
        if self._show_debug_vis and hasattr(self.env.unwrapped, "update_visualizers"):
            assert self._vis_task is not None
            self.env.unwrapped.update_visualizers(self._vis_task)
        # Re-assert the visualization-flag baseline every frame so the native
        # viewer's built-in single-key flag toggles can't alter the scene
        if viewer is not None and self._baseline_visflags is not None:
            viewer.opt.flags[:] = self._baseline_visflags
        self._reset_visflags()
        if self._ghost_model is None or self._ghost_qpos is None:
            return
        assert self.mjm is not None
        # Feedback color if active, else neutral ghost (cached, no copy).
        current_color = self._get_ghost_color()
        ghost_model = self._get_ghost_model_with_color(current_color)
        ngeom_before = viewer.user_scn.ngeom
        self._vis_ghost.add_ghost_mesh(
            self._ghost_qpos, model=ghost_model, label="ghost"
        )
        # Don't cast shadows for ghost geoms through tagging them as DECOR
        for i in range(ngeom_before, viewer.user_scn.ngeom):
            viewer.user_scn.geoms[i].category = _mujoco.mjtCatBit.mjCAT_DECOR

        # Gripper laser: down projection of the held object, beam plus landing
        # dot. The projection drops from the object's xz (not the grasp
        # site's), so an off-centre grip still previews where the object
        # would land; a translucent box in its orientation marks the pose.
        if self._grasp_site_id >= 0 and self._vis_gripper_laser is not None:
            assert self.mjd is not None
            grasp_pos = self.mjd.site_xpos[self._grasp_site_id].copy()
            if self._pad_geom_ids is None:
                try:
                    self._pad_geom_ids = _find_pad_geom_ids(self.mjm)
                except Exception:
                    self._pad_geom_ids = []
            hit, held_half, held_mat, beam_top = _cast_gripper_ray(
                self.mjm, self.mjd, grasp_pos, self._pad_geom_ids
            )
            if hit is not None:
                self._vis_gripper_laser.add_cylinder(
                    start=beam_top,
                    end=hit,
                    radius=GRIPPER_LASER_ROD_RADIUS,
                    color=GRIPPER_LASER_COLOR,
                )
                dot_bottom = hit.copy()
                dot_bottom[2] += GRIPPER_LASER_DOT_LIFT
                dot_top = dot_bottom.copy()
                dot_top[2] += GRIPPER_LASER_DOT_THICKNESS
                self._vis_gripper_laser.add_cylinder(
                    start=dot_bottom,
                    end=dot_top,
                    radius=GRIPPER_LASER_DOT_RADIUS,
                    color=GRIPPER_LASER_COLOR,
                )
                if held_half is not None and held_mat is not None:
                    fp_half = np.asarray(held_half, dtype=float) + _FOOTPRINT_INFLATE
                    r = np.asarray(held_mat, dtype=float).reshape(3, 3)
                    world_hz = (
                        abs(r[2, 0]) * fp_half[0]
                        + abs(r[2, 1]) * fp_half[1]
                        + abs(r[2, 2]) * fp_half[2]
                    )
                    fp_center = hit.copy()
                    fp_center[2] += GRIPPER_LASER_DOT_LIFT + world_hz
                    self._vis_gripper_laser.add_box(
                        center=fp_center,
                        size=fp_half,
                        mat=r,
                        color=_FOOTPRINT_COLOR,
                    )

        # PiP wrist render, throttled (never at full frame rate).
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
            _disable_pip_shadows(self._wrist_renderer)
            # update_scene rebuilds the scene, so re-add target markers.
            if (
                self._show_debug_vis
                and self._vis_wrist is not None
                and hasattr(self.env.unwrapped, "update_visualizers")
            ):
                self.env.unwrapped.update_visualizers(self._vis_wrist)
            img = self._wrist_renderer.render()
            if VIEW_CYCLE[self._view_idx] == VIEW_FRONT:
                img = img[::-1, ::-1]
            QImage, QPixmap = _get_qimage_and_qpixmap()
            h, w = img.shape[0], img.shape[1]
            qimg = QImage(img.tobytes(), w, h, QImage.Format_RGB888)
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

    # ── HUD (native-specific) ────────────────────────────────────────────────

    def _set_status_overlay(self, viewer: "_mujoco.viewer.Handle") -> None:  # type: ignore[name-defined]
        """Send base overlay plus recording HUD in one set_texts call."""
        now = time.perf_counter()
        if now - self._last_overlay_time < self.OVERLAY_INTERVAL:
            return
        self._last_overlay_time = now

        captured: list = []
        shim = SimpleNamespace(set_texts=captured.append)
        NativeMujocoViewer._set_status_overlay(self, shim)  # type: ignore[arg-type]
        labels, values = self._hud_text()
        hud = (
            _mujoco.mjtFontScale.mjFONTSCALE_150.value,
            _mujoco.mjtGridPos.mjGRID_BOTTOMLEFT.value,
            labels,
            values,
        )
        viewer.set_texts(captured + [hud])

    # ── Key handling ─────────────────────────────────────────────────────────

    def _on_key(self, key: int) -> None:
        self._reset_visflags()
        if key == KEY_SAVE_SUCCESS:
            self.request_action("CUSTOM", "save_success")
        elif key == KEY_SAVE_FAILURE:
            self.request_action("CUSTOM", "save_failure")
        elif key == KEY_DISCARD:
            self.request_action("CUSTOM", "discard")
        elif key == KEY_UNDO:
            self.request_action("CUSTOM", "undo")
        elif key == KEY_CYCLE_VIEW:
            # Only mutates the viewer's own camera (never env/sim), so it is
            # safe to handle directly on the viewer thread.
            self._cycle_view()
        elif key == KEY_QUIT:
            self._interrupted = True

    def _cycle_view(self) -> None:
        """Step the viewer camera to the next view in VIEW_CYCLE."""
        self._view_idx = (self._view_idx + 1) % len(VIEW_CYCLE)
        self._apply_view(VIEW_CYCLE[self._view_idx])

    def _apply_view(self, view: str) -> None:
        """Point the viewer camera at the named view.

        A view whose fixed camera is missing from the model degrades to the
        orbit free-cam, so the cycle never lands on an invalid camera.
        """
        v = self.viewer
        if v is None or self.mjm is None:
            return

        if view == VIEW_WRIST:
            if self._wrist_cam_id < 0:
                self._wrist_cam_id = _find_camera_id(self.mjm, WRIST_CAMERA_NAME)
            if self._wrist_cam_id >= 0:
                v.cam.type = _mujoco.mjtCamera.mjCAMERA_FIXED.value
                v.cam.fixedcamid = self._wrist_cam_id
                print("[INFO] View: wrist camera (press C to cycle)")
                return
            print(f"[WARN] No '{WRIST_CAMERA_NAME}' camera in model — skipping")
            view = VIEW_ORBIT

        if view == VIEW_FRONT and self._front_cam_id >= 0:
            v.cam.type = _mujoco.mjtCamera.mjCAMERA_FIXED.value
            v.cam.fixedcamid = self._front_cam_id
            print("[INFO] View: front camera (press C to cycle)")
            return

        v.cam.type = _mujoco.mjtCamera.mjCAMERA_FREE.value
        v.cam.lookat = ORBIT_CAM_LOOKAT
        v.cam.distance = ORBIT_CAM_DISTANCE
        v.cam.azimuth = ORBIT_CAM_AZIMUTH
        v.cam.elevation = ORBIT_CAM_ELEVATION
        print("[INFO] View: orbit / over-the-shoulder (press C to cycle)")

    # Recording state machine (_handle_custom_action, _end_episode, _handle_done,
    # _execute_step, ghost-color helpers) is inherited from RecordingViewerMixin.


def _disable_shadows_global(mj_model: "_mujoco.MjModel | None") -> None:
    """Turn off shadow casting at the model level (all renderers at once).

    The main viewer additionally clears its RND_SHADOW scene flag via
    ViewerConfig(enable_shadows=False); the wrist PiP clears its own scene
    flag in _build_wrist_pip / the PiP update. Setting light_castshadow here
    is what makes it truly global: no light casts, so neither renderer nor
    any training camera can produce shadows.
    """
    if mj_model is None:
        return
    try:
        mj_model.light_castshadow[:] = False
    except Exception as exc:  # pragma: no cover - model-variant guard
        print(f"[WARN] Could not disable light shadows: {exc}")


def _disable_pip_shadows(wrist_renderer: "_mujoco.Renderer | None") -> None:
    """Clear the PiP scene's shadow flag (cheap, idempotent)."""
    if wrist_renderer is None:
        return
    with contextlib.suppress(Exception):
        wrist_renderer.scene.flags[_mujoco.mjtRndFlag.mjRND_SHADOW] = 0


def _find_camera_id(mj_model: "_mujoco.MjModel", suffix: str) -> int:
    """Return first camera matching suffix; -1 if none (prefix-aware)."""
    for ci in range(mj_model.ncam):
        name = _mujoco.mj_id2name(mj_model, _mujoco.mjtObj.mjOBJ_CAMERA, ci) or ""
        if name == suffix or name.endswith("/" + suffix):
            return ci
    return -1


def _find_site_id(mj_model: "_mujoco.MjModel", suffix: str) -> int:
    """Return the id of the first site whose name is, or ends with, ``suffix``.

    Same suffix-match rationale as _find_camera_id: the compiled model
    prefixes entity sites (e.g. "robot/grasp_site"). Returns -1 if none match.
    """
    for si in range(mj_model.nsite):
        name = _mujoco.mj_id2name(mj_model, _mujoco.mjtObj.mjOBJ_SITE, si) or ""
        if name == suffix or name.endswith("/" + suffix):
            return si
    return -1


def _find_pad_geom_ids(mjm: "_mujoco.MjModel") -> list[int]:
    """Geom ids of the finger pads (suffix match: model prefixes robot geoms)."""
    ids = []
    for gi in range(int(mjm.ngeom)):
        name = _mujoco.mj_id2name(mjm, _mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if name.endswith(_PAD_GEOM_SUFFIXES):
            ids.append(gi)
    return ids


def _geom_is_small(mjm: "_mujoco.MjModel", gid: int) -> bool:
    try:
        return bool(
            np.max(np.asarray(mjm.geom_size[gid], dtype=float)[:3])
            < _SMALL_GEOM_MAX_HALF
        )
    except Exception:
        return False


def _body_is_robot(mjm: "_mujoco.MjModel", body_id: int) -> bool:
    try:
        name = _mujoco.mj_id2name(mjm, _mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or ""
    except Exception:
        return False
    return name.startswith(ROBOT_ENTITY_PREFIX)


def _body_has_small_geom(mjm: "_mujoco.MjModel", body_id: int) -> bool:
    try:
        bodyids = np.asarray(mjm.geom_bodyid)
        sizes = np.asarray(mjm.geom_size, dtype=float)[:, :3]
    except Exception:
        return False
    small = np.max(sizes, axis=1) < _SMALL_GEOM_MAX_HALF
    return bool(np.any(small & (bodyids == int(body_id))))


def _find_held_body(
    mjm: "_mujoco.MjModel",
    mjd: "_mujoco.MjData",
    origin: np.ndarray,
    pad_geom_ids: "list[int] | None" = None,
) -> "int | None":
    """Body id of the object in the gripper, else None.

    Pad contacts first: while the pads touch a small object it is held no
    matter how badly (tilted / edge-held / slipped) it sits in the fingers.
    Proximity fallback: the nearest small body riding with the grasp site.
    World body 0, robot bodies and large geoms (table/board) never qualify.
    """
    if pad_geom_ids:
        pad = {int(g) for g in pad_geom_ids}
        try:
            ncon = int(mjd.ncon)
        except Exception:
            ncon = 0
        best: int | None = None
        best_d = float("inf")
        for i in range(ncon):
            try:
                c = mjd.contact[i]
                g1, g2, dist = int(c.geom1), int(c.geom2), float(c.dist)
            except Exception:
                continue
            if dist >= _CONTACT_TOUCH_DIST:
                continue
            if g1 in pad:
                other = g2
            elif g2 in pad:
                other = g1
            else:
                continue
            try:
                body_id = int(mjm.geom_bodyid[other])
            except Exception:
                continue
            if body_id == 0 or _body_is_robot(mjm, body_id):
                continue
            if not _geom_is_small(mjm, other):
                continue
            try:
                d = float(np.linalg.norm(np.asarray(mjd.xpos[body_id]) - origin))
            except Exception:
                d = float("inf")
            if d < best_d:
                best, best_d = body_id, d
        if best is not None:
            return best
    try:
        nbody = int(mjm.nbody)
    except Exception:
        return None
    best = None
    best_d = _HELD_PROXIMITY
    for bid in range(1, nbody):
        try:
            pos = np.asarray(mjd.xpos[bid], dtype=float)
        except Exception:
            continue
        d = float(np.linalg.norm(pos - origin))
        if d >= best_d or _body_is_robot(mjm, bid):
            continue
        if not _body_has_small_geom(mjm, bid):
            continue
        best, best_d = bid, d
    return best


def _held_footprint_geom(mjm: "_mujoco.MjModel", body_id: int) -> "int | None":
    """First small box geom of the body (its footprint size), else None."""
    try:
        bodyids = np.asarray(mjm.geom_bodyid)
        box_type = int(_mujoco.mjtGeom.mjGEOM_BOX.value)
        for gid in np.where(bodyids == int(body_id))[0]:
            gid = int(gid)
            if _geom_is_small(mjm, gid) and int(mjm.geom_type[gid]) == box_type:
                return gid
    except Exception:
        pass
    return None


def _cast_down_ray(
    mjm: "_mujoco.MjModel",
    mjd: "_mujoco.MjData",
    origin: np.ndarray,
    skip_body_ids: "set[int] | None" = None,
) -> "np.ndarray | None":
    """Down ray from origin; pierces robot geoms and skipped bodies."""
    skip = skip_body_ids or set()
    vec = np.array([0.0, 0.0, -1.0])
    pnt = origin.copy()
    geomid = np.zeros(1, dtype=np.int32)
    for _ in range(_GRIPPER_RAY_MAX_BOUNCES):
        dist = _mujoco.mj_ray(mjm, mjd, pnt, vec, None, True, -1, geomid)
        if dist < 0:
            return None
        hit = pnt + dist * vec
        if dist < _GRIPPER_RAY_NUDGE:
            # Origin inside the geom: jump, don't crawl, so the loop exits
            # within its bounce budget.
            pnt = pnt + _GRIPPER_RAY_EXIT_STEP * vec
            continue
        gid = int(geomid[0])
        try:
            body_id = int(mjm.geom_bodyid[gid])
        except Exception:
            return hit
        if body_id in skip or _body_is_robot(mjm, body_id):
            pnt = hit + _GRIPPER_RAY_NUDGE * vec
            continue
        return hit
    return None


def _cast_gripper_ray(
    mjm: "_mujoco.MjModel",
    mjd: "_mujoco.MjData",
    origin: np.ndarray,
    pad_geom_ids: "list[int] | None" = None,
) -> tuple[
    "np.ndarray | None",
    "np.ndarray | None",
    "np.ndarray | None",
    np.ndarray,
]:
    """Down projection of the held object (gripper laser).

    Returns ``(hit, held_half, held_mat, beam_top)``: ``hit`` is None on
    miss; ``held_half``/``held_mat`` size/orient the footprint preview and
    are None with an empty gripper. The projection drops from the *object's*
    xz, not the grasp site's: an off-centre (end-held / tilted) block lands
    its footprint under itself, where it would actually land on release.
    """
    held = _find_held_body(mjm, mjd, origin, pad_geom_ids)
    if held is None:
        return _cast_down_ray(mjm, mjd, origin), None, None, origin.copy()
    try:
        centre = np.asarray(mjd.xpos[held], dtype=float).reshape(3)
    except Exception:
        return _cast_down_ray(mjm, mjd, origin), None, None, origin.copy()
    beam_top = np.array([centre[0], centre[1], origin[2]])
    hit = _cast_down_ray(mjm, mjd, beam_top, skip_body_ids={held})
    gid = _held_footprint_geom(mjm, held)
    if gid is None:
        held_half: np.ndarray | None = np.array(_FOOTPRINT_FALLBACK_HALF, dtype=float)
    else:
        try:
            held_half = np.asarray(mjm.geom_size[gid], dtype=float).copy()
        except Exception:
            held_half = np.array(_FOOTPRINT_FALLBACK_HALF, dtype=float)
    try:
        held_mat: np.ndarray | None = np.asarray(mjd.xmat[held], dtype=float).reshape(
            3, 3
        )
    except Exception:
        held_mat = None
    return hit, held_half, held_mat, beam_top


def _build_wrist_pip(mj_model: "_mujoco.MjModel | None"):
    """Wrist PiP with aspect-locked resize."""
    if mj_model is None:
        return None, -1, None, None, None
    cam_id = _find_camera_id(mj_model, WRIST_CAMERA_NAME)
    if cam_id < 0:
        return None, -1, None, None, None

    from PyQt5.QtWidgets import QLabel, QMainWindow, QSizePolicy

    class _AspectWindow(QMainWindow):
        def resizeEvent(self, event):  # noqa: N802
            super().resizeEvent(event)
            central = self.centralWidget()
            if central is None:
                return
            want = int(round(central.width() * _PIP_HEIGHT / _PIP_WIDTH))
            if abs(central.height() - want) > 1:
                self.resize(self.width(), want + self.height() - central.height())

    renderer = _mujoco.Renderer(mj_model, height=_PIP_HEIGHT, width=_PIP_WIDTH)
    _disable_pip_shadows(renderer)
    app = _get_pip_app()
    window = _AspectWindow()
    label = QLabel()
    label.setScaledContents(True)
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    label.setMinimumSize(320, 240)
    window.setCentralWidget(label)
    window.setWindowTitle("wrist")
    window.resize(_PIP_WIDTH, _PIP_HEIGHT)
    with contextlib.suppress(Exception):
        window.move(app.primaryScreen().availableGeometry().topLeft())
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
        default="LOOT-Place-FixedS-FixedT-Dk1",
        help="MJLab task name",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help=(
            "Output base directory. Defaults to out/demos/ at the repo root. "
            "Episodes are saved to <base>/success/<task>/<timestamp>/ and "
            "<base>/failure/<task>/<timestamp>/ (data is never overwritten)."
        ),
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="Stop after N saved episodes (0 = record until Q is pressed).",
    )
    parser.add_argument(
        "--start-pose-noise",
        type=float,
        default=None,
        help=(
            "Override the task's start-pose sample half-width [rad]. Omit it: "
            "the task's own value is used, which is what makes a demo start "
            "where an evaluation episode starts. Pass 0.0 for a bit-identical "
            "start every episode. Keep any value well under the 0.2 rad "
            "alignment gate, which is measured against the nominal pose."
        ),
    )
    parser.add_argument(
        "--debug-timing",
        action="store_true",
        help=(
            "Log a per-stage breakdown for any recording step over budget "
            "(prints [STEP-SPIKE] lines) to diagnose intermittent FPS drops."
        ),
    )
    parser.add_argument(
        "--debug-obs",
        action="store_true",
        help="Append-only debug panel (keep full per-frame history).",
    )
    args = parser.parse_args()

    # 0. Identify operator for session metadata.
    operator_id = ask_operator()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1. Load task config under the *training* episode length, not play's
    #    endless one. The recorder used to build with play=True, so it was the
    #    one place with no timeout while training and eval always capped —
    #    which is how 20 of the 99 place demos came to run past the budget and
    #    lose their successful endings at conversion. An over-long episode is
    #    now visibly truncated while the operator flies it (ADR-016).
    #    `time_out=False` keeps the truncation out of `terminated`, so the
    #    success termination still auto-saves as success.
    cfg = load_env_cfg(args.task, play=False)
    cfg.terminations["time_out"].time_out = False
    # First action steps back to nominal pose; keep it as real data.
    if args.start_pose_noise is not None:
        cfg.events["reset_robot_joints"].params["position_range"] = (
            -args.start_pose_noise,
            args.start_pose_noise,
        )
    noise_lo, noise_hi = cfg.events["reset_robot_joints"].params["position_range"]
    print(f"[INFO] Start-pose noise: {noise_lo:+.3f} .. {noise_hi:+.3f} rad")

    # 2. Create environment.
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env = RslRlVecEnvWrapper(env)
    print("[INFO] Environment created")

    # Static world-frame framing: the orbit leg of the C view cycle.
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
    # Global shadow kill: no light casts, so the main viewer, the wrist PiP
    # and any camera render shadow-free (cfg.enable_shadows=False already
    # clears the main viewer's scene flag; PiP clears its own flag).
    _disable_shadows_global(mj_model)

    # Dataset dirs are created lazily; never wipe existing data.
    base_dir = Path(args.dataset_dir) if args.dataset_dir else DEMOS_DIR
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    success_root = base_dir / "success" / args.task / stamp
    failure_root = base_dir / "failure" / args.task / stamp
    for _path in (success_root, failure_root):
        if _path.exists():
            raise SystemExit(
                f"[ERROR] {_path} already exists (refusing to overwrite recorded data)."
            )

    # 5. Set up recorder.
    obs_dim = int(env.get_observations()["actor"].shape[-1])
    qpos_dim = int(env.env.sim.data.qpos.shape[-1])
    qvel_dim = int(env.env.sim.data.qvel.shape[-1])
    recorder = Recorder(
        repo_id="live_record",
        fps=round(1.0 / env.env.step_dt),
        obs_dim=obs_dim,
        success_root=str(success_root),
        failure_root=str(failure_root),
        qpos_dim=qpos_dim,
        qvel_dim=qvel_dim,
        device=device,
        start_pose_noise=noise_hi,
    )
    print(
        f"[INFO] Recording to {base_dir}/{{success,failure}}/{args.task}/{stamp} "
        f"(obs_dim={obs_dim}, qpos_dim={qpos_dim}, qvel_dim={qvel_dim})"
    )
    print(
        "[INFO] Debug panel: "
        + ("append-only (--debug-obs)" if args.debug_obs else "in-place overwrite")
    )

    # 6. Shared single-element lists (GIL-safe last-writer-wins).
    action_list: list = [None]
    robot_obs_list: list = [None]
    shutdown_flag: list = [False]
    hw_failed: list = [False]

    # Build PiP before HW thread (one-time init can block). See helper.
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
    live_policy = _LivePolicy(env, action_list, device)

    print("[INFO] Starting native MuJoCo viewer")
    print("[INFO] Move the leader arm to control the sim and follower")
    print("[INFO] Keys (see also the HUD in the viewer):")
    print("[INFO]   V → save current episode as SUCCESS")
    print("[INFO]   F → save current episode as FAILURE")
    print(
        "[INFO]   D → discard/redo current episode "
        "(also discards a collision within 2s)"
    )
    print("[INFO]   U → undo last saved episode (single-level)")
    print("[INFO]   Q → stop recording session")
    print("[INFO]   C → cycle view (front → wrist → orbit)")
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
            debug_obs_append=args.debug_obs,
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
