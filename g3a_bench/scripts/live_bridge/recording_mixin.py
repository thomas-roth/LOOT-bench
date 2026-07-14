"""Viewer-agnostic recording state machine (record → reset phase → collision label).
Only viewer-specific drawing differs; everything else here to avoid drift."""

import copy
import time
import traceback

import mujoco as _mujoco
import numpy as np
import torch
from mjlab.viewer.base import VerbosityLevel

GHOST_ROBOT_ENTITY = "robot"  # scene entity key whose geoms get ghosted
GHOST_COLOR = (0.5, 0.6, 0.9, 0.5)  # translucent blue, the neutral/reset-target color
GHOST_COLOR_SUCCESS = (0.3, 0.9, 0.3, 0.8)  # bright green for success feedback
GHOST_COLOR_FAILURE = (0.9, 0.3, 0.3, 0.8)  # bright red for failure feedback
RESET_THRESHOLD_RAD = 0.2  # all 6 arm joints must be within this many rad of home


class RecordingViewerMixin:
    """Multi-episode recording + reset phase. States: RECORDING → (collision:
    PROVISIONAL_FAILURE, auto-saves after a 2s grace period unless discarded)
    → RESET_PHASE → next episode."""

    def _init_recording(
        self,
        recorder,
        robot_obs_list: list,
        hw_failed: list,
        mj_model: "_mujoco.MjModel | None",
        live_rest_pose_rel: np.ndarray,
        task: str,
        num_episodes: int,
        debug_timing: bool,
        sim_rest_pose: dict,
        home_joint_pos: dict,
    ) -> None:
        """Set up recording state; call from subclass __init__."""
        self._recorder = recorder
        self._robot_obs_list = robot_obs_list
        self._hw_failed = hw_failed
        self._debug_timing = debug_timing
        self._live_rest_pose_rel = live_rest_pose_rel
        self._spike_ms = 35.0
        self._worst_step_ms = 0.0
        self._ghost_model: _mujoco.MjModel | None = None
        self._ghost_qpos: np.ndarray | None = None
        # Stable object identity per color (Viser caches meshes by id)
        self._ghost_model_cache: dict[tuple, _mujoco.MjModel] = {}
        if mj_model is not None:
            self._ghost_model, self._ghost_qpos = build_ghost_model_and_qpos(
                mj_model, sim_rest_pose, home_joint_pos
            )
        self._task = task
        self._num_episodes = num_episodes
        self.episodes_saved = 0
        self._in_reset_phase = False
        self._feedback_color: tuple[float, float, float, float] | None = None
        # Auto-save failure after ~2s if user doesn't press N (discard) first
        self._provisional_failure = False
        self._provisional_failure_time = 0.0
        self._current_ep_seed: int | None = None

    # Ghost arm helpers

    def _get_ghost_color(self) -> tuple[float, float, float, float]:
        """Success/failure color persists until the arm is realigned to the
        reset position (_exit_reset_phase clears it); otherwise the neutral
        reset-target color."""
        if self._feedback_color is None:
            return GHOST_COLOR
        return self._feedback_color

    def _get_ghost_model_with_color(
        self, color: tuple[float, float, float, float]
    ) -> "_mujoco.MjModel | None":
        """Return ghost model recolored, or None. Cached per color (stable id)."""
        if self._ghost_model is None:
            return None
        key = tuple(round(float(c), 4) for c in color)
        cached = self._ghost_model_cache.get(key)
        if cached is not None:
            return cached
        ghost_model = copy.deepcopy(self._ghost_model)
        prefix = f"{GHOST_ROBOT_ENTITY}/"
        for gi in range(ghost_model.ngeom):
            body_name = (
                _mujoco.mj_id2name(
                    ghost_model, _mujoco.mjtObj.mjOBJ_BODY, ghost_model.geom_bodyid[gi]
                )
                or ""
            )
            is_robot = body_name.startswith(prefix)
            is_collision = (
                ghost_model.geom_contype[gi] != 0
                or ghost_model.geom_conaffinity[gi] != 0
            )
            if is_robot and not is_collision:
                ghost_model.geom_rgba[gi] = color
            else:
                ghost_model.geom_rgba[gi, 3] = 0
        self._ghost_model_cache[key] = ghost_model
        return ghost_model

    # Reset phase

    def _enter_reset_phase(self) -> None:
        self._in_reset_phase = True
        self._provisional_failure = False
        # Suppress verbose per-tick spam during arm alignment
        self.verbosity = VerbosityLevel.INFO
        print(
            f"[INFO] Reset phase — move the arm in the sim to the ghost position "
            f"(within {RESET_THRESHOLD_RAD:.2f} rad per joint) to start next episode"
        )

    def _exit_reset_phase(self) -> None:
        self._in_reset_phase = False
        self._feedback_color = None
        self.verbosity = VerbosityLevel.DEBUG

    # Custom actions (GUI buttons / keys)

    def _handle_custom_action(self, action, payload) -> bool:  # type: ignore[override]
        if payload == "save_success":
            self._end_episode(success=True)
        elif payload == "save_failure":
            self._end_episode(success=False)
        elif payload == "discard":
            frames = self._recorder.num_frames_in_episode
            self._recorder.discard_episode()
            self._provisional_failure = False
            print(f"[INFO] Episode discarded ({frames} frames dropped)")
            self._feedback_color = GHOST_COLOR_FAILURE
            self._enter_reset_phase()
        else:
            # Delegate to concrete viewer (checkpoint, GUI-reset, etc.)
            return super()._handle_custom_action(action, payload)
        return True

    def _end_episode(self, success: bool) -> None:
        frames = self._recorder.num_frames_in_episode
        if frames == 0:
            print("[INFO] Nothing to save (no frames recorded yet)")
            return
        self._recorder.save_episode(success=success, seed=self._current_ep_seed)
        self.episodes_saved += 1
        label = "SUCCESS" if success else "FAILURE"
        color_label = "✓ SUCCESS" if success else "✗ FAILURE"
        print(
            f"[INFO] Episode {self.episodes_saved} saved as {label} ({frames} frames)"
        )

        self._feedback_color = GHOST_COLOR_SUCCESS if success else GHOST_COLOR_FAILURE

        folder = "success" if success else "failure"
        print(f"[{color_label}] Episode saved to {folder}/ folder")

        if self._num_episodes and self.episodes_saved >= self._num_episodes:
            print(f"[INFO] Reached {self._num_episodes} episodes — stopping")
            self._interrupted = True
        else:
            self._enter_reset_phase()

    def _handle_done(self) -> None:
        """Distinguish success from collision."""
        # Env auto-resets before done reaches us; read success term from termination manager
        try:
            tm = self.env.env.termination_manager
            is_success = bool(tm.get_term("success")[0].item())
        except Exception:
            is_success = False

        if is_success:
            print("[INFO] Task succeeded — auto-saving as success")
            self._end_episode(success=True)
        else:
            print(
                "[WARN] Ground collision — will save as failure in ~2s "
                "(discard within 2s to redo)"
            )
            self._provisional_failure = True
            self._provisional_failure_time = time.perf_counter()
            self._feedback_color = GHOST_COLOR_FAILURE
            self.verbosity = VerbosityLevel.INFO

    # Step loop

    def _log_step_timing(self, t0, t1, t2, t3, t4) -> None:
        """Log stage breakdown for slow steps."""
        total_ms = (t4 - t0) * 1e3
        if total_ms <= self._spike_ms:
            return
        self._worst_step_ms = max(self._worst_step_ms, total_ms)
        print(
            f"[STEP-SPIKE] {total_ms:4.0f} ms  obs={(t1 - t0) * 1e3:3.0f} "
            f"pol={(t2 - t1) * 1e3:3.0f} step={(t3 - t2) * 1e3:4.0f} "
            f"add={(t4 - t3) * 1e3:3.0f}  | worst={self._worst_step_ms:.0f} ms "
            f"(step#{self._step_count})"
        )

    def _execute_step(self) -> bool:
        if self._hw_failed[0]:
            self.log(
                "[ERROR] Hardware thread failed — aborting session.",
                VerbosityLevel.SILENT,
            )
            self._interrupted = True
            return False

        # Provisional failure: waiting for discard or auto-save
        if self._provisional_failure:
            timeout = 2.0  # seconds to wait before auto-saving
            elapsed = time.perf_counter() - self._provisional_failure_time
            if elapsed > timeout:
                print(f"[INFO] Saving failure episode (timeout {timeout}s reached)")
                self._provisional_failure = False
                self._end_episode(success=False)
            return True

        # Reset phase: ghost visible, not recorded; operator aligns arm to ghost
        if self._in_reset_phase:
            try:
                with torch.no_grad():
                    obs = self.env.get_observations()
                    actions = self.policy(obs)
                    self.env.step(actions)
                    self._step_count += 1
                    self._stats_steps += 1
                    joint_pos_rel = obs["actor"][0, :6].cpu().numpy()
                    if np.all(
                        np.abs(joint_pos_rel - self._live_rest_pose_rel)
                        < RESET_THRESHOLD_RAD
                    ):
                        self._exit_reset_phase()
                        print(
                            f"[INFO] === Episode {self.episodes_saved + 1} "
                            "started — recording resumed ==="
                        )
            except Exception:
                self._last_error = traceback.format_exc()
                self.log(
                    f"[ERROR] Exception during reset step:\n{self._last_error}",
                    VerbosityLevel.SILENT,
                )
                self.pause()
                return False
            return True

        # Normal recording step
        try:
            with torch.no_grad():
                t0 = time.perf_counter()
                obs = self.env.get_observations()
                t1 = time.perf_counter()
                actions = self.policy(obs)
                t2 = time.perf_counter()
                _obs_next, rew, dones, _extras = self.env.step(actions)
                t3 = time.perf_counter()

                # Only record if both leader action and robot obs are available
                leader_action = getattr(self.policy, "latest_leader_action", None)
                robot_obs = self._robot_obs_list[0]
                if leader_action is not None and robot_obs is not None:
                    # Ground-truth sim state. Cloned because the qpos buffer is
                    # overwritten next step, and kept on-device — the GPU→CPU
                    # copy is deferred to save time so this loop pays no extra
                    # CUDA sync (that stall dropped live fps from 60 to 45).
                    qpos_t = self.env.env.sim.data.qpos[0].clone()
                    qvel_t = self.env.env.sim.data.qvel[0].clone()
                    # rew[0] and dones[0] are GPU tensors — defer .item() to
                    # save_episode() to avoid per-step CUDA syncs.
                    self._recorder.add_frame(
                        {"actor": obs["actor"]},
                        leader_action,
                        rew[0],
                        dones[0],
                        task=self._task,
                        robot_obs=robot_obs,
                        qpos=qpos_t,
                        qvel=qvel_t,
                    )
                t4 = time.perf_counter()

                if self._debug_timing:
                    self._log_step_timing(t0, t1, t2, t3, t4)

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


def build_ghost_model_and_qpos(
    mj_model: "_mujoco.MjModel",
    sim_rest_pose: dict,
    home_joint_pos: dict,
) -> "tuple[_mujoco.MjModel, np.ndarray]":
    """Build ghost model: robot visual geoms only, colored, at rest pose."""
    ghost_model = copy.deepcopy(mj_model)
    prefix = f"{GHOST_ROBOT_ENTITY}/"
    for gi in range(ghost_model.ngeom):
        body_name = (
            _mujoco.mj_id2name(
                ghost_model, _mujoco.mjtObj.mjOBJ_BODY, ghost_model.geom_bodyid[gi]
            )
            or ""
        )
        is_robot = body_name.startswith(prefix)
        is_collision = (
            ghost_model.geom_contype[gi] != 0 or ghost_model.geom_conaffinity[gi] != 0
        )
        if is_robot and not is_collision:
            ghost_model.geom_rgba[gi] = GHOST_COLOR
        else:
            ghost_model.geom_rgba[gi, 3] = 0

    qpos = ghost_model.qpos0.copy()
    ghost_pose = {**home_joint_pos, **sim_rest_pose}
    for joint_name, value in ghost_pose.items():
        jid = _mujoco.mj_name2id(
            ghost_model, _mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{joint_name}"
        )
        if jid != -1:
            qpos[ghost_model.jnt_qposadr[jid]] = value
    return ghost_model, qpos
