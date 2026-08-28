"""Viewer-agnostic recording state machine (record → provisional failure →
reset phase). Only viewer-specific drawing differs; everything else here to
avoid drift."""

import copy
import time
import traceback

import mujoco as _mujoco
import numpy as np
import torch
from mjlab.viewer.base import VerbosityLevel

from g3a_bench.scripts.live_bridge.debug_panel import LiveDebugPanel
from g3a_bench.scripts.live_bridge.mapper import leader_is_aligned

GHOST_ROBOT_ENTITY = "robot"  # scene entity key whose geoms get ghosted
GHOST_COLOR = (0.5, 0.6, 0.9, 0.5)  # translucent blue, the neutral/reset-target color
GHOST_COLOR_SUCCESS = (0.3, 0.9, 0.3, 0.8)  # bright green for success feedback
GHOST_COLOR_FAILURE = (0.9, 0.3, 0.3, 0.8)  # bright red for failure feedback
GHOST_COLOR_PROMPT = (0.95, 0.6, 0.1, 0.8)  # orange during the save-or-discard window


class RecordingViewerMixin:
    """Multi-episode recording + reset phase. States: RECORDING → (non-success
    termination: PROVISIONAL_FAILURE, auto-saves after a 2s grace period unless
    discarded) → RESET_PHASE → next episode."""

    def _init_recording(
        self,
        recorder,
        robot_obs_list: list,
        hw_failed: list,
        mj_model: "_mujoco.MjModel | None",
        task: str,
        num_episodes: int,
        debug_timing: bool,
        sim_start_pose: dict,
        home_joint_pos: dict,
        env=None,
        debug_obs_append: bool = False,
    ) -> None:
        """Set up recording state; call from subclass __init__.

        The debug panel is always on (see debug_panel.py) — ``env`` is read
        once here to build its obs-term layout (self.env isn't assigned yet
        at this point in the subclass __init__, see live_record.py).
        ``debug_obs_append`` (``--debug-obs``) switches it from the default
        in-place-overwrite style to append-only (nothing ever overwritten).
        """
        self._recorder = recorder
        self._robot_obs_list = robot_obs_list
        self._hw_failed = hw_failed
        self._debug_timing = debug_timing
        self._debug_panel = LiveDebugPanel(env, append_only=debug_obs_append)
        self._spike_ms = 35.0
        self._worst_step_ms = 0.0
        self._ghost_model: _mujoco.MjModel | None = None
        self._ghost_qpos: np.ndarray | None = None
        # Stable object identity per color (Viser caches meshes by id)
        self._ghost_model_cache: dict[tuple, _mujoco.MjModel] = {}
        if mj_model is not None:
            self._ghost_model, self._ghost_qpos = build_ghost_model_and_qpos(
                mj_model, sim_start_pose, home_joint_pos
            )
        self._task = task
        self._num_episodes = num_episodes
        self._saved_success = 0
        self._saved_failure = 0
        self._in_reset_phase = False
        self._feedback_color: tuple[float, float, float, float] | None = None
        # Auto-save failure after ~2s if user doesn't press D (discard) first
        self._provisional_failure = False
        self._provisional_failure_time = 0.0
        self._current_ep_seed: int | None = None

    @property
    def episodes_saved(self) -> int:
        return self._saved_success + self._saved_failure

    # Ghost arm helpers

    def _get_ghost_color(self) -> tuple[float, float, float, float]:
        """Orange during the provisional-failure save-or-discard window;
        success/failure color persists until the arm is realigned to the
        reset position (_exit_reset_phase clears it); otherwise the neutral
        reset-target color."""
        if self._provisional_failure:
            return GHOST_COLOR_PROMPT
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
        # Ensure the next episode starts from a clean env reset (especially when an
        # episode is ended/discarded manually without env-driven termination).
        reset = getattr(self, "reset_environment", None)
        if callable(reset):
            reset()

        self._in_reset_phase = True
        self._provisional_failure = False
        # Suppress verbose per-tick spam during arm alignment
        self.verbosity = VerbosityLevel.INFO
        print(
            "[INFO] Reset phase — move the leader arm to the ghost position "
            "(within one action box of it) to start next episode"
        )

    def _exit_reset_phase(self) -> None:
        self._in_reset_phase = False
        self._feedback_color = None
        self.verbosity = VerbosityLevel.DEBUG
        self._debug_panel.reset_episode()

    # Custom actions (GUI buttons / keys)

    def _handle_custom_action(self, action, payload) -> bool:  # type: ignore[override]
        # Guards keep the accepted keys identical to what the HUD advertises
        # as "currently valid" in each state.
        if payload == "save_success":
            self._end_episode(success=True)
        elif payload == "save_failure":
            self._end_episode(success=False)
        elif payload == "discard":
            if self._in_reset_phase:
                print("[INFO] Nothing to discard in the reset phase — ignored")
                return True
            frames = self._recorder.num_frames_in_episode
            self._recorder.discard_episode()
            print(f"[INFO] Episode discarded ({frames} frames dropped)")
            if frames > 0:
                self._debug_panel.finish_episode("DISCARDED")
            self._feedback_color = GHOST_COLOR_FAILURE
            self._enter_reset_phase()
        elif payload == "undo":
            if self._provisional_failure:
                print("[INFO] Undo is ignored while an episode awaits save/discard")
                return True
            self._undo_last_save()
        elif payload == "quit":
            if self._provisional_failure:
                print("[INFO] Decide the pending episode first (F/D/V), then quit")
                return True
            self._interrupted = True
        else:
            # Delegate to concrete viewer (checkpoint, GUI-reset, etc.)
            return super()._handle_custom_action(action, payload)
        return True

    def _undo_last_save(self) -> None:
        """Drop the last saved (still staged) episode — single-level undo."""
        label = self._recorder.undo_last_save()
        if label is None:
            print("[INFO] Nothing to undo (no episode staged)")
            return
        if label:
            self._saved_success -= 1
        else:
            self._saved_failure -= 1
        print(
            f"[INFO] Undid last save (was {'SUCCESS' if label else 'FAILURE'}) — "
            f"{self.episodes_saved} episode(s) remain"
        )

    def _end_episode(self, success: bool) -> None:
        frames = self._recorder.num_frames_in_episode
        if frames == 0:
            print("[INFO] Nothing to save (no frames recorded yet)")
            return
        self._recorder.save_episode(success=success, seed=self._current_ep_seed)
        if success:
            self._saved_success += 1
        else:
            self._saved_failure += 1
        label = "SUCCESS" if success else "FAILURE"
        color_label = "✓ SUCCESS" if success else "✗ FAILURE"
        print(
            f"[INFO] Episode {self.episodes_saved} saved as {label} ({frames} frames)"
        )
        self._debug_panel.finish_episode(label)

        self._feedback_color = GHOST_COLOR_SUCCESS if success else GHOST_COLOR_FAILURE

        folder = "success" if success else "failure"
        print(f"[{color_label}] Episode saved to {folder}/ folder (U undoes)")

        if self._num_episodes and self.episodes_saved >= self._num_episodes:
            print(f"[INFO] Reached {self._num_episodes} episodes — stopping")
            self._interrupted = True
        else:
            self._enter_reset_phase()

    def _read_success_termination(self) -> bool:
        """Read whether the success termination fired (env auto-resets before
        done reaches us, so ask the termination manager)."""
        try:
            tm = self.env.env.termination_manager
            return bool(tm.get_term("success")[0].item())
        except Exception:
            return False

    def _read_fired_terminations(self) -> str:
        """Names of the termination terms that fired for the viewed env,
        e.g. "out_of_bounds" — same post-done read as the success check."""
        try:
            tm = self.env.env.termination_manager
            fired = [n for n in tm.active_terms if bool(tm.get_term(n)[0].item())]
            return ", ".join(fired) or "unknown"
        except Exception:
            return "unknown"

    def _handle_done(self, is_success: bool) -> None:
        """Episode label = which termination fired: success termination saves
        as success; any other termination enters PROVISIONAL_FAILURE (auto-saves
        as failure after the grace period unless F saves now or D discards)."""
        if is_success:
            print("[INFO] Task succeeded — auto-saving as success")
            self._end_episode(success=True)
        elif self._recorder.num_frames_in_episode == 0:
            # Nothing recorded (e.g. leader action not available yet): a
            # prompt would have nothing to save, so skip straight to reset.
            print("[INFO] Episode ended with no recorded frames — nothing to decide")
            self._enter_reset_phase()
        else:
            print(
                f"[INFO] Episode ended by termination: "
                f"{self._read_fired_terminations()} — auto-saving as FAILURE "
                f"in 2s (F saves now as failure, D discards, "
                f"V overrides to success)"
            )
            self._provisional_failure = True
            self._provisional_failure_time = time.perf_counter()

    # HUD

    def _hud_text(self) -> tuple[str, str]:
        """Two-column HUD text (labels, values) for the viewer's text overlay:
        state, saved counts, staged/undo info, and the currently valid keys."""
        if self._provisional_failure:
            state = "SAVE OR DISCARD?"
            keys = "V save success   F save failure   D discard   C wrist cam"
        else:
            state = f"RECORDING  episode {self.episodes_saved + 1}"
            keys = (
                "V save success   F save failure   D discard   "
                "U undo   Q quit   C wrist cam"
            )
        staged = "last save undoable (U)" if self._recorder.has_staged else "-"
        labels = "State\nSaved\nUndo\nKeys"
        values = (
            f"{state}\n"
            f"{self._saved_success} success / {self._saved_failure} failure\n"
            f"{staged}\n"
            f"{keys}"
        )
        return labels, values

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
                    # Gate on the leader's pose, not the sim arm's: the env
                    # reset inside _enter_reset_phase yanked the arm to the
                    # start pose, which would satisfy a sim-based gate before
                    # the operator has aligned the leader. The leader must
                    # reach the start pose itself before recording resumes.
                    #
                    # The gate is one action box wide (leader_is_aligned), not
                    # a joint-space tolerance: that is exactly the condition
                    # under which the episode's opening action does not clamp,
                    # so every stored action stays one the simulator took and
                    # a bounded policy could emit.
                    leader_action = getattr(self.policy, "latest_leader_action", None)
                    if leader_action is not None and leader_is_aligned(leader_action):
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

                # Only record if both the action and the robot obs are there.
                # The stored action is the *raw action* the policy just handed
                # to env.step(), not the leader pose behind it: under delta
                # control an action means "move this far from where you are",
                # so it cannot be re-derived later from the leader pose alone.
                # Storing it directly is what keeps ADR-011's "the recording
                # and the simulator saw one MDP" literally true.
                raw_action = getattr(self.policy, "latest_raw_action", None)
                robot_obs = self._robot_obs_list[0]
                if raw_action is not None and robot_obs is not None:
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
                        raw_action,
                        rew[0],
                        dones[0],
                        task=self._task,
                        robot_obs=robot_obs,
                        qpos=qpos_t,
                        qvel=qvel_t,
                    )
                    # Mirrors exactly what was just handed to add_frame above,
                    # so the panel can never show something other than what
                    # actually landed in the dataset. status is the viewer's
                    # own already-computed FPS/step/speed/RTF (get_status()
                    # just reads fields updated on its own 0.5s timer) — free
                    # to pass along, no extra sync.
                    get_status = getattr(self, "get_status", None)
                    self._debug_panel.update(
                        obs["actor"][0],
                        raw_action,
                        rew[0],
                        dones[0],
                        robot_state=robot_obs,
                        qpos=qpos_t,
                        qvel=qvel_t,
                        status=get_status() if get_status is not None else None,
                    )
                t4 = time.perf_counter()

                if self._debug_timing:
                    self._log_step_timing(t0, t1, t2, t3, t4)

                self._step_count += 1
                self._stats_steps += 1

                if dones[0]:
                    self._handle_done(self._read_success_termination())

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
    sim_start_pose: dict,
    home_joint_pos: dict,
) -> "tuple[_mujoco.MjModel, np.ndarray]":
    """Build ghost model: robot visual geoms only, colored, at the start pose."""
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
    ghost_pose = {**home_joint_pos, **sim_start_pose}
    for joint_name, value in ghost_pose.items():
        jid = _mujoco.mj_name2id(
            ghost_model, _mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{joint_name}"
        )
        if jid != -1:
            qpos[ghost_model.jnt_qposadr[jid]] = value
    return ghost_model, qpos
