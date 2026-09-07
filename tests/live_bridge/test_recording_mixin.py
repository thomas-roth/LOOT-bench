"""State-machine tests for RecordingViewerMixin, driven with fake recorder/env.

External behavior only: which recorder calls fire, which state the mixin ends
up in, and what the HUD/ghost color report. No viewer, no simulator.

States: RECORDING → (non-success termination: PROVISIONAL_FAILURE, auto-saves
as failure after a 2s grace period unless F/D/V decides first) → RESET_PHASE.
"""

import time

import numpy as np
import pytest
import torch

from loot_bench.scripts.live_bridge.recording_mixin import (
    GHOST_COLOR,
    GHOST_COLOR_FAILURE,
    GHOST_COLOR_PROMPT,
    GHOST_COLOR_SUCCESS,
    RecordingViewerMixin,
)


class FakeRecorder:
    def __init__(self, num_frames: int = 5):
        self.num_frames_in_episode = num_frames
        self.saved: list[bool] = []
        self.discards = 0
        self.has_staged = False
        self._staged_label: bool | None = None

    def save_episode(self, success: bool, seed: int | None = None) -> None:
        self.saved.append(success)
        self.has_staged = True
        self._staged_label = success
        self.num_frames_in_episode = 0

    def discard_episode(self) -> None:
        self.discards += 1
        self.num_frames_in_episode = 0

    def undo_last_save(self) -> bool | None:
        if not self.has_staged:
            return None
        self.has_staged = False
        label = self._staged_label
        self._staged_label = None
        return label

    def add_frame(self, *args, **kwargs) -> None:
        self.num_frames_in_episode += 1


class FakeEnv:
    """Minimal env: steps, returns zeros, and counts how often it was asked."""

    def __init__(self):
        self.steps = 0

    def get_observations(self):
        return {"actor": torch.zeros(1, 10)}

    def step(self, actions):
        self.steps += 1
        return (
            {"actor": torch.zeros(1, 10)},
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            {},
        )


class FakePolicy:
    """The two attributes the recording path reads off the live policy.

    ``latest_raw_action`` is the 7-wide action ``env.step()`` was given and the
    one a frame stores; ``latest_leader_action`` is the leader pose behind it,
    which only the alignment gate reads. ``None`` is the pre-hardware state --
    no leader reading yet -- and holds the reset phase open.
    """

    latest_raw_action = np.zeros(7, dtype=np.float32)
    latest_leader_action = None

    def __call__(self, obs):
        return torch.zeros(1, 7)


class Harness(RecordingViewerMixin):
    """Concrete host standing in for the real viewer subclass."""

    def __init__(self, recorder, num_episodes: int = 0):
        self.env = FakeEnv()
        self.policy = FakePolicy()
        self._interrupted = False
        self._step_count = 0
        self._stats_steps = 0
        self._last_error = None
        self.verbosity = 0
        self.resets = 0
        self._init_recording(
            recorder=recorder,
            robot_obs_list=[None],
            hw_failed=[False],
            mj_model=None,
            task="test-task",
            num_episodes=num_episodes,
            debug_timing=False,
            sim_start_pose={},
            home_joint_pos={},
        )

    def log(self, message, level=0):
        pass

    def pause(self):
        pass

    def reset_environment(self):
        self.resets += 1


def make_harness(num_frames: int = 5, num_episodes: int = 0):
    rec = FakeRecorder(num_frames=num_frames)
    return Harness(rec, num_episodes=num_episodes), rec


class TestTerminationHandling:
    def test_success_termination_autosaves_success(self):
        """Success termination → episode staged as success, reset phase entered."""
        h, rec = make_harness()
        h._handle_done(is_success=True, extras={})
        assert rec.saved == [True]
        assert h._in_reset_phase
        assert not h._provisional_failure

    def test_failure_termination_enters_provisional(self):
        """Non-success termination saves nothing yet. The grace window opens
        so the operator can discard or relabel first."""
        h, rec = make_harness()
        h._handle_done(is_success=False, extras={})
        assert h._provisional_failure
        assert rec.saved == []
        assert not h._in_reset_phase


class TestProvisionalFailureActions:
    def test_save_in_grace_window_stages_failure(self):
        """F during the grace window: the episode is staged as failure now,
        without waiting for the timeout."""
        h, rec = make_harness()
        h._handle_done(is_success=False, extras={})
        assert h._handle_custom_action("CUSTOM", "save_failure")
        assert rec.saved == [False]
        assert not h._provisional_failure
        assert h._in_reset_phase

    def test_override_to_success_in_grace_window(self):
        """V overrides a termination-labeled failure to success (e.g. the task
        succeeded but a stray termination fired)."""
        h, rec = make_harness()
        h._handle_done(is_success=False, extras={})
        assert h._handle_custom_action("CUSTOM", "save_success")
        assert rec.saved == [True]
        assert not h._provisional_failure
        assert h._in_reset_phase

    def test_discard_in_grace_window_drops_episode(self):
        h, rec = make_harness()
        h._handle_done(is_success=False, extras={})
        assert h._handle_custom_action("CUSTOM", "discard")
        assert rec.discards == 1
        assert rec.saved == []
        assert not h._provisional_failure
        assert h._in_reset_phase

    def test_grace_timeout_autosaves_failure(self):
        """Undecided after the grace period: _execute_step auto-saves failure."""
        h, rec = make_harness()
        h._handle_done(is_success=False, extras={})
        h._provisional_failure_time = time.perf_counter() - 10.0
        assert h._execute_step()
        assert rec.saved == [False]
        assert not h._provisional_failure
        assert h._in_reset_phase

    def test_grace_window_waits_before_timeout(self):
        """Within the grace period nothing is saved. The window stays open."""
        h, rec = make_harness()
        h._handle_done(is_success=False, extras={})
        assert h._execute_step()
        assert rec.saved == []
        assert h._provisional_failure


class TestManualEnd:
    def test_save_mid_episode_ends_it(self):
        """V/F mid-episode end the recording manually with that label."""
        h, rec = make_harness()
        assert h._handle_custom_action("CUSTOM", "save_success")
        assert rec.saved == [True]
        assert h._in_reset_phase

    def test_save_without_frames_is_noop(self):
        h, rec = make_harness(num_frames=0)
        assert h._handle_custom_action("CUSTOM", "save_failure")
        assert rec.saved == []
        assert not h._in_reset_phase

    def test_discard_mid_episode_enters_reset(self):
        h, rec = make_harness()
        assert h._handle_custom_action("CUSTOM", "discard")
        assert rec.discards == 1
        assert h._in_reset_phase


class TestUndo:
    def test_undo_drops_last_save_and_decrements_counts(self):
        h, rec = make_harness()
        h._handle_done(is_success=True, extras={})
        assert h.episodes_saved == 1
        assert h._handle_custom_action("CUSTOM", "undo")
        assert h.episodes_saved == 0
        assert not rec.has_staged

    def test_undo_with_nothing_staged_is_noop(self):
        h, rec = make_harness()
        assert h._handle_custom_action("CUSTOM", "undo")
        assert h.episodes_saved == 0

    def test_undo_restores_per_label_counts(self):
        h, rec = make_harness()
        h._handle_done(is_success=True, extras={})
        h._exit_reset_phase()
        rec.num_frames_in_episode = 5
        h._handle_custom_action("CUSTOM", "save_failure")
        assert (h._saved_success, h._saved_failure) == (1, 1)
        h._handle_custom_action("CUSTOM", "undo")
        assert (h._saved_success, h._saved_failure) == (1, 0)


class TestGhostColor:
    def test_grace_window_color_is_distinct(self):
        """The grace window gets its own ghost color, so an undecided episode
        can't be mistaken for the red saved-failure feedback."""
        h, _ = make_harness()
        h._handle_done(is_success=False, extras={})
        assert h._get_ghost_color() == GHOST_COLOR_PROMPT
        assert GHOST_COLOR_PROMPT not in (
            GHOST_COLOR,
            GHOST_COLOR_SUCCESS,
            GHOST_COLOR_FAILURE,
        )

    def test_save_flashes_success_color(self):
        h, _ = make_harness()
        h._handle_done(is_success=True, extras={})
        assert h._get_ghost_color() == GHOST_COLOR_SUCCESS

    def test_feedback_color_clears_on_reset_exit(self):
        h, _ = make_harness()
        h._handle_done(is_success=True, extras={})
        h._exit_reset_phase()
        assert h._get_ghost_color() == GHOST_COLOR


class TestHud:
    def test_hud_reflects_recording_state(self):
        h, _ = make_harness()
        labels, values = h._hud_text()
        assert "RECORDING" in values
        assert "V save success" in values
        assert "U undo" in values

    def test_hud_reflects_grace_window_state_and_keys(self):
        h, _ = make_harness()
        h._handle_done(is_success=False, extras={})
        _, values = h._hud_text()
        assert "F save failure" in values
        assert "D discard" in values
        assert "RECORDING" not in values
        # Keys the grace window does not accept must not be advertised.
        assert "U undo" not in values
        assert "Q quit" not in values

    def test_hud_shows_saved_counts(self):
        h, rec = make_harness()
        h._handle_done(is_success=True, extras={})
        _, values = h._hud_text()
        assert "1 success / 0 failure" in values


class TestStateGuards:
    """Accepted keys must match what the HUD advertises per state."""

    def test_zero_frame_termination_skips_grace_window(self):
        """A termination before any frame was recorded has nothing to decide.
        It must go straight to the reset phase."""
        h, rec = make_harness(num_frames=0)
        h._handle_done(is_success=False, extras={})
        assert not h._provisional_failure
        assert h._in_reset_phase

    def test_discard_in_reset_phase_is_ignored(self):
        h, rec = make_harness()
        h._enter_reset_phase()
        resets_before = h.resets
        assert h._handle_custom_action("CUSTOM", "discard")
        assert rec.discards == 0
        assert h.resets == resets_before

    def test_undo_in_grace_window_is_ignored(self):
        h, rec = make_harness()
        h._handle_done(is_success=True, extras={})
        h._exit_reset_phase()
        rec.num_frames_in_episode = 5
        h._handle_done(is_success=False, extras={})
        assert h._handle_custom_action("CUSTOM", "undo")
        assert rec.has_staged  # staged episode untouched
        assert h.episodes_saved == 1

    def test_quit_in_grace_window_is_blocked(self):
        """Q must not bypass an open grace window. The un-decided episode
        would be silently dropped at finalize."""
        h, rec = make_harness()
        h._handle_done(is_success=False, extras={})
        assert h._handle_custom_action("CUSTOM", "quit")
        assert not h._interrupted
        h._handle_custom_action("CUSTOM", "discard")
        assert h._handle_custom_action("CUSTOM", "quit")
        assert h._interrupted


class TestSessionEnd:
    def test_num_episodes_reached_interrupts(self):
        h, rec = make_harness(num_episodes=1)
        h._handle_done(is_success=True, extras={})
        assert h._interrupted

    def test_quit_action_interrupts(self):
        h, rec = make_harness()
        assert h._handle_custom_action("CUSTOM", "quit")
        assert h._interrupted


class TestResetPhaseAlignmentGate:
    """What ends the reset phase (ADR-016).

    The gate reads the *leader*, and it is one action box wide rather than a
    joint-space tolerance: that is exactly the condition under which the
    episode's opening action does not clamp, so every stored action stays one
    the simulator took and a bounded policy could have emitted. ADR-016
    measured the alternative — 66 of 99 demos opened with the arm snapping
    toward an already-parked leader.

    ``leader_is_aligned`` itself is pinned in ``tests/live_bridge/test_mapper.py``
    against the real forward kinematics; here it is patched, because compiling
    the DK1 is what this file's "no viewer, no simulator" contract rules out.
    """

    def test_an_aligned_leader_ends_the_reset_phase(self, monkeypatch):
        h, _ = make_harness()
        h._handle_done(is_success=True, extras={})
        assert h._in_reset_phase
        h.policy.latest_leader_action = np.zeros(7, dtype=np.float32)
        monkeypatch.setattr(
            "loot_bench.scripts.live_bridge.recording_mixin.leader_is_aligned",
            lambda _pose: True,
        )
        assert h._execute_step()
        assert not h._in_reset_phase

    def test_an_unaligned_leader_keeps_it_open(self, monkeypatch):
        h, _ = make_harness()
        h._handle_done(is_success=True, extras={})
        h.policy.latest_leader_action = np.zeros(7, dtype=np.float32)
        monkeypatch.setattr(
            "loot_bench.scripts.live_bridge.recording_mixin.leader_is_aligned",
            lambda _pose: False,
        )
        assert h._execute_step()
        assert h._in_reset_phase

    def test_no_leader_reading_keeps_it_open(self, monkeypatch):
        """Before the hardware has reported once there is nothing to compare,
        and the gate must not be answered by the sim arm — the env reset
        already parked *that* at the start pose."""
        h, _ = make_harness()
        h._handle_done(is_success=True, extras={})
        h.policy.latest_leader_action = None
        monkeypatch.setattr(
            "loot_bench.scripts.live_bridge.recording_mixin.leader_is_aligned",
            lambda _pose: pytest.fail("the gate must not run without a leader pose"),
        )
        assert h._execute_step()
        assert h._in_reset_phase

    def test_the_reset_phase_still_steps_the_simulator(self):
        """Held open, the arm must keep being driven — a frozen sim is what
        the operator is aligning against."""
        h, _ = make_harness()
        h._handle_done(is_success=True, extras={})
        before = h.env.steps
        h._execute_step()
        assert h.env.steps == before + 1
