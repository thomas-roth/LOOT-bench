"""State-machine tests for RecordingViewerMixin, driven with fake recorder/env.

External behavior only: which recorder calls fire, which state the mixin ends
up in, and what the HUD/ghost color report — no viewer, no simulator.
"""

import numpy as np
import torch

from g3a_bench.scripts.live_bridge.recording_mixin import (
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

    def save_episode(self, success: bool) -> None:
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
    """Minimal env: obs at rest pose so the reset phase exits immediately."""

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
            live_rest_pose_rel=np.zeros(6, dtype=np.float32),
            task="test-task",
            num_episodes=num_episodes,
            debug_timing=False,
            sim_rest_pose={},
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
        h._handle_done(is_success=True)
        assert rec.saved == [True]
        assert h._in_reset_phase
        assert not h._awaiting_decision


class TestPromptActions:
    def test_save_in_prompt_stages_failure(self):
        """F in the prompt: the episode is staged as failure (label comes from
        the termination, never from the operator)."""
        h, rec = make_harness()
        h._handle_done(is_success=False)
        assert h._handle_custom_action("CUSTOM", "save_failure")
        assert rec.saved == [False]
        assert not h._awaiting_decision
        assert h._in_reset_phase

    def test_discard_in_prompt_drops_episode(self):
        h, rec = make_harness()
        h._handle_done(is_success=False)
        assert h._handle_custom_action("CUSTOM", "discard")
        assert rec.discards == 1
        assert rec.saved == []
        assert not h._awaiting_decision
        assert h._in_reset_phase

    def test_save_outside_prompt_is_ignored(self):
        """F only acts inside the prompt — a stray press mid-episode must not
        end or mislabel the recording."""
        h, rec = make_harness()
        assert h._handle_custom_action("CUSTOM", "save_failure")
        assert rec.saved == []
        assert not h._in_reset_phase


class TestManualEndEpisode:
    def test_end_episode_opens_prompt(self):
        """E ends a hopeless episode manually: same prompt as a termination."""
        h, rec = make_harness()
        assert h._handle_custom_action("CUSTOM", "end_episode")
        assert h._awaiting_decision
        assert rec.saved == []

    def test_end_episode_without_frames_is_noop(self):
        h, rec = make_harness(num_frames=0)
        assert h._handle_custom_action("CUSTOM", "end_episode")
        assert not h._awaiting_decision

    def test_end_episode_in_reset_phase_is_noop(self):
        h, rec = make_harness()
        h._enter_reset_phase()
        assert h._handle_custom_action("CUSTOM", "end_episode")
        assert not h._awaiting_decision


class TestUndo:
    def test_undo_drops_last_save_and_decrements_counts(self):
        h, rec = make_harness()
        h._handle_done(is_success=True)
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
        h._handle_done(is_success=True)
        h._exit_reset_phase()
        rec.num_frames_in_episode = 5
        h._handle_custom_action("CUSTOM", "end_episode")
        h._handle_custom_action("CUSTOM", "save_failure")
        assert (h._saved_success, h._saved_failure) == (1, 1)
        h._handle_custom_action("CUSTOM", "undo")
        assert (h._saved_success, h._saved_failure) == (1, 0)


class TestGhostColor:
    def test_prompt_color_is_distinct_and_persistent(self):
        """The prompt gets its own ghost color that does not time out, so a
        frozen prompt can't be mistaken for the red saved-failure flash."""
        h, _ = make_harness()
        h._handle_done(is_success=False)
        assert h._get_ghost_color() == GHOST_COLOR_PROMPT
        assert GHOST_COLOR_PROMPT not in (
            GHOST_COLOR,
            GHOST_COLOR_SUCCESS,
            GHOST_COLOR_FAILURE,
        )
        # Still the prompt color long after any flash would have expired.
        h._feedback_color_start_time = -1e9
        assert h._get_ghost_color() == GHOST_COLOR_PROMPT

    def test_save_flashes_success_color(self):
        h, _ = make_harness()
        h._handle_done(is_success=True)
        assert h._get_ghost_color() == GHOST_COLOR_SUCCESS


class TestHud:
    def test_hud_reflects_recording_state(self):
        h, _ = make_harness()
        labels, values = h._hud_text()
        assert "RECORDING" in values
        assert "E end episode" in values

    def test_hud_reflects_prompt_state_and_keys(self):
        h, _ = make_harness()
        h._handle_done(is_success=False)
        _, values = h._hud_text()
        assert "F save failure" in values
        assert "D discard" in values
        assert "RECORDING" not in values
        # Keys the prompt does not accept must not be advertised.
        assert "U undo" not in values
        assert "Q quit" not in values

    def test_hud_shows_saved_counts(self):
        h, rec = make_harness()
        h._handle_done(is_success=True)
        _, values = h._hud_text()
        assert "1 success / 0 failure" in values


class TestStateGuards:
    """Accepted keys must match what the HUD advertises per state."""

    def test_zero_frame_termination_skips_prompt(self):
        """A termination before any frame was recorded has nothing to decide —
        it must not open a prompt whose S key could never exit."""
        h, rec = make_harness(num_frames=0)
        h._handle_done(is_success=False)
        assert not h._awaiting_decision
        assert h._in_reset_phase

    def test_discard_in_reset_phase_is_ignored(self):
        h, rec = make_harness()
        h._enter_reset_phase()
        resets_before = h.resets
        assert h._handle_custom_action("CUSTOM", "discard")
        assert rec.discards == 0
        assert h.resets == resets_before

    def test_undo_in_prompt_is_ignored(self):
        h, rec = make_harness()
        h._handle_done(is_success=True)
        h._exit_reset_phase()
        rec.num_frames_in_episode = 5
        h._handle_custom_action("CUSTOM", "end_episode")
        assert h._handle_custom_action("CUSTOM", "undo")
        assert rec.has_staged  # staged episode untouched
        assert h.episodes_saved == 1

    def test_quit_in_prompt_is_blocked(self):
        """Q must not bypass an open prompt — the un-decided episode would be
        silently dropped at finalize."""
        h, rec = make_harness()
        h._handle_done(is_success=False)
        assert h._handle_custom_action("CUSTOM", "quit")
        assert not h._interrupted
        h._handle_custom_action("CUSTOM", "discard")
        assert h._handle_custom_action("CUSTOM", "quit")
        assert h._interrupted


class TestSessionEnd:
    def test_num_episodes_reached_interrupts(self):
        h, rec = make_harness(num_episodes=1)
        h._handle_done(is_success=True)
        assert h._interrupted

    def test_quit_action_interrupts(self):
        h, rec = make_harness()
        assert h._handle_custom_action("CUSTOM", "quit")
        assert h._interrupted
