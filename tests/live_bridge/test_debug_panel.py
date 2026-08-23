"""Unit tests for the --debug-obs live dashboard (no viewer, no simulator)."""

from types import SimpleNamespace

import numpy as np
import torch

from g3a_bench.scripts.live_bridge.debug_panel import (
    LiveDebugPanel,
    _aggregate_buffered,
    _ObsLayout,
    _RunningStats,
    _xyz,
)


def _fake_env(names: list[str], dims: list[tuple[int, ...]]):
    """Stands in for the real env.env.observation_manager shape used by
    _ObsLayout.from_env (see mjlab's ObservationManager.active_terms /
    group_obs_term_dim)."""
    om = SimpleNamespace(
        active_terms={"actor": names},
        group_obs_term_dim={"actor": dims},
    )
    return SimpleNamespace(env=SimpleNamespace(observation_manager=om))


PLACE_NAMES = ["joint_pos", "joint_vel", "ee_pose", "cube_pose", "command", "actions"]
PLACE_DIMS = [(8,), (8,), (9,), (7,), (3,), (7,)]


class TestObsLayout:
    def test_from_env_builds_offsets(self):
        layout = _ObsLayout.from_env(_fake_env(PLACE_NAMES, PLACE_DIMS))
        assert layout is not None
        assert layout.names == PLACE_NAMES
        assert layout.dims == [8, 8, 9, 7, 3, 7]
        assert layout.total == 42
        assert layout.offsets == [0, 8, 16, 25, 32, 35]

    def test_identifies_ee_goal_object_terms(self):
        layout = _ObsLayout.from_env(_fake_env(PLACE_NAMES, PLACE_DIMS))
        assert layout.ee_term == "ee_pose"
        assert layout.goal_term == "command"
        assert layout.object_term == "cube_pose"  # first term outside the common set

    def test_domino_first_task_term_is_object(self):
        """domino's task_obs_terms puts domino_poses (positions) first, then
        slot_placed/phase — object_term must pick the positional one."""
        names = [
            "joint_pos",
            "joint_vel",
            "ee_pose",
            "domino_poses",
            "slot_placed",
            "command",
            "actions",
        ]
        dims = [(8,), (8,), (9,), (14,), (2,), (2,), (7,)]
        layout = _ObsLayout.from_env(_fake_env(names, dims))
        assert layout.object_term == "domino_poses"

    def test_missing_command_term_leaves_goal_none(self):
        names = ["joint_pos", "joint_vel", "ee_pose", "cube_pose", "actions"]
        dims = [(8,), (8,), (9,), (7,), (7,)]
        layout = _ObsLayout.from_env(_fake_env(names, dims))
        assert layout.goal_term is None

    def test_slice_returns_the_right_chunk(self):
        layout = _ObsLayout.from_env(_fake_env(PLACE_NAMES, PLACE_DIMS))
        obs = np.arange(42, dtype=np.float64)
        chunk = layout.slice(obs, "command")
        assert list(chunk) == [32.0, 33.0, 34.0]

    def test_from_env_returns_none_on_unrecognized_env(self):
        """Any env shape this doesn't understand disables the named
        breakdown rather than crashing the recording session."""
        assert _ObsLayout.from_env(object()) is None
        assert _ObsLayout.from_env(SimpleNamespace(env=SimpleNamespace())) is None


class TestXyz:
    def test_formats_first_three_values(self):
        s, bad = _xyz(np.array([0.1, -0.2, 0.3, 999.0]))
        assert not bad
        assert "x=" in s and "y=" in s and "z=" in s
        assert "999" not in s  # only the first 3 values are shown

    def test_flags_nan(self):
        s, bad = _xyz(np.array([0.0, np.nan, 0.0]))
        assert bad
        assert s == "NON-FINITE"

    def test_falls_back_when_too_short(self):
        s, bad = _xyz(np.array([1.0]))
        assert not bad
        assert "n=1" in s


class TestRunningStats:
    def test_aggregates_across_multiple_updates(self):
        """The whole point: mean/min/max are over every value seen across
        all update() calls, not just the last one."""
        stats = _RunningStats()
        stats.update(np.array([0.0, 2.0]))
        stats.update(np.array([4.0, 6.0]))
        assert stats.min == 0.0
        assert stats.max == 6.0
        assert stats.count == 4
        assert stats.sum == 12.0
        s = stats.format()
        assert "mean= +3.000" in s
        assert "min= +0.000" in s
        assert "max= +6.000" in s

    def test_accepts_torch_tensor(self):
        stats = _RunningStats()
        stats.update(torch.tensor([1.0, 2.0, 3.0]))
        assert stats.count == 3
        assert stats.dim == 3

    def test_non_finite_values_excluded_from_aggregate_but_counted(self):
        stats = _RunningStats()
        bad = stats.update(np.array([1.0, np.nan, 3.0]))
        assert bad is True
        assert stats.count == 2  # only the finite values
        assert stats.non_finite == 1
        assert stats.min == 1.0
        assert stats.max == 3.0
        assert "1 non-finite values seen" in stats.format()

    def test_update_returns_false_for_a_clean_frame(self):
        stats = _RunningStats()
        assert stats.update(np.array([1.0, 2.0])) is False

    def test_format_before_any_update(self):
        stats = _RunningStats()
        assert "no finite values seen" in stats.format()


class TestAggregateBuffered:
    def test_empty_list(self):
        stats, bad = _aggregate_buffered([])
        assert stats.dim == 0
        assert bad is False

    def test_aggregates_all_frames(self):
        stats, bad = _aggregate_buffered([np.array([0.0, 1.0]), np.array([2.0, 3.0])])
        assert bad is False
        assert stats.min == 0.0
        assert stats.max == 3.0
        assert stats.count == 4

    def test_flags_any_bad_frame(self):
        stats, bad = _aggregate_buffered(
            [np.array([0.0, 1.0]), np.array([np.nan, 3.0])]
        )
        assert bad is True
        assert stats.non_finite == 1


class TestLiveDebugPanelPerFrame:
    """Per-frame output: EEF/GOAL/OBJECT/REWARD/DONE/ANOMALIES + optional FPS
    line only — no obs/action/qpos/qvel breakdown (that's finish_episode's
    job, see TestFinishEpisodeSummary)."""

    def _panel(self) -> LiveDebugPanel:
        return LiveDebugPanel(_fake_env(PLACE_NAMES, PLACE_DIMS))

    def test_update_does_not_raise_and_prints(self, capsys):
        panel = self._panel()
        obs = torch.zeros(42)
        panel.update(
            obs,
            action=np.zeros(7, dtype=np.float32),
            reward=0.5,
            done=False,
            robot_state=np.zeros(7, dtype=np.float32),
            qpos=torch.zeros(15),
            qvel=torch.zeros(14),
        )
        out = capsys.readouterr().out
        assert "EEF pos" in out
        assert "GOAL" in out
        assert "OBJECT pos" in out
        assert "REWARD" in out
        assert "DONE" in out

    def test_no_per_frame_obs_or_saved_breakdown(self, capsys):
        """The min/mean/max breakdown must NOT appear every step — only in
        the once-per-episode summary from finish_episode."""
        panel = self._panel()
        obs = torch.zeros(42)
        panel.update(obs, np.zeros(7, np.float32), 0.0, False, qpos=torch.zeros(15))
        out = capsys.readouterr().out
        assert "obs terms" not in out
        assert "also saved to the dataset" not in out
        assert "qpos" not in out

    def test_episode_reward_accumulates_until_reset(self, capsys):
        panel = self._panel()
        obs = torch.zeros(42)
        for r in (0.5, 0.25):
            panel.update(obs, np.zeros(7, np.float32), r, False)
        out = capsys.readouterr().out
        assert "+0.7500" in out

        panel.reset_episode()
        panel.update(obs, np.zeros(7, np.float32), 1.0, False)
        out = capsys.readouterr().out
        assert "+1.0000" in out
        assert "+1.7500" not in out

    def test_nan_in_obs_is_not_flagged_per_frame(self, capsys):
        """A NaN in joint_pos is checked live (cheap), but not shown until
        the episode-end summary — no per-frame ANOMALIES line exists."""
        panel = self._panel()
        bad_obs = torch.zeros(42)
        bad_obs[0] = float("nan")  # falls inside joint_pos, offset 0
        panel.update(bad_obs, np.zeros(7, np.float32), 0.0, False)
        out = capsys.readouterr().out
        assert "ANOMALIES" not in out

        capsys.readouterr()
        panel.finish_episode("SUCCESS")
        out = capsys.readouterr().out
        assert "ANOMALIES this episode:" in out
        assert "non-finite value in 'joint_pos' — 1 frame(s)" in out

    def test_works_without_a_resolvable_obs_layout(self, capsys):
        """A task/env shape the panel can't introspect still renders a
        (less detailed) per-frame line instead of crashing the session."""
        panel = LiveDebugPanel(object())
        panel.update(torch.zeros(10), np.zeros(7, np.float32), 0.0, False)
        out = capsys.readouterr().out
        assert "n/a" in out

    def test_falls_back_to_append_when_not_a_tty(self, capsys):
        """capsys isn't a terminal, so isatty() is False here — this is the
        piped/redirected-output fallback: no ANSI codes, plain scrolling."""
        panel = self._panel()
        obs = torch.zeros(42)
        for _ in range(3):
            panel.update(obs, np.zeros(7, np.float32), 0.1, False)
        out = capsys.readouterr().out
        assert "\x1b[" not in out
        assert out.count("DONE") == 3  # all 3 frames still present

    def test_overwrites_in_place_on_a_tty(self, capsys, monkeypatch):
        """The old overwrite style is the default when actually attached to
        a terminal: redraw via cursor-up + clear-line, not scroll."""
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        panel = self._panel()
        obs = torch.zeros(42)
        panel.update(obs, np.zeros(7, np.float32), 0.1, False)
        capsys.readouterr()  # drain frame 1 (nothing to move the cursor over yet)
        panel.update(obs, np.zeros(7, np.float32), 0.1, False)
        out = capsys.readouterr().out
        assert out.startswith("\x1b[")  # cursor-up over frame 1's block
        assert "\x1b[2K" in out  # clear-line before rewriting

    def test_append_only_never_overwrites_even_on_a_tty(self, capsys, monkeypatch):
        """--debug-obs (append_only=True) must keep every frame in the
        scrollback, even when attached to a real terminal."""
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        panel = LiveDebugPanel(_fake_env(PLACE_NAMES, PLACE_DIMS), append_only=True)
        obs = torch.zeros(42)
        panel.update(obs, np.zeros(7, np.float32), 0.1, False)
        panel.update(obs, np.zeros(7, np.float32), 0.1, False)
        out = capsys.readouterr().out
        assert "\x1b[" not in out
        assert out.count("DONE") == 2

    def test_includes_viewer_status_when_given(self, capsys):
        """The FPS/step/speed/RTF line mjlab's base viewer normally prints on
        its own timer — folded into the panel instead of a second, floating
        print — appears only when a status object is passed."""
        panel = self._panel()
        status = SimpleNamespace(
            step_count=42,
            smoothed_fps=58.3,
            speed_label="1x",
            actual_realtime=0.98,
            target_realtime=1.0,
        )
        panel.update(
            torch.zeros(42), np.zeros(7, np.float32), 0.0, False, status=status
        )
        out = capsys.readouterr().out
        assert "step=42" in out
        assert "fps=58" in out
        assert "speed=1x" in out
        assert "rtf=0.98x/1.00x" in out

    def test_no_status_line_when_status_omitted(self, capsys):
        panel = self._panel()
        panel.update(torch.zeros(42), np.zeros(7, np.float32), 0.0, False)
        out = capsys.readouterr().out
        assert "fps=" not in out


class TestDeferredQposQvel:
    """qpos/qvel are the expensive fields (GPU sync) — update() must only
    buffer them, never touch/convert them; the real aggregation + NaN check
    happens once, in finish_episode."""

    def _panel(self) -> LiveDebugPanel:
        return LiveDebugPanel(_fake_env(PLACE_NAMES, PLACE_DIMS))

    def test_update_only_buffers_qpos_qvel(self, capsys):
        panel = self._panel()
        obs = torch.zeros(42)
        panel.update(
            obs,
            np.zeros(7, np.float32),
            0.0,
            False,
            qpos=torch.zeros(15),
            qvel=torch.zeros(14),
        )
        assert len(panel._qpos_frames) == 1
        assert len(panel._qvel_frames) == 1
        out = capsys.readouterr().out
        assert "ANOMALIES" not in out  # no per-frame anomaly line at all

    def test_nan_in_qpos_not_flagged_until_finish(self, capsys):
        panel = self._panel()
        bad_qpos = torch.zeros(15)
        bad_qpos[0] = float("nan")
        panel.update(
            torch.zeros(42), np.zeros(7, np.float32), 0.0, False, qpos=bad_qpos
        )
        out = capsys.readouterr().out
        assert "ANOMALIES" not in out  # not caught live — that's the point

        panel.finish_episode("SUCCESS")
        out = capsys.readouterr().out
        assert "ANOMALIES this episode:" in out
        assert "non-finite value in 'qpos' (checked at episode end)" in out
        assert "consider U to undo" in out

    def test_finish_episode_aggregates_qpos_over_all_frames(self, capsys):
        panel = self._panel()
        obs = torch.zeros(42)
        panel.update(obs, np.zeros(7, np.float32), 0.0, False, qpos=torch.zeros(15))
        qpos_high = torch.zeros(15)
        qpos_high[0] = 5.0
        panel.update(obs, np.zeros(7, np.float32), 0.0, False, qpos=qpos_high)
        capsys.readouterr()

        panel.finish_episode("SUCCESS")
        out = capsys.readouterr().out
        assert "qpos" in out
        assert "max= +5.000" in out  # from the 2nd frame, aggregated with the 1st


class TestFinishEpisodeSummary:
    def _panel(self) -> LiveDebugPanel:
        return LiveDebugPanel(_fake_env(PLACE_NAMES, PLACE_DIMS))

    def test_noop_before_any_frame(self, capsys):
        """Nothing recorded yet -> nothing to summarize, no banner printed."""
        panel = self._panel()
        panel.finish_episode("DISCARDED")
        assert capsys.readouterr().out == ""

    def test_summary_aggregates_over_the_whole_episode_not_last_frame(self, capsys):
        """The reported min/mean/max must span every frame, not just
        whatever the last frame happened to contain."""
        panel = self._panel()
        obs_low = torch.zeros(42)
        obs_high = torch.zeros(42)
        obs_high[0] = 10.0  # joint_pos[0], offset 0
        panel.update(obs_low, np.zeros(7, np.float32), 0.0, False)
        panel.update(obs_high, np.zeros(7, np.float32), 0.0, False)
        capsys.readouterr()

        panel.finish_episode("SUCCESS")
        out = capsys.readouterr().out
        assert "obs terms" in out
        assert "joint_pos" in out
        assert "max=+10.000" in out  # from the first update, not just the last
        assert "min= +0.000" in out

    def test_summary_includes_saved_fields_only_when_provided(self, capsys):
        panel = self._panel()
        obs = torch.zeros(42)
        panel.update(
            obs,
            np.zeros(7, np.float32),
            0.0,
            False,
            robot_state=np.zeros(7, np.float32),
            qpos=torch.zeros(15),
            # qvel intentionally omitted
        )
        capsys.readouterr()

        panel.finish_episode("SUCCESS")
        out = capsys.readouterr().out
        assert "also saved to the dataset" in out
        assert "action" in out
        assert "robot_state" in out
        assert "qpos" in out
        assert "qvel" not in out  # never given this episode -> not reported

    def test_summary_falls_back_when_layout_unresolvable(self, capsys):
        panel = LiveDebugPanel(object())
        panel.update(torch.zeros(10), np.zeros(7, np.float32), 0.0, False)
        capsys.readouterr()
        panel.finish_episode("SUCCESS")
        out = capsys.readouterr().out
        assert "obs-term layout unavailable" in out

    def test_summary_reports_no_anomalies_when_clean(self, capsys):
        panel = self._panel()
        panel.update(torch.zeros(42), np.zeros(7, np.float32), 0.0, False)
        capsys.readouterr()
        panel.finish_episode("SUCCESS")
        out = capsys.readouterr().out
        assert "ANOMALIES this episode: none" in out

    def test_prints_boundary_banner(self, capsys):
        panel = self._panel()
        obs = torch.zeros(42)
        panel.update(obs, np.zeros(7, np.float32), 1.0, True)
        capsys.readouterr()  # drain the per-frame output
        panel.finish_episode("SUCCESS")
        out = capsys.readouterr().out
        assert "EPISODE 1 FINISHED: SUCCESS" in out
        assert "1 frames" in out
        assert "+1.0000" in out

    def test_discard_and_success_labels_are_distinct(self, capsys):
        panel = self._panel()
        obs = torch.zeros(42)
        panel.update(obs, np.zeros(7, np.float32), 0.0, False)
        capsys.readouterr()
        panel.finish_episode("DISCARDED")
        out = capsys.readouterr().out
        assert "DISCARDED" in out
        assert "SUCCESS" not in out

    def test_stats_reset_between_episodes(self, capsys):
        panel = self._panel()
        obs_high = torch.zeros(42)
        obs_high[0] = 100.0
        panel.update(obs_high, np.zeros(7, np.float32), 0.0, False)
        capsys.readouterr()
        panel.finish_episode("SUCCESS")
        capsys.readouterr()

        panel.reset_episode()
        panel.update(torch.zeros(42), np.zeros(7, np.float32), 0.0, False)
        capsys.readouterr()
        panel.finish_episode("SUCCESS")
        out = capsys.readouterr().out
        assert "100" not in out  # episode 1's extreme value must not leak in
