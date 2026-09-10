"""Terminal dashboard for live_record.py, on by default.
Per-frame line plus episode-end aggregate; read-only view of saved data.
"""

import sys

import numpy as np

# First task-specific term after common ones is an object pose (xyz first).
_EE_TERM_NAME = "ee_pose"
_GOAL_TERM_NAME = "command"
_COMMON_TERMS = {
    "joint_pos",
    "joint_vel",
    _EE_TERM_NAME,
    # Proprioceptive head, not an object: the release gate's own number.
    "pad_contact_force",
    _GOAL_TERM_NAME,
}

_PANEL_WIDTH = 92


class _ObsLayout:
    """Name to offset map for one flat obs vector (matches saved layout)."""

    def __init__(self, names: list[str], dims: list[int]):
        self.names = names
        self.dims = dims
        self.offsets: list[int] = []
        off = 0
        for d in dims:
            self.offsets.append(off)
            off += d
        self.total = off
        self.ee_term = _EE_TERM_NAME if _EE_TERM_NAME in names else None
        self.goal_term = _GOAL_TERM_NAME if _GOAL_TERM_NAME in names else None
        self.object_term = next((n for n in names if n not in _COMMON_TERMS), None)

    def slice(self, obs: np.ndarray, name: str) -> np.ndarray:
        i = self.names.index(name)
        start = self.offsets[i]
        return obs[start : start + self.dims[i]]

    @classmethod
    def from_env(cls, env) -> "_ObsLayout | None":
        """Best-effort layout; unknown envs disable the breakdown."""
        try:
            om = env.env.observation_manager
            names = list(om.active_terms["actor"])
            dims = [int(np.prod(shape)) for shape in om.group_obs_term_dim["actor"]]
            return cls(names, dims)
        except Exception:
            return None


def _to_numpy(arr) -> np.ndarray:
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _xyz(arr) -> tuple[str, bool]:
    """First 3 values as x/y/z (positions, goals, object poses)."""
    a = _to_numpy(arr)
    if a.size < 3:
        finite = np.isfinite(a)
        if not finite.all():
            return f"n={a.size} NON-FINITE={int((~finite).sum())}", True
        return f"n={a.size} " + " ".join(f"{v:+7.3f}" for v in a), False
    a = a[:3]
    if not np.isfinite(a).all():
        return "NON-FINITE", True
    return f"x={a[0]:+7.3f}  y={a[1]:+7.3f}  z={a[2]:+7.3f}", False


class _RunningStats:
    """Running min/max/mean over an episode, O(1) memory."""

    def __init__(self) -> None:
        self.dim = 0  # size of one frame's array, for display
        self.count = 0  # finite values seen so far
        self.non_finite = 0  # non-finite values seen so far (excluded above)
        self.sum = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, arr) -> bool:
        """Fold one frame's array in; True if it held a non-finite value."""
        a = _to_numpy(arr)
        self.dim = a.size
        finite = np.isfinite(a)
        bad = int((~finite).sum())
        self.non_finite += bad
        good = a[finite]
        if good.size:
            self.count += good.size
            self.sum += float(good.sum())
            self.min = min(self.min, float(good.min()))
            self.max = max(self.max, float(good.max()))
        return bad > 0

    def format(self) -> str:
        if self.count == 0:
            return f"n={self.dim:<3d} no finite values seen"
        s = (
            f"n={self.dim:<3d} mean={self.sum / self.count:+7.3f} "
            f"min={self.min:+7.3f} max={self.max:+7.3f}"
        )
        if self.non_finite:
            s += f"  ({self.non_finite} non-finite values seen this episode)"
        return s


def _aggregate_buffered(frames: list) -> tuple[_RunningStats, bool]:
    """Aggregate buffered per-frame tensors once at episode end."""
    stats = _RunningStats()
    bad = False
    for frame in frames:
        if stats.update(frame):
            bad = True
    return stats, bad


class LiveDebugPanel:
    """Per-frame update; aggregate breakdown prints at finish_episode."""

    def __init__(self, env, append_only: bool = False):
        self._layout = _ObsLayout.from_env(env)
        self._append_only = append_only
        self._printed_lines = 0  # for the in-place-overwrite bookkeeping
        self._episode_num = 0
        self.reset_episode()

    def reset_episode(self) -> None:
        self._frame = 0
        self._episode_reward = 0.0
        # Anomaly counts per term; shown once in the episode summary.
        self._anomaly_frames: dict[str, int] = {}
        self._printed_lines = 0
        self._episode_num += 1
        self._term_stats: dict[str, _RunningStats] = (
            {name: _RunningStats() for name in self._layout.names}
            if self._layout is not None
            else {}
        )
        self._action_stats = _RunningStats()
        self._robot_state_stats = _RunningStats()
        # Buffer qpos/qvel refs; caller already cloned per step.
        self._qpos_frames: list = []
        self._qvel_frames: list = []

    def update(
        self,
        obs,
        action,
        reward,
        done,
        robot_state=None,
        qpos=None,
        qvel=None,
        status=None,
    ) -> None:
        """Status is precomputed by the viewer; reading it is free."""
        self._frame += 1
        # Obs sync is required; reward/done reuse the same flush.
        obs_np = _to_numpy(obs)
        reward_v = float(reward.item()) if hasattr(reward, "item") else float(reward)
        done_v = bool(done.item()) if hasattr(done, "item") else bool(done)
        self._episode_reward += reward_v

        # Obs/action are host-side already; no extra GPU sync here.
        bad_terms: list[str] = []
        layout = self._layout
        if layout is not None:
            for name in layout.names:
                if self._term_stats[name].update(layout.slice(obs_np, name)):
                    bad_terms.append(name)
        if self._action_stats.update(action):
            bad_terms.append("action")
        if robot_state is not None and self._robot_state_stats.update(robot_state):
            bad_terms.append("robot_state")

        # Buffer qpos/qvel; aggregate once at episode end.
        if qpos is not None:
            self._qpos_frames.append(qpos)
        if qvel is not None:
            self._qvel_frames.append(qvel)

        for name in bad_terms:
            self._anomaly_frames[name] = self._anomaly_frames.get(name, 0) + 1

        lines = self._render_frame_line(obs_np, reward_v, done_v, status)
        self._print_overwrite(lines)

    def finish_episode(self, label: str) -> None:
        """Print aggregate breakdown and banner; no-op if no frames."""
        if self._frame == 0:
            return
        qpos_stats, qpos_bad = _aggregate_buffered(self._qpos_frames)
        qvel_stats, qvel_bad = _aggregate_buffered(self._qvel_frames)
        summary = self._render_summary(qpos_stats, qvel_stats, qpos_bad, qvel_bad)

        # The last frame is already on screen; stop treating it as
        # overwritable before appending the summary below it.
        self._printed_lines = 0
        self._print_lines(summary)
        banner = (
            f" EPISODE {self._episode_num} FINISHED: {label}  "
            f"({self._frame} frames, cumulative_reward={self._episode_reward:+.4f}) "
        ).center(_PANEL_WIDTH, "#")
        sys.stdout.write(banner + "\n\n")
        sys.stdout.flush()

    def _render_frame_line(self, obs, reward_v, done_v, status) -> list[str]:
        w = _PANEL_WIDTH
        layout = self._layout
        header = f" frame {self._frame:5d} "
        if status is not None:
            header = (
                f" frame {self._frame:5d}  "
                f"step={getattr(status, 'step_count', '?')}  "
                f"fps={getattr(status, 'smoothed_fps', float('nan')):.0f}  "
                f"speed={getattr(status, 'speed_label', '?')}  "
                f"rtf={getattr(status, 'actual_realtime', float('nan')):.2f}x/"
                f"{getattr(status, 'target_realtime', float('nan')):.2f}x "
            )
        lines = [header.center(w, "=")]

        def pos_line(label: str, term_name: str | None, fallback: str) -> str:
            if layout is None or term_name is None:
                return f" {label:<11s} n/a ({fallback})"
            chunk = layout.slice(obs, term_name)
            s, _ = _xyz(chunk)
            ref = f"{term_name}[0:3]" if len(chunk) > 3 else term_name
            return f" {label:<11s} {s}   <- {ref}"

        lines.append(
            pos_line(
                "EEF pos",
                layout.ee_term if layout else None,
                "no ee_pose term in this task's obs",
            )
        )
        lines.append(
            pos_line(
                "GOAL",
                layout.goal_term if layout else None,
                "command not exposed in this task's obs",
            )
        )
        lines.append(
            pos_line(
                "OBJECT pos",
                layout.object_term if layout else None,
                "no task-specific pose term found",
            )
        )

        lines.append(
            f" REWARD      step={reward_v:+9.4f}   episode_cumulative={self._episode_reward:+10.4f}"
        )
        lines.append(f" DONE        {done_v}")
        lines.append("=" * w)
        return lines

    def _render_summary(
        self,
        qpos_stats: _RunningStats,
        qvel_stats: _RunningStats,
        qpos_bad: bool,
        qvel_bad: bool,
    ) -> list[str]:
        w = _PANEL_WIDTH
        lines = [
            (
                f" EPISODE {self._episode_num} SUMMARY — {self._frame} frames, "
                "aggregated min/mean/max over the whole episode "
            ).center(w, "=")
        ]
        if self._term_stats:
            total = sum(s.dim for s in self._term_stats.values())
            lines.append(f" obs terms (n={total}, exactly observation.state):")
            for name, stats in self._term_stats.items():
                lines.append(f"   {name:<14s} [{stats.dim:3d}]  {stats.format()}")
        else:
            lines.append(" observation.state  (obs-term layout unavailable)")

        lines.append("-" * w)
        lines.append(" also saved to the dataset:")
        for key, stats in (
            ("action", self._action_stats),
            ("robot_state", self._robot_state_stats),
            ("qpos", qpos_stats),
            ("qvel", qvel_stats),
        ):
            if stats.dim == 0:
                continue  # never provided this episode (e.g. no qpos_dim)
            lines.append(f"   {key:<14s} [{stats.dim:3d}]  {stats.format()}")

        lines.append("-" * w)
        if self._anomaly_frames or qpos_bad or qvel_bad:
            lines.append(" ANOMALIES this episode:")
            for name, count in self._anomaly_frames.items():
                lines.append(f"   non-finite value in '{name}' — {count} frame(s)")
            if qpos_bad:
                lines.append("   non-finite value in 'qpos' (checked at episode end)")
            if qvel_bad:
                lines.append("   non-finite value in 'qvel' (checked at episode end)")
            lines.append("   this episode's data is suspect — consider U to undo")
        else:
            lines.append(" ANOMALIES this episode: none")
        return lines

    def _print_overwrite(self, lines: list[str]) -> None:
        """Redraw in place; append-only when piped or append_only set."""
        out = sys.stdout
        overwrite = (not self._append_only) and sys.stdout.isatty()
        if overwrite and self._printed_lines:
            out.write(f"\x1b[{self._printed_lines}A")
        for line in lines:
            if overwrite:
                out.write("\x1b[2K")
            out.write(line + "\n")
        out.flush()
        self._printed_lines = len(lines) if overwrite else 0

    def _print_lines(self, lines: list[str]) -> None:
        """Plain append for the episode summary (never overwritten)."""
        out = sys.stdout
        for line in lines:
            out.write(line + "\n")
        out.flush()
