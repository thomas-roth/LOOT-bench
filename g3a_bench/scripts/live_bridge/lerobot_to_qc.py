"""Convert a locally-recorded LeRobot dataset to the QC transition format.

LeRobot stores one timestep per row.  QC needs explicit (s_t, a_t, r_t, done_t,
s_{t+1}) transition arrays.  This script bridges that gap.

``live_record.py`` writes two *separate* LeRobotDataset roots per session —
``<base>/success/<task>/<stamp>/`` and ``<base>/failure/<task>/<stamp>/`` —
rather than one dataset filtered by a ``success`` column, so this script
takes the success root (required) and an optional failure root to include
those episodes too.

The output .npz contains two logical sections:

    QC training keys (pass only these to Dataset.create()):
        observations, next_observations, actions, rewards, terminals, masks

    Metadata (preserved for analysis; strip before Dataset.create()):
        episode_index, frame_index, timestamp, success,
        action_min, action_max  (denormalize at deployment),
        action_norm_method, reward_max, task_id
        (the reward convention this file was built under; g3a_bench.qc
        checks it against the live env before training)


Loading for QC training::

    QC_KEYS = {"observations", "next_observations", "actions",
               "rewards", "terminals", "masks"}
    raw = np.load("dataset.npz")
    train_dataset = Dataset.create(**{k: raw[k] for k in QC_KEYS})

Passing metadata keys directly to Dataset.create() would break the offline→online
transition: QC's ReplayBuffer expects every key in the offline dataset to also appear
in online transition dicts, which only have the 6 standard keys.

Action normalization
--------------------
The stored leader_action (real-robot joint positions) is first converted to the
actual sim action that was sent to env.step() during recording via
``lerobot_to_mjlab_raw_action(gravity_compensate=True)``.
That sim action is then normalized to [-1, 1] per joint.

Two normalization methods are supported (``--action-norm`` / ``action_norm`` arg):

  ``physical``  (default) — bounds are the full DK1 joint range expressed in
                raw_action space: ``(MJCF_ctrlrange - HOME) / ACTION_SCALE``.
                Fixed and session-independent; safe to combine datasets.
                Data may not fill [-1, 1] for joints far from HOME at rest.

  ``data``      — bounds are per-joint min/max of *this* dataset.
                Guarantees all training actions fill [-1, 1]; best learning
                quality for a single dataset.  Normalization differs between
                sessions — combine datasets only after re-normalizing together.

Usage:
    # only successful episodes, physical normalization (defaults)
    uv run python g3a_bench/scripts/live_bridge/lerobot_to_qc.py \\
        --success-dir out/demos/success/G3A-Stack-Low-Dk1_uni/2026-06-29_10-00-00 \\
        --output dataset.npz

    # data-driven normalization
    uv run python g3a_bench/scripts/live_bridge/lerobot_to_qc.py \\
        --success-dir ... --output dataset.npz --action-norm data

    # include failure episodes too
    uv run python g3a_bench/scripts/live_bridge/lerobot_to_qc.py \\
        --success-dir out/demos/success/G3A-Stack-Low-Dk1_uni/2026-06-29_10-00-00 \\
        --failure-dir out/demos/failure/G3A-Stack-Low-Dk1_uni/2026-06-29_10-00-00 \\
        --output dataset.npz
"""

import argparse
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from g3a_bench.scripts.live_bridge.mapper import (
    PHYSICAL_RAW_ACTION_MAX,
    PHYSICAL_RAW_ACTION_MIN,
    ActionNormalizer,
    lerobot_to_mjlab_raw_action,
)

QC_KEYS = frozenset(
    {"observations", "next_observations", "actions", "rewards", "terminals", "masks"}
)

ACTION_NORM_METHODS = ("physical", "data")


def _resolve_reward_max(task_id: str | None, override: float | None) -> float:
    """Per-step reward maximum of the task these demos were recorded on.

    This is the constant the env subtracts (``tasks/common/rewards.py``): 2.0
    for the ``progress`` + ``success`` categories, 1.0 for the success-only
    ones.  Resolved from the registered task cfg so the dataset cannot drift
    from the env it will be trained against.

    Cannot be inferred from the recording itself: rewards live in
    ``[-reward_max, 0]``, but the floor is only reached at ``progress == 0``,
    which real demos never sit at.
    """
    if override is not None:
        return float(override)
    if not task_id:
        raise ValueError(
            "Could not determine the task id from the recording; pass --task "
            "(e.g. --task G3A-Stack-Low-Dk1_uni) or --reward-max explicitly."
        )

    from mjlab.tasks.registry import load_env_cfg

    import g3a_bench  # noqa: F401  -- registers the tasks
    from g3a_bench.tasks.common.rewards import reward_max

    try:
        cfg = load_env_cfg(task_id, play=False)
    except Exception as exc:  # noqa: BLE001 -- surfaced with actionable advice
        raise ValueError(
            f"Task {task_id!r} (read from the recording) is not registered, so "
            f"its reward_max cannot be resolved. Old recordings may name a "
            f"since-renamed task. Pass --task with the current id, or "
            f"--reward-max to set it directly."
        ) from exc
    return reward_max(cfg.rewards)


def _check_recorded_reward_range(
    rewards: np.ndarray, reward_max: float, task_id: str | None
) -> None:
    """Reject a recording that is not on the task's current reward convention.

    The env emits ``progress + success - reward_max``, so every recorded reward
    must lie in ``[-reward_max, 0]``.  Both bounds catch a real failure:

    - **above 0**: the recording predates the negative convention, when rewards
      were the raw positive term sum scaled by ``step_dt``.
    - **below -reward_max**: the task had *more* reward terms when it was
      recorded than it has now (e.g. Place demos carry a ``progress`` term that
      was removed from the task in d02dc77).

    Neither is recoverable by rescaling -- the second is a different MDP, and
    training on it reintroduces exactly the offline/online reward split this
    convention exists to remove.  Re-record against the current task cfg.
    """
    if not rewards.size:
        return
    lo, hi = float(rewards.min()), float(rewards.max())
    tol = 1e-4
    if hi <= tol and lo >= -reward_max - tol:
        return
    raise ValueError(
        f"Recorded rewards span [{lo:.6f}, {hi:.6f}], outside the "
        f"[{-reward_max}, 0] this task's reward cfg produces "
        f"(task={task_id!r}, reward_max={reward_max}). "
        + (
            "Positive rewards mean the recording predates the negative reward "
            "convention. "
            if hi > tol
            else "Rewards below the floor mean the task had more reward terms "
            "when this was recorded than it has now. "
        )
        + "Neither is fixable by rescaling; re-record against the current task "
        "cfg (or pass --reward-max if you are certain the recording is right "
        "and the lookup is wrong)."
    )


def _check_absorbing_frames_scored_success(
    rewards: np.ndarray, masks: np.ndarray, episode_idx: np.ndarray
) -> None:
    """Reject episodes whose terminal is absorbing but was never scored a success.

    ``masks`` and ``rewards`` come from two *different* sources that nothing
    else cross-checks:

    - ``masks == 0`` is derived from the operator's save-as-success keypress
      (``recorder.save_episode(success=...)``, V vs F).
    - ``rewards`` is what the env emitted.

    A ``masks == 0`` frame is a genuine terminal MDP state, so Q-learning uses
    its reward *as* the target with no bootstrap. If the operator saved an
    episode as a success but the env's ``SuccessGate`` had not fired on the
    last recorded frame, that target becomes ``-reward_max`` instead of ~0 --
    silently teaching the critic that reaching the goal is worth as little as
    never reaching it, which is the exact anchor this whole reward convention
    rests on. ``sim_done`` exists to spot this (operator-stopped episodes have
    ``next.done == False`` on their last frame) but is metadata only.

    The test: ``success`` contributes exactly its weight (1.0 in every
    category) when it fires, so a scored frame sits 1.0 above the same state
    unscored. Half that is a clean separator, and it comfortably tolerates the
    one category where a success frame is not exactly 0 -- Stack's success is a
    two-sided tolerance band (``|height - target| <= 0.01``) while its
    ``progress`` is ``(best/target).clamp(0, 1)``, so an in-band undershoot
    scores up to ``tol / target_height`` (<= 0.083) short.
    """
    absorbing = masks == 0.0
    if not absorbing.any():
        return
    bad = absorbing & (rewards < -0.5)
    if not bad.any():
        return
    bad_eps = np.unique(episode_idx[bad])
    raise ValueError(
        f"{int(bad.sum())} frame(s) across episode(s) {bad_eps.tolist()} are "
        f"marked as terminal successes (masks == 0) but scored "
        f"{rewards[bad].min():.4f}..{rewards[bad].max():.4f}, i.e. the env's "
        f"success term never fired there. Q-learning would treat those as "
        f"absorbing states worth that much instead of ~0. This happens when an "
        f"episode is saved with the success key before the task's success gate "
        f"latches. Drop or re-record those episodes (check `sim_done` in the "
        f"recording to find them)."
    )


def _compute_terminals_and_masks(
    episode_idx: np.ndarray, success: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Derive QC's `terminals` and `masks` arrays from episode boundaries + outcome.

    These are two distinct signals, easy to conflate:

    - `terminals[t] == 1` marks the last frame of an episode. This is pure
      trajectory-boundary bookkeeping, used by ``Dataset.sample_sequence`` to
      avoid sampling an action-chunk window that crosses into the next episode.
    - `masks[t] == 0` marks a genuine terminal MDP state, so Q-learning does not
      bootstrap through it. Only a *successful* episode's last frame is a true
      terminal state; a timeout or aborted-as-failure episode just stopped
      recording without reaching one, so it must keep `masks == 1` (bootstrap as
      normal). This matches OGBench's own relabeling convention
      (``ogbench/relabel_utils.py``: ``masks = 1.0 - success``), which computes
      `masks` from task success, never from trajectory boundaries.

    Args:
        episode_idx: (N,) episode index per frame, sorted so each episode's rows
            are contiguous.
        success: (N,) bool, whether the episode succeeded (constant per episode).

    Returns:
        (terminals, masks), both (N,) float32.
    """
    n = len(episode_idx)
    terminals = np.zeros(n, dtype=np.float32)
    last_frames = np.where(episode_idx[:-1] != episode_idx[1:])[0]
    terminal_idx = np.concatenate([last_frames, [n - 1]])
    terminals[terminal_idx] = 1.0

    masks = np.ones(n, dtype=np.float32)
    masks[terminal_idx] = 1.0 - success[terminal_idx].astype(np.float32)
    return terminals, masks


def _load_dataset_columns(dataset_root: str, episode_offset: int) -> dict:
    """Load one LeRobotDataset root's columns as numpy arrays.

    ``episode_offset`` is added to this root's episode_index so that episodes
    from the success and failure roots don't collide when concatenated (both
    are independently indexed from 0 by the recorder).
    """
    ds = LeRobotDataset("live_record", root=dataset_root)
    hf = ds.hf_dataset
    # `task` is the MJLab task id live_record.py was run with; LeRobot stores
    # it as the index of the tasks table. Used to look up the task's reward_max.
    task_ids = list(ds.meta.tasks.index)
    return {
        "task_id": task_ids[0] if len(task_ids) == 1 else None,
        "episode_index": np.array(hf["episode_index"]) + episode_offset,
        "frame_index": np.array(hf["frame_index"]),
        "obs": np.array(hf["observation.state"], dtype=np.float32),
        "leader_actions": np.array(hf["action"], dtype=np.float32),
        "rewards": np.array(hf["next.reward"], dtype=np.float32),
        "success": np.array(hf["success"]).astype(bool),
        "timestamp": np.array(hf["timestamp"], dtype=np.float64),
        # next.done is kept as metadata only — it is unreliable as a terminal
        # signal because operator-stopped episodes have next.done=False on
        # their last frame. Terminals are recomputed from episode_index
        # boundaries after sorting/concatenation.
        "sim_done": np.array(hf["next.done"]).astype(bool),
    }


def convert(
    success_dir: str,
    output_path: str | Path,
    failure_dir: str | None = None,
    action_norm: str = "physical",
    task: str | None = None,
    reward_max: float | None = None,
) -> dict:
    """Convert live_record.py LeRobot dataset(s) to QC transition format.

    Reads the success dataset at ``success_dir`` (and, if given, the failure
    dataset at ``failure_dir``), assembles explicit (s_t, a_t, r_t, done_t,
    s_{t+1}) transition tuples, and saves the result as a ``.npz`` file.

    The file contains 6 QC training keys plus metadata keys (episode_index,
    frame_index, timestamp, success, action_min, action_max).  When loading for
    QC training, filter to ``QC_KEYS`` before calling ``Dataset.create()``;
    see module docstring.

    Rewards are the recorded per-step env reward, passed through **unchanged**.
    No arithmetic is applied and there is no reward-mode switch: the env itself
    emits ``progress + success - reward_max`` (see ``tasks/common/rewards.py``),
    so whatever ``live_record`` stored is already the training convention, and
    the dataset is identical to what the env emits online by construction.
    Recordings predating that convention are rejected rather than converted --
    their reward config no longer describes the task, and no rescaling recovers
    a term the current env does not emit.

    Args:
        success_dir: Path to the success LeRobotDataset root, e.g.
            ``out/demos/success/G3A-Stack-Low-Dk1_uni/2026-06-29_10-00-00``.
        output_path: Where to write the ``.npz`` file.
        failure_dir: Path to the sibling failure LeRobotDataset root. When
            given, failure episodes are included alongside the successes;
            when omitted (default), only the success dataset is converted.
        action_norm: Normalization method for actions.  ``"physical"`` (default)
            uses the full DK1 joint range in raw_action space — fixed and
            session-independent. ``"data"`` uses per-joint min/max of this
            dataset — fills [-1, 1] fully but is session-specific.
        task: MJLab task id these demos were recorded on. Defaults to the id
            stored in the recording. Used only to look up ``reward_max``, which
            is stamped into the ``.npz`` for the training-time convention check.
        reward_max: Override for the task's per-step reward maximum (1.0 or
            2.0), skipping the registry lookup. Rarely needed.

    Returns:
        The full assembled dict (QC keys + metadata) as numpy arrays.
    """
    if action_norm not in ACTION_NORM_METHODS:
        raise ValueError(
            f"action_norm must be one of {ACTION_NORM_METHODS}, got {action_norm!r}"
        )
    # 1. Load dataset(s). Success and failure roots are independently indexed
    # from 0 by the recorder, so offset the failure episode_index past the
    # success episodes to keep episode boundaries (used for terminals) global.
    success_cols = _load_dataset_columns(success_dir, episode_offset=0)
    n_success_eps = int(len(np.unique(success_cols["episode_index"])))
    parts = [success_cols]
    n_failure_eps = 0
    if failure_dir is not None:
        failure_cols = _load_dataset_columns(failure_dir, episode_offset=n_success_eps)
        # Both roots must be the same task: reward_max is looked up from the
        # success root's id, and mismatched roots would put two reward scales
        # (and two observation layouts) in one dataset.
        if (
            failure_cols["task_id"] is not None
            and success_cols["task_id"] is not None
            and failure_cols["task_id"] != success_cols["task_id"]
        ):
            raise ValueError(
                f"success-dir is task {success_cols['task_id']!r} but "
                f"failure-dir is {failure_cols['task_id']!r}; these cannot be "
                f"merged into one dataset."
            )
        n_failure_eps = int(len(np.unique(failure_cols["episode_index"])))
        parts.append(failure_cols)

    episode_idx = np.concatenate([p["episode_index"] for p in parts])
    frame_idx = np.concatenate([p["frame_index"] for p in parts])
    obs = np.concatenate([p["obs"] for p in parts])
    leader_actions = np.concatenate([p["leader_actions"] for p in parts])
    rewards = np.concatenate([p["rewards"] for p in parts])
    success = np.concatenate([p["success"] for p in parts])
    timestamp = np.concatenate([p["timestamp"] for p in parts])
    sim_done = np.concatenate([p["sim_done"] for p in parts])

    # 1b. Resolve the task's reward_max, the constant the env subtracts. Both
    # reward modes are expressed relative to it, and it is stored in the .npz
    # so training can reject a dataset built against a different reward cfg.
    task_id = task or success_cols["task_id"]
    resolved_reward_max = _resolve_reward_max(task_id, reward_max)

    print(
        f"[INFO] Loaded {len(obs)} frames across {n_success_eps + n_failure_eps} "
        f"episodes ({n_success_eps} successes, {n_failure_eps} failures)"
    )

    # 2. Reconstruct the actual sim action sent to env.step() during recording.
    # live_record._LivePolicy called lerobot_to_mjlab_raw_action with
    # gravity_compensate=True. The stored leader_action is in real-robot space;
    # this recovers the exact raw_action that caused each obs[t] → obs[t+1] transition.
    sim_actions = np.array(
        [
            lerobot_to_mjlab_raw_action(a, gravity_compensate=True)
            for a in leader_actions
        ],
        dtype=np.float32,
    )  # (N, 7)

    # 3. Sort by (episode_index, frame_index) to guarantee correct order.
    order = np.lexsort((frame_idx, episode_idx))
    episode_idx = episode_idx[order]
    frame_idx = frame_idx[order]
    obs = obs[order]
    sim_actions = sim_actions[order]
    rewards = rewards[order]
    success = success[order]
    timestamp = timestamp[order]
    sim_done = sim_done[order]

    # 4. Normalize sim_actions to [-1, 1].
    if action_norm == "physical":
        a_min = PHYSICAL_RAW_ACTION_MIN
        a_max = PHYSICAL_RAW_ACTION_MAX
    else:  # "data"
        a_min = sim_actions.min(axis=0)
        a_max = sim_actions.max(axis=0)
    actions = ActionNormalizer(a_min, a_max).normalize(sim_actions).astype(np.float32)

    # 5. Build next_observations — never cross episode boundaries.
    # For non-terminal rows whose next row is in the same episode: next_obs = obs[i+1].
    # For terminal rows: next_obs stays zero — harmless, because `terminals` (below)
    # already tells Dataset.sample_sequence not to sample a chunk across this
    # boundary, so this dummy value is never read.
    next_obs = np.zeros_like(obs)
    same_ep = episode_idx[:-1] == episode_idx[1:]  # (N-1,) bool
    valid = np.where(same_ep)[0]
    next_obs[valid] = obs[valid + 1]

    # 6. terminals = episode-boundary bookkeeping; masks = Q-bootstrap signal.
    # See _compute_terminals_and_masks docstring — these are NOT complements of
    # each other in general (only success-terminated episodes get masks=0).
    terminals, masks = _compute_terminals_and_masks(episode_idx, success)

    # 6b. Reward: the recorded per-step env reward, passed through untouched.
    # `rewards` is already `progress + success - reward_max` because the env
    # applies the shift itself and live_record stores what env.step() returned.
    # All this block does is check the recording is on that convention: the
    # right scale overall, and a reward consistent with `masks` at each
    # absorbing frame.
    _check_recorded_reward_range(rewards, resolved_reward_max, task_id)
    _check_absorbing_frames_scored_success(rewards, masks, episode_idx)

    # 7. Assemble and save.
    out = {
        # ── QC training keys — pass only these to Dataset.create() ──────────
        "observations": obs,
        "next_observations": next_obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "masks": masks,
        # ── metadata — preserved in the file, strip before Dataset.create() ─
        "episode_index": episode_idx.astype(np.int32),
        "frame_index": frame_idx.astype(np.int32),
        "timestamp": timestamp,
        "success": success,
        "sim_done": sim_done,  # True only when sim itself terminated (not operator stop)
        # Denormalize at deployment: raw_action = (norm+1)/2*(action_max-action_min)+action_min
        "action_min": a_min.astype(np.float32),
        "action_max": a_max.astype(np.float32),
        "action_norm_method": np.array(action_norm),
        # Reward convention this file was built under. g3a_bench.qc checks it
        # against the live env and refuses to train on a mismatch.
        "reward_max": np.array(resolved_reward_max, dtype=np.float32),
        "task_id": np.array(task_id if task_id else ""),
    }
    np.savez(output_path, **out)

    print(
        f"[INFO] Saved to {output_path}  "
        f"(action_norm={action_norm!r}, task={task_id!r}, "
        f"reward_max={resolved_reward_max})"
    )
    print("       QC keys → pass to Dataset.create()")
    print(f"         observations:      {obs.shape} {obs.dtype}")
    print(f"         next_observations: {next_obs.shape} {next_obs.dtype}")
    print(
        f"         actions:           {actions.shape} {actions.dtype}  "
        f"min={actions.min():.3f} max={actions.max():.3f}"
    )
    print(
        f"         rewards:           {rewards.shape}  "
        f"min={rewards.min():.3f} max={rewards.max():.3f}"
    )
    print(
        f"         terminals:         {terminals.shape}  "
        f"sum={int(terminals.sum())} (= episodes)"
    )
    print(f"         masks:             {masks.shape}")
    print("       metadata → strip before Dataset.create()")
    print(f"         episode_index:     {episode_idx.shape} int32")
    print(f"         frame_index:       {frame_idx.shape} int32")
    print(f"         timestamp:         {timestamp.shape} {timestamp.dtype}")
    print(f"         success:           {success.shape} bool")
    print(
        f"         sim_done:          {sim_done.shape} bool  (True when sim terminated, not operator-stop)"
    )
    print(f"         action_min:        {a_min}")
    print(f"         action_max:        {a_max}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert live_record.py LeRobot dataset(s) to QC transition format (.npz)"
    )
    parser.add_argument(
        "--success-dir",
        required=True,
        help=(
            "Path to the success LeRobotDataset root, "
            "e.g. out/demos/success/G3A-Stack-Low-Dk1_uni/2026-06-29_10-00-00"
        ),
    )
    parser.add_argument(
        "--failure-dir",
        default=None,
        help=(
            "Path to the sibling failure LeRobotDataset root. When given, "
            "failure episodes are included too. Default: successes only."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npz file path.",
    )
    parser.add_argument(
        "--action-norm",
        default="physical",
        choices=list(ACTION_NORM_METHODS),
        help=(
            "Action normalization method. "
            "'physical' (default): fixed bounds from DK1 joint limits — session-independent. "
            "'data': per-joint min/max of this dataset — fills [-1,1] fully but session-specific."
        ),
    )
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "MJLab task id the demos were recorded on, e.g. "
            "G3A-Stack-Low-Dk1_uni. Default: read from the recording. Used to "
            "look up the task's reward_max."
        ),
    )
    parser.add_argument(
        "--reward-max",
        type=float,
        default=None,
        help=(
            "Override the task's per-step reward maximum instead of looking it "
            "up in the task registry (2.0 for progress+success categories, 1.0 "
            "for success-only). Rarely needed."
        ),
    )
    args = parser.parse_args()

    convert(
        success_dir=args.success_dir,
        output_path=args.output,
        failure_dir=args.failure_dir,
        action_norm=args.action_norm,
        task=args.task,
        reward_max=args.reward_max,
    )
