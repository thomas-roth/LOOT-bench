"""Convert a locally-recorded LeRobot dataset to the QC transition format.

LeRobot stores one timestep per row.  QC needs explicit (s_t, a_t, r_t, done_t,
s_{t+1}) transition arrays.  This script bridges that gap.

The output .npz contains two logical sections:

    QC training keys (pass only these to Dataset.create()):
        observations, next_observations, actions, rewards, terminals, masks

    Metadata (preserved for analysis; strip before Dataset.create()):
        episode_index, frame_index, timestamp, success,
        action_min, action_max  (denormalize at deployment)

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
        --dataset-dir logs/live_record/2026-06-29_10-00-00 \\
        --output dataset.npz

    # data-driven normalization
    uv run python g3a_bench/scripts/live_bridge/lerobot_to_qc.py \\
        --dataset-dir ... --output dataset.npz --action-norm data

    # include failure episodes too
    uv run python g3a_bench/scripts/live_bridge/lerobot_to_qc.py \\
        --dataset-dir ... --output dataset.npz --all-episodes
"""

import argparse
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from g3a_bench.scripts.live_bridge.mapper import (
    PHYSICAL_RAW_ACTION_MAX,
    PHYSICAL_RAW_ACTION_MIN,
    lerobot_to_mjlab_raw_action,
)

QC_KEYS = frozenset(
    {"observations", "next_observations", "actions", "rewards", "terminals", "masks"}
)

ACTION_NORM_METHODS = ("physical", "data")


def convert(
    dataset_root: str,
    output_path: str | Path,
    only_successes: bool = True,
    action_norm: str = "physical",
) -> dict:
    """Convert a LeRobot dataset recorded by live_record.py to QC transition format.

    Reads the session stored at ``dataset_root``, assembles explicit
    (s_t, a_t, r_t, done_t, s_{t+1}) transition tuples, and saves the result as
    a ``.npz`` file.

    The file contains 6 QC training keys plus metadata keys (episode_index,
    frame_index, timestamp, success, action_min, action_max).  When loading for
    QC training, filter to ``QC_KEYS`` before calling ``Dataset.create()``;
    see module docstring.

    Args:
        dataset_root: Path to the timestamped session directory, e.g.
            ``logs/live_record/2026-06-29_10-00-00``.
        output_path: Where to write the ``.npz`` file.
        only_successes: When True (default) only episodes labelled as success
            by the operator are included.  Failure / discarded episodes are dropped.
        action_norm: Normalization method for actions.  ``"physical"`` (default)
            uses the full DK1 joint range in raw_action space — fixed and
            session-independent.  ``"data"`` uses per-joint min/max of this
            dataset — fills [-1, 1] fully but is session-specific.

    Returns:
        The full assembled dict (QC keys + metadata) as numpy arrays.
    """
    if action_norm not in ACTION_NORM_METHODS:
        raise ValueError(
            f"action_norm must be one of {ACTION_NORM_METHODS}, got {action_norm!r}"
        )
    # 1. Load dataset
    ds = LeRobotDataset("live_record", root=dataset_root)
    hf = ds.hf_dataset

    # 2. Extract columns as numpy.
    # Multi-dim features (observation.state, action) are list columns in HuggingFace;
    # np.array() converts them to 2D arrays.  Scalar columns (next.reward, next.done,
    # success, timestamp, indices) come out as (N,) directly — no squeeze needed.
    episode_idx = np.array(hf["episode_index"])
    frame_idx = np.array(hf["frame_index"])
    obs = np.array(hf["observation.state"], dtype=np.float32)  # (N, obs_dim)
    leader_actions = np.array(
        hf["action"], dtype=np.float32
    )  # (N, 7) real-robot format
    rewards = np.array(hf["next.reward"], dtype=np.float32)  # (N,)
    success = np.array(hf["success"]).astype(bool)  # (N,)
    timestamp = np.array(hf["timestamp"], dtype=np.float64)  # (N,)
    # next.done is kept as metadata only — it is unreliable as a terminal signal
    # because operator-stopped episodes have next.done=False on their last frame.
    # Terminals are recomputed from episode_index boundaries after sorting/filtering.
    sim_done = np.array(hf["next.done"]).astype(bool)  # (N,) metadata only

    # Reconstruct the actual sim action sent to env.step() during recording.
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

    n_episodes = len(np.unique(episode_idx))
    n_success_eps = int(len(np.unique(episode_idx[success])))
    n_failure_eps = n_episodes - n_success_eps
    print(
        f"[INFO] Loaded {len(obs)} frames across {n_episodes} episodes "
        f"({n_success_eps} successes, {n_failure_eps} failures)"
    )

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

    # 4. Filter to successful episodes only (if requested).
    if only_successes:
        keep = success
        episode_idx = episode_idx[keep]
        frame_idx = frame_idx[keep]
        obs = obs[keep]
        sim_actions = sim_actions[keep]
        rewards = rewards[keep]
        success = success[keep]
        timestamp = timestamp[keep]
        sim_done = sim_done[keep]
        n_kept_eps = int(len(np.unique(episode_idx)))
        print(
            f"[INFO] Kept {int(keep.sum())} frames from {n_kept_eps} episodes "
            f"(only_successes=True)"
        )
    else:
        print(f"[INFO] Keeping all {len(obs)} frames ({n_episodes} episodes)")

    # 5. Normalize sim_actions to [-1, 1].
    if action_norm == "physical":
        a_min = PHYSICAL_RAW_ACTION_MIN
        a_max = PHYSICAL_RAW_ACTION_MAX
    else:  # "data"
        a_min = sim_actions.min(axis=0)
        a_max = sim_actions.max(axis=0)
    a_range = np.where(a_max - a_min > 1e-6, a_max - a_min, 1.0)
    actions = (2.0 * (sim_actions - a_min) / a_range - 1.0).astype(np.float32)

    # 6. Build next_observations — never cross episode boundaries.
    # For non-terminal rows whose next row is in the same episode: next_obs = obs[i+1].
    # For terminal rows: next_obs stays zero; masks=0 prevents bootstrapping there,
    # so the content of next_obs at terminal positions is never used.
    next_obs = np.zeros_like(obs)
    same_ep = episode_idx[:-1] == episode_idx[1:]  # (N-1,) bool
    valid = np.where(same_ep)[0]
    next_obs[valid] = obs[valid + 1]

    # 7. Terminals and masks derived from episode boundaries, not next.done.
    # next.done is False on the last frame of operator-stopped episodes, so using it
    # would leave masks=1.0 at terminal states, causing incorrect Q-value bootstrapping.
    terminals = np.zeros(len(obs), dtype=np.float32)
    last_frames = np.where(episode_idx[:-1] != episode_idx[1:])[0]
    terminals[last_frames] = 1.0
    terminals[-1] = 1.0  # last frame of the last episode
    masks = (1.0 - terminals).astype(np.float32)

    # 8. Assemble and save.
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
    }
    np.savez(output_path, **out)

    print(f"[INFO] Saved to {output_path}  (action_norm={action_norm!r})")
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
        description="Convert a live_record.py LeRobot session to QC transition format (.npz)"
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help=(
            "Path to the timestamped session directory, "
            "e.g. logs/live_record/2026-06-29_10-00-00"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npz file path.",
    )
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Include failure episodes. Default: only successful episodes are converted.",
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
    args = parser.parse_args()

    convert(
        dataset_root=args.dataset_dir,
        output_path=args.output,
        only_successes=not args.all_episodes,
        action_norm=args.action_norm,
    )
