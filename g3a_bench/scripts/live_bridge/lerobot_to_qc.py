"""Convert a locally-recorded LeRobot dataset to the QC transition format.

LeRobot stores one timestep per row.  QC needs explicit (s_t, a_t, r_t, done_t,
s_{t+1}) transition arrays.  This script bridges that gap.

``live_record.py`` writes two *separate* LeRobotDataset roots per session —
``<base>/success/<stamp>/`` and ``<base>/failure/<stamp>/`` — rather than one
dataset filtered by a ``success`` column, so this script takes the success
root (required) and an optional failure root to include those episodes too.

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
        --success-dir out/demos/success/2026-06-29_10-00-00 \\
        --output dataset.npz

    # data-driven normalization
    uv run python g3a_bench/scripts/live_bridge/lerobot_to_qc.py \\
        --success-dir ... --output dataset.npz --action-norm data

    # include failure episodes too
    uv run python g3a_bench/scripts/live_bridge/lerobot_to_qc.py \\
        --success-dir out/demos/success/2026-06-29_10-00-00 \\
        --failure-dir out/demos/failure/2026-06-29_10-00-00 \\
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


def _load_dataset_columns(dataset_root: str, episode_offset: int) -> dict:
    """Load one LeRobotDataset root's columns as numpy arrays.

    ``episode_offset`` is added to this root's episode_index so that episodes
    from the success and failure roots don't collide when concatenated (both
    are independently indexed from 0 by the recorder).
    """
    ds = LeRobotDataset("live_record", root=dataset_root)
    hf = ds.hf_dataset
    return {
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
) -> dict:
    """Convert live_record.py LeRobot dataset(s) to QC transition format.

    Reads the success dataset at ``success_dir`` (and, if given, the failure
    dataset at ``failure_dir``), assembles explicit (s_t, a_t, r_t, done_t,
    s_{t+1}) transition tuples, and saves the result as a ``.npz`` file.

    The file contains 6 QC training keys plus metadata keys (episode_index,
    frame_index, timestamp, success, action_min, action_max).  When loading for
    QC training, filter to ``QC_KEYS`` before calling ``Dataset.create()``;
    see module docstring.

    Args:
        success_dir: Path to the success LeRobotDataset root, e.g.
            ``out/demos/success/2026-06-29_10-00-00``.
        output_path: Where to write the ``.npz`` file.
        failure_dir: Path to the sibling failure LeRobotDataset root. When
            given, failure episodes are included alongside the successes;
            when omitted (default), only the success dataset is converted.
        action_norm: Normalization method for actions.  ``"physical"`` (default)
            uses the full DK1 joint range in raw_action space — fixed and
            session-independent. ``"data"`` uses per-joint min/max of this
            dataset — fills [-1, 1] fully but is session-specific.

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
    # For terminal rows: next_obs stays zero; masks=0 prevents bootstrapping there,
    # so the content of next_obs at terminal positions is never used.
    next_obs = np.zeros_like(obs)
    same_ep = episode_idx[:-1] == episode_idx[1:]  # (N-1,) bool
    valid = np.where(same_ep)[0]
    next_obs[valid] = obs[valid + 1]

    # 6. Terminals and masks derived from episode boundaries, not next.done.
    # next.done is False on the last frame of operator-stopped episodes, so using it
    # would leave masks=1.0 at terminal states, causing incorrect Q-value bootstrapping.
    terminals = np.zeros(len(obs), dtype=np.float32)
    last_frames = np.where(episode_idx[:-1] != episode_idx[1:])[0]
    terminals[last_frames] = 1.0
    terminals[-1] = 1.0  # last frame of the last episode
    masks = (1.0 - terminals).astype(np.float32)

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
        description="Convert live_record.py LeRobot dataset(s) to QC transition format (.npz)"
    )
    parser.add_argument(
        "--success-dir",
        required=True,
        help=(
            "Path to the success LeRobotDataset root, "
            "e.g. out/demos/success/2026-06-29_10-00-00"
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
    args = parser.parse_args()

    convert(
        success_dir=args.success_dir,
        output_path=args.output,
        failure_dir=args.failure_dir,
        action_norm=args.action_norm,
    )
