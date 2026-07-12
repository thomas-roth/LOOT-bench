"""Merge multiple lerobot_to_qc .npz files into one combined dataset.

All input files must use the same --action-norm method (default: physical).
Episode indices are automatically renumbered to stay unique across files,
and terminals/masks remain correct because each file already marks its last
frame as terminal.

Usage:
    uv run python g3a_bench/scripts/live_bridge/merge_datasets.py \
        out/demos/success/2026-07-12_16-18-02_dataset.npz \
        out/demos/success/2026-07-12_17-15-06_dataset.npz \
        out/demos/success/2026-07-12_18-32-00_dataset.npz \
        --output combined_45demos.npz
"""

import argparse

import numpy as np

QC_KEYS = [
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "terminals",
    "masks",
]
META_KEYS = ["episode_index", "frame_index", "timestamp", "success", "sim_done"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple dataset .npz files")
    parser.add_argument("inputs", nargs="+", help=".npz files to merge")
    parser.add_argument("--output", required=True, help="combined .npz file")
    args = parser.parse_args()

    loaded = [np.load(f) for f in args.inputs]

    # Verify all use the same action_norm method
    methods = {str(d.get("action_norm_method", b"physical"), "utf-8") for d in loaded}
    if len(methods) > 1:
        raise ValueError(f"Mixed action_norm methods: {methods} — cannot merge")

    combined: dict[str, np.ndarray] = {}

    # QC training keys — simple concatenation
    for key in QC_KEYS:
        combined[key] = np.concatenate([d[key] for d in loaded])

    # Episode index needs offset to stay unique across files
    ep_arrays: list[np.ndarray] = []
    ep_off = 0
    for d in loaded:
        ep_arrays.append(d["episode_index"] + ep_off)
        ep_off = int(ep_arrays[-1].max()) + 1
    combined["episode_index"] = np.concatenate(ep_arrays)

    for key in META_KEYS[1:]:
        combined[key] = np.concatenate([d[key] for d in loaded])

    # Action norm metadata — same across all files, copy from first
    combined["action_min"] = loaded[0]["action_min"]
    combined["action_max"] = loaded[0]["action_max"]
    combined["action_norm_method"] = np.atleast_1d(
        np.array(str(loaded[0]["action_norm_method"]))
    )

    np.savez(
        args.output,
        observations=combined["observations"],
        next_observations=combined["next_observations"],
        actions=combined["actions"],
        rewards=combined["rewards"],
        terminals=combined["terminals"],
        masks=combined["masks"],
        episode_index=combined["episode_index"],
        frame_index=combined["frame_index"],
        timestamp=combined["timestamp"],
        success=combined["success"],
        sim_done=combined["sim_done"],
        action_min=combined["action_min"],
        action_max=combined["action_max"],
        action_norm_method=combined["action_norm_method"],
    )

    n_episodes = int(combined["terminals"].sum())
    n_frames = len(combined["observations"])
    print(
        f"Merged {len(loaded)} files ({n_episodes} episodes, {n_frames} frames) → {args.output}"
    )
    print(f"  action_norm: {methods.pop()!r}")


if __name__ == "__main__":
    main()
