"""Fuse several recorded LeRobot sessions into one LeRobot root.

``loot_bench.qc.convert`` already fuses sessions at ``.npz`` time: several
roots go in one call and their episodes are offset apart. That covers
training. What it does not cover is every consumer of the LeRobot layout
itself -- ``split_demos``, the replay readers, inspection -- which take one
root. This script is that fuse: a thin wrapper around lerobot's
``merge_datasets`` that additionally merges this repo's sidecar
(``loot_extra_info.json``), which lerobot does not know about.

Sidecar rules mirror :func:`loot_bench.qc.convert.convert`:

* ``start_pose_noise`` must agree across roots; a missing sidecar counts as
  ``0.0``, as in convert. Mixed noise refuses: one fused root stamps one
  noise.
* ``episode_seeds`` are reindexed with a running episode offset. Every source
  numbers from zero, so without that two episodes would claim one seed.
* ``seed_device``/``operator_id``/``split_from`` describe the session, so they
  survive only when every source agrees; a mixed fuse omits them rather than
  stamping one session's claim on another's episodes.

Usage:
    uv run loot-fuse-demos \\
        --src out/demos/success/LOOT-Stack-1Henge-Dk1/2026-09-13_18-16-20 \\
        --src out/demos/success/LOOT-Stack-1Henge-Dk1/2026-09-13_19-53-22 \\
        --out out/demos/fused/LOOT-Stack-1Henge-Dk1/<stamp>
    uv run loot-convert-demos --success-dir <out> --output dataset/foo.npz
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

# Importing any loot_bench submodule pulls in mjlab first via
# loot_bench/__init__.py, so only the process entry point can order lerobot
# before mjlab. If a segfault appears here, move the ordering there.
import lerobot.datasets.lerobot_dataset  # noqa: F401
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from loot_bench.scripts.live_bridge.dataset_extra_info import (
    load_extra_info,
    write_extra_info,
)

_DEFAULT_REPO_ID = "live_record"
"""Every recording session is written under this id; convert reopens with it."""

# Session-level sidecar keys: carried only when every source agrees.
_UNANIMOUS_EXTRA_KEYS = ("seed_device", "operator_id", "split_from")


def fuse_extra_info(extra_infos: Sequence[dict], episode_counts: Sequence[int]) -> dict:
    """Merge several sidecars into the fused root's one.

    Args:
        extra_infos: one sidecar dict per source, in fuse order. Empty for a
            root without a sidecar (counts as ``start_pose_noise`` 0.0, as in
            convert; synthetic test roots have none).
        episode_counts: one episode count per source, in the same order.

    Raises:
        ValueError: on a missing ``start_pose_noise`` key, on disagreement
            across roots, or on a length mismatch.
    """
    if len(extra_infos) != len(episode_counts):
        raise ValueError(
            f"{len(extra_infos)} sidecars for {len(episode_counts)} sources"
        )
    if not any(extra_infos):
        return {}

    noises: list[float] = []
    for info in extra_infos:
        if not info:
            noises.append(0.0)
            continue
        if "start_pose_noise" not in info:
            raise ValueError(
                "a loot_extra_info.json has no start_pose_noise; "
                "was it written before this stamp existed?"
            )
        noises.append(float(info["start_pose_noise"]))
    if len(set(noises)) > 1:
        raise ValueError(f"mixed start_pose_noise across dataset roots: {noises}")

    extra: dict = {"start_pose_noise": noises[0]}
    seeds: dict[str, int] = {}
    offset = 0
    for info, count in zip(extra_infos, episode_counts, strict=True):
        for key, seed in (info.get("episode_seeds") or {}).items():
            seeds[str(int(key) + offset)] = int(seed)
        offset += int(count)
    if seeds:
        extra["episode_seeds"] = seeds

    for key in _UNANIMOUS_EXTRA_KEYS:
        values = [info[key] for info in extra_infos if key in info]
        if len(values) == len(extra_infos) and all(v == values[0] for v in values):
            extra[key] = values[0]
    return extra


def fuse_datasets(
    src_roots: Sequence[str | Path],
    out_root: str | Path,
    repo_id: str = _DEFAULT_REPO_ID,
) -> LeRobotDataset:
    """Fuse several LeRobot roots into one, sidecar included.

    Episodes land in fuse order, reindexed from zero; use the fused root
    wherever one root is expected (``split_demos``, replay, ``convert``).
    """
    srcs = [Path(root) for root in src_roots]
    if not srcs:
        raise ValueError("no source roots given")
    out = Path(out_root)
    if out.exists():
        raise FileExistsError(
            f"{out} already exists (refusing to overwrite fused data)."
        )
    for src in srcs:
        # Checked before opening, because of how lerobot fails otherwise: see
        # convert._load_columns for why this message exists.
        if not (src / "meta").is_dir():
            raise FileNotFoundError(
                f"{src} is not a LeRobot dataset root: no meta/ directory. A "
                "session is only readable once the recorder has closed it."
            )

    datasets = [LeRobotDataset(repo_id, root=str(src)) for src in srcs]
    counts = [int(dataset.meta.total_episodes) for dataset in datasets]
    for src, count in zip(srcs, counts, strict=True):
        if count == 0:
            raise ValueError(f"{src} holds no episodes")

    merged = merge_datasets(datasets, output_repo_id=repo_id, output_dir=str(out))
    extra = fuse_extra_info([load_extra_info(src) for src in srcs], counts)
    if extra:
        write_extra_info(out, extra)

    print(f"[fuse] wrote {out}")
    print(
        f"       {merged.meta.total_episodes} episodes, "
        f"{merged.meta.total_frames} frames from {len(srcs)} roots"
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        required=True,
        action="append",
        help="Source LeRobot dataset root, in fuse order (repeatable).",
    )
    parser.add_argument("--out", required=True, help="Output root.")
    parser.add_argument(
        "--repo-id",
        default=_DEFAULT_REPO_ID,
        help="Repo id for the fused dataset (default: live_record).",
    )
    args = parser.parse_args()

    # Expected user errors (missing roots, existing output, mixed sidecars)
    # get a clean message rather than a traceback.
    try:
        fuse_datasets(args.src, args.out, repo_id=args.repo_id)
    except (
        ValueError,
        AssertionError,
        FileExistsError,
        FileNotFoundError,
    ) as error:
        raise SystemExit(f"[fuse] {error}") from None


if __name__ == "__main__":
    main()
