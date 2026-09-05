"""A teleop recording -> the arrays ``qc`` trains on.

``live_record.py`` writes a LeRobot dataset: one row per timestep, holding the
observation the operator acted from, the leader arm's pose, and the reward that
step returned. ``qc`` wants transition arrays. :func:`convert` bridges the two.

The episode-structure rules, what is absorbing, which rows survive, where an
episode ends, all live in :mod:`g3a_bench.qc.transitions` and are tested
without a simulator. This module adds the three things around them:

* **The action.** The recording stores the *leader arm's* joint positions, not
  the action the simulator was stepped with. ``_LivePolicy`` converted one to
  the other at record time, and this repeats that conversion exactly, same
  function, same gravity compensation. Anything else records one trajectory and
  trains on another.
* **The order.** Episode structure is read off row order, so rows are sorted by
  ``(episode, frame)`` before anything looks at them.
* **The stamp.** The action scale and offset that were in force, written into
  the file. ``dataset.check_dataset_matches_env`` compares them against the
  environment a training run actually builds. A dataset recorded under a
  different convention is the failure mode this exists for: nothing raises, the
  loss falls, and evaluation scores zero.

Reading a dataset back lives in :mod:`g3a_bench.qc.dataset`, which imports no
lerobot: the ``qc`` venv has to load a dataset and must not carry the recording
stack.

Observations are copied through untouched. They are the task's ``actor`` group,
which carries no noise on either side, no task declares any, and the adapter
sets the flag as well, so the offline and online observations come from one
distribution.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

# Importing any g3a_bench submodule pulls in mjlab first via
# g3a_bench/__init__.py, so only the process entry point can order lerobot
# before mjlab. If a segfault appears here, move the ordering there.
import lerobot.datasets.lerobot_dataset  # noqa: F401  # pyright: ignore[reportMissingImports]
import numpy as np
from numpy.typing import NDArray

from g3a_bench.qc.dataset import tensor_or_array_to_float32_vector
from g3a_bench.qc.transitions import QC_KEYS, assemble_transitions
from g3a_bench.scripts.live_bridge.dataset_extra_info import load_extra_info
from g3a_bench.scripts.live_bridge.mapper import (
    action_scale_offset,
    lerobot_to_mjlab_raw_action,
)


def _action_scale_offset_from_task(
    task: str, device: str = "cpu"
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Read ``(scale, offset)`` off the built action term for ``task``.

    This is the same term ``g3a_bench.qc.dataset.check_dataset_matches_env``
    reads at train time, so a recording and the simulator cannot disagree.
    """
    from g3a_bench.qc.env import G3aEnv

    env = G3aEnv(task, device=device)
    try:
        term = env.action_term()
        scale = tensor_or_array_to_float32_vector(term.scale)
        offset = tensor_or_array_to_float32_vector(term.offset)
        return scale, offset
    finally:
        env.close()


def _task_from_meta(dataset: object, root: Path) -> str | None:
    """The single task id registered in one LeRobot root, or None.

    lerobot 0.6 keeps the task id out of the frame data, the frames carry a
    ``task_index``, and in ``meta/tasks.parquet`` instead, which
    ``dataset.meta.tasks`` exposes as a DataFrame indexed by task id. The
    metadata loads with the dataset, so this costs nothing beyond the open
    the columns already need.

    Raises:
        ValueError: if one root registered more than one task.
    """
    tasks = getattr(getattr(dataset, "meta", None), "tasks", None)
    if tasks is None or len(tasks) == 0:
        return None
    if len(tasks) > 1:
        raise ValueError(f"mixed tasks inside {root}: {list(tasks.index)}")
    return str(tasks.index[0]) or None


def build_dataset(
    *,
    episode_index: NDArray[np.integer],
    frame_index: NDArray[np.integer],
    observations: NDArray[np.floating],
    leader_actions: NDArray[np.floating],
    rewards: NDArray[np.floating],
    gravity_compensate: bool = True,
    start_pose_noise: float = 0.0,
    task: str | None = None,
    action_scale: NDArray[np.floating] | None = None,
    action_offset: NDArray[np.floating] | None = None,
    device: str = "cpu",
) -> dict[str, NDArray]:
    """Assemble one recording's columns into the training arrays.

    Kept apart from the file handling so the conversion can be tested without
    writing a LeRobot dataset to disk.

    Args:
        episode_index: which episode each row belongs to.
        frame_index: position of each row inside its episode.
        observations: the task's ``actor`` observation, one row per step.
        leader_actions: the leader arm's joint pose, as recorded, radians for
            the arm, normalized ``[0, 1]`` for the gripper.
        rewards: the reward that step returned.
        gravity_compensate: repeat the record-time gravity compensation.
            Leave it on: ``_LivePolicy`` applied it, so switching it off here
            would train on actions the simulator never saw.
        start_pose_noise: half-width [rad] of the start-pose sample the
            episodes were recorded under. Stamped into the ``.npz`` so
            ``dataset.check_dataset_matches_env`` can refuse a dataset that
            was recorded at a different start distribution than the training
            env (``ADR-008``).
        task: task id to read the action stamp from (e.g.
            ``G3A-Stack-Low-Dk1_uni``). When given, ``scale``/``offset`` are
            read off the built action term rather than the DK1-only
            ``mapper.action_scale_offset()``, and the id itself is stamped
            into the output so ``dataset.check_dataset_matches_env`` can
            refuse a dataset loaded against the wrong task. When omitted,
            the legacy DK1 path is used (kept for synthetic tests).
        action_scale: explicit scale to use. When given, ``action_offset`` must
            also be given and ``task`` is ignored. Mainly for tests that want
            to inject a fake convention.
        action_offset: explicit offset to use.
        device: device to build the task on when resolving ``task``.

    Returns:
        :data:`QC_KEYS`, :data:`STAMP_KEYS`, and ``episode_index``.
    """
    if (action_scale is None) != (action_offset is None):
        raise ValueError("action_scale and action_offset must be given together")
    if action_scale is not None and action_offset is not None:
        action_scale = tensor_or_array_to_float32_vector(action_scale)
        action_offset = tensor_or_array_to_float32_vector(action_offset)
    elif task is not None:
        action_scale, action_offset = _action_scale_offset_from_task(
            task, device=device
        )
    else:
        action_scale, action_offset = action_scale_offset()

    row_order = np.lexsort((np.asarray(frame_index), np.asarray(episode_index)))
    leader_actions_sorted = np.asarray(leader_actions, dtype=np.float32)[row_order]

    actions = np.array(
        [
            lerobot_to_mjlab_raw_action(
                action,
                gravity_compensate=gravity_compensate,
                scale=action_scale,
                offset=action_offset,
            )
            for action in leader_actions_sorted
        ],
        dtype=np.float32,
    )
    # Clamp to the action box: the stored action must match what the sim applied.
    np.clip(actions, -1.0, 1.0, out=actions)

    transitions = assemble_transitions(
        episode_index=np.asarray(episode_index)[row_order],
        observations=np.asarray(observations, dtype=np.float32)[row_order],
        actions=actions,
        rewards=np.asarray(rewards, dtype=np.float32).reshape(-1)[row_order],
    )

    dataset: dict[str, NDArray] = {
        **transitions,
        "action_scale": action_scale,
        "action_offset": action_offset,
        "start_pose_noise": np.array(float(start_pose_noise), dtype=np.float32),
    }
    # Task stamp checked by dataset.py. Missing on synthetic builds (warns, no errors).
    if task is not None:
        dataset["task"] = np.array(task, dtype=np.str_)

    return dataset


def _load_columns(
    dataset_root: str | Path, episode_offset: int = 0
) -> tuple[dict[str, NDArray], str | None]:
    """Read one LeRobot dataset root into plain arrays.

    Returns the columns and the task id the root was recorded with (``None``
    when it registered none). Both come off the one open below, so a root is
    read from disk once.
    """
    from lerobot.datasets.lerobot_dataset import (  # pyright: ignore[reportMissingImports]
        LeRobotDataset,
    )

    # Check before opening: lerobot falls back to the HuggingFace Hub with a 401
    # when local metadata is missing, hiding the real cause.
    # A recording is unfinished until Recorder.close() calls finalize().
    dataset_root = Path(dataset_root)
    if not (dataset_root / "meta").is_dir():
        raise FileNotFoundError(
            f"{dataset_root} is not a LeRobot dataset root: no meta/ directory. "
            "A session is only readable once the recorder has closed it."
        )

    dataset = LeRobotDataset("live_record", root=str(dataset_root))
    frames = dataset.hf_dataset
    columns = {
        "episode_index": np.array(frames["episode_index"]) + episode_offset,
        "frame_index": np.array(frames["frame_index"]),
        "observations": np.array(frames["observation.state"], dtype=np.float32),
        "leader_actions": np.array(frames["action"], dtype=np.float32),
        "rewards": np.array(frames["next.reward"], dtype=np.float32).reshape(-1),
    }
    task = _task_from_meta(dataset, dataset_root)

    return columns, task


def _load_many(
    dataset_roots: Sequence[str | Path],
) -> tuple[dict[str, NDArray], str | None]:
    """Read several LeRobot dataset roots into one set of columns.

    Returns the concatenated columns and the task id all the roots carry
    (``None`` when none of them carries one). Each root is numbered from
    episode zero by the recorder, so every root after the first is pushed past
    the ones before it. Without that, episodes from two sessions merge into
    one and the transitions across the join are built from unrelated frames.

    Raises:
        ValueError: if the roots disagree on the task id.
    """
    per_root_columns: list[dict[str, NDArray]] = []
    recorded_tasks: list[str] = []
    episode_offset = 0
    for dataset_root in dataset_roots:
        root_columns, task = _load_columns(dataset_root, episode_offset=episode_offset)
        if len(root_columns["episode_index"]) == 0:
            raise ValueError(f"{dataset_root} holds no frames")
        episode_offset = int(root_columns["episode_index"].max()) + 1

        per_root_columns.append(root_columns)

        if (
            task is not None
        ):  # A root that registered no task adds nothing to the agreement check.
            recorded_tasks.append(task)

    if not per_root_columns:
        raise ValueError("no recording roots given")

    if len(set(recorded_tasks)) > 1:
        raise ValueError(
            f"mixed tasks across dataset roots: {recorded_tasks} from {dataset_roots}"
        )

    columns = {
        key: np.concatenate([root_columns[key] for root_columns in per_root_columns])
        for key in per_root_columns[0]
    }
    return columns, (recorded_tasks[0] if recorded_tasks else None)


def _read_start_pose_noise(dataset_roots: Sequence[str | Path]) -> float:
    """Half-width [rad] the roots were recorded under.

    Every root's ``g3a_extra_info.json`` must agree; merging a ``0.0`` session
    with a ``0.05`` one would train on two start distributions while evaluating
    on one (``ADR-008``). Synthetic test datasets have no sidecar and default
    to ``0.0`` so unit tests need no extra fixture.

    Raises:
        ValueError: on a missing key or on disagreement across roots.
    """
    start_pose_noises: list[float] = []
    for dataset_root in dataset_roots:
        extra_info = load_extra_info(dataset_root)
        if not extra_info:
            start_pose_noises.append(0.0)
            continue
        if "start_pose_noise" not in extra_info:
            raise ValueError(
                f"{dataset_root} has a g3a_extra_info.json but no start_pose_noise; "
                "was it written before this stamp existed?"
            )
        start_pose_noises.append(float(extra_info["start_pose_noise"]))

    if not start_pose_noises:
        return 0.0

    if len(set(start_pose_noises)) > 1:
        raise ValueError(
            f"mixed start_pose_noise across dataset roots: {start_pose_noises}"
        )

    return start_pose_noises[0]


def convert(
    success_dirs: Sequence[str | Path],
    output_path: str | Path,
    failure_dirs: Sequence[str | Path] = (),
    task: str | None = None,
    device: str = "cpu",
) -> dict[str, NDArray]:
    """Convert one or more recording sessions to a ``.npz``.

    Args:
        success_dirs: the sessions' success roots.
        output_path: where to write the ``.npz``.
        failure_dirs: the sibling failure roots.
        task: task id the recordings were made with. When given, the action
            stamp is read off that task's built action term (robot-agnostic).
            When omitted, the task is inferred from the task id the datasets
            registered; if that fails the legacy DK1-only stamp is used.
        device: device to build the task on when resolving ``task``.
    """
    all_roots = [*success_dirs, *failure_dirs]
    start_pose_noise = _read_start_pose_noise(all_roots)
    columns, inferred_task = _load_many(all_roots)

    if task is not None and inferred_task is not None and inferred_task != task:
        raise ValueError(f"task {task!r} but datasets carry task {inferred_task!r}")
    if task is None:
        task = inferred_task

    data = build_dataset(
        **columns,  # pyright: ignore[reportArgumentType]
        start_pose_noise=start_pose_noise,
        task=task,
        device=device,
    )

    # allow_pickle=False matches np.load's default: an object array written here
    # would be unloadable by dataset.py, so fail at write time instead.
    np.savez(output_path, allow_pickle=False, **data)

    _report_dataset_stats(data, output_path)

    return data


def _report_dataset_stats(data: dict[str, NDArray], output_path: str | Path) -> None:
    rewards = data["rewards"]
    episodes = np.unique(data["episode_index"])
    success_count = int((data["masks"] == 0.0).sum())

    print(f"[INFO] wrote {output_path}")
    print(f"       transitions: {len(rewards)} over {len(episodes)} episodes")
    print(f"       successes:   {success_count} of {len(episodes)}")
    print(f"       observations {data['observations'].shape}")
    print(
        f"       actions      {data['actions'].shape} "
        f"min={data['actions'].min():.3f} max={data['actions'].max():.3f}"
    )
    print(f"       rewards      min={rewards.min():.3f} max={rewards.max():.3f}")
    print(f"       training keys: {sorted(QC_KEYS)}")


def main() -> None:
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0] if doc else "")
    parser.add_argument(
        "--success-dir",
        required=True,
        nargs="+",
        help="LeRobot dataset roots holding the successful episodes.",
    )
    parser.add_argument(
        "--failure-dir",
        nargs="*",
        default=(),
        help="Sibling dataset roots holding the failed episodes. Included when given.",
    )
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument(
        "--task",
        default=None,
        help="Task id to read the action stamp from (e.g. G3A-Stack-Low-Dk1_uni). "
        "When omitted, inferred from the task id the datasets registered; "
        "falls back to DK1-only.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to build the task on when resolving --task (default: cpu).",
    )
    args = parser.parse_args()

    convert(
        args.success_dir,
        args.output,
        failure_dirs=args.failure_dir,
        task=args.task,
        device=args.device,
    )


if __name__ == "__main__":
    main()
