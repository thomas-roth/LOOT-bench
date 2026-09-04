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
import lerobot.datasets.lerobot_dataset  # noqa: F401
import numpy as np
from numpy.typing import NDArray

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
        scale = term.scale  # type: ignore[attr-defined]
        offset = term.offset  # type: ignore[attr-defined]
        if hasattr(scale, "detach"):
            scale = scale.detach().cpu().numpy()  # type: ignore[union-attr]
        if hasattr(offset, "detach"):
            offset = offset.detach().cpu().numpy()  # type: ignore[union-attr]
        scale = np.asarray(scale, dtype=np.float32).reshape(-1)
        offset = np.asarray(offset, dtype=np.float32).reshape(-1)
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
        scale = np.asarray(action_scale, dtype=np.float32).reshape(-1)
        offset = np.asarray(action_offset, dtype=np.float32).reshape(-1)
    elif task is not None:
        scale, offset = _action_scale_offset_from_task(task, device=device)
    else:
        scale, offset = action_scale_offset()

    order = np.lexsort((np.asarray(frame_index), np.asarray(episode_index)))
    leader_sorted = np.asarray(leader_actions, dtype=np.float32)[order]

    actions = np.array(
        [
            lerobot_to_mjlab_raw_action(
                pose,
                gravity_compensate=gravity_compensate,
                scale=scale,
                offset=offset,
            )
            for pose in leader_sorted
        ],
        dtype=np.float32,
    )
    # Clamp to the action box: the stored action must match what the sim applied.
    np.clip(actions, -1.0, 1.0, out=actions)

    out: dict[str, NDArray] = {
        **assemble_transitions(
            episode_index=np.asarray(episode_index)[order],
            observations=np.asarray(observations, dtype=np.float32)[order],
            actions=actions,
            rewards=np.asarray(rewards, dtype=np.float32).reshape(-1)[order],
        ),
        "action_scale": scale,
        "action_offset": offset,
        "start_pose_noise": np.array(float(start_pose_noise), dtype=np.float32),
    }
    # Task stamp checked by dataset.py. Missing on synthetic builds (warns, no errors).
    if task is not None:
        out["task"] = np.array(task, dtype=np.str_)
    return out


def load_columns(
    dataset_root: str | Path, episode_offset: int = 0
) -> tuple[dict[str, NDArray], str | None]:
    """Read one LeRobot dataset root into plain arrays.

    Returns the columns and the task id the root was recorded with (``None``
    when it registered none). Both come off the one open below, so a root is
    read from disk once.

    Args:
        dataset_root: a root written by ``live_record.py``.
        episode_offset: added to every episode index. The success and failure
            roots of one session are each numbered from zero, so the second one
            has to be pushed past the first or their episodes merge into one.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Check before opening: lerobot falls back to the HuggingFace Hub with a 401
    # when local metadata is missing, hiding the real cause.
    # A recording is unfinished until Recorder.close() calls finalize().
    root = Path(dataset_root)
    if not (root / "meta").is_dir():
        raise FileNotFoundError(
            f"{root} is not a LeRobot dataset root: no meta/ directory. A "
            "session is only readable once the recorder has closed it."
        )

    dataset = LeRobotDataset("live_record", root=str(root))
    frames = dataset.hf_dataset
    columns = {
        "episode_index": np.array(frames["episode_index"]) + episode_offset,
        "frame_index": np.array(frames["frame_index"]),
        "observations": np.array(frames["observation.state"], dtype=np.float32),
        "leader_actions": np.array(frames["action"], dtype=np.float32),
        "rewards": np.array(frames["next.reward"], dtype=np.float32).reshape(-1),
    }
    return columns, _task_from_meta(dataset, root)


def load_many(roots: Sequence[str | Path]) -> tuple[dict[str, NDArray], str | None]:
    """Read several LeRobot roots into one set of columns.

    Returns the concatenated columns and the task id all the roots carry
    (``None`` when none of them carries one). Each root is numbered from
    episode zero by the recorder, so every root after the first is pushed past
    the ones before it. Without that, episodes from two sessions merge into
    one and the transitions across the join are built from unrelated frames.

    Raises:
        ValueError: if the roots disagree on the task id.
    """
    parts: list[dict[str, NDArray]] = []
    tasks: list[str | None] = []
    offset = 0
    for root in roots:
        part, task = load_columns(root, episode_offset=offset)
        if len(part["episode_index"]) == 0:
            raise ValueError(f"{root} holds no frames")
        offset = int(part["episode_index"].max()) + 1
        parts.append(part)
        tasks.append(task)
    if not parts:
        raise ValueError("no recording roots given")
    present = [t for t in tasks if t is not None]
    if present and any(t != present[0] for t in present):
        raise ValueError(f"mixed tasks across roots: {present} from {roots}")
    columns = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    return columns, (present[0] if present else None)


def _read_start_pose_noise(roots: Sequence[str | Path]) -> float:
    """Half-width [rad] the roots were recorded under.

    Every root's ``g3a_extra_info.json`` must agree; merging a ``0.0`` session
    with a ``0.05`` one would train on two start distributions while evaluating
    on one (``ADR-008``). Synthetic test datasets have no sidecar and default
    to ``0.0`` so unit tests need no extra fixture.

    Raises:
        ValueError: on a missing key or on disagreement across roots.
    """
    noises: list[float] = []
    for root in roots:
        info = load_extra_info(root)
        if not info:
            noises.append(0.0)
            continue
        if "start_pose_noise" not in info:
            raise ValueError(
                f"{root} has a g3a_extra_info.json but no start_pose_noise; "
                "was it written before this stamp existed?"
            )
        noises.append(float(info["start_pose_noise"]))
    if not noises:
        return 0.0
    first = noises[0]
    if any(n != first for n in noises):
        raise ValueError(f"mixed start_pose_noise across roots: {noises}")
    return first


def convert(
    success_dirs: Sequence[str | Path],
    output_path: str | Path,
    failure_dirs: Sequence[str | Path] = (),
    task: str | None = None,
    device: str = "cpu",
) -> dict[str, NDArray]:
    """Convert one or more recording sessions to a ``.npz``.

    Several sessions go in one call rather than through a separate merge step.
    A merge tool would have to know the file layout as well as this module
    does, and the two would then have to be kept in agreement, for a job that
    is one concatenation.

    Args:
        success_dirs: the sessions' success roots.
        output_path: where to write the ``.npz``.
        failure_dirs: the sibling failure roots. Include them to train on
            failed attempts too, they carry the only evidence of what not to
            do, and :mod:`~g3a_bench.qc.transitions` treats them correctly: a
            failed episode ends without reaching an absorbing state, so its
            last row is dropped rather than bootstrapped through.
        task: task id the recordings were made with. When given, the action
            stamp is read off that task's built action term (robot-agnostic).
            When omitted, the task is inferred from the task id the datasets
            registered; if that fails the legacy DK1-only stamp is used.
        device: device to build the task on when resolving ``task``.
    """
    all_roots = [*success_dirs, *failure_dirs]
    start_pose_noise = _read_start_pose_noise(all_roots)
    # The task id comes off the same open as the columns, so a root is read
    # from disk once.
    columns, inferred = load_many(all_roots)
    if task is not None and inferred is not None and inferred != task:
        raise ValueError(f"task {task!r} but datasets carry task {inferred!r}")
    if task is None:
        task = inferred
    # build_dataset resolves and stamps scale/offset/task; dataset.py checks them.
    data = build_dataset(
        **columns,  # type: ignore[arg-type]
        start_pose_noise=start_pose_noise,
        task=task,
        device=device,
    )

    # The splat hides savez's own keyword arguments from the type checker,
    # which then reads an array as a candidate for allow_pickle.
    np.savez(output_path, **data)  # type: ignore[arg-type]
    _report(data, output_path)
    return data


def _report(data: dict[str, NDArray], output_path: str | Path) -> None:
    rewards = data["rewards"]
    episodes = np.unique(data["episode_index"])
    successes = int((data["masks"] == 0.0).sum())
    print(f"[INFO] wrote {output_path}")
    print(f"       transitions: {len(rewards)} over {len(episodes)} episodes")
    print(f"       successes:   {successes} of {len(episodes)}")
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
        help="Sibling roots holding the failed episodes. Included when given.",
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
