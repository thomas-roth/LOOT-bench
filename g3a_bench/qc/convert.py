"""A teleop recording -> the arrays ``qc`` trains on.

``live_record.py`` writes a LeRobot dataset: one row per timestep, holding the
observation the operator acted from, the action the simulator was stepped
with, and the reward that step returned. ``qc`` wants transition arrays. :func:`convert` bridges the two.

The episode-structure rules -- what is absorbing, which rows survive, where an
episode ends -- all live in :mod:`g3a_bench.qc.transitions` and are tested
without a simulator. This module adds the three things around them:

* **The action.** The recording stores the raw action ``env.step()`` was given,
  so this module copies it through rather than re-deriving it. It has to: under
  delta control an action means "move this far from where you are", and the
  pose it was measured against is not in the file. The old converter re-ran the
  leader-pose conversion here, which only worked because the action was
  absolute (ADR-016 licenses the change: "both terms exist at record time, so
  the stored action is still exactly what the simulator stepped with").
* **The order.** Episode structure is read off row order, so rows are sorted by
  ``(episode, frame)`` before anything looks at them.
* **The stamp.** The convention record in force at record time
  (:data:`~g3a_bench.qc.dataset.STAMP_KEYS`), written into the file.
  ``dataset.check_dataset_matches_env`` compares it against the environment a
  training run actually builds. A dataset recorded under a different MDP is the
  failure mode this exists for: nothing raises, the loss falls, and evaluation
  scores zero.

Reading a dataset back lives in :mod:`g3a_bench.qc.dataset`, which imports no
lerobot: the ``qc`` venv has to load a dataset and must not carry the recording
stack.

Observations are copied through untouched. They are the task's ``actor`` group,
which carries no noise on either side -- no task declares any, and the adapter
sets the flag as well -- so the offline and online observations come from one
distribution.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Imported here because the module needs lerobot, and for no stronger reason.
#
# There is a standing warning that lerobot must load before mjlab or the native
# import order segfaults. **No module inside this package can honour it**:
# importing any g3a_bench submodule runs g3a_bench/__init__.py first, which
# registers the tasks and therefore imports mjlab before a line of this file
# runs. Only the first import of the whole process can order the two, which is
# what tests/qc/conftest.py does for the test run.
#
# Measured on 2026-08-14: importing mjlab first, then lerobot, then compiling
# the DK1 and reading a dataset raised nothing. That is not proof the warning
# is stale -- it may need the camera or codec paths this converter never
# touches -- so treat a segfault here as a real possibility and move the
# ordering to the entry point if one appears.
import lerobot.datasets.lerobot_dataset  # noqa: F401
import numpy as np
from numpy.typing import NDArray

from g3a_bench.qc.dataset import CONVENTION_NAME
from g3a_bench.qc.dataset import STAMP_KEYS as STAMP_KEYS
from g3a_bench.qc.dataset import check_dataset_matches_env as check_dataset_matches_env
from g3a_bench.qc.dataset import load_dataset as load_dataset
from g3a_bench.qc.transitions import QC_KEYS, assemble_transitions
from g3a_bench.scripts.live_bridge.mapper import gripper_scale_offset
from g3a_bench.tasks.common.actions import DELTA_ORI_SCALE, DELTA_POS_SCALE
from g3a_bench.tasks.common.env_cfg import CONTROL_STEP_S


def action_convention(task_id: str) -> dict[str, object]:
    """The convention record a recording made right now is stamped with.

    Read from the same constants the environment builds its action terms from,
    so this is the recording's claim about the MDP rather than a copy of one.
    ``check_dataset_matches_env`` compares it against what an env actually
    reports.
    """
    gripper_scale, gripper_offset = gripper_scale_offset()
    return {
        "convention": CONVENTION_NAME,
        "delta_pos_scale": DELTA_POS_SCALE,
        "delta_ori_scale": DELTA_ORI_SCALE,
        "gripper_scale": gripper_scale,
        "gripper_offset": gripper_offset,
        "control_step_s": CONTROL_STEP_S,
        "task_id": task_id,
    }


def build_dataset(
    *,
    episode_index: NDArray[np.integer],
    frame_index: NDArray[np.integer],
    observations: NDArray[np.floating],
    actions: NDArray[np.floating],
    rewards: NDArray[np.floating],
    task_id: str,
) -> dict[str, Any]:
    """Assemble one recording's columns into the training arrays.

    Kept apart from the file handling so the conversion can be tested without
    writing a LeRobot dataset to disk.

    Args:
        episode_index: which episode each row belongs to.
        frame_index: position of each row inside its episode.
        observations: the task's ``actor`` observation, one row per step.
        actions: the raw action ``env.step()`` was given, as recorded. Copied
            through, not re-derived -- see the module docstring.
        rewards: the reward that step returned.
        task_id: the registered task the episodes were flown on. Stamped, so a
            ``RandS-RandT`` recording cannot silently train against a
            ``FixedS-FixedT`` env whose widths happen to agree.

    Returns:
        :data:`QC_KEYS`, :data:`STAMP_KEYS`, and ``episode_index``.
    """
    order = np.lexsort((np.asarray(frame_index), np.asarray(episode_index)))
    sorted_actions = np.asarray(actions, dtype=np.float32)[order]

    # A recorded action came from a mapping that clamps and was applied by a
    # term that clamps, so an out-of-box row means the two disagreed -- which
    # is the one thing the seam exists to prevent. Loud, not silently clipped.
    outside = np.abs(sorted_actions) > 1.0 + 1e-5
    if outside.any():
        worst = float(np.abs(sorted_actions).max())
        raise ValueError(
            f"{int(outside.any(axis=1).sum())} recorded actions lie outside the "
            f"action box (worst |a| = {worst:.4f}); the recorder's mapping and "
            "the action term disagree, and no bounded policy can reproduce them"
        )

    return {
        **assemble_transitions(
            episode_index=np.asarray(episode_index)[order],
            observations=np.asarray(observations, dtype=np.float32)[order],
            actions=sorted_actions,
            rewards=np.asarray(rewards, dtype=np.float32).reshape(-1)[order],
        ),
        **action_convention(task_id),
    }


def load_columns(
    dataset_root: str | Path, episode_offset: int = 0
) -> dict[str, NDArray]:
    """Read one LeRobot dataset root into plain arrays.

    Args:
        dataset_root: a root written by ``live_record.py``.
        episode_offset: added to every episode index. The success and failure
            roots of one session are each numbered from zero, so the second one
            has to be pushed past the first or their episodes merge into one.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Checked before opening, because of how lerobot fails otherwise: when the
    # local metadata is missing or unfinished it falls back to the Hugging Face
    # Hub and reports a 401 against the repo id below, burying the real cause.
    # A recording is unfinished until Recorder.close() calls finalize().
    root = Path(dataset_root)
    if not (root / "meta").is_dir():
        raise FileNotFoundError(
            f"{root} is not a LeRobot dataset root: no meta/ directory. A "
            "session is only readable once the recorder has closed it."
        )

    dataset = LeRobotDataset("live_record", root=str(root))
    frames = dataset.hf_dataset
    # lerobot stores a per-frame ``task_index`` into ``meta.tasks``, a frame
    # indexed *by the task string*, so the id comes back through a reverse map
    # rather than through positional lookup.
    task_by_index = {
        int(index): str(task)
        for task, index in dataset.meta.tasks["task_index"].items()
    }
    return {
        "episode_index": np.array(frames["episode_index"]) + episode_offset,
        "frame_index": np.array(frames["frame_index"]),
        "observations": np.array(frames["observation.state"], dtype=np.float32),
        "actions": np.array(frames["action"], dtype=np.float32),
        "rewards": np.array(frames["next.reward"], dtype=np.float32).reshape(-1),
        # The task each frame was flown on. The stamp needs one value, so
        # load_many insists every row of every session agrees.
        "task_ids": np.array(
            [task_by_index[int(index)] for index in frames["task_index"]], dtype=object
        ),
    }


def load_many(roots: Sequence[str | Path]) -> dict[str, NDArray]:
    """Read several LeRobot roots into one set of columns.

    Each root is numbered from episode zero by the recorder, so every root
    after the first is pushed past the ones before it. Without that, episodes
    from two sessions merge into one and the transitions across the join are
    built from unrelated frames.
    """
    parts: list[dict[str, NDArray]] = []
    offset = 0
    for root in roots:
        part = load_columns(root, episode_offset=offset)
        if len(part["episode_index"]) == 0:
            raise ValueError(f"{root} holds no frames")
        offset = int(part["episode_index"].max()) + 1
        parts.append(part)
    if not parts:
        raise ValueError("no recording roots given")
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def convert(
    success_dirs: Sequence[str | Path],
    output_path: str | Path,
    failure_dirs: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Convert one or more recording sessions to a ``.npz``.

    Several sessions go in one call rather than through a separate merge step.
    A merge tool would have to know the file layout as well as this module
    does, and the two would then have to be kept in agreement -- for a job that
    is one concatenation.

    Args:
        success_dirs: the sessions' success roots.
        output_path: where to write the ``.npz``.
        failure_dirs: the sibling failure roots. Include them to train on
            failed attempts too -- they carry the only evidence of what not to
            do, and :mod:`~g3a_bench.qc.transitions` treats them correctly: a
            failed episode ends without reaching an absorbing state, so its
            last row is dropped rather than bootstrapped through.
    """
    columns = load_many([*success_dirs, *failure_dirs])
    task_ids = set(columns.pop("task_ids").tolist())
    # One dataset, one task. Every `place` variant shares its observation and
    # action widths, so a mixed file would load against any of them without a
    # word -- which is how fourteen runs of the alpha ladder trained a
    # RandS-RandT recording against a FixedS-FixedT env.
    if len(task_ids) != 1:
        raise ValueError(
            f"recordings span {len(task_ids)} tasks ({sorted(task_ids)}); a "
            "dataset stamps one task id, so convert them separately"
        )
    data = build_dataset(task_id=task_ids.pop(), **columns)  # type: ignore[arg-type]

    # The splat hides savez's own keyword arguments from the type checker,
    # which then reads an array as a candidate for allow_pickle.
    np.savez(output_path, **data)  # type: ignore[arg-type]
    _report(data, output_path)
    return data


def _report(data: dict[str, Any], output_path: str | Path) -> None:
    rewards = data["rewards"]
    episodes = np.unique(data["episode_index"])
    successes = int((data["masks"] == 0.0).sum())
    print(f"[INFO] wrote {output_path}")
    print(f"       task:        {data['task_id']}")
    print(
        f"       convention:  {data['convention']} ({data['control_step_s']:g} s/step)"
    )
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    args = parser.parse_args()
    convert(args.success_dir, args.output, failure_dirs=args.failure_dir)


if __name__ == "__main__":
    main()
