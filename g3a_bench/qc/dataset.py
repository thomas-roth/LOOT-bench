"""Reading a converted dataset back, and checking it against an environment.

Separate from :mod:`g3a_bench.qc.convert` for one reason: **this module must
not import lerobot**. Writing a dataset happens in this repo's venv, which has
lerobot; reading one happens in the ``qc`` venv, which does not and must not
(installing lerobot's dependency tree there is how a cu130 torch gets in, and
JAX then aborts at import). A training run therefore reaches
:func:`load_dataset` without paying for the recording stack.

The check itself is the point. A dataset recorded under a different action
convention is a silent failure: nothing raises, the loss falls, and every
evaluation episode scores zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from g3a_bench.qc.transitions import QC_KEYS as QC_KEYS

STAMP_KEYS = frozenset({"action_scale", "action_offset"})
"""Keys recording the action convention the dataset was built under.

Metadata, like ``episode_index`` -- strip everything outside :data:`QC_KEYS`
before ``Dataset.create``, or qc's replay buffer will demand these keys from
every online transition too.
"""


class _ActionTerm(Protocol):
    scale: Any
    offset: Any


class _Env(Protocol):
    """What :func:`check_dataset_matches_env` needs of an environment."""

    observation_space: Any
    action_space: Any

    def action_term(self) -> _ActionTerm: ...


def _as_row(value: Any) -> NDArray[np.float32]:
    """A float32 1-D copy of a stamp value or an action-term buffer.

    The term's buffers are torch tensors on the simulation device, and
    ``np.asarray`` refuses a CUDA tensor outright -- so the check that exists
    to catch a stale dataset would otherwise raise ``TypeError`` on every GPU
    run, which is every real run.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def check_dataset_matches_env(data: dict[str, NDArray], env: _Env) -> None:
    """Raise unless ``data`` was recorded under the environment's conventions.

    Every check here guards a failure that is otherwise silent: training runs
    to completion, the loss falls, and every evaluation episode scores zero.

    It does **not** check task identity. Two tasks whose observation and action
    widths agree would pass -- across categories the widths differ, so this
    only leaves variants of one category unguarded. Stamping the recording's
    task name (``LeRobotDatasetMetadata.tasks``) would close it; that is left
    until there is a real recording to read it from.

    Args:
        data: a dataset from ``convert.build_dataset``, or one loaded back from
            the ``.npz`` it was saved to.
        env: the environment a training run would use.

    Raises:
        ValueError: on any disagreement, naming the two values.
    """
    term = env.action_term()
    stamped_scale = _as_row(data["action_scale"])
    stamped_offset = _as_row(data["action_offset"])
    env_scale = _as_row(term.scale)
    env_offset = _as_row(term.offset)

    obs_width = int(np.asarray(data["observations"]).shape[1])
    env_obs_width = int(env.observation_space.shape[0])
    if obs_width != env_obs_width:
        raise ValueError(
            f"dataset observation width {obs_width} but the environment's is "
            f"{env_obs_width}; the task's observation terms changed since the "
            "recording"
        )

    action_width = int(np.asarray(data["actions"]).shape[1])
    env_action_width = int(env.action_space.shape[0])
    if action_width != env_action_width:
        raise ValueError(
            f"dataset action width {action_width} but the environment's is "
            f"{env_action_width}"
        )

    if not np.allclose(stamped_scale, env_scale, atol=1e-5):
        raise ValueError(
            f"dataset action scale {stamped_scale} but the environment's is "
            f"{env_scale}; the recording predates the current action convention"
        )
    if not np.allclose(stamped_offset, env_offset, atol=1e-5):
        raise ValueError(
            f"dataset action offset {stamped_offset} but the environment's is "
            f"{env_offset}; the recording predates the current action convention"
        )


def load_dataset(path: str | Path, env: _Env | None = None) -> dict[str, NDArray]:
    """Load a converted ``.npz``, checked against ``env`` when one is given.

    The way a training run should read a dataset: passing the environment it is
    about to train against turns a convention mismatch into an exception at
    startup instead of a flat evaluation curve hours later. Filter the result
    to :data:`QC_KEYS` before handing it to ``Dataset.create``.
    """
    with np.load(path) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    if env is not None:
        check_dataset_matches_env(data, env)
    return data


def training_columns(data: dict[str, NDArray]) -> dict[str, NDArray]:
    """Only the keys ``qc``'s ``Dataset.create`` may be given.

    The stamp and ``episode_index`` are metadata. Left in, qc's replay buffer
    would require them from every online transition dict as well, and those
    carry the six training keys and nothing else.
    """
    return {key: value for key, value in data.items() if key in QC_KEYS}
