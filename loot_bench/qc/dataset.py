"""Reading a converted dataset back, and checking it against an environment.

Separate from convert so this module never imports lerobot. The qc venv
has no lerobot; installing it pulls cu130 torch and JAX aborts at import.

A convention mismatch is silent: loss falls and every eval episode scores zero.
Every field in :data:`STAMP_KEYS` is one way that can happen.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from loot_bench.qc.transitions import QC_KEYS as QC_KEYS

CONVENTION_NAME = "joint-absolute-20hz"
"""The action convention this repo currently records and trains under.

Bump it whenever the meaning of an action changes in a way the other stamped
numbers would not catch. It is checked first, so a dataset from an older
convention says *that* rather than reporting whichever number happens to
differ.
"""

NUMERIC_STAMP_FIELDS = (
    "control_step_s",
    "start_pose_noise",
)
"""Scalar stamp fields compared with a tolerance rather than by equality."""

VECTOR_STAMP_FIELDS = (
    "action_scale",
    "action_offset",
)
"""Vector stamp fields compared elementwise with a tolerance."""

_STAMP_FIELD_ORDER = (
    "convention",
    *VECTOR_STAMP_FIELDS,
    *NUMERIC_STAMP_FIELDS,
    "task_id",
)
"""Every stamp field, ``convention`` first so a stale dataset names itself
before any number below is compared. The single source both
:func:`check_dataset_matches_env` and :data:`STAMP_KEYS` derive from --
add a field here and it is checked, not just stamped."""

STAMP_KEYS = frozenset(_STAMP_FIELD_ORDER)
"""Keys recording the MDP the dataset was built under.

* ``convention`` — the name above. Catches every change at once.
* ``action_scale`` / ``action_offset`` — the normalized joint mapping. This
  is the ADR-011 stamp: a recording made under a different convention
  describes different motion.
* ``control_step_s`` — [s] per action. Never stamped before ADR-016, so
  changing the rate used to be silent, which is exactly the failure ADR-011
  exists for.
* ``start_pose_noise`` — [rad] half-width of the start-pose sample (ADR-008).
  Recording and evaluation have to begin at the same place.
* ``task_id`` — closes an already-realised hole: every ``place`` variant shares
  its observation and action widths, so a ``RandS-RandT`` dataset trained
  silently against a ``FixedS-FixedT`` env for all fourteen runs of the alpha
  ladder.

Metadata, like ``episode_index`` — strip everything outside :data:`QC_KEYS`
before ``Dataset.create``, or qc's replay buffer will demand these keys from
every online transition too.
"""


class _Env(Protocol):
    """What :func:`check_dataset_matches_env` needs of an environment."""

    observation_space: Any
    action_space: Any

    def action_convention(self) -> Mapping[str, Any]: ...


def _as_scalar(value: Any) -> Any:
    """One stamp value, out of whatever container it arrived in.

    np.load gives 0-d arrays and an action term gives CUDA tensors, which
    np.asarray refuses outright -- so the check would raise on every real run.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.reshape(()).item() if array.size == 1 else array


def _as_vector(value: Any) -> NDArray[np.float32]:
    """One vector stamp value as float32, out of whatever container."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def check_dataset_matches_env(data: dict[str, NDArray], env: _Env) -> None:
    """Raise unless ``data`` was recorded under the environment's conventions.

    Every check here guards a failure that is otherwise silent: training runs
    to completion, the loss falls, and every evaluation episode scores zero.

    Args:
        data: a dataset from ``convert.build_dataset``, or one loaded back from
            the ``.npz`` it was saved to.
        env: the environment a training run would use.

    Raises:
        ValueError: on any disagreement, naming the field that differs.
    """
    convention = env.action_convention()

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

    # Name first: a dataset from an older convention should say so, rather
    # than reporting whichever of the numbers below happens to differ.
    for field in _STAMP_FIELD_ORDER:
        if field not in data:
            raise ValueError(
                f"dataset carries no {field!r}; it predates the current action "
                f"convention {convention['convention']!r} and has to be "
                "re-recorded"
            )
        if field in VECTOR_STAMP_FIELDS:
            stamped = _as_vector(data[field])
            expected = _as_vector(convention[field])
            same = stamped.shape == expected.shape and bool(
                np.allclose(stamped, expected, atol=1e-5)
            )
        elif field in NUMERIC_STAMP_FIELDS:
            stamped = _as_scalar(data[field])
            expected = _as_scalar(convention[field])
            same = bool(np.isclose(stamped, expected, atol=1e-6))
        else:
            stamped = _as_scalar(data[field])
            expected = _as_scalar(convention[field])
            same = stamped == expected
        if not same:
            raise ValueError(
                f"dataset {field} {stamped!r} but the environment's is "
                f"{expected!r}; the recording describes a different MDP"
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


def get_training_columns(data: dict[str, NDArray]) -> dict[str, NDArray]:
    """Only the keys ``qc``'s ``Dataset.create`` may be given.

    The stamp and ``episode_index`` are metadata. Left in, qc's replay buffer
    would require them from every online transition dict as well, and those
    carry the six training keys and nothing else.
    """
    return {key: value for key, value in data.items() if key in QC_KEYS}
