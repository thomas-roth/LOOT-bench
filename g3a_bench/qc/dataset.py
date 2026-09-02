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

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from g3a_bench.qc.transitions import QC_KEYS

STAMP_KEYS = frozenset({"action_scale", "action_offset", "start_pose_noise", "task"})
"""Keys recording the conventions the dataset was built under.

``action_scale``/``action_offset`` record the action convention,
``start_pose_noise`` the start-pose sample the demos were recorded under
(``ADR-008``), and ``task`` the task id the dataset was recorded for.
All are metadata, like ``episode_index`` -- strip everything
outside :data:`QC_KEYS` before ``Dataset.create``, or qc's replay buffer will
demand these keys from every online transition too.
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


def _as_scalar(value: Any) -> float:
    """One float from a 0-d or 1-d stamp entry."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"expected a scalar stamp value, got {value!r}")
    return float(arr[0])


def _as_task_str(value: Any) -> str:
    """Decode a task stamp to a plain string.

    The stamp is written as ``np.array(task, dtype=np.str_)`` but can be
    read back as bytes depending on numpy version and npz handling.
    """
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError(f"expected a task stamp, got {value!r}")
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode()
    # numpy scalar string types (np.str_, np.bytes_)
    if isinstance(item, np.bytes_):
        return item.tobytes().decode()
    return str(item)


def _env_task_id(env: _Env) -> str | None:
    """Task id the env was built for, or None if not exposed.

    ``G3aEnv`` stores it in ``task_id``; fakes in tests may use any of the
    names below. None means the task check is skipped (the env cannot
    answer it), which is right for a fake that models no task.
    """
    for attr in ("task_id", "task", "_task_id"):
        candidate = getattr(env, attr, None)
        if candidate is not None:
            val = candidate() if callable(candidate) else candidate  # type: ignore[arg-type]
            if isinstance(val, (str, bytes)):
                return val.decode() if isinstance(val, bytes) else val
            if val is not None:
                return str(val)
    return None


def _noise_from_cfg(cfg: Any) -> float:
    """Half-width [rad] read off the mjlab cfg's reset event.

    The source for an env that exposes the simulator (``unwrapped_env``)
    without the ``start_pose_noise`` seam. Every level is checked by name,
    so a cfg that changed shape says which key is missing instead of the
    caller hearing only that the env is not a G3aEnv.
    """
    events = getattr(cfg, "events", None)
    if not isinstance(events, Mapping):
        raise ValueError("cfg.events is not a mapping of named event terms")
    term = events.get("reset_robot_joints")
    if term is None:
        raise ValueError(
            f"cfg.events has no reset_robot_joints term; found {sorted(events)}"
        )
    params = getattr(term, "params", None)
    if not isinstance(params, Mapping) or "position_range" not in params:
        raise ValueError("reset_robot_joints carries no position_range param")
    rng = params["position_range"]
    if len(rng) < 2:
        raise ValueError(f"position_range {rng!r} has no upper bound to read")
    return float(rng[1])


def _env_start_pose_noise(env: Any) -> float:
    """Half-width [rad] the env samples its start pose from.

    ``env`` is typed ``Any`` on purpose: every source below is probed with
    ``getattr``, so a partial fake is checkable. That is the point -- the
    check must reach an env that is missing the seam, and name what it
    found or did not.

    Three sources, first match wins:

    1. ``env.start_pose_noise`` -- the seam ``G3aEnv`` publishes, a method
       or an attribute. Every real run takes this path.
    2. ``env._start_pose_noise`` -- the private attribute ``G3aEnv`` stores
       the value in, for a fake that exposes no method.
    3. ``env.unwrapped_env.cfg`` -- the mjlab cfg, for an env that exposes
       the simulator without the seam.

    A source that is present but broken raises naming the source. The final
    error means none of the three was found at all.
    """
    for attr in ("start_pose_noise", "_start_pose_noise"):
        candidate = getattr(env, attr, None)
        if candidate is not None:
            value = candidate() if callable(candidate) else candidate
            try:
                return _as_scalar(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"env.{attr} is not a half-width [rad]: {exc}"
                ) from exc
    unwrapped = getattr(env, "unwrapped_env", None)
    cfg = getattr(unwrapped, "cfg", None) if unwrapped is not None else None
    if cfg is not None:
        return _noise_from_cfg(cfg)
    raise ValueError(
        "cannot read start_pose_noise from env: no start_pose_noise seam, "
        "no _start_pose_noise attribute, and no unwrapped_env.cfg; "
        "is it a G3aEnv?"
    )


def check_dataset_matches_env(data: dict[str, NDArray], env: _Env) -> None:
    """Raise unless ``data`` was recorded under the environment's conventions.

    Every check here guards a failure that is otherwise silent: training runs
    to completion, the loss falls, and every evaluation episode scores zero.
    Task identity is checked when the dataset carries a ``task`` stamp
    (new recordings); old datasets without the stamp warn but still pass
    (minimal breakage, rebuild with ``g3a-convert-demos`` to gain the check).

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

    if "start_pose_noise" not in data:
        raise ValueError(
            "dataset has no start_pose_noise stamp; it predates the stamp "
            "(rebuild it with g3a-convert-demos)"
        )
    stamped_noise = _as_scalar(data["start_pose_noise"])
    env_noise = _env_start_pose_noise(env)
    if not np.isclose(stamped_noise, env_noise, atol=1e-5):
        raise ValueError(
            f"dataset start_pose_noise {stamped_noise} but the environment's is "
            f"{env_noise}; recording and evaluation must begin at the same place "
            f"(ADR-008)"
        )

    # Task identity — minimal breakage: old datasets without the stamp
    # pass with a warning; new datasets must match exactly.
    if "task" not in data:
        warnings.warn(
            "dataset has no task stamp; it predates the stamp "
            "(rebuild it with g3a-convert-demos to enable task checking)",
            UserWarning,
            stacklevel=2,
        )
    else:
        stamped_task = _as_task_str(data["task"])
        env_task = _env_task_id(env)
        if env_task is not None and stamped_task != env_task:
            raise ValueError(
                f"dataset task {stamped_task!r} but the environment's is "
                f"{env_task!r}; a dataset from one variant cannot train "
                "another (rebuild with the correct --task)"
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
