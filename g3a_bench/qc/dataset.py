"""Load a converted dataset and check it against an environment.

Separate from convert so this module never imports lerobot. The qc venv
has no lerobot; installing it pulls cu130 torch and JAX aborts at import.
A convention mismatch is silent: loss falls and every eval episode scores zero.
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
"""Stamp keys recording build conventions (ADR-011).

action_scale/offset record the action convention, start_pose_noise the
recording noise (ADR-008), task the task id. Strip non-QC_KEYS before
Dataset.create to avoid the replay buffer demanding these keys.
"""


class _ActionTerm(Protocol):
    scale: Any
    offset: Any


class _Env(Protocol):
    """What :func:`check_dataset_matches_env` needs of an environment."""

    observation_space: Any
    action_space: Any

    def action_term(self) -> _ActionTerm: ...


def tensor_or_array_to_float32_vector(value: Any) -> NDArray[np.float32]:
    """Float32 1-D copy of a stamp value or action-term buffer."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _tensor_or_array_to_float(value: Any) -> float:
    """One float from a 0-d or 1-d stamp entry."""
    vector = tensor_or_array_to_float32_vector(value)
    if vector.size == 0:
        raise ValueError(f"expected a scalar stamp value, got {value!r}")
    return float(vector[0])


def _array_to_task_string(value: Any) -> str:
    """Decode a task stamp to a plain string (handles bytes forms)."""
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
    """Task id the env was built for, or None if not exposed."""
    for attr_name in ("task_id", "task", "_task_id"):
        task_attr = getattr(env, attr_name, None)
        if task_attr is not None:
            task_value = task_attr() if callable(task_attr) else task_attr
            if isinstance(task_value, (str, bytes)):
                return (
                    task_value.decode() if isinstance(task_value, bytes) else task_value
                )
            if task_value is not None:
                return str(task_value)
    return None


def _get_noise_from_cfg(cfg: Any) -> float:
    """Half-width [rad] read off the mjlab cfg reset event."""
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

    Probe start_pose_noise, _start_pose_noise, then mjlab_env.cfg.
    """
    for attr_name in ("start_pose_noise", "_start_pose_noise"):
        noise_attr = getattr(env, attr_name, None)
        if noise_attr is not None:
            noise_value = noise_attr() if callable(noise_attr) else noise_attr
            try:
                return _tensor_or_array_to_float(noise_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"env.{attr_name} is not a half-width [rad]: {exc}"
                ) from exc

    mjlab_env = getattr(env, "mjlab_env", None)
    cfg = getattr(mjlab_env, "cfg", None) if mjlab_env is not None else None
    if cfg is None:
        raise ValueError(
            "cannot read start_pose_noise from env: no start_pose_noise seam, "
            "no _start_pose_noise attribute, and no mjlab_env.cfg; "
            "is it a G3aEnv?"
        )

    start_pose_noise = _get_noise_from_cfg(cfg)
    return start_pose_noise


def check_dataset_matches_env(data: dict[str, NDArray], env: _Env) -> None:
    """Raise unless ``data`` was recorded under the env conventions.

    Missing task stamp warns and passes; present stamp must match exactly.

    Args:
        data: a dataset from ``convert.build_dataset``, or one loaded back from
            the ``.npz`` it was saved to.
        env: the environment a training run would use.

    Raises:
        ValueError: on any disagreement, naming the two values.
    """
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
            f"{env_action_width}; the recording used a different robot or a "
            "different action term"
        )

    action_term = env.action_term()
    stamped_scale = tensor_or_array_to_float32_vector(data["action_scale"])
    env_scale = tensor_or_array_to_float32_vector(action_term.scale)
    if not np.allclose(stamped_scale, env_scale, atol=1e-5):
        raise ValueError(
            f"dataset action scale {stamped_scale} but the environment's is "
            f"{env_scale}; the recording predates the current action convention"
        )

    stamped_offset = tensor_or_array_to_float32_vector(data["action_offset"])
    env_offset = tensor_or_array_to_float32_vector(action_term.offset)
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
    stamped_noise = _tensor_or_array_to_float(data["start_pose_noise"])
    env_noise = _env_start_pose_noise(env)
    if not np.isclose(stamped_noise, env_noise, atol=1e-5):
        raise ValueError(
            f"dataset start_pose_noise {stamped_noise} but the environment's is "
            f"{env_noise}; recording and evaluation must begin at the same place "
            f"(ADR-008)"
        )

    # Old datasets without task stamp pass with a warning.
    if "task" not in data:
        warnings.warn(
            "dataset has no task stamp; it predates the stamp "
            "(rebuild it with g3a-convert-demos to enable task checking)",
            UserWarning,
            stacklevel=2,
        )
    else:
        stamped_task = _array_to_task_string(data["task"])
        env_task = _env_task_id(env)
        if env_task is not None and stamped_task != env_task:
            raise ValueError(
                f"dataset task {stamped_task!r} but the environment's is "
                f"{env_task!r}; a dataset from one variant cannot train "
                "another (rebuild with the correct --task)"
            )


def load_dataset(path: str | Path, env: _Env | None = None) -> dict[str, NDArray]:
    """Load a converted ``.npz``, checked against ``env`` when given."""
    with np.load(path) as loaded_dataset:
        data = {key: loaded_dataset[key] for key in loaded_dataset.files}

    if env is not None:
        check_dataset_matches_env(data, env)

    return data


def get_training_columns(data: dict[str, NDArray]) -> dict[str, NDArray]:
    """Only the keys qc Dataset.create accepts (drops stamps)."""
    return {key: value for key, value in data.items() if key in QC_KEYS}
