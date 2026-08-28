"""Reading a converted dataset back, and checking it against an environment.

Separate from :mod:`g3a_bench.qc.convert` for one reason: **this module must
not import lerobot**. Writing a dataset happens in this repo's venv, which has
lerobot; reading one happens in the ``qc`` venv, which does not and must not
(installing lerobot's dependency tree there is how a cu130 torch gets in, and
JAX then aborts at import). A training run therefore reaches
:func:`load_dataset` without paying for the recording stack.

The check itself is the point. A dataset recorded under a different MDP is a
silent failure: nothing raises, the loss falls, and every evaluation episode
scores zero. Every field in :data:`STAMP_KEYS` is one way that has already
happened or could happen — see the per-field notes there.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from g3a_bench.qc.transitions import QC_KEYS as QC_KEYS

CONVENTION_NAME = "ee-delta-20hz"
"""The action convention this repo currently records and trains under.

Bump it whenever the meaning of an action changes in a way the other stamped
numbers would not catch. It is checked first, so a dataset from an older
convention says *that* rather than reporting whichever number happens to
differ.
"""

STAMP_KEYS = frozenset(
    {
        "convention",
        "delta_pos_scale",
        "delta_ori_scale",
        "gripper_scale",
        "gripper_offset",
        "control_step_s",
        "task_id",
    }
)
"""Keys recording the MDP the dataset was built under.

* ``convention`` — the name above. Catches every change at once.
* ``delta_pos_scale`` / ``delta_ori_scale`` — [m] and [rad] one action unit
  asks for. A recording made under a different box describes different motion.
* ``gripper_scale`` / ``gripper_offset`` — the normalized stroke mapping. This
  is the old ADR-011 stamp, narrowed to the one term that still has a scale.
* ``control_step_s`` — [s] per action. Never stamped before, so changing the
  rate used to be silent, which is exactly the failure ADR-011 exists for.
* ``task_id`` — closes an already-realised hole: every ``place`` variant shares
  its observation and action widths, so a ``RandS-RandT`` dataset trained
  silently against a ``FixedS-FixedT`` env for all fourteen runs of the alpha
  ladder.

Metadata, like ``episode_index`` — strip everything outside :data:`QC_KEYS`
before ``Dataset.create``, or qc's replay buffer will demand these keys from
every online transition too.
"""

_NUMERIC_FIELDS = (
    "delta_pos_scale",
    "delta_ori_scale",
    "gripper_scale",
    "gripper_offset",
    "control_step_s",
)


class _Env(Protocol):
    """What :func:`check_dataset_matches_env` needs of an environment."""

    observation_space: Any
    action_space: Any

    def action_convention(self) -> Mapping[str, Any]: ...


def _as_scalar(value: Any) -> Any:
    """One stamp value, out of whatever container it arrived in.

    ``np.load`` hands back 0-d arrays, an action term hands back torch tensors
    on the simulation device, and ``np.asarray`` refuses a CUDA tensor outright
    -- so the check that exists to catch a stale dataset would otherwise raise
    ``TypeError`` on every GPU run, which is every real run.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.reshape(()).item() if array.size == 1 else array


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
    for field in ("convention", *_NUMERIC_FIELDS, "task_id"):
        if field not in data:
            raise ValueError(
                f"dataset carries no {field!r}; it predates the current action "
                f"convention {convention['convention']!r} and has to be "
                "re-recorded"
            )
        stamped = _as_scalar(data[field])
        expected = _as_scalar(convention[field])
        same = (
            bool(np.isclose(stamped, expected, atol=1e-6))
            if field in _NUMERIC_FIELDS
            else stamped == expected
        )
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


def training_columns(data: dict[str, NDArray]) -> dict[str, NDArray]:
    """Only the keys ``qc``'s ``Dataset.create`` may be given.

    The stamp and ``episode_index`` are metadata. Left in, qc's replay buffer
    would require them from every online transition dict as well, and those
    carry the six training keys and nothing else.
    """
    return {key: value for key, value in data.items() if key in QC_KEYS}
