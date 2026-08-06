"""OGBench-style QC-FQL adapter for G3A/MJLab tasks.

The upstream QC code expects a Gym-like single environment whose policy
actions live in ``[-1, 1]`` and a plain transition dictionary with the six
arrays consumed by ``utils.datasets.Dataset``.  G3A recordings are converted
to that transition format by ``scripts/live_bridge/lerobot_to_qc.py``; this
module is the small public factory that lets upstream QC consume them
directly.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Protocol

import numpy as np

QC_DATASET_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "terminals",
    "masks",
)

SUPPORTED_ACTION_NORM_METHOD = "physical"


class _ShapeSpace(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


class _ShapeEnv(Protocol):
    @property
    def observation_space(self) -> _ShapeSpace: ...

    @property
    def action_space(self) -> _ShapeSpace: ...


class _RewardEnv(Protocol):
    """Just the reward-convention surface ``validate_reward_convention`` reads."""

    @property
    def task_id(self) -> str: ...

    @property
    def reward_max(self) -> float | None: ...


def _decode_str_metadata(value: Any) -> str:
    """Decode scalar/string numpy metadata into a plain Python string."""
    return str(np.atleast_1d(np.asarray(value))[0])


def _observation_array(value: Any) -> np.ndarray:
    """Convert observation-like data to numpy while preserving uint8 pixels."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.dtype == np.uint8:
        return arr
    return arr.astype(np.float32, copy=False)


def _scalar_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def _scalar_bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_manager_env(task_id: str, device: str, render_mode: str | None):
    """Construct the underlying MJLab env.

    Kept as a helper so tests can monkeypatch it without importing MJLab.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(task_id, play=False)
    env_cfg.scene.num_envs = 1
    env_cfg.auto_reset = False
    env_cfg.observations["actor"].enable_corruption = False
    return ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)


class G3AMjlabEnv:
    """Gym-style single-env wrapper over one MJLab G3A task.

    QC actions are normalized to ``[-1, 1]``.  Before stepping MJLab, they are
    mapped back to the DK1 physical raw-action range, the same convention used
    by ``lerobot_to_qc.py --action-norm physical``.
    """

    metadata = {"render_modes": [None, "rgb_array"]}

    def __init__(
        self,
        task_id: str,
        *,
        device: str = "auto",
        obs_group: str = "actor",
        render_mode: str | None = None,
    ) -> None:
        from gymnasium.spaces import Box

        from g3a_bench.scripts.live_bridge.mapper import ActionNormalizer

        self.task_id = task_id
        self.device = _resolve_device(device)
        self.obs_group = obs_group
        self.render_mode = render_mode
        self.env = _make_manager_env(task_id, self.device, render_mode)
        # Per-step reward is in [-reward_max, 0], 0 exactly at success (the
        # env applies the shift itself; see tasks/common/rewards.py). Recorded
        # datasets must have been produced under the same value -- checked by
        # validate_reward_convention().
        self.reward_max = _env_reward_max(self.env)
        self._normalizer = ActionNormalizer.physical()
        self._reset_episode_stats()
        self.total_timesteps = 0

        obs, _ = self.reset()
        if obs.dtype == np.uint8:
            obs_low = np.zeros_like(obs, dtype=np.uint8)
            obs_high = np.full_like(obs, np.iinfo(np.uint8).max, dtype=np.uint8)
            obs_dtype = np.uint8
        else:
            obs_low = np.full(obs.shape, -np.inf, dtype=np.float32)
            obs_high = np.full(obs.shape, np.inf, dtype=np.float32)
            obs_dtype = np.float32

        action_shape = self._normalizer.action_min.shape
        self.observation_space = Box(
            low=obs_low,
            high=obs_high,
            shape=obs.shape,
            dtype=obs_dtype,
        )
        self.action_space = Box(
            low=-np.ones(action_shape, dtype=np.float32),
            high=np.ones(action_shape, dtype=np.float32),
            shape=action_shape,
            dtype=np.float32,
        )

    @property
    def unwrapped(self):
        return self

    def _reset_episode_stats(self) -> None:
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def _extract_obs(self, obs_dict: dict) -> np.ndarray:
        if self.obs_group not in obs_dict:
            raise KeyError(
                f"Observation group {self.obs_group!r} not found. "
                f"Available groups: {sorted(obs_dict.keys())}"
            )
        obs = _observation_array(obs_dict[self.obs_group])
        # MJLab keeps a leading num_envs dimension; this adapter is single-env.
        if obs.shape[0] != 1:
            raise ValueError(
                f"Expected single MJLab env, got observation shape {obs.shape}"
            )
        return obs[0]

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        expected_shape = self._normalizer.action_min.shape
        if action.shape != expected_shape:
            raise ValueError(
                f"Expected normalized action shape {expected_shape}, got {action.shape}"
            )
        return self._normalizer.denormalize(action).astype(np.float32, copy=False)

    def _get_success(self) -> float | None:
        try:
            command = self.env.command_manager.get_term("task")
            return _scalar_float(command.episode_success[0])
        except (KeyError, AttributeError, TypeError):
            return None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        self._reset_episode_stats()
        if seed is None:
            obs_dict, _info = self.env.reset()
        else:
            obs_dict, _info = self.env.reset(seed=seed)
        return self._extract_obs(obs_dict), {}

    def step(self, action: np.ndarray):
        import torch

        raw_action = self._denormalize_action(action)
        action_t = torch.from_numpy(raw_action).float().unsqueeze(0).to(self.device)

        obs_dict, reward, terminated, truncated, _extras = self.env.step(action_t)
        next_obs = self._extract_obs(obs_dict)
        reward_value = _scalar_float(reward[0])
        terminated_flag = _scalar_bool(terminated[0])
        truncated_flag = _scalar_bool(truncated[0])
        done = terminated_flag or truncated_flag

        self.reward_sum += reward_value
        self.episode_length += 1
        self.total_timesteps += 1

        info: dict[str, Any] = {"total": {"timesteps": self.total_timesteps}}
        success = self._get_success()
        if success is not None:
            info["success"] = success
        if done:
            info["episode"] = {
                "final_reward": reward_value,
                "return": self.reward_sum,
                "length": self.episode_length,
                "duration": time.time() - self.start_time,
            }

        return next_obs, reward_value, terminated_flag, truncated_flag, info

    def render(self):
        return self.env.render()

    def get_state(self) -> dict[str, np.ndarray]:
        state: dict[str, np.ndarray] = {}
        sim_data = getattr(getattr(self.env, "sim", None), "data", None)
        if sim_data is None:
            return state
        for key in ("qpos", "qvel"):
            if hasattr(sim_data, key):
                arr = _observation_array(getattr(sim_data, key))
                state[key] = arr[0].copy() if arr.ndim > 1 else arr.copy()
        return state

    def close(self) -> None:
        self.env.close()


def _env_reward_max(manager_env: Any) -> float | None:
    """``reward_max`` the MJLab env was built with, or ``None`` if unknown.

    Read back off the constructed cfg rather than recomputed, so it tracks
    whatever ``make_manipulation_env_cfg`` actually assembled.  Returns
    ``None`` for envs without the shared offset term (test doubles, or a task
    cfg built outside the shared skeleton) so callers can skip the check
    instead of failing on a missing attribute.
    """
    from g3a_bench.tasks.common.rewards import REWARD_OFFSET_TERM

    cfg = getattr(manager_env, "cfg", None)
    rewards = getattr(cfg, "rewards", None)
    if not isinstance(rewards, dict) or REWARD_OFFSET_TERM not in rewards:
        return None
    return -float(rewards[REWARD_OFFSET_TERM].weight)


def read_dataset_reward_max(dataset_path: str | Path) -> float | None:
    """``reward_max`` recorded in a QC ``.npz``, or ``None`` if not stamped."""
    with np.load(dataset_path) as raw:
        if "reward_max" not in raw:
            return None
        return float(np.asarray(raw["reward_max"]).reshape(-1)[0])


def validate_reward_convention(
    dataset_path: str | Path,
    env: _RewardEnv,
) -> None:
    """Fail fast when a dataset's reward convention disagrees with the env.

    The offline dataset and the online env feed the *same* replay buffer, so
    they must agree on what a step is worth.  Nothing else catches a
    disagreement: QC trains happily on either scale and only the learned
    policy degrades, which is what made the previous ``{0,+1}`` dataset vs
    ``{0,+0.02}`` env mismatch invisible until online fine-tuning.

    Raises:
        ValueError: when the dataset's ``reward_max`` differs from the env's,
            or when the dataset predates that metadata (and is therefore on the
            old positive, dt-scaled convention).
    """
    env_max = env.reward_max
    if env_max is None:
        # Fail open rather than hard, so test doubles and cfgs built outside
        # the shared skeleton stay usable -- but say so, because this is the
        # only check standing between a mis-scaled dataset and a silently
        # degraded policy.
        warnings.warn(
            f"{env.task_id} has no {'max_reward_offset'!r} reward term, so the "
            f"reward convention of {dataset_path} cannot be verified against "
            f"it. Tasks built via make_manipulation_env_cfg always have one.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    dataset_max = read_dataset_reward_max(dataset_path)

    if dataset_max is None:
        raise ValueError(
            f"{dataset_path} predates the reward_max metadata, so it is on the "
            f"old positive, dt-scaled reward convention and cannot be trained "
            f"against {env.task_id} (which emits rewards in [{-env_max}, 0]). "
            f"Regenerate it with lerobot_to_qc.py from a current recording."
        )
    if abs(dataset_max - env_max) > 1e-6:
        raise ValueError(
            f"Reward convention mismatch: {dataset_path} was built with "
            f"reward_max={dataset_max}, but {env.task_id} emits rewards in "
            f"[{-env_max}, 0] (reward_max={env_max}). Offline and online "
            f"transitions would disagree on what a step is worth. Regenerate "
            f"the dataset against the current task cfg."
        )


def load_qc_dataset(dataset_path: str | Path) -> dict[str, np.ndarray]:
    """Load a LeRobot-to-QC ``.npz`` as the six arrays upstream QC consumes."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    with np.load(dataset_path) as raw:
        missing = [key for key in QC_DATASET_KEYS if key not in raw]
        if missing:
            raise ValueError(
                f"{dataset_path} is missing QC dataset keys: {', '.join(missing)}"
            )
        if "action_norm_method" not in raw:
            raise ValueError(
                f"{dataset_path} is missing action_norm_method metadata; "
                f"expected {SUPPORTED_ACTION_NORM_METHOD!r}"
            )
        action_norm_method = _decode_str_metadata(raw["action_norm_method"])
        if action_norm_method != SUPPORTED_ACTION_NORM_METHOD:
            raise ValueError(
                f"Unsupported action_norm_method={action_norm_method!r}; "
                f"G3A QC integration currently requires "
                f"{SUPPORTED_ACTION_NORM_METHOD!r}"
            )

        dataset = {
            "observations": _observation_array(raw["observations"]),
            "actions": raw["actions"].astype(np.float32, copy=False),
            "next_observations": _observation_array(raw["next_observations"]),
            "rewards": raw["rewards"].astype(np.float32, copy=False),
            "terminals": raw["terminals"].astype(np.float32, copy=False),
            "masks": raw["masks"].astype(np.float32, copy=False),
        }

    size = len(dataset["observations"])
    bad_lengths = {
        key: len(value) for key, value in dataset.items() if len(value) != size
    }
    if bad_lengths:
        raise ValueError(
            f"{dataset_path} has inconsistent dataset lengths: "
            f"observations={size}, {bad_lengths}"
        )
    if dataset["observations"].shape != dataset["next_observations"].shape:
        raise ValueError(
            "observations and next_observations must have identical shapes; "
            f"got {dataset['observations'].shape} and "
            f"{dataset['next_observations'].shape}"
        )
    return dataset


def validate_qc_dataset_against_env(
    dataset: dict[str, np.ndarray],
    env: _ShapeEnv,
) -> None:
    obs_shape = tuple(dataset["observations"].shape[1:])
    next_obs_shape = tuple(dataset["next_observations"].shape[1:])
    env_obs_shape = tuple(env.observation_space.shape)
    if obs_shape != env_obs_shape or next_obs_shape != env_obs_shape:
        raise ValueError(
            "Dataset observation shape does not match the MJLab env: "
            f"observations={obs_shape}, next_observations={next_obs_shape}, "
            f"env={env_obs_shape}. Regenerate the QC dataset with the current "
            "task observation config."
        )

    action_shape = tuple(dataset["actions"].shape[1:])
    env_action_shape = tuple(env.action_space.shape)
    if action_shape != env_action_shape:
        raise ValueError(
            "Dataset action shape does not match the MJLab env: "
            f"actions={action_shape}, env={env_action_shape}."
        )


def make_env_and_datasets(
    dataset_name: str,
    *,
    dataset_path: str | Path | None = None,
    env_only: bool = False,
    dataset_only: bool = False,
    cur_env: G3AMjlabEnv | None = None,
    device: str = "auto",
    obs_group: str = "actor",
    render_mode: str | None = None,
    eval_render_mode: str | None = None,
    validate_dataset: bool = True,
    **_env_kwargs: Any,
):
    """Make a G3A MJLab env and load a QC-format training dataset.

    Mirrors the useful part of ``ogbench.make_env_and_datasets`` for upstream
    QC.  G3A has no validation split, so the validation dataset is always
    ``None``.
    """
    env = cur_env

    train_dataset = None
    if dataset_path is not None:
        train_dataset = load_qc_dataset(dataset_path)

    if not dataset_only:
        env = G3AMjlabEnv(
            dataset_name,
            device=device,
            obs_group=obs_group,
            render_mode=render_mode,
        )

    if env_only:
        return env

    if dataset_path is None or train_dataset is None:
        raise ValueError("dataset_path is required for G3A QC datasets")

    if validate_dataset and env is not None:
        validate_qc_dataset_against_env(train_dataset, env)
        # Both halves of QC's replay buffer must agree on the reward scale.
        validate_reward_convention(dataset_path, env)

    if dataset_only:
        return train_dataset, None
    return (
        env,
        G3AMjlabEnv(
            dataset_name,
            device=device,
            obs_group=obs_group,
            render_mode=eval_render_mode
            if eval_render_mode is not None
            else render_mode,
        ),
        train_dataset,
        None,
    )
