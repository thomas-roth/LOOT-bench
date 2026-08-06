from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from g3a_bench import qc


@dataclass
class _FakeSpace:
    shape: tuple[int, ...]


@dataclass
class _FakeShapeEnv:
    observation_space: _FakeSpace
    action_space: _FakeSpace


def _dataset_arrays(
    *,
    size: int = 4,
    obs_shape: tuple[int, ...] = (3,),
    action_shape: tuple[int, ...] = (7,),
) -> dict[str, np.ndarray]:
    return {
        "observations": np.ones((size, *obs_shape), dtype=np.float64),
        "actions": np.zeros((size, *action_shape), dtype=np.float64),
        "next_observations": np.full((size, *obs_shape), 2.0, dtype=np.float64),
        "rewards": np.arange(size, dtype=np.float64),
        "terminals": np.array([0, 0, 0, 1], dtype=np.bool_),
        "masks": np.array([1, 1, 1, 0], dtype=np.float64),
    }


def _write_dataset(path, **overrides) -> dict[str, np.ndarray]:
    arrays = overrides.pop("arrays", None)
    if arrays is None:
        arrays = _dataset_arrays()
    np.savez(
        path,
        **arrays,
        action_norm_method=overrides.pop("action_norm_method", np.array("physical")),
        episode_index=np.arange(len(arrays["observations"]), dtype=np.int64),
        action_min=np.full((7,), -1.0, dtype=np.float32),
        action_max=np.full((7,), 1.0, dtype=np.float32),
        **overrides,
    )
    return arrays


def test_load_qc_dataset_strips_metadata(tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    _write_dataset(dataset_path)

    dataset = qc.load_qc_dataset(dataset_path)

    assert tuple(dataset.keys()) == qc.QC_DATASET_KEYS
    assert dataset["observations"].dtype == np.float32
    assert dataset["actions"].dtype == np.float32
    assert dataset["next_observations"].dtype == np.float32
    assert dataset["terminals"].dtype == np.float32
    assert "episode_index" not in dataset
    assert "action_min" not in dataset
    assert "action_max" not in dataset


def test_load_qc_dataset_preserves_uint8_observations(tmp_path):
    dataset_path = tmp_path / "pixels.npz"
    arrays = _dataset_arrays(obs_shape=(4, 4, 3))
    arrays["observations"] = np.zeros((4, 4, 4, 3), dtype=np.uint8)
    arrays["next_observations"] = np.ones((4, 4, 4, 3), dtype=np.uint8)
    _write_dataset(dataset_path, arrays=arrays)

    dataset = qc.load_qc_dataset(dataset_path)

    assert dataset["observations"].dtype == np.uint8
    assert dataset["next_observations"].dtype == np.uint8


def test_load_qc_dataset_rejects_missing_required_keys(tmp_path):
    dataset_path = tmp_path / "missing.npz"
    arrays = _dataset_arrays()
    del arrays["masks"]
    _write_dataset(dataset_path, arrays=arrays)

    with pytest.raises(ValueError, match="missing QC dataset keys: masks"):
        qc.load_qc_dataset(dataset_path)


def test_load_qc_dataset_rejects_data_action_normalization(tmp_path):
    dataset_path = tmp_path / "data_norm.npz"
    _write_dataset(dataset_path, action_norm_method=np.array("data"))

    with pytest.raises(ValueError, match="Unsupported action_norm_method=.*data"):
        qc.load_qc_dataset(dataset_path)


def test_validate_qc_dataset_against_env_rejects_shape_mismatch():
    dataset = _dataset_arrays(obs_shape=(3,), action_shape=(7,))
    env = _FakeShapeEnv(
        observation_space=_FakeSpace(shape=(3,)),
        action_space=_FakeSpace(shape=(7,)),
    )
    qc.validate_qc_dataset_against_env(dataset, env)

    bad_obs = dict(
        dataset,
        observations=np.zeros((4, 4), dtype=np.float32),
        next_observations=np.zeros((4, 4), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="observation shape"):
        qc.validate_qc_dataset_against_env(bad_obs, env)

    bad_actions = dict(dataset, actions=np.zeros((4, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="action shape"):
        qc.validate_qc_dataset_against_env(bad_actions, env)


@dataclass
class _FakeRewardEnv:
    """Just the surface ``validate_reward_convention`` reads off an env."""

    reward_max: float | None
    task_id: str = "G3A-Fake-v0"


class TestRewardConvention:
    """Offline dataset and online env must agree on what a step is worth.

    Nothing else catches a disagreement: QC trains happily on either scale and
    only the learned policy degrades, which is what kept the earlier
    ``{0,+1}``-dataset / ``{0,+0.02}``-env split invisible until online
    fine-tuning.
    """

    @staticmethod
    def _env(reward_max: float | None) -> _FakeRewardEnv:
        return _FakeRewardEnv(reward_max=reward_max)

    def test_accepts_matching_convention(self, tmp_path):
        path = tmp_path / "dataset.npz"
        _write_dataset(path, reward_max=np.array(2.0, dtype=np.float32))
        qc.validate_reward_convention(path, self._env(2.0))

    def test_rejects_mismatched_reward_max(self, tmp_path):
        """e.g. a shaped-task dataset against a task whose progress term went."""
        path = tmp_path / "dataset.npz"
        _write_dataset(path, reward_max=np.array(2.0, dtype=np.float32))
        with pytest.raises(ValueError, match="Reward convention mismatch"):
            qc.validate_reward_convention(path, self._env(1.0))

    def test_rejects_dataset_without_reward_metadata(self, tmp_path):
        """Files written before the convention landed are on the old scale."""
        path = tmp_path / "dataset.npz"
        _write_dataset(path)
        with pytest.raises(ValueError, match="predates the reward_max metadata"):
            qc.validate_reward_convention(path, self._env(1.0))

    def test_warns_but_passes_when_env_has_no_offset_term(self, tmp_path):
        """Test doubles and cfgs outside the shared skeleton opt out — loudly.

        Failing open is deliberate, but silence is not: this is the only check
        between a mis-scaled dataset and a silently degraded policy.
        """
        path = tmp_path / "dataset.npz"
        _write_dataset(path)
        with pytest.warns(RuntimeWarning, match="cannot be verified"):
            qc.validate_reward_convention(path, self._env(None))


class _FakeCommand:
    episode_success = np.array([1.0], dtype=np.float32)


class _FakeCommandManager:
    def get_term(self, name):
        if name != "task":
            raise KeyError(name)
        return _FakeCommand()


class _FakeManagerEnv:
    def __init__(self):
        self.command_manager = _FakeCommandManager()
        self.sim = SimpleNamespace(
            data=SimpleNamespace(
                qpos=np.array([[0.1, 0.2]], dtype=np.float32),
                qvel=np.array([[0.3, 0.4]], dtype=np.float32),
            )
        )
        self.last_action = None
        self.closed = False

    def reset(self, seed=None):
        del seed
        return {"actor": np.array([[0.1, 0.2, 0.3]], dtype=np.float32)}, {}

    def step(self, action):
        self.last_action = action.detach().cpu().numpy()
        return (
            {"actor": np.array([[0.4, 0.5, 0.6]], dtype=np.float32)},
            np.array([0.25], dtype=np.float32),
            np.array([False]),
            np.array([True]),
            {},
        )

    def render(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def test_g3a_mjlab_env_reset_step_returns_gym_values(monkeypatch):
    fake_env = _FakeManagerEnv()
    monkeypatch.setattr(qc, "_make_manager_env", lambda *args: fake_env)

    env = qc.G3AMjlabEnv("G3A-Fake-v0", device="cpu")
    obs, reset_info = env.reset()
    assert obs.shape == (3,)
    assert reset_info == {}

    next_obs, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    assert next_obs.shape == (3,)
    assert reward == pytest.approx(0.25)
    assert terminated is False
    assert truncated is True
    assert info["success"] == pytest.approx(1.0)
    assert info["episode"]["length"] == 1
    assert fake_env.last_action.shape == (1, 7)
    np.testing.assert_array_equal(env.render(), np.zeros((2, 2, 3), dtype=np.uint8))
    assert env.get_state()["qpos"].shape == (2,)

    env.close()
    assert fake_env.closed
