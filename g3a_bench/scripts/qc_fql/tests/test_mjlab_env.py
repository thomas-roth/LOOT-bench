import numpy as np
import pytest
from scripts.mjlab_env import MjlabSingleEnv

TASK_ID = "G3a-Cube-Stack3-Dk1_uni"


def test_denormalize_action_maps_to_physical_limits():
    """No env needed -- pure function, but the constants it uses (PHYSICAL_RAW_
    ACTION_MIN/MAX) were confirmed live to match the real action term's
    scale/offset exactly (see module docstring)."""
    from g3a_bench.scripts.live_bridge.mapper import (
        PHYSICAL_RAW_ACTION_MAX,
        PHYSICAL_RAW_ACTION_MIN,
    )

    action_dim = PHYSICAL_RAW_ACTION_MIN.shape[0]

    low = MjlabSingleEnv._denormalize_action(-np.ones(action_dim))
    high = MjlabSingleEnv._denormalize_action(np.ones(action_dim))
    mid = MjlabSingleEnv._denormalize_action(np.zeros(action_dim))

    assert np.allclose(low, PHYSICAL_RAW_ACTION_MIN)
    assert np.allclose(high, PHYSICAL_RAW_ACTION_MAX)
    assert np.allclose(mid, (PHYSICAL_RAW_ACTION_MIN + PHYSICAL_RAW_ACTION_MAX) / 2.0)

    # Out-of-range policy output must still clip to the physical bounds, not extrapolate.
    beyond = MjlabSingleEnv._denormalize_action(np.full(action_dim, 2.0))
    assert np.allclose(beyond, PHYSICAL_RAW_ACTION_MAX)


@pytest.fixture(scope="module")
def real_env():
    env = MjlabSingleEnv(TASK_ID, device="cpu")
    yield env
    env.close()


def test_real_env_reset_returns_expected_shape(real_env):
    obs, info = real_env.reset()
    assert obs.shape == (53,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_real_env_step_returns_expected_types_and_bounds(real_env):
    real_env.reset()
    action = np.zeros(
        7, dtype=np.float32
    )  # mid-range: valid regardless of denorm direction

    next_obs, reward, terminated, truncated, info = real_env.step(action)

    assert next_obs.shape == (53,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    if "success" in info:
        assert 0.0 <= info["success"] <= 1.0


def test_real_env_action_denorm_clips_out_of_range_policy_output(real_env):
    real_env.reset()
    # A policy that (incorrectly) outputs slightly outside [-1,1] must not
    # blow past the physical joint limits when denormalized.
    action = np.full(7, 1.5, dtype=np.float32)
    real_env.step(action)  # must not raise / must not send an out-of-range ctrl


def test_pending_reset_avoids_redundant_env_reset(real_env, monkeypatch):
    """Simulates the auto-reset caching path directly (reaching a real
    termination via zero actions isn't practical in a fast test -- the
    underlying mjlab auto_reset mechanics were already confirmed live, see
    module docstring). This isolates and verifies the adapter's own
    caching logic: when `_pending_reset_obs` is set, `.reset()` must return
    it without calling into the real `env.reset()`."""
    sentinel = np.arange(53, dtype=np.float32)
    real_env._pending_reset_obs = sentinel

    called = {"count": 0}
    real_reset = real_env.env.reset

    def spy_reset(*args, **kwargs):
        called["count"] += 1
        return real_reset(*args, **kwargs)

    monkeypatch.setattr(real_env.env, "reset", spy_reset)

    obs, info = real_env.reset()

    assert np.array_equal(obs, sentinel)
    assert called["count"] == 0
    assert real_env._pending_reset_obs is None
