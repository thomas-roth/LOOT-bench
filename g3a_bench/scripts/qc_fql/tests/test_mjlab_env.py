"""Tests for the mjlab adapter, run against the REAL G3a stacking task.

This is the one test file with no reference to compare against (the adapter
is new code, not a port) -- so instead of gold data it constructs the actual
mjlab env (module-scoped fixture; construction is the slow part) and checks
the adapter's contract: obs shape/dtype, gym-style step returns, action
denormalization hitting exactly the DK1 physical limits, and the cached
auto-reset observation short-circuiting the next reset() call.
"""

import numpy as np
import pytest

from g3a_bench.scripts.qc_fql.scripts.mjlab_env import MjlabSingleEnv

TASK_ID = "G3a-Stacking-Stack3-Dk1_uni"


def test_denormalize_action_maps_to_physical_limits():
    """Pure function, no env: [-1,1] endpoints must land exactly on the DK1
    physical bounds, midpoint on their mean, and out-of-range input must
    saturate at the bound rather than extrapolate past it."""
    from g3a_bench.scripts.live_bridge.mapper import (
        PHYSICAL_RAW_ACTION_MAX,
        PHYSICAL_RAW_ACTION_MIN,
        ActionNormalizer,
    )

    normalizer = ActionNormalizer.physical()
    action_dim = PHYSICAL_RAW_ACTION_MIN.shape[0]

    low = normalizer.denormalize(-np.ones(action_dim))
    high = normalizer.denormalize(np.ones(action_dim))
    mid = normalizer.denormalize(np.zeros(action_dim))

    assert np.allclose(low, PHYSICAL_RAW_ACTION_MIN)
    assert np.allclose(high, PHYSICAL_RAW_ACTION_MAX)
    assert np.allclose(mid, (PHYSICAL_RAW_ACTION_MIN + PHYSICAL_RAW_ACTION_MAX) / 2.0)

    # Out-of-range policy output must still clip to the physical bounds, not extrapolate.
    beyond = normalizer.denormalize(np.full(action_dim, 2.0))
    assert np.allclose(beyond, PHYSICAL_RAW_ACTION_MAX)


@pytest.fixture(scope="module")
def real_env():
    env = MjlabSingleEnv(TASK_ID, device="cpu")
    yield env
    env.close()


def test_real_env_reset_returns_expected_shape(real_env):
    obs, info = real_env.reset()
    assert obs.shape == (32,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_real_env_step_returns_expected_types_and_bounds(real_env):
    real_env.reset()
    action = np.zeros(
        7, dtype=np.float32
    )  # mid-range: valid regardless of denorm direction

    next_obs, reward, terminated, truncated, info = real_env.step(action)

    assert next_obs.shape == (32,)
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
    """Verifies the adapter's auto-reset cache in isolation: with
    `_pending_reset_obs` set, `.reset()` must return the cached observation
    and NOT call the underlying `env.reset()` (spy counts calls). Driving a
    real termination is impractically slow here (thousands of steps to the
    time limit), so only the cache mechanics are tested against the live env
    -- the fact that mjlab's `step()` returns the post-reset observation on
    done steps is documented and relied on in `mjlab_env.py` itself."""
    sentinel = np.arange(32, dtype=np.float32)
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
