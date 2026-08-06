"""The shared reward convention: per-step reward in ``[-reward_max, 0]``.

Covers the ``reward_max`` kernel and the wiring in ``make_manipulation_env_cfg``
that every registered task inherits. See ``tasks/common/rewards.py`` for why the
convention is negative and why dt-scaling has to be off for it to hold.
"""

import pytest
from mjlab.managers.reward_manager import RewardTermCfg

from g3a_bench.tasks.common.rewards import REWARD_OFFSET_TERM, reward_max


def _term(weight: float) -> RewardTermCfg:
    return RewardTermCfg(func=lambda env: None, weight=weight)


class TestRewardMax:
    def test_sums_progress_and_success_weights(self):
        assert reward_max({"progress": _term(1.0), "success": _term(1.0)}) == 2.0

    def test_success_only_category(self):
        assert reward_max({"success": _term(1.0)}) == 1.0

    def test_ignores_the_offset_term(self):
        """Stays correct when called on an already-assembled reward dict."""
        assembled = {
            "progress": _term(1.0),
            "success": _term(1.0),
            REWARD_OFFSET_TERM: _term(-2.0),
        }
        assert reward_max(assembled) == 2.0


# (task id, expected reward_max): one category per reward shape.
_TASKS = [
    ("G3A-Stack-Low-Dk1_uni", 2.0),  # progress + success
    ("G3A-Place-Easy-Dk1_uni", 1.0),  # success only
]


@pytest.mark.parametrize(("task_id", "expected_max"), _TASKS)
def test_registered_task_reward_is_shifted_to_non_positive(task_id, expected_max):
    """The offset term is wired in with the right magnitude, and dt-scaling off.

    Both halves matter and neither is visible from the category cfgs: mjlab
    defaults ``scale_rewards_by_dt`` to True, which would shrink the task terms
    *and* this offset by ``step_dt`` (0.02 s), leaving rewards 50x too small.
    """
    from mjlab.tasks.registry import load_env_cfg

    import g3a_bench  # noqa: F401  -- registers the tasks

    cfg = load_env_cfg(task_id, play=False)

    assert cfg.scale_rewards_by_dt is False
    assert REWARD_OFFSET_TERM in cfg.rewards
    assert cfg.rewards[REWARD_OFFSET_TERM].weight == pytest.approx(-expected_max)
    assert reward_max(cfg.rewards) == pytest.approx(expected_max)
