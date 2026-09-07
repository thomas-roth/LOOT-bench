"""Unit tests for ``tasks/common/rewards.py``.

These pin the reward convention the offline pipeline depends on:
``r`` lies in ``[-1, 0]``, and ``r == 0`` exactly at success.
"""

import torch

from loot_bench.tasks.common.rewards import MIN_SHORTFALL, shaped_task_reward


class TestShapedTaskReward:
    def test_range_and_zero_only_at_success(self):
        progress = torch.linspace(0.0, 1.0, 21)
        for success_val in (False, True):
            success = torch.full((21,), success_val, dtype=torch.bool)
            out = shaped_task_reward(progress, success)
            assert bool((out >= -1.0).all()) and bool((out <= 0.0).all())
            if success_val:
                assert bool((out == 0.0).all())
            else:
                assert bool((out < 0.0).all()), "only success may reach exactly 0"

    def test_full_progress_without_success_is_not_zero(self):
        """The invariant qc's ``--sparse`` flag rests on.

        ``--sparse`` maps ``(rewards != 0.0) * -1.0``. A task that reports
        progress 1.0 while its success gate has not latched (objects placed
        but still moving, say) would otherwise be relabelled a success.
        """
        out = shaped_task_reward(torch.ones(4), torch.zeros(4, dtype=torch.bool))
        assert bool((out == -MIN_SHORTFALL).all())
        assert bool((out != 0.0).all())

    def test_monotone_in_progress(self):
        progress = torch.linspace(0.0, 1.0, 50)
        out = shaped_task_reward(progress, torch.zeros(50, dtype=torch.bool))
        assert bool((out.diff() >= 0).all())

    def test_out_of_range_progress_is_clamped(self):
        progress = torch.tensor([-5.0, -0.1, 1.1, 7.0])
        out = shaped_task_reward(progress, torch.zeros(4, dtype=torch.bool))
        assert bool((out >= -1.0).all()) and bool((out < 0.0).all())
        assert out[0] == -1.0 and out[1] == -1.0

    def test_zero_progress_gives_the_sparse_task(self):
        """How the categories with no progress signal (ball) are served."""
        success = torch.tensor([False, False, True, False])
        out = shaped_task_reward(torch.zeros(4), success)
        assert torch.equal(out, torch.tensor([-1.0, -1.0, 0.0, -1.0]))

    def test_shape_and_dtype_follow_progress(self):
        progress = torch.rand(3, 5)
        out = shaped_task_reward(progress, torch.zeros(3, 5, dtype=torch.bool))
        assert out.shape == (3, 5) and out.dtype == progress.dtype

    def test_accepts_float_success(self):
        """Command terms expose ``success`` as float in some categories."""
        out = shaped_task_reward(torch.zeros(2), torch.tensor([0.0, 1.0]))
        assert torch.equal(out, torch.tensor([-1.0, 0.0]))
