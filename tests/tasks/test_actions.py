"""Tests for the shared action convention.

The pure tests need no simulator. The ones that build a task earn their cost:
they are the only place that checks the convention actually reached the
compiled model, and a wrong action range does not raise -- it just puts the
task out of reach and reads back as a policy that never learns.

``LOOT-Place-FixedS-FixedT-Dk1`` rather than a stack variant, because the
episode-budget and observation-width assertions below are ``place``'s own.
"""

import numpy as np
import pytest
import torch

from loot_bench.tasks.common.actions import (
    JOINT_POS_ACTION_NAME,
    normalized_scale_offset,
)
from loot_bench.tasks.common.env_cfg import CONTROL_STEP_S

TASK_ID = "LOOT-Place-FixedS-FixedT-Dk1"
PLACE_EPISODE_LENGTH_S = 15.0
PLACE_OBS_WIDTH = 36


class TestNormalizedScaleOffset:
    def test_minus_one_is_the_lower_limit(self):
        lower = torch.tensor([-2.0, -0.087, 0.0])
        upper = torch.tensor([2.0, 3.14, 1.0])
        scale, offset = normalized_scale_offset(lower, upper)
        torch.testing.assert_close(-1.0 * scale + offset, lower)

    def test_plus_one_is_the_upper_limit(self):
        lower = torch.tensor([-2.0, -0.087, 0.0])
        upper = torch.tensor([2.0, 3.14, 1.0])
        scale, offset = normalized_scale_offset(lower, upper)
        torch.testing.assert_close(1.0 * scale + offset, upper)

    def test_zero_is_the_midpoint_not_the_origin(self):
        """The whole point of the offset. A joint whose range does not
        straddle zero still gets the full box."""
        lower = torch.tensor([1.0])
        upper = torch.tensor([3.0])
        scale, offset = normalized_scale_offset(lower, upper)
        assert float(0.0 * scale + offset) == pytest.approx(2.0)

    def test_scale_is_positive_for_an_asymmetric_range(self):
        lower = torch.tensor([-0.087])
        upper = torch.tensor([4.712])
        scale, _ = normalized_scale_offset(lower, upper)
        assert float(scale) > 0.0


@pytest.fixture(scope="module")
def built_env():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import loot_bench  # noqa: F401  (registers the tasks)

    cfg = load_env_cfg(TASK_ID, play=False)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield env
    env.close()


class TestCompiledTask:
    """The convention as the simulator actually sees it."""

    def test_the_action_is_seven_wide(self, built_env):
        assert built_env.action_manager.total_action_dim == 7
        assert built_env.action_manager.get_term(JOINT_POS_ACTION_NAME).action_dim == 7

    def test_the_control_step_is_twenty_hertz(self, built_env):
        """Joint-absolute control at the 20 Hz rate: 0.05 s per step."""
        assert built_env.step_dt == pytest.approx(CONTROL_STEP_S)
        assert built_env.step_dt == pytest.approx(0.05)

    def test_action_box_spans_every_joint_limit(self, built_env):
        term = built_env.action_manager.get_term(JOINT_POS_ACTION_NAME)
        limits = built_env.scene["robot"].data.joint_pos_limits
        assert limits is not None
        expected = limits[:, term.target_ids]
        torch.testing.assert_close(-1.0 * term.scale + term.offset, expected[..., 0])
        torch.testing.assert_close(1.0 * term.scale + term.offset, expected[..., 1])

    def test_the_start_pose_is_inside_the_action_box(self, built_env):
        """The demos start here, so a policy that cannot command it cannot
        reproduce a single demonstrated frame."""
        robot = built_env.scene["robot"]
        term = built_env.action_manager.get_term(JOINT_POS_ACTION_NAME)
        start = robot.data.default_joint_pos
        assert start is not None
        action = (start[:, term.target_ids] - term.offset) / term.scale
        assert torch.all(action.abs() <= 1.0), action

    def test_the_whole_table_reach_is_commandable(self, built_env):
        """Sanity check against the old residual scale, under which the arm
        could not leave a 0.07 rad band around the keyframe."""
        term = built_env.action_manager.get_term(JOINT_POS_ACTION_NAME)
        span = 2.0 * term.scale
        assert float(span[:, :6].min()) > 1.0, "arm joints must sweep >1 rad"

    def test_the_observation_carries_no_action_term(self, built_env):
        terms = built_env.observation_manager.active_terms["actor"]
        assert "actions" not in terms
        assert built_env.observation_manager.group_obs_dim["actor"] == (
            PLACE_OBS_WIDTH,
        )

    def test_an_action_outside_the_box_is_clamped(self, built_env):
        """The box has to bind the recorder too, not only a squashed policy.
        Teleop with gravity compensation lands outside it near a joint limit."""
        term = built_env.action_manager.get_term(JOINT_POS_ACTION_NAME)
        width = built_env.action_manager.total_action_dim
        built_env.reset()
        obs, *_ = built_env.step(torch.full((1, width), 3.0))
        assert float(term.raw_action.max()) <= 1.0
        torch.testing.assert_close(
            term._processed_actions,  # pyright: ignore[reportPrivateUsage]
            1.0 * term.scale + term.offset,
        )
        # And the observation says so too. The `actions` term is the last one
        # in the group, so it is the final `width` columns -- unless the task
        # drops it, in which case there is nothing further to assert here.
        terms = built_env.observation_manager.active_terms["actor"]
        if "actions" in terms:
            reported = obs["actor"][:, -width:]
            assert float(reported.max()) <= 1.0, (
                "observation reports an unapplied action"
            )

    def test_the_teleop_mapper_agrees_with_the_action_term(self, built_env):
        """The recording/evaluation seam, in one assertion.

        ``live_bridge/mapper.py`` turns a leader-arm pose into an action while
        recording; the action term turns an action into a joint target while
        evaluating. If the two ever disagree, the demos describe a trajectory
        the policy is never scored on, and nothing raises.
        """
        from loot_bench.scripts.live_bridge.mapper import (
            ACTUATED_JOINT_NAMES,
            action_scale_offset,
        )

        term = built_env.action_manager.get_term(JOINT_POS_ACTION_NAME)
        assert list(term.target_names) == list(ACTUATED_JOINT_NAMES)
        scale, offset = action_scale_offset()
        np.testing.assert_array_almost_equal(term.scale[0].numpy(), scale, decimal=5)
        np.testing.assert_array_almost_equal(term.offset[0].numpy(), offset, decimal=5)

    def test_actions_reach_the_commanded_target(self, built_env):
        """End to end: a held action settles the arm at the target that the
        scale and offset predict."""
        term = built_env.action_manager.get_term(JOINT_POS_ACTION_NAME)
        action = torch.zeros(1, built_env.action_manager.total_action_dim)
        built_env.reset()
        for _ in range(200):
            built_env.step(action)
        target = (0.0 * term.scale + term.offset)[0, :6]
        reached = built_env.scene["robot"].data.joint_pos[0, :6]
        np.testing.assert_allclose(
            reached.numpy(), target.numpy(), atol=0.1, err_msg="arm did not track"
        )

    def test_a_place_episode_times_out_at_its_budget(self, built_env):
        """15 s is 300 steps at 20 Hz."""
        expected = round(PLACE_EPISODE_LENGTH_S / built_env.step_dt)
        assert int(built_env.max_episode_length) == expected == 300

    def test_the_teleop_mapper_reaches_what_the_leader_asked_for(self, built_env):
        """The recording/evaluation seam, closed loop.

        ``live_bridge/mapper.py`` turns a leader-arm pose into a joint action;
        the action term turns that action into joint targets. Drive the
        converted action open-loop and check the arm settles where the leader
        asked.
        """
        from loot_bench.scripts.live_bridge.mapper import lerobot_to_mjlab_raw_action

        leader = np.zeros(7, dtype=np.float32)
        leader[:6] = [0.1, 1.62, 1.75, -1.5708, 0.0, 0.0]
        leader[6] = 0.5
        action = torch.from_numpy(lerobot_to_mjlab_raw_action(leader)).unsqueeze(0)

        term = built_env.action_manager.get_term(JOINT_POS_ACTION_NAME)
        assert float(action.abs().max()) <= 1.0 + 1e-5

        built_env.reset()
        for _ in range(200):
            built_env.step(action)
        target = (action * term.scale + term.offset)[0].numpy()
        reached = built_env.scene["robot"].data.joint_pos[0].detach().cpu().numpy()
        np.testing.assert_allclose(reached[:6], target[:6], atol=0.15)
