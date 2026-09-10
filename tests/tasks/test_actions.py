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
    DELTA_ORI_SCALE,
    DELTA_POS_SCALE,
    EE_DELTA_ACTION_NAME,
    GRIPPER_ACTION_NAME,
    delta_action_from_poses,
    normalized_scale_offset,
    pose_within_one_delta,
)
from loot_bench.tasks.common.env_cfg import CONTROL_STEP_S

TASK_ID = "LOOT-Place-FixedS-FixedT-Dk1"
PLACE_EPISODE_LENGTH_S = 15.0
PLACE_OBS_WIDTH = 36
SUCCESS_TOLERANCE_M = 0.02
"""The place task's own position tolerance. A mapping error larger than this
cannot be distinguished from a policy that missed."""

_IDENTITY_QUAT = torch.tensor([[1.0, 0.0, 0.0, 0.0]])


class TestNormalizedScaleOffset:
    """The gripper's half of the convention."""

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


class TestDeltaFromPoses:
    """The mapping both the recorder and the action term are built on."""

    def test_the_same_pose_is_a_zero_action(self):
        pos = torch.tensor([[0.3, 0.0, 0.2]])
        delta = delta_action_from_poses(pos, _IDENTITY_QUAT, pos, _IDENTITY_QUAT)
        torch.testing.assert_close(delta, torch.zeros(1, 6), atol=1e-6, rtol=0)

    def test_one_box_of_travel_is_a_unit_action(self):
        pos = torch.tensor([[0.3, 0.0, 0.2]])
        target = pos + torch.tensor([[DELTA_POS_SCALE, 0.0, 0.0]])
        delta = delta_action_from_poses(pos, _IDENTITY_QUAT, target, _IDENTITY_QUAT)
        assert float(delta[0, 0]) == pytest.approx(1.0, abs=1e-5)

    def test_a_pose_beyond_the_box_clamps(self):
        """A bounded policy can only emit [-1, 1]. The recorder has to obey the
        same bound as the term, or the two describe different MDPs."""
        pos = torch.tensor([[0.3, 0.0, 0.2]])
        target = pos + torch.tensor([[10.0 * DELTA_POS_SCALE, 0.0, 0.0]])
        delta = delta_action_from_poses(pos, _IDENTITY_QUAT, target, _IDENTITY_QUAT)
        assert float(delta[0, 0]) == pytest.approx(1.0)
        assert float(delta.abs().max()) <= 1.0

    def test_a_rotation_of_one_box_is_a_unit_action(self):
        pos = torch.zeros(1, 3)
        half = DELTA_ORI_SCALE / 2.0
        target_quat = torch.tensor([[np.cos(half), np.sin(half), 0.0, 0.0]]).float()
        delta = delta_action_from_poses(pos, _IDENTITY_QUAT, pos, target_quat)
        assert float(delta[0, 3]) == pytest.approx(1.0, abs=1e-5)


class TestWithinOneDelta:
    """The recorder's episode-start gate."""

    def test_the_same_pose_is_inside(self):
        pos = torch.tensor([[0.3, 0.0, 0.2]])
        assert bool(pose_within_one_delta(pos, _IDENTITY_QUAT, pos, _IDENTITY_QUAT)[0])

    def test_just_beyond_one_box_is_outside(self):
        pos = torch.tensor([[0.3, 0.0, 0.2]])
        target = pos + torch.tensor([[1.01 * DELTA_POS_SCALE, 0.0, 0.0]])
        assert not bool(
            pose_within_one_delta(pos, _IDENTITY_QUAT, target, _IDENTITY_QUAT)[0]
        )


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


def _grasp_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    """The grasp site's world-frame pose, as the solver reads it."""
    return env.action_manager.get_term(EE_DELTA_ACTION_NAME).frame_pose


def _settle(env, action: torch.Tensor, steps: int) -> None:
    for _ in range(steps):
        env.step(action)


class TestCompiledTask:
    """The convention as the simulator actually sees it."""

    def test_the_action_is_seven_wide(self, built_env):
        assert built_env.action_manager.total_action_dim == 7
        assert (
            built_env.action_manager.get_term(EE_DELTA_ACTION_NAME).action_dim == 6
        ), "a 6-D delta needs orientation_weight > 0 and relative mode"
        assert built_env.action_manager.get_term(GRIPPER_ACTION_NAME).action_dim == 1

    def test_the_control_step_is_twenty_hertz(self, built_env):
        """QC's chunk of 5 spans 0.25 s here, which is both reference domains.
        At the old 50 Hz it spanned 0.10 s, near the failing end of QC's own
        sweep."""
        assert built_env.step_dt == pytest.approx(CONTROL_STEP_S)
        assert built_env.step_dt == pytest.approx(0.05)

    def test_the_observation_carries_no_action_term(self, built_env):
        """ "Repeat the last action" was a trivially strong regressor: 0.0242
        held-out against a BC gate bar of 0.0766."""
        terms = built_env.observation_manager.active_terms["actor"]
        assert "actions" not in terms
        assert built_env.observation_manager.group_obs_dim["actor"] == (
            PLACE_OBS_WIDTH,
        )

    def test_a_zero_action_holds_the_end_effector_still(self, built_env):
        built_env.reset()
        _settle(built_env, torch.zeros(1, 7), 20)
        before, _ = _grasp_pose(built_env)
        before = before.clone()
        _settle(built_env, torch.zeros(1, 7), 20)
        after, _ = _grasp_pose(built_env)
        moved = float((after - before).norm())
        assert moved < 0.005, f"a zero action drifted {moved:.4f} m"

    def test_one_action_unit_is_one_delta_scale_of_travel(self, built_env):
        """The number a researcher reads off the action space has to be the
        number the arm travels.

        Driven closed-loop against a *fixed* goal, because the term is
        relative: it re-reads the current pose every control step, so a single
        action commands 5 cm and then the next action re-baselines. Holding a
        goal means re-deriving the delta from where the arm now is, which is
        what teleoperation and a policy both do.
        """
        built_env.reset()
        _settle(built_env, torch.zeros(1, 7), 20)
        start, start_quat = _grasp_pose(built_env)
        goal = start.clone() + torch.tensor([[DELTA_POS_SCALE, 0.0, 0.0]])

        first = delta_action_from_poses(start, start_quat, goal, start_quat)
        assert float(first[0, 0]) == pytest.approx(1.0, abs=1e-4), (
            "a goal one delta scale away must be exactly one action unit"
        )

        for _ in range(40):
            pos, quat = _grasp_pose(built_env)
            delta = delta_action_from_poses(pos, quat, goal, quat)
            action = torch.zeros(1, 7)
            action[:, :6] = delta
            built_env.step(action)

        reached, _ = _grasp_pose(built_env)
        error = float((reached - goal).norm())
        assert error < 0.005, (
            f"a one-unit goal left the arm {error:.4f} m short of {DELTA_POS_SCALE} m"
        )

    def test_an_action_outside_the_box_is_clamped(self, built_env):
        """The box has to be a bound, not a unit. mjlab's own DiffIK term
        applies its scale to whatever it is handed, so without the clamp an
        action of 3.0 would honestly ask for three boxes of travel -- and the
        dataset would hold actions no tanh-squashed policy can emit.

        Asserted as an equality rather than a threshold: 3.0 and 1.0 must move
        the arm identically, because both mean the same commanded delta.
        """
        term = built_env.action_manager.get_term(EE_DELTA_ACTION_NAME)

        def travel(magnitude: float) -> float:
            built_env.reset()
            _settle(built_env, torch.zeros(1, 7), 20)
            before, _ = _grasp_pose(built_env)
            before = before.clone()
            action = torch.zeros(1, 7)
            action[0, 0] = magnitude
            built_env.step(action)
            assert float(term.raw_action.abs().max()) <= 1.0, (
                "raw_action must report the clamped value the simulator applied"
            )
            after, _ = _grasp_pose(built_env)
            return float((after - before)[0, 0])

        at_the_edge = travel(1.0)
        far_outside = travel(3.0)
        assert far_outside == pytest.approx(at_the_edge, abs=1e-4), (
            f"a 3.0 action moved {far_outside:.4f} m against {at_the_edge:.4f} m "
            "for 1.0; the clamp did not bind"
        )

    def test_a_place_episode_times_out_at_its_budget(self, built_env):
        """15 s is 300 steps at 20 Hz -- exactly robomimic `can`. The old 10 s
        severed 20 of the 99 recorded demos' successful endings."""
        expected = round(PLACE_EPISODE_LENGTH_S / built_env.step_dt)
        assert int(built_env.max_episode_length) == expected == 300

    def test_driving_into_a_joint_limit_does_not_break_through_it(self, built_env):
        """Differential IK steers with the *soft* limits, and the robots set
        the soft factor to 1.0 so those are the hard ones -- ADR-005's outer
        slice stays teleoperable.

        The bar is 0.1 rad rather than zero, and that is not slack. MuJoCo's
        own joint-limit constraint is soft, so a saturated action pressing into
        a limit always seats slightly past it, and the IK residual is zero
        strictly inside the limits (it restores at the wall, it does not
        decelerate before it -- see ``IK_JOINT_LIMIT_WEIGHT``). Measured on
        this task: 0.089 rad with the residual off, 0.060 rad with it on, and
        no further improvement at 5x the weight. The assertion is that the arm
        seats against its limits rather than driving through them.
        """
        robot = built_env.scene["robot"]
        limits = robot.data.joint_pos_limits
        assert limits is not None

        worst = 0.0
        for axis, sign in ((2, -1.0), (2, 1.0), (0, 1.0)):
            built_env.reset()
            action = torch.zeros(1, 7)
            action[0, axis] = sign
            for _ in range(60):
                built_env.step(action)
                joint_pos = robot.data.joint_pos
                worst = min(
                    worst,
                    float((joint_pos - limits[..., 0]).min()),
                    float((limits[..., 1] - joint_pos).min()),
                )
        assert worst > -0.1, f"joint limit broken through by {worst:.5f} rad"

    def test_the_teleop_mapper_reaches_what_the_leader_asked_for(self, built_env):
        """The recording/evaluation seam, in one assertion.

        ``live_bridge/mapper.py`` turns a leader-arm pose into an action while
        recording; the action term turns that action into joint targets while
        evaluating. If the two ever disagree, the demos describe a trajectory
        the policy is never scored on -- and nothing raises.

        Driven closed-loop, the way teleoperation actually runs: the delta is
        re-derived from the *current* pose every step. An open-loop single step
        cannot land a 5 cm target through a soft PD arm rate-limited to
        0.04 rad per physics tick, so it would need a tolerance loose enough to
        stop testing anything.
        """
        from loot_bench.scripts.live_bridge.mapper import (
            leader_grasp_pose,
            leader_to_delta_action,
        )

        # A leader pose a few centimetres from the reset pose: reachable, and
        # far enough that a broken mapping cannot pass by standing still.
        leader = np.zeros(7, dtype=np.float32)
        leader[:6] = [0.1, 1.62, 1.75, -1.5708, 0.0, 0.0]
        wanted_pos, _ = leader_grasp_pose(leader)

        built_env.reset()
        origins = built_env.scene.env_origins
        for _ in range(60):
            pos_w, quat_w = _grasp_pose(built_env)
            action = leader_to_delta_action(
                leader,
                (pos_w - origins)[0].numpy(),
                quat_w[0].numpy(),
            )
            built_env.step(torch.from_numpy(action).unsqueeze(0))

        pos_w, _ = _grasp_pose(built_env)
        reached = (pos_w - origins)[0].numpy()
        error = float(np.linalg.norm(reached - wanted_pos))
        assert error < SUCCESS_TOLERANCE_M, (
            f"leader asked for {wanted_pos}, arm reached {reached} "
            f"({error:.4f} m off, tolerance {SUCCESS_TOLERANCE_M})"
        )
