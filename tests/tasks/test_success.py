"""Unit tests for the success criteria in the per-category ``mdp/success.py``
modules, the shared ``SuccessGate`` in ``tasks/common/success.py``, and the
gripper-open gate in ``tasks/common/commands.py``."""

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import NoReturn, cast

import pytest
import torch
from mjlab.entity import Entity

from g3a_bench.tasks.ball.mdp.success import ball_in_target_zone
from g3a_bench.tasks.common.commands import GripperGateCommandCfg, GripperOpenGate
from g3a_bench.tasks.common.success import SuccessGate
from g3a_bench.tasks.domino.mdp.success import domino_fallen, domino_upright_aligned
from g3a_bench.tasks.shape_sorter.mdp.success import SorterLatch
from g3a_bench.tasks.shape_sorter.variants import SHAPE_SORTER_VARIANTS, allowed_faces
from g3a_bench.tasks.stack.mdp.success import (
    block_top_heights,
    stack_progress,
    stack_reached,
    two_stacks_progress,
    two_stacks_success,
)


def _quat_axis_angle(axis: tuple[float, float, float], angle: float) -> torch.Tensor:
    """(4,) wxyz quaternion for a rotation of *angle* about *axis* (unit)."""
    half = angle / 2
    s = math.sin(half)
    return torch.tensor(
        [math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=torch.float32
    )


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# domino_upright_aligned
# ---------------------------------------------------------------------------


class TestDominoUprightAligned:
    def _check(self, quat: torch.Tensor, yaw: float) -> bool:
        q = quat.view(1, 1, 4)
        row_yaw = torch.tensor([yaw])
        return bool(domino_upright_aligned(q, row_yaw, threshold_deg=15.0)[0, 0])

    def test_upright_aligned_passes(self):
        yaw = math.pi / 2
        quat = _quat_axis_angle((0.0, 0.0, 1.0), yaw)
        assert self._check(quat, yaw)

    @pytest.mark.parametrize("tilt_deg,expected", [(14.0, True), (16.0, False)])
    def test_tilt_boundary(self, tilt_deg: float, expected: bool):
        # Tilt about the row direction keeps the thin side aligned, so only
        # the upright check decides.
        yaw = math.pi / 2
        row_axis = (math.cos(yaw), math.sin(yaw), 0.0)
        quat = _quat_mul(
            _quat_axis_angle(row_axis, math.radians(tilt_deg)),
            _quat_axis_angle((0.0, 0.0, 1.0), yaw),
        )
        assert self._check(quat, yaw) is expected

    @pytest.mark.parametrize("off_deg,expected", [(14.0, True), (16.0, False)])
    def test_yaw_alignment_boundary(self, off_deg: float, expected: bool):
        yaw = 0.3
        quat = _quat_axis_angle((0.0, 0.0, 1.0), yaw + math.radians(off_deg))
        assert self._check(quat, yaw) is expected

    def test_alignment_is_mod_pi(self):
        # Thin side alignment is direction-free: a domino rotated pi about z
        # still counts as aligned.
        yaw = 0.7
        quat = _quat_axis_angle((0.0, 0.0, 1.0), yaw + math.pi)
        assert self._check(quat, yaw)


# ---------------------------------------------------------------------------
# ball_in_target_zone
# ---------------------------------------------------------------------------


class TestBallInTargetZone:
    def _at(self, pos, *, hole: bool) -> tuple[bool, float]:
        target = torch.tensor([[0.5, 0.0, 0.1]])
        at, d_xy = ball_in_target_zone(
            torch.tensor([pos]),
            target,
            radius=0.05,
            surface_z=0.1,
            is_hole=hole,
            ball_radius=0.02,
        )
        return bool(at[0]), float(d_xy[0])

    def test_marked_target_on_surface(self):
        at, d = self._at([0.52, 0.0, 0.12], hole=False)
        assert at and d == pytest.approx(0.02)

    def test_marked_target_rejects_airborne_ball(self):
        at, _ = self._at([0.5, 0.0, 0.3], hole=False)
        assert not at

    def test_hole_requires_drop_below_surface(self):
        assert self._at([0.5, 0.0, 0.05], hole=True)[0]
        assert not self._at([0.5, 0.0, 0.12], hole=True)[0]

    def test_outside_radius_fails(self):
        assert not self._at([0.6, 0.0, 0.1], hole=False)[0]


# ---------------------------------------------------------------------------
# SorterLatch
# ---------------------------------------------------------------------------


def _latch() -> SorterLatch:
    return SorterLatch(
        num_envs=1,
        num_sockets=1,
        device="cpu",
        hole_radius=torch.tensor([0.05]),
        piece_half=torch.tensor([0.03]),
        s_cross=torch.tensor([-0.05]),
        tilt_cos=math.cos(math.radians(30.0)),
    )


def _step(
    latch: SorterLatch, axial: float, lateral: float = 0.0, cos_tilt: float = 1.0
):
    one = torch.ones(1, 1)
    return latch.update(axial * one, lateral * one, cos_tilt * one)


class TestSorterLatch:
    def test_latches_on_crossing_and_holds_while_tumbling(self):
        latch = _latch()
        assert not _step(latch, 0.02)[0, 0]  # in channel, above the plane
        assert _step(latch, -0.06)[0, 0]  # crossed the plane
        # Piece tumbles around inside the hollow cube, far from the channel.
        assert _step(latch, -0.2, lateral=0.3, cos_tilt=0.1)[0, 0]

    def test_no_latch_without_prior_channel_step(self):
        # Appearing below the plane without an in-channel step above it (e.g.
        # entering through a foreign hole) must not latch.
        latch = _latch()
        assert not _step(latch, -0.06)[0, 0]

    def test_tilted_crossing_does_not_latch(self):
        latch = _latch()
        _step(latch, 0.02)
        assert not _step(latch, -0.06, cos_tilt=0.5)[0, 0]

    def test_reset_clears_latch_and_history(self):
        latch = _latch()
        _step(latch, 0.02)
        assert _step(latch, -0.06)[0, 0]
        latch.reset(torch.tensor([0]))
        assert not latch.latched[0, 0]
        # History cleared: the piece sitting below the plane cannot re-latch.
        assert not _step(latch, -0.06)[0, 0]


# ---------------------------------------------------------------------------
# SuccessGate
# ---------------------------------------------------------------------------


class TestSuccessGate:
    def _gate(self) -> SuccessGate:
        return SuccessGate(num_envs=2, device="cpu", vel_threshold=0.05)

    def test_requires_done_and_rest_and_release(self):
        gate = self._gate()
        done = torch.tensor([True, True])
        slow = torch.tensor([0.01, 0.01])
        fast = torch.tensor([0.01, 0.2])
        released = torch.tensor([True, True])
        held = torch.tensor([True, False])
        assert gate.update(done, fast, released).tolist() == [1.0, 0.0]
        assert gate.update(done, slow, held).tolist() == [1.0, 0.0]
        assert gate.update(~done, slow, released).tolist() == [0.0, 0.0]

    def test_multi_object_speed_reduces_over_max(self):
        gate = self._gate()
        done = torch.tensor([True, True])
        released = torch.tensor([True, True])
        speed = torch.tensor([[0.01, 0.2], [0.01, 0.02]])
        assert gate.update(done, speed, released).tolist() == [0.0, 1.0]

    def test_episode_latch_survives_later_failure_until_reset(self):
        gate = self._gate()
        done = torch.tensor([True, False])
        slow = torch.tensor([0.0, 0.0])
        released = torch.tensor([True, True])
        gate.update(done, slow, released)
        gate.update(torch.tensor([False, False]), slow, released)
        assert gate.episode_success.tolist() == [1.0, 0.0]
        gate.reset(torch.tensor([0]))
        assert gate.episode_success.tolist() == [0.0, 0.0]
        assert gate.success.tolist() == [0.0, 0.0]


# ---------------------------------------------------------------------------
# GripperOpenGate
# ---------------------------------------------------------------------------


def _fake_robot(names: list[str], joint_pos: torch.Tensor) -> Entity:
    """Duck-typed Entity stand-in: the gate only reads joint_names and
    data.joint_pos."""
    return cast(
        Entity,
        SimpleNamespace(joint_names=names, data=SimpleNamespace(joint_pos=joint_pos)),
    )


@dataclass(kw_only=True)
class _GateCfg(GripperGateCommandCfg):
    """Concrete stand-in: the gate reads only the two gripper fields."""

    def build(self, env: object) -> NoReturn:
        raise NotImplementedError


def _gate_cfg(joints: tuple[str, ...], threshold: float) -> GripperGateCommandCfg:
    return _GateCfg(
        resampling_time_range=(1.0, 1.0),
        gripper_joints=joints,
        gripper_open_threshold=threshold,
    )


class TestGripperOpenGate:
    def test_open_beyond_threshold(self):
        gate = GripperOpenGate(_gate_cfg(("gripper_left",), -0.015))
        robot = _fake_robot(
            ["q1", "gripper_left", "gripper_right"],
            torch.tensor([[0.0, -0.01, 0.0], [0.0, -0.02, 0.0]]),
        )
        assert gate(robot).tolist() == [True, False]

    def test_all_listed_joints_must_be_open(self):
        gate = GripperOpenGate(_gate_cfg(("left_g", "right_g"), -0.015))
        robot = _fake_robot(
            ["left_g", "right_g"],
            torch.tensor([[0.0, 0.0], [0.0, -0.02]]),
        )
        assert gate(robot).tolist() == [True, False]

    def test_unwired_cfg_rejected(self):
        # The env cfg leaves gripper_joints empty; manipulator wiring must
        # fill it before the command term is built.
        with pytest.raises(AssertionError):
            GripperOpenGate(_gate_cfg((), 0.0))

    def test_unknown_joint_raises(self):
        gate = GripperOpenGate(_gate_cfg(("nope",), 0.0))
        with pytest.raises(ValueError):
            gate(_fake_robot(["q1"], torch.zeros(1, 1)))


# ---------------------------------------------------------------------------
# stack kernels
# ---------------------------------------------------------------------------


class TestStackHeight:
    def test_top_heights_add_half_heights(self):
        pos_z = torch.tensor([[0.02, 0.06]])
        half_h = torch.tensor([0.02, 0.03])
        top = block_top_heights(pos_z, half_h)
        assert top[0].tolist() == pytest.approx([0.04, 0.09])

    @pytest.mark.parametrize(
        "top,expected",
        [(0.11, True), (0.13, True), (0.1099, False), (0.1301, False)],
    )
    def test_reached_boundary_at_tol(self, top: float, expected: bool):
        top_z = torch.tensor([[top]])
        assert bool(stack_reached(top_z, 0.0, 0.12, 0.01)[0, 0]) is expected

    def test_any_block_counts_regardless_of_position(self):
        # Height-only semantics: exactly the blocks whose top face sits in
        # the tol band count, wherever they are in xy; the rest do not.
        half_h = torch.full((3,), 0.02)
        pos_z = torch.tensor([[0.02, 0.02, 0.10]])  # 2 on the table, 1 on top
        reached = stack_reached(block_top_heights(pos_z, half_h), 0.0, 0.12, 0.01)
        assert reached.tolist() == [[False, False, True]]
        assert bool(reached.any(dim=-1)[0])

    def test_progress_clamps_to_unit_interval(self):
        assert stack_progress(torch.tensor([[0.20]]), 0.0, 0.12)[0] == 1.0
        assert stack_progress(torch.tensor([[-0.05]]), 0.0, 0.12)[0] == 0.0

    def test_progress_tracks_tallest_block(self):
        top_z = torch.tensor([[0.04, 0.08]])
        assert stack_progress(top_z, 0.0, 0.16)[0].item() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# two-stacks kernels
# ---------------------------------------------------------------------------


class TestTwoStacks:
    def _success(self, reached: list[bool], xy: list[list[float]]) -> bool:
        return bool(
            two_stacks_success(
                torch.tensor([reached]), torch.tensor([xy]), min_separation=0.06
            )[0]
        )

    @pytest.mark.parametrize("dist,expected", [(0.06, True), (0.059, False)])
    def test_separation_boundary(self, dist: float, expected: bool):
        xy = [[0.0, 0.0], [dist, 0.0]]
        assert self._success([True, True], xy) is expected

    def test_requires_two_reached_blocks(self):
        xy = [[0.0, 0.0], [0.2, 0.0]]
        assert not self._success([True, False], xy)
        assert self._success([True, True], xy)

    def test_progress_ignores_unseparated_second_stack(self):
        # Both height leaders within min_separation: one stack, second term 0.
        top_z = torch.tensor([[0.12, 0.12]])
        xy = torch.tensor([[[0.0, 0.0], [0.03, 0.0]]])
        p = two_stacks_progress(top_z, xy, 0.0, 0.12, 0.06)
        assert p[0].item() == pytest.approx(0.5)

    def test_progress_mean_of_two_best_separated_leaders(self):
        top_z = torch.tensor([[0.12, 0.06, 0.03]])
        xy = torch.tensor([[[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]])
        p = two_stacks_progress(top_z, xy, 0.0, 0.12, 0.06)
        assert p[0].item() == pytest.approx(0.75)

    def test_progress_clamps_at_two_full_stacks(self):
        top_z = torch.tensor([[0.15, 0.14]])
        xy = torch.tensor([[[0.0, 0.0], [0.1, 0.0]]])
        p = two_stacks_progress(top_z, xy, 0.0, 0.12, 0.06)
        assert p[0].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# domino_fallen
# ---------------------------------------------------------------------------


class TestDominoFallen:
    def _fallen(self, tilt_deg: float) -> bool:
        quat = _quat_axis_angle((1.0, 0.0, 0.0), math.radians(tilt_deg)).view(1, 1, 4)
        return bool(domino_fallen(quat, tilt_deg=45.0)[0, 0])

    def test_standing_domino_is_not_fallen(self):
        assert not self._fallen(0.0)

    @pytest.mark.parametrize("tilt_deg,expected", [(44.0, False), (46.0, True)])
    def test_tilt_boundary_at_45_degrees(self, tilt_deg: float, expected: bool):
        assert self._fallen(tilt_deg) is expected

    def test_fallen_is_direction_free(self):
        # Toppling either way about any horizontal axis counts the same.
        quat = _quat_axis_angle((0.0, 1.0, 0.0), math.radians(90.0)).view(1, 1, 4)
        assert bool(domino_fallen(quat, tilt_deg=45.0)[0, 0])


# ---------------------------------------------------------------------------
# shape_sorter: no_rotation allowed-face sets
# ---------------------------------------------------------------------------


class TestNoRotationFaces:
    def _allowed(self, name: str) -> set[int]:
        return set(allowed_faces(SHAPE_SORTER_VARIANTS[name].shapes))

    def test_allowed_faces_shrink_and_nest_with_shape_count(self):
        a3 = self._allowed("ShapeSorter3NoRotation")
        a4 = self._allowed("ShapeSorter4NoRotation")
        a5 = self._allowed("ShapeSorter5NoRotation")
        assert a5 <= a4 <= a3
        assert len(a5) < len(a4) < len(a3)

    def test_allowed_faces_never_empty(self):
        # The star socket is never a required shape, so its face index is
        # always allowed, however many shapes the variant needs.
        assert self._allowed("ShapeSorter5NoRotation")
