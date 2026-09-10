"""Unit tests for the success criteria in the per-category ``mdp/success.py``
modules, the shared ``SuccessGate`` in ``tasks/common/success.py``, and the
contact release gate in ``tasks/common/commands.py``."""

import math
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.sensor import ContactSensor

from loot_bench.tasks.ball.mdp.success import ball_in_target_zone
from loot_bench.tasks.common.commands import (
    MIN_GRIP_FORCE,
    ContactReleaseGate,
)
from loot_bench.tasks.common.success import SuccessGate
from loot_bench.tasks.domino.mdp.success import (
    chain_topple_success,
    domino_fallen_along_row,
    domino_upright_aligned,
)
from loot_bench.tasks.place.mdp.success import place_progress, place_reached
from loot_bench.tasks.shape_sorter.mdp.success import SorterLatch
from loot_bench.tasks.shape_sorter.variants import SHAPE_SORTER_VARIANTS, allowed_faces
from loot_bench.tasks.stack.mdp.success import (
    block_top_heights,
    stack_progress,
    stack_reached,
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

    @pytest.mark.parametrize(
        "tilt_deg,expected",
        [(14.0, True), (16.0, False), (166.0, True), (164.0, False)],
    )
    def test_tilt_boundary(self, tilt_deg: float, expected: bool):
        # Tilt about the row direction keeps the thin side aligned, so only
        # the upright check decides. The band is symmetric about both poles:
        # a domino tilted near 180 deg is standing upside-down, not fallen.
        yaw = math.pi / 2
        row_axis = (math.cos(yaw), math.sin(yaw), 0.0)
        quat = _quat_mul(
            _quat_axis_angle(row_axis, math.radians(tilt_deg)),
            _quat_axis_angle((0.0, 0.0, 1.0), yaw),
        )
        assert self._check(quat, yaw) is expected

    def test_upside_down_passes(self):
        # A domino has no visual/physical top-bottom distinction, so a clean
        # 180 deg flip about the row-aligned axis still counts as standing.
        yaw = math.pi / 2
        row_axis = (math.cos(yaw), math.sin(yaw), 0.0)
        quat = _quat_mul(
            _quat_axis_angle(row_axis, math.pi),
            _quat_axis_angle((0.0, 0.0, 1.0), yaw),
        )
        assert self._check(quat, yaw)

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
# place_reached / place_progress
# ---------------------------------------------------------------------------


class TestPlaceReached:
    def _at(self, dist: float, threshold: float = 0.02) -> bool:
        return bool(place_reached(torch.tensor([dist]), threshold)[0])

    def test_inside_threshold(self):
        assert self._at(0.019)

    def test_threshold_is_exclusive(self):
        assert not self._at(0.02)

    def test_outside_threshold(self):
        assert not self._at(0.05)

    def test_margin_widens_the_tolerance(self):
        assert not self._at(0.03)
        assert self._at(0.03, threshold=0.02 * 2.0)


class TestPlaceProgress:
    def _p(self, dist: float, max_dist: float = 0.4) -> float:
        return float(place_progress(torch.tensor([dist]), max_dist)[0])

    def test_one_at_the_target(self):
        assert self._p(0.0) == pytest.approx(1.0)

    def test_zero_at_max_travel(self):
        assert self._p(0.4) == pytest.approx(0.0)

    def test_linear_between(self):
        assert self._p(0.1) == pytest.approx(0.75)

    def test_clamped_beyond_max_travel(self):
        # A cube knocked outside the workspace cannot score negative progress,
        # which would drive the reward below -1.
        assert self._p(1.0) == pytest.approx(0.0)


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
# ContactReleaseGate
# ---------------------------------------------------------------------------


def _fake_contact_sensor(force: torch.Tensor) -> ContactSensor:
    """Duck-typed ContactSensor stand-in: the gate only reads data.force."""
    return cast(ContactSensor, SimpleNamespace(data=SimpleNamespace(force=force)))


class TestContactReleaseGate:
    def test_released_when_no_contact(self):
        gate = ContactReleaseGate(min_force=0.5)
        force = torch.zeros(2, 2, 3)  # (E=2, P=2 pads, 3)
        assert gate(_fake_contact_sensor(force)).tolist() == [True, True]

    def test_not_released_when_any_pad_over_threshold(self):
        gate = ContactReleaseGate(min_force=0.5)
        force = torch.zeros(2, 2, 3)
        force[0, 1] = torch.tensor([1.0, 0.0, 0.0])  # env 0, right pad: 1 N
        assert gate(_fake_contact_sensor(force)).tolist() == [False, True]

    def test_boundary_at_threshold_counts_as_released(self):
        gate = ContactReleaseGate(min_force=0.5)
        force = torch.zeros(1, 2, 3)
        force[0, 0] = torch.tensor([0.5, 0.0, 0.0])  # exactly at threshold
        assert gate(_fake_contact_sensor(force)).tolist() == [True]

    def test_default_threshold_is_the_grip_force(self):
        """The gate ships reading the env's release threshold, not a copy."""
        gate = ContactReleaseGate()
        loaded = torch.zeros(1, 2, 3)
        loaded[0, 0] = torch.tensor([MIN_GRIP_FORCE + 1e-3, 0.0, 0.0])
        assert gate(_fake_contact_sensor(loaded)).tolist() == [False]
        freed = torch.zeros(1, 2, 3)
        freed[0, 0] = torch.tensor([MIN_GRIP_FORCE, 0.0, 0.0])
        assert gate(_fake_contact_sensor(freed)).tolist() == [True]

    def test_reduction_is_the_max_over_pads(self):
        """One loaded pad vetoes release, however lightly the other rests."""
        gate = ContactReleaseGate()
        force = torch.zeros(1, 3, 3)  # (E=1, P=3 pads, 3)
        force[0, 2] = torch.tensor([0.0, MIN_GRIP_FORCE + 1.0, 0.0])
        assert gate(_fake_contact_sensor(force)).tolist() == [False]


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

    def test_progress_excludes_held_block(self):
        # The tallest block is held in the gripper: ignored, so progress
        # reflects the tallest *resting* block, not the lifted one.
        top_z = torch.tensor([[0.04, 0.16]])
        exclude = torch.tensor([[False, True]])
        assert stack_progress(top_z, 0.0, 0.16)[0].item() == pytest.approx(1.0)
        p = stack_progress(top_z, 0.0, 0.16, exclude=exclude)
        assert p[0].item() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# domino_fallen_along_row / chain_topple_success
# ---------------------------------------------------------------------------


class TestDominoFallenAlongRow:
    def _fallen(self, quat, row_yaw=0.0, tilt_deg=45.0):
        yaw = torch.tensor([row_yaw])
        return domino_fallen_along_row(quat.view(1, -1, 4), yaw, tilt_deg)

    def test_standing_is_not_fallen(self):
        fallen, sign = self._fallen(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        assert not fallen.any() and (sign == 0).all()

    def test_fallen_forward_along_row(self):
        # Row along +x; tip 80 deg about the y-axis -> leans toward +x.
        q = _quat_axis_angle((0.0, 1.0, 0.0), math.radians(80.0))
        fallen, sign = self._fallen(q.view(1, 1, 4))
        assert fallen.all() and (sign == 1.0).all()

    def test_fallen_backward_along_row(self):
        q = _quat_axis_angle((0.0, 1.0, 0.0), math.radians(-80.0))
        fallen, sign = self._fallen(q.view(1, 1, 4))
        assert fallen.all() and (sign == -1.0).all()

    def test_fallen_sideways_does_not_count(self):
        # Tipped past 45 deg but perpendicular to the row: the old bug.
        q = _quat_axis_angle((1.0, 0.0, 0.0), math.radians(80.0))
        fallen, sign = self._fallen(q.view(1, 1, 4))
        assert not fallen.any() and (sign == 0).all()

    def test_row_yaw_rotates_the_along_axis(self):
        # Same sideways tip counts once the row itself runs along y.
        q = _quat_axis_angle((1.0, 0.0, 0.0), math.radians(80.0))
        fallen, sign = self._fallen(q.view(1, 1, 4), row_yaw=math.pi / 2)
        assert fallen.all() and (sign == -1.0).all()

    def test_upside_down_is_not_fallen(self):
        q = _quat_axis_angle((1.0, 0.0, 0.0), math.pi)
        fallen, sign = self._fallen(q.view(1, 1, 4))
        assert not fallen.any() and (sign == 0).all()

    def test_near_upside_down_is_not_fallen(self):
        torch.manual_seed(0)
        n = 2000
        phi = torch.rand(n) * 2 * math.pi
        eps = (torch.rand(n) * 2 - 1) * math.radians(
            1.0
        )  # up to +/-1 deg off horizontal
        axis = torch.stack(
            [
                torch.cos(phi) * torch.cos(eps),
                torch.sin(phi) * torch.cos(eps),
                torch.sin(eps),
            ],
            dim=-1,
        )
        half = math.pi / 2  # 180 deg flip
        quat = torch.cat(
            [torch.full((n, 1), math.cos(half)), axis * math.sin(half)], dim=-1
        )
        fallen, sign = domino_fallen_along_row(
            quat.view(1, n, 4), torch.zeros(1), tilt_deg=45.0
        )
        assert not fallen.any() and (sign == 0).all()


class TestChainToppleSuccess:
    def _run(self, signs, order):
        fallen = torch.tensor([[s != 0 for s in signs]])
        sign = torch.tensor([signs], dtype=torch.float32)
        return chain_topple_success(
            fallen, sign, torch.tensor([order], dtype=torch.float32)
        )

    def test_forward_chain_in_order_succeeds(self):
        assert self._run([1, 1, 1], [3.0, 5.0, 8.0]).all()

    def test_backward_chain_in_reverse_order_succeeds(self):
        assert self._run([-1, -1, -1], [8.0, 5.0, 3.0]).all()

    def test_simultaneous_falls_allowed(self):
        assert self._run([1, 1, 1], [3.0, 3.0, 3.0]).all()

    def test_out_of_order_fails(self):
        assert not self._run([1, 1, 1], [5.0, 3.0, 8.0]).any()

    def test_mixed_directions_fail(self):
        assert not self._run([1, -1, 1], [3.0, 5.0, 8.0]).any()

    def test_not_all_fallen_fails(self):
        assert not self._run([1, 1, 0], [3.0, 5.0, 1e9]).any()


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
