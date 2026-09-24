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
from loot_bench.tasks.sorter.mdp.success import SorterLatch
from loot_bench.tasks.sorter.variants import SORTER_VARIANTS, allowed_faces
from loot_bench.tasks.stack.mdp.success import (
    block_is_horizontal,
    pi_levels_complete,
    pi_progress,
)
from loot_bench.tasks.stack.variants import (
    BLOCK_HALF_EXTENTS,
    BLOCK_LENGTH,
    BLOCK_WIDTH,
    HEIGHT_TOL,
    LEVEL_HEIGHT,
    PI_VARIANTS,
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
    )


def _step(latch: SorterLatch, axial: float, lateral: float = 0.0):
    one = torch.ones(1, 1)
    return latch.update(axial * one, lateral * one)


class TestSorterLatch:
    def test_latches_on_crossing_and_holds_while_tumbling(self):
        latch = _latch()
        assert not _step(latch, 0.02)[0, 0]  # in channel, above the plane
        assert _step(latch, -0.06)[0, 0]  # crossed the plane
        # Piece tumbles around inside the hollow cube, far from the channel.
        assert _step(latch, -0.2, lateral=0.3)[0, 0]

    def test_no_latch_without_prior_channel_step(self):
        # Appearing below the plane without an in-channel step above it (e.g.
        # entering through a foreign hole) must not latch.
        latch = _latch()
        assert not _step(latch, -0.06)[0, 0]

    def test_tilted_crossing_latches(self):
        # Orientation is intentionally unchecked: centered + fully deep +
        # channel approach already proves hole passage (a centroid cannot get
        # there without going through). A corner-first shove that ends up
        # fully inside must count.
        latch = _latch()
        _step(latch, 0.02)
        assert _step(latch, -0.06)[0, 0]

    def test_reset_clears_latch_and_history(self):
        latch = _latch()
        _step(latch, 0.02)
        assert _step(latch, -0.06)[0, 0]
        latch.reset(torch.tensor([0]))
        assert not latch.latched[0, 0]
        # History cleared: the piece sitting below the plane cannot re-latch.
        assert not _step(latch, -0.06)[0, 0]

    def test_near_miss_outside_channel_does_not_latch(self):
        # Just outside the hole radius on the approach step: the crossing
        # step never counts, however deep it goes.
        latch = _latch()
        assert not _step(latch, 0.02, lateral=0.051)[0, 0]
        assert not _step(latch, -0.06)[0, 0]

    def test_near_miss_above_window_does_not_latch(self):
        # Above the channel's top edge (piece_half + margin) on the approach
        # step: not in-channel, so the crossing step cannot latch.
        latch = _latch()
        assert not _step(latch, 0.05)[0, 0]  # 0.03 + 0.01 = 0.04 edge
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
# pi kernels (the stack category)
# ---------------------------------------------------------------------------

_UP = torch.tensor([1.0, 0.0, 0.0, 0.0])
_L1 = PI_VARIANTS["Pi1"].level_centre_height(1)  # 0.075
_LEVELS3 = torch.tensor(PI_VARIANTS["Pi3"].level_centre_heights)


def _lintel(tilt_deg: float = 0.0) -> torch.Tensor:
    """A block lying along x (90 deg about y), pitched a further *tilt_deg*.

    Pitching about y tips the long axis out of the horizontal plane by
    exactly *tilt_deg*; a roll about the long axis itself would not.
    """
    return _quat_axis_angle((0.0, 1.0, 0.0), math.pi / 2 + math.radians(tilt_deg))


def _levels(
    heights: list[float], quats: list[torch.Tensor], exclude=None
) -> torch.Tensor:
    """(K,) bool for one env, K = 3 levels, at the shipped tolerance."""
    return pi_levels_complete(
        torch.tensor([heights]),
        torch.stack(quats).unsqueeze(0),
        0.0,
        _LEVELS3,
        HEIGHT_TOL,
        exclude=None if exclude is None else torch.tensor([exclude]),
    )[0]


class TestBlockIsHorizontal:
    def test_upright_is_not_horizontal_and_flat_is(self):
        assert not block_is_horizontal(_UP.view(1, 1, 4))[0, 0]
        assert block_is_horizontal(_lintel().view(1, 1, 4))[0, 0]

    @pytest.mark.parametrize("tilt_deg,expected", [(4.0, True), (8.0, False)])
    def test_gate_is_five_point_seven_degrees_not_fifteen(self, tilt_deg, expected):
        # The old 15 deg boundary would have passed 8 deg. This is the test
        # that catches a revert of TILT_COS_MAX to the other categories' 0.26.
        assert bool(block_is_horizontal(_lintel(tilt_deg).view(1, 1, 4))[0, 0]) is (
            expected
        )


class TestPiLevelsComplete:
    def test_trilithon_scores(self):
        hz = BLOCK_HALF_EXTENTS[2]
        levels = _levels([hz, hz, _L1], [_UP, _UP, _lintel()])
        assert levels.tolist() == [True, False, False]

    def test_t_scores_too_the_criterion_encodes_no_structure(self):
        # Deliberately a near-copy of the trilithon case: one upright, same
        # lintel. Both build forms are legal (ADR-020); do not delete this
        # as a duplicate.
        hz = BLOCK_HALF_EXTENTS[2]
        levels = _levels([hz, _L1], [_UP, _lintel()])
        assert levels.tolist() == [True, False, False]

    def test_upright_at_the_target_height_does_not_score(self):
        levels = _levels([_L1], [_UP])
        assert not levels.any()

    def test_height_tolerance_is_inclusive_and_tight(self):
        assert _levels([_L1 + HEIGHT_TOL - 1e-6], [_lintel()])[0]
        assert not _levels([_L1 + HEIGHT_TOL + 1e-4], [_lintel()])[0]

    def test_held_lintel_earns_nothing_until_released(self):
        # ADR-017: the same block, same pose; only the mask changes.
        assert not _levels([_L1], [_lintel()], exclude=[True])[0]
        assert _levels([_L1], [_lintel()], exclude=[False])[0]

    def test_two_pile_near_miss_is_rejected_on_tilt_not_height(self):
        # The PRD's Family-1 case: three flats (60 mm) and one upright
        # (65 mm) carry a lintel at their mean, 2.5 mm under level 1, tilted
        # by atan(5 / 45) = 6.34 deg over the 45 mm span. The centre is
        # inside HEIGHT_TOL, so a height-only criterion would have scored
        # it; the tilt gate is what refuses it.
        tilt = math.degrees(math.atan(0.005 / (BLOCK_LENGTH - BLOCK_WIDTH)))
        assert tilt == pytest.approx(6.34, abs=0.01)
        centre = _L1 - 0.0025
        assert abs(centre - _L1) <= HEIGHT_TOL
        assert not block_is_horizontal(_lintel(tilt).view(1, 1, 4))[0, 0]
        assert not _levels([centre], [_lintel(tilt)]).any()

    def test_progress_drops_on_collapse(self):
        hz = BLOCK_HALF_EXTENTS[2]
        l2 = float(_LEVELS3[1])
        standing = _levels(
            [hz, _L1, LEVEL_HEIGHT + hz, l2], [_UP, _lintel(), _UP, _lintel()]
        )
        assert pi_progress(standing.unsqueeze(0))[0].item() == pytest.approx(2 / 3)
        # Level 1's lintel is knocked off; level 2's happens to still stand.
        collapsed = _levels(
            [hz, hz, LEVEL_HEIGHT + hz, l2], [_UP, _lintel(), _UP, _lintel()]
        )
        assert pi_progress(collapsed.unsqueeze(0))[0].item() == pytest.approx(1 / 3)

    def test_success_needs_all_levels_not_just_the_top(self):
        l3 = float(_LEVELS3[2])
        top_only = _levels([l3], [_lintel()])
        assert top_only.tolist() == [False, False, True]
        assert not top_only.all()
        every = _levels([_L1, float(_LEVELS3[1]), l3], [_lintel()] * 3)
        assert every.all()

    def test_table_height_is_subtracted_per_env(self):
        table_z = torch.tensor([0.0, 1.0])
        centre_z = torch.tensor([[_L1], [1.0 + _L1]])
        quat = _lintel().view(1, 1, 4).expand(2, 1, 4)
        levels = pi_levels_complete(centre_z, quat, table_z, _LEVELS3, HEIGHT_TOL)
        assert levels[:, 0].tolist() == [True, True]


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
# sorter: no_rotation allowed-face sets
# ---------------------------------------------------------------------------


class TestNoRotationFaces:
    def _allowed(self, name: str) -> set[int]:
        return set(allowed_faces(SORTER_VARIANTS[name].shapes))

    def test_allowed_faces_shrink_and_nest_with_shape_count(self):
        a2 = self._allowed("Sorter2NoRotation")
        a3 = self._allowed("Sorter3NoRotation")
        a4 = self._allowed("Sorter4NoRotation")
        a5 = self._allowed("Sorter5NoRotation")
        assert a5 <= a4 <= a3 <= a2
        assert len(a5) < len(a4) < len(a3) < len(a2)

    def test_allowed_faces_never_empty(self):
        # The star socket is never a required shape, so its face index is
        # always allowed, however many shapes the variant needs.
        assert self._allowed("Sorter5NoRotation")


# ---------------------------------------------------------------------------
# sorter variants: hardest count, tail-extension, single-source tolerances
# ---------------------------------------------------------------------------


class TestSorterVariants:
    def test_hardest_count_matches_largest_variant(self):
        from loot_bench.tasks.sorter.variants import HARDEST_COUNT

        assert HARDEST_COUNT == 5
        assert max(v.num_pieces for v in SORTER_VARIANTS.values()) == HARDEST_COUNT

    def test_shapes_tail_extend(self):
        shapes5 = SORTER_VARIANTS["Sorter5Full"].shapes
        for name in ("Sorter2Full", "Sorter3Full", "Sorter4Full"):
            shapes = SORTER_VARIANTS[name].shapes
            assert shapes == shapes5[: len(shapes)]

    def test_rejects_non_prefix_shapes(self):
        from loot_bench.tasks.sorter.variants import SorterVariantCfg

        with pytest.raises(AssertionError):
            SorterVariantCfg(
                name="Bad",
                shapes=("triangle", "circle", "rect"),  # type: ignore[list-item]
                spawn="full",
            )

    def test_tolerances_are_single_sourced(self):
        """The command term reads the variants module's tolerances."""
        from loot_bench.tasks.sorter.mdp import commands as sorter_commands
        from loot_bench.tasks.sorter.variants import HOLE_SLACK

        assert sorter_commands.SorterCommandCfg.hole_slack == HOLE_SLACK
        assert pytest.approx(0.005) == HOLE_SLACK
