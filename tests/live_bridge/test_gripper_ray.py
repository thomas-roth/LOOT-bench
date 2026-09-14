"""Unit tests for the gripper-laser down projection (live_record).

Covers the two reported screenshots:
- off-centre held block: the footprint must land under the *object*, not the
  grasp site (heldObjectToProjectionOffset).
- imperfectly gripped (tilted / beside-the-beam) block: the projection must
  still appear via the pads' grip contact (NotPerfectlyGrasped...).
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from loot_bench.scripts.live_bridge import live_record
from loot_bench.scripts.live_bridge.live_record import (
    _cast_gripper_ray,
    _find_held_body,
    _find_pad_geom_ids,
    _find_sorter_body_id,
    _point_box_surface_dist,
    _ray_box_param,
)

_MJUCO = live_record._mujoco
BOX = int(_MJUCO.mjtGeom.mjGEOM_BOX.value)
GEOM_OBJ = _MJUCO.mjtObj.mjOBJ_GEOM
BODY_OBJ = _MJUCO.mjtObj.mjOBJ_BODY

ROBOT_BODY, HELD_BODY, TABLE_BODY, STACK_BODY, SORTER_BODY = 1, 3, 7, 9, 11
LPAD_GID, RPAD_GID, HELD_GID, TABLE_GID, STACK_GID, SORTER_GID = 1, 2, 10, 20, 30, 40

HELD_HALF = (0.010, 0.010, 0.0325)
HELD_TOP, HELD_BOTTOM = 0.30, 0.28
STACK_TOP, STACK_BOTTOM = 0.20, 0.18
SORTER_HALF = (0.05, 0.05, 0.05)


def _ray_box_dist(pnt, vec, center, half):
    """Nearest non-negative ray distance to an axis-aligned box, else None."""
    t_min, t_max = 0.0, float("inf")
    for i in range(3):
        if abs(vec[i]) < 1e-12:
            if abs(pnt[i] - center[i]) > half[i]:
                return None
        else:
            t1 = (center[i] - half[i] - pnt[i]) / vec[i]
            t2 = (center[i] + half[i] - pnt[i]) / vec[i]
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
    if t_max < t_min:
        return None
    return t_min


def _contact(g1, g2, dist=-1e-4):
    return SimpleNamespace(geom1=g1, geom2=g2, dist=dist)


class _SlabWorld:
    """Fake MjModel/MjData: slabs with xy extents hit by the down ray."""

    def __init__(
        self,
        held_xy=(0.0, 0.0),
        with_stack=False,
        contacts=(),
        bury_once=False,
        sorter_xy=None,
        sorter_z=0.29,
        sorter_mat=None,
    ):
        hx, hy = HELD_HALF[0], HELD_HALF[1]
        self.slabs = [(HELD_TOP, HELD_BOTTOM, held_xy[0], held_xy[1], hx, hy, HELD_GID)]
        if with_stack:
            self.slabs.append((STACK_TOP, STACK_BOTTOM, 0.0, 0.0, hx, hy, STACK_GID))
        self._bury_once = bury_once
        self._buried = False
        self.sorter_xy = sorter_xy
        ngeom = SORTER_GID + 1
        geom_bodyid = [0] * ngeom
        geom_bodyid[LPAD_GID] = ROBOT_BODY
        geom_bodyid[RPAD_GID] = ROBOT_BODY
        geom_bodyid[HELD_GID] = HELD_BODY
        geom_bodyid[TABLE_GID] = TABLE_BODY
        geom_bodyid[STACK_GID] = STACK_BODY
        geom_bodyid[SORTER_GID] = SORTER_BODY
        geom_size: list[np.ndarray[Any, Any]] = [np.zeros(3)] * ngeom
        geom_size[HELD_GID] = np.array(HELD_HALF)
        geom_size[TABLE_GID] = np.array([0.5, 0.5, 0.05])
        geom_size[STACK_GID] = np.array(HELD_HALF)
        geom_size[SORTER_GID] = np.array(SORTER_HALF)
        geom_type = [BOX] * ngeom
        self.mjm = SimpleNamespace(
            ngeom=ngeom,
            nbody=12,
            geom_bodyid=geom_bodyid,
            geom_size=geom_size,
            geom_type=geom_type,
        )
        xpos: dict[int, np.ndarray[Any, Any]] = {b: np.zeros(3) for b in range(12)}
        xpos[HELD_BODY] = np.array([held_xy[0], held_xy[1], 0.29])
        xpos[TABLE_BODY] = np.array([0.0, 0.0, -0.05])
        xpos[STACK_BODY] = np.array([0.0, 0.0, 0.19])
        if sorter_xy is not None:
            xpos[SORTER_BODY] = np.array([sorter_xy[0], sorter_xy[1], sorter_z])
        else:
            xpos[SORTER_BODY] = np.array([5.0, 5.0, sorter_z])
        self.mjd = SimpleNamespace(
            xpos=xpos,
            xmat={b: np.eye(3).reshape(-1) for b in range(12)},
            ncon=len(contacts),
            contact=list(contacts),
        )
        if sorter_mat is not None:
            self.mjd.xmat[SORTER_BODY] = np.asarray(sorter_mat, dtype=float).reshape(-1)
        self.geom_names = {
            LPAD_GID: "robot/finger_left_pad_collision",
            RPAD_GID: "robot/finger_right_pad_collision",
            HELD_GID: "cube/block_geom",
            TABLE_GID: "table/surface_geom",
            STACK_GID: "cube/block_1_geom",
            SORTER_GID: "cube/sorter_geom",
        }
        self.body_names = {
            ROBOT_BODY: "robot/finger",
            HELD_BODY: "cube/block_0",
            TABLE_BODY: "table/surface",
            STACK_BODY: "cube/block_1",
            SORTER_BODY: "cube/sorter",
        }

    def set_held_centre(self, pos):
        self.mjd.xpos[HELD_BODY] = np.array(pos)

    # Fake mujoco functions -------------------------------------------------
    def mj_ray(self, mjm, mjd, pnt, vec, *args):
        del mjm, mjd
        geomid = args[-1]
        v = np.array([float(vec[0]), float(vec[1]), float(vec[2])])
        if abs(v[2]) < 0.5:
            # Horizontal ray: intersect the held and sorter boxes.
            p = np.array([float(pnt[0]), float(pnt[1]), float(pnt[2])])
            best = None
            held_c = np.asarray(self.mjd.xpos[HELD_BODY], dtype=float)
            for center, half, gid in (
                (held_c, np.array(HELD_HALF), HELD_GID),
                (
                    np.asarray(self.mjd.xpos[SORTER_BODY], dtype=float),
                    np.array(SORTER_HALF),
                    SORTER_GID,
                ),
            ):
                dist = _ray_box_dist(p, v, center, half)
                if dist is not None and (best is None or dist < best[0]):
                    best = (dist, gid)
            if best is None:
                return -1.0
            geomid[0] = best[1]
            return best[0]
        x, y, z = float(pnt[0]), float(pnt[1]), float(pnt[2])
        best = None
        for top, bottom, cx, cy, hx, hy, gid in self.slabs:
            if abs(x - cx) > hx or abs(y - cy) > hy:
                continue
            if z > top:
                cand = z - top
            elif z > bottom:
                if self._bury_once and not self._buried:
                    self._buried = True
                    geomid[0] = gid
                    return 0.0
                cand = z - bottom
            else:
                continue
            if best is None or cand < best[0]:
                best = (cand, gid)
        if z > 0.0 and (best is None or z < best[0]):
            best = (z, TABLE_GID)
        if best is None:
            return -1.0
        geomid[0] = best[1]
        return best[0]

    def mj_id2name(self, mjm, objtype, idx):
        del mjm
        idx = int(idx)
        if objtype == GEOM_OBJ:
            return self.geom_names.get(idx, "")
        if objtype == BODY_OBJ:
            return self.body_names.get(idx, "")
        return ""


def _install(monkeypatch, world):
    monkeypatch.setattr(_MJUCO, "mj_ray", world.mj_ray)
    monkeypatch.setattr(_MJUCO, "mj_id2name", world.mj_id2name)


def test_pad_geom_lookup(monkeypatch):
    world = _SlabWorld()
    _install(monkeypatch, world)
    assert sorted(_find_pad_geom_ids(world.mjm)) == [LPAD_GID, RPAD_GID]


def test_perfect_grip_projects_at_grasp_xz(monkeypatch):
    world = _SlabWorld(contacts=(_contact(LPAD_GID, HELD_GID),))
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.35])
    assert _find_held_body(world.mjm, world.mjd, origin, [LPAD_GID]) == HELD_BODY
    hit, held_half, _, beam_top = _cast_gripper_ray(
        world.mjm, world.mjd, origin, [LPAD_GID]
    )
    assert hit is not None and tuple(hit) == (0.0, 0.0, 0.0)
    assert held_half is not None and np.allclose(held_half, HELD_HALF)
    assert tuple(beam_top) == (0.0, 0.0, 0.35)


def test_off_centre_grip_projects_under_object(monkeypatch):
    """heldObjectToProjectionOffset: end-held block sticks out sideways."""
    world = _SlabWorld(held_xy=(0.06, 0.0), contacts=(_contact(LPAD_GID, HELD_GID),))
    world.set_held_centre((0.06, 0.0, 0.29))
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.35])
    hit, held_half, _, beam_top = _cast_gripper_ray(
        world.mjm, world.mjd, origin, [LPAD_GID]
    )
    assert hit is not None
    assert hit[0] == 0.06 and hit[2] == 0.0, f"footprint not under object: {hit}"
    assert held_half is not None
    assert beam_top[0] == 0.06


def test_imperfect_grip_beside_beam_still_projects(monkeypatch):
    """NotPerfectlyGrasped: tilted block hangs beside the grasp-site ray."""
    world = _SlabWorld(held_xy=(0.05, 0.02), contacts=(_contact(RPAD_GID, HELD_GID),))
    world.set_held_centre((0.05, 0.02, 0.27))  # outside the reach ball
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.35])
    assert float(np.linalg.norm(world.mjd.xpos[HELD_BODY] - origin)) > 0.08
    # The grasp-site ray alone misses the block entirely...
    assert _find_held_body(world.mjm, world.mjd, origin, pad_geom_ids=None) is None
    # ...but the pad contact still identifies it for the projection.
    hit, held_half, _, beam_top = _cast_gripper_ray(
        world.mjm, world.mjd, origin, [LPAD_GID, RPAD_GID]
    )
    assert hit is not None and hit[2] == 0.0, "no projection for gripped block"
    assert held_half is not None, "no footprint for gripped block"
    assert (beam_top[0], beam_top[1]) == (0.05, 0.02)


def test_proximity_fallback_without_contacts(monkeypatch):
    world = _SlabWorld()
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.35])
    assert _find_held_body(world.mjm, world.mjd, origin, []) == HELD_BODY
    hit, held_half, _, _ = _cast_gripper_ray(world.mjm, world.mjd, origin, [])
    assert hit is not None and hit[2] == 0.0
    assert held_half is not None


def test_empty_gripper_stops_on_resting_block(monkeypatch):
    world = _SlabWorld()
    world.set_held_centre((0.5, 0.5, 0.28))  # slab stays, centre far away
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.65])
    hit, held_half, _, beam_top = _cast_gripper_ray(world.mjm, world.mjd, origin, [])
    assert hit is not None and hit[2] == HELD_TOP
    assert held_half is None
    assert tuple(beam_top) == tuple(origin)


def test_pads_touching_table_is_not_held(monkeypatch):
    world = _SlabWorld(contacts=(_contact(LPAD_GID, TABLE_GID),))
    _install(monkeypatch, world)
    world.set_held_centre((0.5, 0.5, 0.28))
    origin = np.array([0.0, 0.0, 0.65])
    assert _find_held_body(world.mjm, world.mjd, origin, [LPAD_GID]) is None
    hit, held_half, _, _ = _cast_gripper_ray(world.mjm, world.mjd, origin, [LPAD_GID])
    assert hit is not None and held_half is None


def test_buried_projection_origin_exits_block(monkeypatch):
    world = _SlabWorld(
        held_xy=(0.06, 0.0),
        contacts=(_contact(LPAD_GID, HELD_GID),),
        bury_once=True,
    )
    world.set_held_centre((0.06, 0.0, 0.29))
    _install(monkeypatch, world)
    hit, held_half, _, _ = _cast_gripper_ray(
        world.mjm, world.mjd, np.array([0.06, 0.0, 0.29]), [LPAD_GID]
    )
    assert hit is not None and hit[2] == 0.0
    assert held_half is not None


def test_stack_below_held_stops_at_stack(monkeypatch):
    world = _SlabWorld(with_stack=True, contacts=(_contact(LPAD_GID, HELD_GID),))
    _install(monkeypatch, world)
    hit, held_half, _, _ = _cast_gripper_ray(
        world.mjm, world.mjd, np.array([0.0, 0.0, 0.35]), [LPAD_GID]
    )
    assert hit is not None and hit[2] == STACK_TOP, "ray must stop on the stack"
    assert held_half is not None and np.allclose(held_half, HELD_HALF)


def test_sorter_body_lookup_by_suffix(monkeypatch):
    world = _SlabWorld()
    _install(monkeypatch, world)
    assert _find_sorter_body_id(world.mjm) == SORTER_BODY


def test_empty_gripper_stays_vertical_near_cube(monkeypatch):
    world = _SlabWorld(sorter_xy=(0.2, 0.0))
    world.set_held_centre((0.5, 0.5, 0.28))  # slab stays, centre far away
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.35])
    assert _find_held_body(world.mjm, world.mjd, origin, []) is None
    hit, held_half, _, beam_top = _cast_gripper_ray(
        world.mjm, world.mjd, origin, [], SORTER_BODY
    )
    assert hit is not None and hit[2] == HELD_TOP, "empty gripper must stay vertical"
    assert held_half is None
    assert tuple(beam_top) == tuple(origin)


def test_held_cube_closer_goes_horizontal_through_object(monkeypatch):
    # Sorter centre (0.2, 0, 0.29), gripper at cube height in the plus:
    # closest face ~0.15 m away, closer than the floor 0.32 m below.
    world = _SlabWorld(contacts=(_contact(LPAD_GID, HELD_GID),), sorter_xy=(0.2, 0.0))
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.32])
    hit, held_half, _, beam_top = _cast_gripper_ray(
        world.mjm, world.mjd, origin, [LPAD_GID], SORTER_BODY
    )
    assert hit is not None
    assert hit[2] == beam_top[2], f"beam must be horizontal: {hit} vs {beam_top}"
    assert hit[0] == pytest.approx(0.15), f"beam must stop on the cube wall: {hit}"
    assert held_half is not None
    assert tuple(beam_top) == (0.0, 0.0, 0.29), "beam must start at the held centre"


def test_face_level_object_switches_despite_high_wrist(monkeypatch):
    # The grasp site (origin) rides well above the cube top, but the held
    # object is level with the cube's face: the switch anchors on the held
    # object's centre, so the beam must still go horizontal.
    world = _SlabWorld(contacts=(_contact(LPAD_GID, HELD_GID),), sorter_xy=(0.2, 0.0))
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.45])  # wrist above the cube top (0.34)
    hit, held_half, _, beam_top = _cast_gripper_ray(
        world.mjm, world.mjd, origin, [LPAD_GID], SORTER_BODY
    )
    assert hit is not None
    assert hit[2] == beam_top[2], f"beam must be horizontal: {hit} vs {beam_top}"
    assert hit[0] == pytest.approx(0.15), f"beam must stop on the cube wall: {hit}"
    assert held_half is not None
    assert tuple(beam_top) == (0.0, 0.0, 0.29), "beam starts at the held centre"


def test_held_cube_between_floor_and_weight_goes_horizontal(monkeypatch):
    # Cube face 0.50 m away is farther than the floor (0.29 m) but within
    # the 2x weight (0.58 m): the beam must still go horizontal.
    world = _SlabWorld(contacts=(_contact(LPAD_GID, HELD_GID),), sorter_xy=(0.55, 0.0))
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.32])
    hit, held_half, _, beam_top = _cast_gripper_ray(
        world.mjm, world.mjd, origin, [LPAD_GID], SORTER_BODY
    )
    assert hit is not None
    assert hit[2] == beam_top[2], f"beam must be horizontal: {hit} vs {beam_top}"
    assert hit[0] == pytest.approx(0.5), f"beam must stop on the cube wall: {hit}"
    assert held_half is not None


def test_held_floor_closer_stays_vertical(monkeypatch):
    # Sorter 2 m away: Euclidean farther than the floor below.
    world = _SlabWorld(contacts=(_contact(LPAD_GID, HELD_GID),), sorter_xy=(2.0, 0.0))
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.32])
    hit, held_half, _, beam_top = _cast_gripper_ray(
        world.mjm, world.mjd, origin, [LPAD_GID], SORTER_BODY
    )
    assert hit is not None and hit[2] == 0.0, "beam must stay vertical"
    assert held_half is not None
    assert tuple(beam_top) == (0.0, 0.0, 0.32)


def test_held_above_cube_stays_vertical(monkeypatch):
    # Gripper over the cube footprint, higher than its top face: back to
    # vertical even though the top face is the closest surface.
    world = _SlabWorld(
        contacts=(_contact(LPAD_GID, HELD_GID),),
        sorter_xy=(0.0, 0.0),
        sorter_z=0.10,
    )
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.35])
    hit, held_half, _, beam_top = _cast_gripper_ray(
        world.mjm, world.mjd, origin, [LPAD_GID], SORTER_BODY
    )
    assert hit is not None and hit[2] == 0.0, "beam must be vertical above the cube"
    assert held_half is not None
    assert tuple(beam_top) == (0.0, 0.0, 0.35)


def test_held_above_cube_top_outside_footprint_stays_vertical(monkeypatch):
    # Higher than the cube top but off to the side: still vertical, the
    # plus only extends sideways at/below cube-top height.
    world = _SlabWorld(
        contacts=(_contact(LPAD_GID, HELD_GID),),
        sorter_xy=(0.3, 0.0),
        sorter_z=0.10,
    )
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.35])
    hit, held_half, _, beam_top = _cast_gripper_ray(
        world.mjm, world.mjd, origin, [LPAD_GID], SORTER_BODY
    )
    assert hit is not None and hit[2] == 0.0, (
        "beam must be vertical above cube-top height"
    )
    assert held_half is not None
    assert tuple(beam_top) == (0.0, 0.0, 0.35)


# Roll-by-pi about x: the live spawn pose of the sorter (star face down).
_ROLL_PI = np.array([1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0])


def test_held_above_flipped_cube_stays_vertical(monkeypatch):
    # The sorter rests with its local +z pointing DOWN. "Above the cube"
    # must mean beyond the face that is actually up in the world, so the
    # beam goes vertical here even though the local z test would read
    # "below the top".
    world = _SlabWorld(
        contacts=(_contact(LPAD_GID, HELD_GID),),
        sorter_xy=(0.0, 0.0),
        sorter_z=0.10,
        sorter_mat=_ROLL_PI,
    )
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.35])
    hit, held_half, _, beam_top = _cast_gripper_ray(
        world.mjm, world.mjd, origin, [LPAD_GID], SORTER_BODY
    )
    assert hit is not None and hit[2] == 0.0, (
        "beam must be vertical above the flipped cube"
    )
    assert held_half is not None
    assert tuple(beam_top) == (0.0, 0.0, 0.35)


def test_held_at_flipped_cube_height_goes_horizontal(monkeypatch):
    # Same flipped cube, but the beam origin is at (not above) cube height
    # and face-aligned: the beam still goes horizontal.
    world = _SlabWorld(
        contacts=(_contact(LPAD_GID, HELD_GID),),
        sorter_xy=(0.2, 0.0),
        sorter_z=0.26,
        sorter_mat=_ROLL_PI,
    )
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.29])  # below the flipped cube's up face (0.31)
    hit, held_half, _, beam_top = _cast_gripper_ray(
        world.mjm, world.mjd, origin, [LPAD_GID], SORTER_BODY
    )
    assert hit is not None
    assert hit[2] == beam_top[2], f"beam must be horizontal: {hit} vs {beam_top}"
    assert hit[0] == pytest.approx(0.15), f"beam must stop on the cube wall: {hit}"
    assert held_half is not None


def test_held_beside_cube_goes_horizontal(monkeypatch):
    # Same height as the side-approach case but offset in y too: the beam
    # must snap to a single world axis, never diagonally.
    world = _SlabWorld(
        held_xy=(0.0, 0.0),
        contacts=(_contact(LPAD_GID, HELD_GID),),
        sorter_xy=(0.0, 0.2),
        sorter_z=0.29,
    )
    world.set_held_centre((0.0, 0.0, 0.29))
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.32])
    hit, held_half, _, beam_top = _cast_gripper_ray(
        world.mjm, world.mjd, origin, [LPAD_GID], SORTER_BODY
    )
    assert hit is not None
    assert hit[2] == beam_top[2], f"beam must be horizontal: {hit} vs {beam_top}"
    assert hit[0] == beam_top[0], f"beam must run straight along y: {hit} vs {beam_top}"
    assert hit[1] == pytest.approx(0.15), f"beam must stop on the cube wall: {hit}"
    assert held_half is not None


def test_diagonal_quadrant_stays_vertical(monkeypatch):
    # Sorter off in a diagonal quadrant (face-aligned in neither x nor y):
    # outside the plus, so vertical even at cube height.
    world = _SlabWorld(
        contacts=(_contact(LPAD_GID, HELD_GID),),
        sorter_xy=(0.3, 0.1),
        sorter_z=0.29,
    )
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.32])
    hit, held_half, _, beam_top = _cast_gripper_ray(
        world.mjm, world.mjd, origin, [LPAD_GID], SORTER_BODY
    )
    assert hit is not None and hit[2] == 0.0, "beam must be vertical off the plus"
    assert held_half is not None
    assert tuple(beam_top) == (0.0, 0.0, 0.32)


def test_off_axis_in_plus_snaps_straight(monkeypatch):
    # Slightly off-axis but inside the plus: the beam snaps to the dominant
    # world axis and runs straight, never diagonally.
    world = _SlabWorld(
        contacts=(_contact(LPAD_GID, HELD_GID),),
        sorter_xy=(0.3, 0.02),
        sorter_z=0.29,
    )
    _install(monkeypatch, world)
    origin = np.array([0.0, 0.0, 0.32])
    hit, held_half, _, beam_top = _cast_gripper_ray(
        world.mjm, world.mjd, origin, [LPAD_GID], SORTER_BODY
    )
    assert hit is not None
    assert hit[2] == beam_top[2], f"beam must be horizontal: {hit} vs {beam_top}"
    assert hit[1] == beam_top[1], f"beam must not run diagonally: {hit} vs {beam_top}"
    assert hit[0] == pytest.approx(0.25), f"beam must stop on the cube wall: {hit}"
    assert held_half is not None


def test_box_helpers():
    centre = np.array([0.2, 0.0, 0.10])
    eye = np.eye(3)
    assert _point_box_surface_dist(
        np.array([0.2, 0.0, 0.35]), centre, eye
    ) == pytest.approx(0.2)
    assert _point_box_surface_dist(
        np.array([0.0, 0.0, 0.35]), centre, eye
    ) == pytest.approx(0.25)
    assert _point_box_surface_dist(centre, centre, eye) == pytest.approx(0.0)
    assert _ray_box_param(
        np.array([0.0, 0.0, 0.29]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.2, 0.0, 0.29]),
        eye,
    ) == pytest.approx(0.15)
    assert (
        _ray_box_param(
            np.array([0.0, 0.0, 0.29]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.10]),
            eye,
        )
        is None
    )
