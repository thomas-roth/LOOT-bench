"""DK1 collision-mode contract tests on the compiled MuJoCo model.

Applies a collision mode to the unimanual DK1 spec exactly the way mjlab does
(``CollisionCfg.edit_spec``) and compiles it. No env, no warp, CPU-only.
Guards the gripper-only invariant: exactly the wrist/finger/pad geoms have
collision enabled, and the finger pads keep their tuned contact parameters.
"""

import re

import mujoco

from loot_bench.robots.trlc_dk1.dk1_collision import (
    DK1_GRIPPER_GEOMS_PATTERN,
    DK1_PAD_GEOM_PATTERN,
    GRIPPER_ONLY_COLLISION,
)
from loot_bench.robots.trlc_dk1.dk1_constants import get_unimanual_spec

# Deliberately restated tuned pad values (independent oracle for the numbers;
# the geom patterns are shared single sources by design, not re-derived here).
_PAD_FRICTION = (1.0, 5e-3, 5e-4)
_PAD_CONDIM = 6
_PAD_PRIORITY = 1


def _compiled_gripper_only_model() -> mujoco.MjModel:
    spec = get_unimanual_spec()
    GRIPPER_ONLY_COLLISION.edit_spec(spec)
    return spec.compile()


def _geom_names(model: mujoco.MjModel) -> list[tuple[int, str]]:
    return [
        (gid, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
        for gid in range(model.ngeom)
    ]


def test_gripper_only_collides_exactly_the_gripper_geoms() -> None:
    model = _compiled_gripper_only_model()
    for gid, name in _geom_names(model):
        # A geom takes part in collision if either mask is nonzero (MuJoCo
        # pairs on contype1 & conaffinity2 OR contype2 & conaffinity1).
        colliding = bool(model.geom_contype[gid] or model.geom_conaffinity[gid])
        expected = bool(re.fullmatch(DK1_GRIPPER_GEOMS_PATTERN, name))
        assert colliding == expected, (
            f"{name!r}: colliding={colliding}, expected {expected}"
        )


def test_gripper_only_keeps_pad_contact_tuning() -> None:
    model = _compiled_gripper_only_model()
    pads = [
        (gid, name)
        for gid, name in _geom_names(model)
        if re.search(DK1_PAD_GEOM_PATTERN, name)
    ]
    assert len(pads) == 2, f"expected 2 finger-pad geoms, found {pads}"
    for gid, name in pads:
        assert model.geom_condim[gid] == _PAD_CONDIM, name
        assert model.geom_priority[gid] == _PAD_PRIORITY, name
        assert tuple(model.geom_friction[gid]) == _PAD_FRICTION, name
