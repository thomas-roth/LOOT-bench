"""MjSpec builders for the ball task category.

The ball is a free-body sphere. The *field* is a fixed-base entity holding the
target structure (marked disk / U-pocket / pit platform) plus the throw wall.
Like the insertion boards, the field is placed via ``EntityCfg.init_state.pos``;
its geoms use env-local workspace coordinates.
"""
# pyright: reportAttributeAccessIssue=false
# (mujoco ships no type stubs for the compiled MjSpec module)

from __future__ import annotations

import mujoco

from g3a_bench.tasks.ball.variants import (
    BALL_MASS,
    BALL_RADIUS,
    HOLE_RADIUS,
    PLATFORM_Z,
    TARGET_X,
    WALL_HALF_X,
    WALL_HALF_Y,
    WALL_HEIGHT,
    WALL_X,
    BallVariantCfg,
)
from g3a_bench.tasks.common.table import TABLE_RGBA

BALL_RGBA = (0.95, 0.35, 0.1, 1.0)
TARGET_RGBA = (0.15, 0.8, 0.25, 0.45)
WALL_RGBA = (0.55, 0.55, 0.6, 1.0)
# Golf platform top: same flat neutral gray as the shared table.
PLATFORM_RGBA = TABLE_RGBA
RIM_RGBA = (0.85, 0.2, 0.2, 1.0)

# Platform footprint for hole variants. Shared across every action, so the
# right edge must clear the farthest target's hole plus a rim margin -- derive
# it from TARGET_X/HOLE_RADIUS instead of hardcoding it, or it silently falls
# out of sync again if a target moves.
_PLAT_RIM = 0.08
PLAT_X = (0.18, max(TARGET_X.values()) + HOLE_RADIUS + _PLAT_RIM)
PLAT_Y = (-0.22, 0.22)


def get_ball_spec(name: str = "ball") -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(BALL_RADIUS, 0.0, 0.0),
        mass=BALL_MASS,
        rgba=BALL_RGBA,
        friction=(0.8, 0.01, 0.001),
        condim=6,
    )
    return spec


def _add_box(body, name, x, y, z, hx, hy, hz, rgba):
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x, y, z),
        size=(hx, hy, hz),
        rgba=rgba,
    )


def _add_marker(body, name, x, y, z, r, rgba, thickness=0.0008):
    """Flat, non-colliding disk used to paint a target/rim onto a surface."""
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=(x, y, z),
        size=(r, thickness, 0.0),
        rgba=rgba,
        contype=0,
        conaffinity=0,
    )


def _add_u_pocket(body, tcx, tcy, sz, r):
    """Three walls forming a U pocket open toward -x (incoming ball)."""
    t, h = 0.012, 2 * BALL_RADIUS
    z = sz + h / 2
    # Back wall (+x), and two side walls (+/-y).
    _add_box(body, "u_back", tcx + r, tcy, z, t, r + t, h / 2, WALL_RGBA)
    _add_box(body, "u_left", tcx, tcy + r, z, r, t, h / 2, WALL_RGBA)
    _add_box(body, "u_right", tcx, tcy - r, z, r, t, h / 2, WALL_RGBA)
    _add_marker(body, "u_mark", tcx, tcy, sz + 0.0008, r, TARGET_RGBA)


def _add_platform_with_hole(body, tcx, tcy):
    """Raised platform spanning the workspace with a square pit at the target."""
    x0, x1 = PLAT_X
    y0, y1 = PLAT_Y
    g = HOLE_RADIUS
    hz = PLATFORM_Z / 2
    yc, yhalf = (y0 + y1) / 2, (y1 - y0) / 2
    # Left / right of the hole (full y span).
    _add_box(
        body,
        "plat_l",
        (x0 + tcx - g) / 2,
        yc,
        hz,
        (tcx - g - x0) / 2,
        yhalf,
        hz,
        PLATFORM_RGBA,
    )
    _add_box(
        body,
        "plat_r",
        (tcx + g + x1) / 2,
        yc,
        hz,
        (x1 - tcx - g) / 2,
        yhalf,
        hz,
        PLATFORM_RGBA,
    )
    # Front / back of the hole (middle x band).
    _add_box(body, "plat_f", tcx, (y0 - g) / 2, hz, g, (-g - y0) / 2, hz, PLATFORM_RGBA)
    _add_box(body, "plat_b", tcx, (y1 + g) / 2, hz, g, (y1 - g) / 2, hz, PLATFORM_RGBA)
    _add_marker(
        body, "hole_rim", tcx, tcy, PLATFORM_Z + 0.001, g + 0.006, RIM_RGBA, 0.001
    )


def get_field_spec(variant: BallVariantCfg, name: str = "field") -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    field = spec.worldbody.add_body(name=name)
    tcx, tcy, sz = variant.target_x, 0.0, variant.surface_z
    r = variant.target_radius

    if variant.target == "marked":
        _add_marker(field, "target_mark", tcx, tcy, sz + 0.0008, r, TARGET_RGBA)
    elif variant.target == "u_boundary":
        _add_u_pocket(field, tcx, tcy, sz, r)
    else:  # hole
        _add_platform_with_hole(field, tcx, tcy)

    if variant.has_wall:
        _add_box(
            field,
            "wall",
            WALL_X,
            0.0,
            WALL_HEIGHT / 2,
            WALL_HALF_X,
            WALL_HALF_Y,
            WALL_HEIGHT / 2,
            WALL_RGBA,
        )
    return spec
