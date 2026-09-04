"""MjSpec builders for ball and field. Field pose comes from init_state."""
# pyright: reportAttributeAccessIssue=false
# (mujoco ships no type stubs for the compiled MjSpec module)

from __future__ import annotations

import math

import mujoco
import numpy as np

from g3a_bench.tasks.ball.variants import (
    BALL_MASS,
    BALL_RADIUS,
    HOLE_RADIUS,
    PLATFORM_Z,
    RAMP_RISE,
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
# Platform top: same flat neutral gray as the shared table.
PLATFORM_RGBA = TABLE_RGBA
RIM_RGBA = (0.85, 0.2, 0.2, 1.0)

# Platform footprint. Right edge clears the hole plus a rim margin.
_PLAT_RIM = 0.08
PLAT_X = (0.18, TARGET_X + HOLE_RADIUS + _PLAT_RIM)
PLAT_Y = (-0.22, 0.22)

# Ramp from low platform to raised section holding the pit.
_RAMP_X0, _RAMP_X1 = 0.64, 0.76
_INCLINE_THICKNESS = 0.02

_RAIL_T = 0.012
# Rail top heights. Ramp uses one uniform higher top.
_RAIL_TOP = PLATFORM_Z + 0.04
_RAIL_TOP_RAMP = PLATFORM_Z + RAMP_RISE + 0.04


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


def _mat2quat(xax: np.ndarray, yax: np.ndarray, zax: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) for the rotation whose columns are the given axes."""
    mat = np.array(
        [
            [xax[0], yax[0], zax[0]],
            [xax[1], yax[1], zax[1]],
            [xax[2], yax[2], zax[2]],
        ]
    ).flatten()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


def _add_incline(body, x0, z0, x1, z1, y_range, thickness, rgba, name):
    """A thin box whose top face bridges (x0, z0) -> (x1, z1)."""
    run, rise = x1 - x0, z1 - z0
    theta = math.atan2(rise, run)
    half_len, half_thick = math.hypot(run, rise) / 2, thickness / 2
    xax = np.array([math.cos(theta), 0.0, math.sin(theta)])
    yax = np.array([0.0, 1.0, 0.0])
    zax = np.cross(xax, yax)
    top_mid = np.array([(x0 + x1) / 2, 0.0, (z0 + z1) / 2])
    center = top_mid - half_thick * zax
    yc, yhalf = (y_range[0] + y_range[1]) / 2, (y_range[1] - y_range[0]) / 2
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(center[0], yc, center[2]),
        quat=_mat2quat(xax, yax, zax),
        size=(half_len, yhalf, half_thick),
        rgba=rgba,
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


def _add_platform_with_hole(
    body, tcx, tcy=0.0, *, x_range=PLAT_X, z_half=PLATFORM_Z / 2, prefix=""
):
    """Platform spanning ``x_range`` with a square pit at (tcx, tcy)."""
    x0, x1 = x_range
    y0, y1 = PLAT_Y
    g = HOLE_RADIUS
    hz = z_half
    yc, yhalf = (y0 + y1) / 2, (y1 - y0) / 2
    # Left / right of the hole (full y span).
    _add_box(
        body,
        f"{prefix}plat_l",
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
        f"{prefix}plat_r",
        (tcx + g + x1) / 2,
        yc,
        hz,
        (x1 - tcx - g) / 2,
        yhalf,
        hz,
        PLATFORM_RGBA,
    )
    # Front / back of the hole (middle x band).
    _add_box(
        body,
        f"{prefix}plat_f",
        tcx,
        (y0 + tcy - g) / 2,
        hz,
        g,
        (tcy - g - y0) / 2,
        hz,
        PLATFORM_RGBA,
    )
    _add_box(
        body,
        f"{prefix}plat_b",
        tcx,
        (y1 + tcy + g) / 2,
        hz,
        g,
        (y1 - tcy - g) / 2,
        hz,
        PLATFORM_RGBA,
    )
    _add_marker(
        body,
        f"{prefix}hole_rim",
        tcx,
        tcy,
        2 * hz + 0.001,
        g + 0.006,
        RIM_RGBA,
        0.001,
    )


def _add_obstacle(body):
    """Box in the open lane, forcing a bank shot through narrow sub-lanes."""
    hz = 0.025
    _add_box(body, "obstacle", 0.52, -0.13, PLATFORM_Z + hz, 0.02, 0.025, hz, WALL_RGBA)


def _add_railing(body, z_top):
    """Perimeter railing on the platform edges (like real minigolf)."""
    x0, x1 = PLAT_X
    y0, y1 = PLAT_Y
    t = _RAIL_T / 2
    zc, hz = (PLATFORM_Z + z_top) / 2, (z_top - PLATFORM_Z) / 2
    xc, xhalf = (x0 + x1) / 2, (x1 - x0) / 2
    yc, yhalf = (y0 + y1) / 2, (y1 - y0) / 2
    _add_box(body, "rail_near", x0 + t, yc, zc, t, yhalf, hz, WALL_RGBA)
    _add_box(body, "rail_far", x1 - t, yc, zc, t, yhalf, hz, WALL_RGBA)
    _add_box(body, "rail_left", xc, y1 - t, zc, xhalf, t, hz, WALL_RGBA)
    _add_box(body, "rail_right", xc, y0 + t, zc, xhalf, t, hz, WALL_RGBA)


def _add_blocker(body):
    """Blocker across the direct line. Forces a bank shot off the railing."""
    t, h = 0.012, 0.05
    _add_box(body, "blocker", 0.58, 0.09, PLATFORM_Z + h / 2, t, 0.13, h / 2, WALL_RGBA)


def _add_ramp_platform(body, tcx, tcy):
    x0, x1 = PLAT_X
    y0, y1 = PLAT_Y
    yc, yhalf = (y0 + y1) / 2, (y1 - y0) / 2
    hz_low = PLATFORM_Z / 2
    _add_box(
        body,
        "plat_low",
        (x0 + _RAMP_X0) / 2,
        yc,
        hz_low,
        (_RAMP_X0 - x0) / 2,
        yhalf,
        hz_low,
        PLATFORM_RGBA,
    )
    _add_incline(
        body,
        _RAMP_X0,
        PLATFORM_Z,
        _RAMP_X1,
        PLATFORM_Z + RAMP_RISE,
        PLAT_Y,
        _INCLINE_THICKNESS,
        PLATFORM_RGBA,
        name="ramp_incline",
    )
    hz_raised = (PLATFORM_Z + RAMP_RISE) / 2
    _add_platform_with_hole(
        body, tcx, tcy, x_range=(_RAMP_X1, x1), z_half=hz_raised, prefix="raised_"
    )


def get_field_spec(variant: BallVariantCfg, name: str = "field") -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    field = spec.worldbody.add_body(name=name)

    if variant.task == "minigolf":
        layout = variant.minigolf_layout
        if layout == "ramp":
            _add_ramp_platform(field, TARGET_X, variant.pit_y)
        else:
            _add_platform_with_hole(field, TARGET_X, variant.pit_y)
        _add_railing(field, _RAIL_TOP_RAMP if layout == "ramp" else _RAIL_TOP)
        if layout != "direct":
            _add_blocker(field)
        if layout in ("obstacle", "ramp"):
            _add_obstacle(field)
    else:  # throw
        r = variant.target_radius * variant.margin
        if variant.throw_target == "u":
            _add_u_pocket(field, TARGET_X, 0.0, 0.0, r)
        else:
            _add_marker(field, "target_mark", TARGET_X, 0.0, 0.0008, r, TARGET_RGBA)
        if not variant.movable_wall:
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


def get_wall_spec(name: str = "wall") -> mujoco.MjSpec:
    """Standalone wall entity for ThrowWallVariation."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, WALL_HEIGHT / 2),
        size=(WALL_HALF_X, WALL_HALF_Y, WALL_HEIGHT / 2),
        rgba=WALL_RGBA,
    )
    return spec
