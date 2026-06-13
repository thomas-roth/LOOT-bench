"""MjSpec builders and board layout for the object-insertion task category.

Boards are *fixed-base* entities: a body with no free joint, which mjlab
auto-wraps as a mocap body pinned to the env origin. Pegs (and the movable
top sheet of the two-sheet variant) are free bodies.

The same ``socket_layout`` is used both to build the bore geometry and by the
command term to compute goal poses and insertion success, so geometry and
success metric can never drift apart.
"""
# pyright: reportAttributeAccessIssue=false
# (mujoco ships no type stubs for the compiled MjSpec module)

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from g3a_bench.tasks.insertion.variants import (
    PEG_HALF_LEN,
    SHAPE_GEOMS,
    InsertionVariantCfg,
    Shape,
    bore_apothem,
)

PALETTE: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.15, 0.15, 1.0),
    "green": (0.15, 0.7, 0.2, 1.0),
    "blue": (0.15, 0.3, 0.85, 1.0),
    "yellow": (0.9, 0.8, 0.1, 1.0),
    "gray": (0.5, 0.5, 0.5, 1.0),
}
BOARD_RGBA = (0.72, 0.6, 0.42, 1.0)  # light wood
PEG_MASS = 0.05
SHEET_MASS = 0.12

# Board placement in the robot workspace (board body frame origin, env-local).
BOARD_POS = (0.34, 0.0, 0.0)

# Bore wall thickness and base-plate thickness.
WALL_T = 0.006
PLATE_HALF_Z = 0.011
# Horizontal bore: centre height above the table and how far its mouth
# protrudes toward the robot from the board origin (-x).
H_BORE_CENTER_Z = 0.075
H_MOUTH_X = 0.06
# Lateral spacing of sorter holes (along board y).
SORTER_SPACING = 0.072
# Thickness of each sheet in the two-sheet variant.
SHEET_HALF_Z = 0.008


@dataclass(frozen=True)
class SocketFrame:
    """A bore's mouth and outward axis, in the board body frame."""

    mouth: np.ndarray  # (3,) board-local position of the bore opening
    axis_out: np.ndarray  # (3,) unit vector pointing out of the bore
    shape: Shape
    color: str


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _frame_axes(axis_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed basis (c0, c1, axis_out) spanning the bore cross section."""
    a = _normalize(axis_out)
    t = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    c0 = _normalize(np.cross(t, a))
    c1 = np.cross(a, c0)
    return c0, c1, a


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


def _wall_boxes(
    mouth: np.ndarray,
    axis_out: np.ndarray,
    shape: Shape,
    bore_depth: float,
) -> list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]]:
    """Boxes forming a polygonal bore. Returns (pos, quat_wxyz, half_size)."""
    c0, c1, a = _frame_axes(axis_out)
    center = mouth - a * (bore_depth / 2.0)  # chimney centre along the bore
    geom = SHAPE_GEOMS[shape]
    out: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]] = []

    if shape == "rect":
        hx, hy = geom.rect_half  # type: ignore[misc]
        clear = bore_apothem("rect") - geom.inradius  # == CLEARANCE
        cx, cy = hx + clear, hy + clear
        for sgn in (1.0, -1.0):
            pos = center + c0 * sgn * (cx + WALL_T / 2)
            yax = np.cross(a, c0)
            out.append(
                (pos, _mat2quat(c0, yax, a), (WALL_T / 2, cy + WALL_T, bore_depth / 2))
            )
        for sgn in (1.0, -1.0):
            pos = center + c1 * sgn * (cy + WALL_T / 2)
            yax = np.cross(a, c1)
            out.append(
                (pos, _mat2quat(c1, yax, a), (WALL_T / 2, cx + WALL_T, bore_depth / 2))
            )
        return out

    n = 16 if shape == "circle" else geom.nsides
    apo = bore_apothem(shape)
    edge_half = apo * math.tan(math.pi / n) + WALL_T
    for k in range(n):
        th = 2.0 * math.pi * k / n
        radial = math.cos(th) * c0 + math.sin(th) * c1
        pos = center + radial * (apo + WALL_T / 2)
        yax = np.cross(a, radial)
        out.append(
            (pos, _mat2quat(radial, yax, a), (WALL_T / 2, edge_half, bore_depth / 2))
        )
    return out


# --- Board layout. --------------------------------------------------------


def socket_layout(variant: InsertionVariantCfg) -> list[SocketFrame]:
    """Mouth pose of every bore in the board body frame."""
    frames: list[SocketFrame] = []
    n = variant.num_sockets
    if variant.orientation == "horizontal":
        # Single horizontal bore facing the robot (-x); the tube cantilevers
        # off a backing slab at the closed (+x) end.
        s = variant.sockets[0]
        mouth = np.array([-H_MOUTH_X, 0.0, H_BORE_CENTER_Z])
        frames.append(SocketFrame(mouth, np.array([-1.0, 0.0, 0.0]), s.shape, s.color))
        return frames

    plate_top = 2 * PLATE_HALF_Z
    for i, s in enumerate(variant.sockets):
        y = (i - (n - 1) / 2.0) * SORTER_SPACING
        mouth = np.array([0.0, y, plate_top + variant.bore_depth])
        frames.append(SocketFrame(mouth, np.array([0.0, 0.0, 1.0]), s.shape, s.color))
    return frames


def _plate_half_xy(variant: InsertionVariantCfg) -> tuple[float, float]:
    n = variant.num_sockets
    half_y = (n - 1) / 2.0 * SORTER_SPACING + 0.05
    return 0.055, half_y


# --- Specs. ---------------------------------------------------------------


def get_peg_spec(shape: Shape, color: str, name: str = "peg") -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_joint")
    geom = SHAPE_GEOMS[shape]
    rgba = PALETTE[color]

    if shape == "circle":
        body.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=(geom.circumradius, PEG_HALF_LEN, 0.0),
            mass=PEG_MASS,
            rgba=rgba,
        )
    elif shape in ("square", "rect"):
        if shape == "square":
            hw = geom.circumradius / math.sqrt(2.0)
            half = (hw, hw, PEG_HALF_LEN)
        else:
            hx, hy = geom.rect_half  # type: ignore[misc]
            half = (hx, hy, PEG_HALF_LEN)
        body.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half,
            mass=PEG_MASS,
            rgba=rgba,
        )
    else:  # triangle, hexagon -> convex-hull prism mesh
        verts: list[float] = []
        for z in (-PEG_HALF_LEN, PEG_HALF_LEN):
            for k in range(geom.nsides):
                ang = 2.0 * math.pi * k / geom.nsides
                verts += [
                    geom.circumradius * math.cos(ang),
                    geom.circumradius * math.sin(ang),
                    z,
                ]
        mesh = spec.add_mesh()
        mesh.name = f"{name}_mesh"
        mesh.uservert = verts
        body.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"{name}_mesh",
            mass=PEG_MASS,
            rgba=rgba,
        )
    return spec


def _add_walls(body, frame: SocketFrame, bore_depth: float, rgba) -> None:
    for j, (pos, quat, size) in enumerate(
        _wall_boxes(frame.mouth, frame.axis_out, frame.shape, bore_depth)
    ):
        body.add_geom(
            name=f"{body.name}_wall_{int(frame.mouth[1] * 1e4)}_{j}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            quat=quat,
            size=size,
            rgba=rgba,
        )


def get_board_spec(variant: InsertionVariantCfg, name: str = "board") -> mujoco.MjSpec:
    """Fixed-base board: base structure plus one bore per socket."""
    spec = mujoco.MjSpec()
    # Body stays at the origin; placement is via EntityCfg.init_state.pos
    # (mjlab overwrites a fixed-base root body's pose from init_state).
    board = spec.worldbody.add_body(name=name)
    frames = socket_layout(variant)

    if variant.orientation == "horizontal":
        # Backing slab closes the blind end and supports the cantilevered tube.
        apo = bore_apothem(variant.sockets[0].shape)
        slab_top = H_BORE_CENTER_Z + apo + 0.018
        blind_x = -H_MOUTH_X + variant.bore_depth
        slab_half_x = 0.018
        board.add_geom(
            name=f"{name}_back",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(blind_x + slab_half_x, 0.0, slab_top / 2),
            size=(slab_half_x, apo + 0.03, slab_top / 2),
            rgba=BOARD_RGBA,
        )
    else:
        hx, hy = _plate_half_xy(variant)
        board.add_geom(
            name=f"{name}_plate",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(0.0, 0.0, PLATE_HALF_Z),
            size=(hx, hy, PLATE_HALF_Z),
            rgba=BOARD_RGBA,
        )

    for frame in frames:
        rgba = PALETTE[frame.color] if variant.kind == "sorter" else BOARD_RGBA
        _add_walls(board, frame, variant.bore_depth, rgba)
    return spec


def get_sheet_board_spec(
    variant: InsertionVariantCfg, name: str = "board"
) -> mujoco.MjSpec:
    """Two-sheet variant: fixed bottom sheet (a thin ring) at the board centre."""
    spec = mujoco.MjSpec()
    board = spec.worldbody.add_body(name=name)
    s = variant.sockets[0]
    frame = SocketFrame(
        np.array([0.0, 0.0, 2 * SHEET_HALF_Z]),
        np.array([0.0, 0.0, 1.0]),
        s.shape,
        "gray",
    )
    _add_walls(board, frame, 2 * SHEET_HALF_Z, PALETTE["gray"])
    return spec


def get_sheet_spec(variant: InsertionVariantCfg, name: str = "sheet") -> mujoco.MjSpec:
    """Two-sheet variant: movable top sheet (a thin ring) as a free body."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_joint")
    s = variant.sockets[0]
    frame = SocketFrame(
        np.array([0.0, 0.0, SHEET_HALF_Z]), np.array([0.0, 0.0, 1.0]), s.shape, s.color
    )
    boxes = _wall_boxes(frame.mouth, frame.axis_out, frame.shape, 2 * SHEET_HALF_Z)
    mass_each = SHEET_MASS / len(boxes)
    for j, (pos, quat, size) in enumerate(boxes):
        body.add_geom(
            name=f"{name}_wall_{j}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            quat=quat,
            size=size,
            mass=mass_each,
            rgba=PALETTE[s.color],
        )
    return spec
