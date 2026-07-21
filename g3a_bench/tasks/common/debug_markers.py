"""Shared status coloring for the command terms' debug-vis goal markers.

Every category draws its goal markers in the target object's own color while
pending, green once that object is at its goal, and a brighter, more opaque
green once the command's full success (including the velocity / gripper /
release gates) holds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer

Rgba = tuple[float, float, float, float]

MARKER_INFLATE = 0.0015
"""Grow markers by this margin (m) so they don't z-fight with the object
surface once it sits exactly at the goal."""

PLACED_RGBA: Rgba = (0.2, 0.8, 0.2, 0.45)
SUCCESS_RGBA: Rgba = (0.1, 1.0, 0.3, 0.7)
PENDING_ALPHA = 0.35

OUTLINE_RGBA: Rgba = (0.85, 0.1, 0.1, 0.95)
"""Ghost/target-marker outline color: a red edge marking a marker as a
virtual target rather than a real object, at any state. Opposite the cool
identity-color hues on the wheel, so it contrasts hard against them."""

OUTLINE_RADIUS = 0.0012
"""Edge-cylinder radius (m) for :func:`add_box_wireframe`.

Earlier attempt, for the record: drawing a second, larger *filled* box
behind the fill box to fake a border. That doesn't work -- two nested solid
boxes don't produce a border, the bigger one fully *encloses* the smaller
one (its surface blocks the view of everything inside), regardless of how
much bigger it is. A border needs edge-only (hollow) geometry, which mjlab's
visualizer doesn't offer as a single primitive, so it's built from 12 thin
cylinders along the box's edges instead."""


def add_box_wireframe(
    visualizer: DebugVisualizer,
    center: np.ndarray,
    half_extents: np.ndarray,
    mat: np.ndarray,
    color: Rgba,
    label_prefix: str,
    radius: float = OUTLINE_RADIUS,
) -> None:
    """Draw an oriented box's 12 edges as thin cylinders.

    Unlike a second, larger filled box (see :data:`OUTLINE_RADIUS`), this
    only occupies the boundary, so it frames a fill box drawn at the same
    center/size instead of hiding it.

    Args:
      center: (3,) world-frame box center.
      half_extents: (3,) local-frame half extents (hx, hy, hz).
      mat: (3, 3) local-to-world rotation matrix.
      color: Edge color.
      label_prefix: Unique label prefix; each edge gets its own suffix.
      radius: Cylinder radius (m).
    """
    hx, hy, hz = half_extents
    signs = np.array(
        [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    local_corners = signs * np.array([hx, hy, hz])
    world_corners = np.asarray(center) + local_corners @ np.asarray(mat).T

    edges = (
        (0, 1), (2, 3), (4, 5), (6, 7),  # along z
        (0, 2), (1, 3), (4, 6), (5, 7),  # along y
        (0, 4), (1, 5), (2, 6), (3, 7),  # along x
    )  # fmt: skip
    for j, (a, b) in enumerate(edges):
        visualizer.add_cylinder(
            start=world_corners[a],
            end=world_corners[b],
            radius=radius,
            color=color,
            label=f"{label_prefix}_edge{j}",
        )


def marker_color(
    base_rgba: Rgba,
    placed: bool,
    success: bool,
    *,
    pending_alpha: float = PENDING_ALPHA,
) -> Rgba:
    """Marker RGBA for one goal slot.

    Args:
      base_rgba: The target object's own color (alpha is replaced).
      placed: Whether this slot's object is at its goal.
      success: Whether the command's full success metric holds.
      pending_alpha: Alpha for the base (pending) color. Override per
        category if the default is too faint/strong for that marker style.
    """
    if success:
        return SUCCESS_RGBA
    if placed:
        return PLACED_RGBA
    return (base_rgba[0], base_rgba[1], base_rgba[2], pending_alpha)
