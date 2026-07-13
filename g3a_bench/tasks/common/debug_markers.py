"""Shared status coloring for the command terms' debug-vis goal markers.

Every category draws its goal markers in the target object's own color while
pending, green once that object is at its goal, and a brighter, more opaque
green once the command's full success (including the velocity / gripper /
release gates) holds.
"""

Rgba = tuple[float, float, float, float]

MARKER_INFLATE = 0.0015
"""Grow markers by this margin (m) so they don't z-fight with the object
surface once it sits exactly at the goal."""

PLACED_RGBA: Rgba = (0.2, 0.8, 0.2, 0.45)
SUCCESS_RGBA: Rgba = (0.1, 1.0, 0.3, 0.7)
PENDING_ALPHA = 0.35


def marker_color(base_rgba: Rgba, placed: bool, success: bool) -> Rgba:
    """Marker RGBA for one goal slot.

    Args:
      base_rgba: The target object's own color (alpha is replaced).
      placed: Whether this slot's object is at its goal.
      success: Whether the command's full success metric holds.
    """
    if success:
        return SUCCESS_RGBA
    if placed:
        return PLACED_RGBA
    return (base_rgba[0], base_rgba[1], base_rgba[2], PENDING_ALPHA)
