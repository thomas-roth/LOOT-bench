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

OUTLINE_RGBA: Rgba = (0.85, 0.1, 0.1, 0.95)
"""Ghost/target-marker outline color: a red edge marking a marker as a
virtual target rather than a real object, at any state. Opposite the cool
identity-color hues on the wheel, so it contrasts hard against them."""

OUTLINE_SCALE = 1.15
"""Outline box size, as a multiple of its fill box's size (per axis).
mjlab's debug visualizer only draws solid boxes (no border primitive), so an
outline is faked by drawing a slightly larger, near-opaque box first and a
smaller, category-colored fill box on top -- normal depth testing then shows
the outer box only on the margin the inner one doesn't cover. A *proportional*
scale, not a fixed additive margin: goal objects here can be highly
non-cubic (e.g. a domino's half-extents are (0.0075, 0.02, 0.035) -- a fixed
few-mm margin would consume nearly half of the thin axis and read as a solid
block instead of a thin border. Scaling keeps the border a consistent
fraction of size on every axis, for any object shape."""


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
