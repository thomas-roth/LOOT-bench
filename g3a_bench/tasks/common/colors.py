"""Shared object color palette for the task categories' asset builders."""

PALETTE: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.15, 0.15, 1.0),
    "green": (0.15, 0.7, 0.2, 1.0),
    "blue": (0.15, 0.3, 0.85, 1.0),
    "yellow": (0.9, 0.8, 0.1, 1.0),
    "purple": (0.6, 0.2, 0.8, 1.0),
    "orange": (0.95, 0.55, 0.1, 1.0),
    "cyan": (0.1, 0.75, 0.8, 1.0),
    # Split-tier ramp (see order_color); darkened against the table.
    "black": (0.05, 0.05, 0.05, 1.0),
    "gray": (0.35, 0.35, 0.35, 1.0),
    "white": (0.95, 0.95, 0.95, 1.0),
    # Cool-hue, brightness-descending arc (see tasks/domino/variants.py's
    # DOMINO_COLORS and docs/DominoTask-Analysis.md): two independent,
    # colorblind-safe ordering cues (hue position + lightness) for
    # identity-coding otherwise-identical pieces.
    "sky": (0.149, 0.851, 0.851, 1.0),
    "cobalt": (0.094, 0.239, 0.667, 1.0),
    "indigo": (0.278, 0.078, 0.482, 1.0),
    "violet": (0.337, 0.063, 0.267, 1.0),
}

SPLIT_TIERS: tuple[str, str, str] = ("black", "gray", "white")
"""Tier colors in split order; index into these to recolor a category."""


def order_color(index: int, first_tier: int, second_tier: int) -> str:
    """Color an object by which split tier first needs it.

    Objects tail-extend across a category's difficulty axis, so the tier
    boundaries are the object counts of the easier tasks. Coloring on them
    shows the teleoperator the collection order and marks where a recording
    can be cut (see ``scripts/split_demos.py``). E.g. Stack5's 10 blocks with
    the Stack3/Stack4 counts (6, 8) read 6 black, 2 gray, 2 white.
    """
    if index < first_tier:
        return SPLIT_TIERS[0]
    if index < second_tier:
        return SPLIT_TIERS[1]
    return SPLIT_TIERS[2]
