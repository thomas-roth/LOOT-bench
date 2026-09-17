"""Shared object color palette for the task categories' asset builders."""

COLOR_VALUES: dict[str, tuple[float, float, float, float]] = {
    "black": (0.1, 0.1, 0.1, 1.0),
    "navy": (0.0471, 0.1294, 0.2784, 1.0),
    "teal": (0.0, 0.3961, 0.4039, 1.0),
    "azure": (0.3529, 0.5725, 0.8392, 1.0),
    "ice": (0.5255, 0.8471, 0.8118, 1.0),
    "white": (1.0, 1.0, 1.0, 1.0),
}
"""Values of the shared color palette.

They are ordered, dark to light, so a piece's place in the row is read off how
light it is; the hue shifts between blue and cyan along the way and backs the
ordering up, but lightness leads.

| Color | Hex | RGB (0-255) | OKLCH (L, C, H) |
| --- | --- | --- | --- |
| `black` | `#1a1a1a` | `rgb(26, 26, 26)` | `0.217, 0.000, 0.0` |
| `navy` | `#0c2147` | `rgb(12, 33, 71)` | `0.256, 0.075, 260.3` |
| `teal` | `#006567` | `rgb(0, 101, 103)` | `0.460, 0.078, 197.3` |
| `azure` | `#5a92d6` | `rgb(90, 146, 214)` | `0.650, 0.119, 254.1` |
| `ice` | `#86d8cf` | `rgb(134, 216, 207)` | `0.827, 0.082, 187.3` |
| `white` | `#ffffff` | `rgb(255, 255, 255)` | `1.000, 0.000, 0.0` |"""

COLOR_NAMES = list(COLOR_VALUES)
"""Names of the colors in the shared palette."""

SPLIT_TIERS: tuple[str, str, str, str] = (
    COLOR_NAMES[1],
    COLOR_NAMES[2],
    COLOR_NAMES[3],
    COLOR_NAMES[5],
)
"""Tier colors in split order; index into these to recolor a category."""


def order_color(
    index: int, first_tier: int, second_tier: int, third_tier: int | None = None
) -> str:
    """Color an object by which split tier first needs it.

    Objects tail-extend across a category's task axis, so the tier
    boundaries are the object counts of the easier tasks. Coloring on them
    shows the teleoperator the collection order and marks where a recording
    can be cut (see ``scripts/split_demos.py``). E.g. 3Pis' 9 blocks with
    the 1Pi/2Pis counts (3, 6) read 3 navy, 3 teal, 3 white, one tier
    per level. Pass ``third_tier`` for a four-task axis: the tiers then read
    navy, teal, azure, white, one tier per easier task.
    """
    if index < first_tier:
        return SPLIT_TIERS[0]
    if index < second_tier:
        return SPLIT_TIERS[1]
    if third_tier is not None and index < third_tier:
        return SPLIT_TIERS[2]
    return SPLIT_TIERS[3]
