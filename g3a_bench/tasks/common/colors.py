"""Shared object color palette for the task categories' asset builders."""

PALETTE: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.15, 0.15, 1.0),
    "green": (0.15, 0.7, 0.2, 1.0),
    "blue": (0.15, 0.3, 0.85, 1.0),
    "yellow": (0.9, 0.8, 0.1, 1.0),
    "purple": (0.6, 0.2, 0.8, 1.0),
    "orange": (0.95, 0.55, 0.1, 1.0),
    "cyan": (0.1, 0.75, 0.8, 1.0),
    "black": (0.05, 0.05, 0.05, 1.0),
    "gray": (0.5, 0.5, 0.5, 1.0),
    "white": (0.95, 0.95, 0.95, 1.0),
}


def order_color(index: int, first_tier: int, second_tier: int) -> str:
    """Color an object by its build-order tier so the collection/stacking
    order is visible in every variant (and demos can be collected in index
    order and split on the color boundaries): the easiest task's objects are
    black, the middle tier's addition gray, the hardest tier's addition (and
    any spares) white. E.g. Stack (first_tier=3, second_tier=4): a 5-high
    stack reads black, black, black, gray, white bottom-to-top.
    """
    if index < first_tier:
        return "black"
    if index < second_tier:
        return "gray"
    return "white"
