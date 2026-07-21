"""Shared object color palette for the task categories' asset builders."""

PALETTE: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.15, 0.15, 1.0),
    "green": (0.15, 0.7, 0.2, 1.0),
    "blue": (0.15, 0.3, 0.85, 1.0),
    "yellow": (0.9, 0.8, 0.1, 1.0),
    "purple": (0.6, 0.2, 0.8, 1.0),
    "orange": (0.95, 0.55, 0.1, 1.0),
    "cyan": (0.1, 0.75, 0.8, 1.0),
    "gray": (0.5, 0.5, 0.5, 1.0),
    "white": (0.95, 0.95, 0.95, 1.0),
    # Cool-hue, brightness-descending arc (see tasks/domino/variants.py's
    # DOMINO_COLORS and docs/DominoTask-Analysis.md): two independent,
    # colorblind-safe ordering cues (hue position + lightness) for
    # identity-coding otherwise-identical pieces.
    "sky": (0.471, 0.855, 0.886, 1.0),
    "cobalt": (0.122, 0.302, 0.839, 1.0),
    "indigo": (0.247, 0.114, 0.647, 1.0),
    "violet": (0.373, 0.098, 0.463, 1.0),
}
