"""Variant definitions for the place task category.

A deliberately trivial pick-and-place sanity check: one cube, carried to a
freshly-sampled table-top target each episode, released there. Exists to
validate the QC-FQL training pipeline (offline-to-online RL) can learn
*something* before attempting harder categories (stack, domino, ...).

There is no difficulty axis to vary -- one cube to one target already is the
complete task -- so a single "Easy" variant is registered. The category once
carried a second, "Dense" variant that added progress shaping on top of a
sparse outcome bonus. That axis is gone: ``tasks/common/rewards.py`` now gives
every category one reward in ``[-1, 0]`` with the progress signal already
inside it, so there is no sparse mode left to shape.
"""

from dataclasses import dataclass

CUBE_HALF_EXTENTS = (0.02, 0.02, 0.02)
"""Matches stack's "cube" form (``FORM_HALF_EXTENTS["cube"]``) -- reuse the
established box geometry rather than inventing new physics constants."""

CUBE_MASS = 0.10
"""Matches stack's ``FORM_MASS["cube"]``."""

CUBE_HEIGHT = 2.0 * CUBE_HALF_EXTENTS[2]

CUBE_COLOR = "red"


@dataclass(frozen=True)
class PlaceVariantCfg:
    name: str
    """Variant identifier used in the task id (e.g. "Easy")."""
    margin: float = 1.0
    """Scale on the position-success tolerance (default = tight, for data
    collection; kept for structural parity with every other category's
    margin convention)."""


PLACE_VARIANTS: dict[str, PlaceVariantCfg] = {
    v.name: v for v in (PlaceVariantCfg(name="Easy"),)
}
