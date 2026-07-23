"""Variant definitions for the place task category.

A deliberately trivial pick-and-place sanity check: one cube, carried to a
freshly-sampled table-top target each episode, released there. Exists to
validate the QC-FQL training pipeline (offline-to-online RL) can learn
*something* before attempting harder categories (stack, domino, ...).

Unlike the other categories there is no natural difficulty axis to vary
(object count, material choice, ...) -- one cube to one target already is
the complete task -- so a single "Easy" variant is registered.
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
