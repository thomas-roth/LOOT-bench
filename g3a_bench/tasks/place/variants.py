"""Variant definitions for the place task category.

A deliberately trivial pick-and-place sanity check: one cube, carried to a
freshly-sampled table-top target each episode, released there. Exists to
validate the QC-FQL training pipeline (offline-to-online RL) can learn
*something* before attempting harder categories (stack, domino, ...).

Unlike the other categories there is no natural difficulty axis to vary
(object count, material choice, ...) -- one cube to one target already is
the complete task. The one axis that *is* varied is the reward mode: "Easy"
ships the sparse outcome signal every real category is planned to use, and
"Easy-Dense" adds the progress shaping, as the lever for the flat-eval
investigation (.scratch/qcfql-flat-success/issues/08-progress-reward-decision.md).
"""

from dataclasses import dataclass
from typing import Literal

RewardMode = Literal["success", "progress"]
"""Named tier of active task reward terms (see CONTEXT.md). ``success`` keeps
only the sparse outcome indicator; ``progress`` adds the completion-fraction
shaping term on top of it."""

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
    reward_mode: RewardMode = "success"
    """Which task reward terms are active. ``progress`` raises ``reward_max``
    from 1.0 to 2.0, so a dataset recorded for one mode is rejected by
    ``qc.validate_reward_convention`` against the other -- the two modes need
    their own ``.npz``."""


PLACE_VARIANTS: dict[str, PlaceVariantCfg] = {
    v.name: v
    for v in (
        PlaceVariantCfg(name="Easy"),
        PlaceVariantCfg(name="EasyDense", reward_mode="progress"),
    )
}
