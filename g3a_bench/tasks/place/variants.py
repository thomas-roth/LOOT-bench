"""Variant definitions for the place task category.

A deliberately trivial pick-and-place sanity check: one cube, carried to a
table-top target, released there. Exists to validate the QC-FQL training
pipeline (offline-to-online RL) can learn *something* before attempting
harder categories (stack, domino, ...).

There is no difficulty axis in the usual sense: one cube to one target
already is the complete task. What the variants vary is how much of the
episode is *sampled*: the cube spawn and the target each are either drawn
fresh every reset ("Rand") or pinned to a constant pose ("Fixed"), giving
the four combinations below. Pinning one or both removes a source of
generalization the policy would otherwise have to cover, which is what makes
``FixedS-FixedT`` the smallest learnable version of the task.

The category once carried a second, "Dense" variant that added progress
shaping on top of a sparse outcome bonus. That axis is gone:
``tasks/common/rewards.py`` now gives every category one reward in
``[-1, 0]`` with the progress signal already inside it, so there is no
sparse mode left to shape.
"""

from dataclasses import dataclass

CUBE_HALF_EXTENTS = (0.02, 0.02, 0.02)
"""Matches stack's "cube" form (``FORM_HALF_EXTENTS["cube"]``). Reuses the
established box geometry rather than inventing new physics constants."""

CUBE_MASS = 0.10
"""Matches stack's ``FORM_MASS["cube"]``."""

CUBE_HEIGHT = 2.0 * CUBE_HALF_EXTENTS[2]

CUBE_COLOR = "red"


@dataclass(frozen=True)
class PlaceVariantCfg:
    name: str
    """Variant identifier used in the task id (e.g. "RandS-RandT")."""
    random_spawn: bool = True
    """Draw the cube's spawn pose fresh each reset. ``False`` pins it to
    ``mdp/commands.py::FIXED_SPAWN_XY`` at yaw 0."""
    random_target: bool = True
    """Draw the target point fresh each reset. ``False`` pins it to
    ``mdp/commands.py::FIXED_TARGET_XY``."""
    margin: float = 1.0
    """Scale on the position-success tolerance (default = tight, for data
    collection; kept for structural parity with every other category's
    margin convention)."""


def _variant_name(random_spawn: bool, random_target: bool) -> str:
    """``"<Rand|Fixed>S-<Rand|Fixed>T"``: S for the cube spawn, T for the
    target. ``config/__init__.py`` splits it back on the hyphen into the task
    id's ``<task>-<variation>`` slots."""
    return (
        f"{'Rand' if random_spawn else 'Fixed'}S-"
        f"{'Rand' if random_target else 'Fixed'}T"
    )


PLACE_VARIANTS: dict[str, PlaceVariantCfg] = {
    v.name: v
    for v in (
        PlaceVariantCfg(
            name=_variant_name(random_spawn, random_target),
            random_spawn=random_spawn,
            random_target=random_target,
        )
        for random_spawn in (True, False)
        for random_target in (True, False)
    )
}
