"""Variant definitions for the cube (piece arrangement) task category.

A variant fixes the piece set and the goal arrangement:

- forms: cube, cylinder, rect (flat cuboid), domino (tall thin cuboid)
- colors: single color per stack, or a specific color order per stack
- arrangement: "stack" (pieces on top of each other) or "domino" (pieces
  standing upright in a row, close enough for the domino effect)
- available pieces: exactly the needed ones, or extra distractor pieces with
  other colors/forms
"""

from dataclasses import dataclass
from typing import Literal

Form = Literal["cube", "cylinder", "rect", "domino"]

# Half extents per form. Cylinders use (radius, half_height); boxes use
# (hx, hy, hz). Heights are what matters for stacking goal computation.
FORM_HALF_EXTENTS: dict[Form, tuple[float, ...]] = {
    "cube": (0.02, 0.02, 0.02),
    "cylinder": (0.02, 0.02),
    "rect": (0.03, 0.02, 0.015),
    "domino": (0.0075, 0.02, 0.035),
}


def form_height(form: Form) -> float:
    return 2.0 * FORM_HALF_EXTENTS[form][-1]


@dataclass(frozen=True)
class PieceCfg:
    form: Form = "cube"
    color: str = "red"


@dataclass(frozen=True)
class CubeVariantCfg:
    name: str
    """Variant identifier used in the task id (e.g. "Stack3")."""
    arrangement: Literal["stack", "domino"]
    stacks: tuple[tuple[PieceCfg, ...], ...]
    """Goal pieces per stack, ordered bottom to top. For "domino" there is a
    single 'stack' holding the row pieces in row order."""
    distractors: tuple[PieceCfg, ...] = ()
    """Extra pieces present in the scene without a goal slot."""
    interchangeable: bool = True
    """If True, pieces of the same form may fill each other's goal slot (the
    command re-matches them to slots by proximity every step), so neither the
    stacking order nor the color order affects success. Set False to enforce
    the exact slot order given in ``stacks``."""
    domino_spacing: float = 0.05
    """Center distance of neighboring dominos; must be smaller than the domino
    height (0.07 m) so toppling one knocks over the next."""

    @property
    def goal_pieces(self) -> tuple[PieceCfg, ...]:
        return tuple(p for stack in self.stacks for p in stack)

    @property
    def pieces(self) -> tuple[PieceCfg, ...]:
        return self.goal_pieces + self.distractors

    @property
    def num_goal_pieces(self) -> int:
        return len(self.goal_pieces)

    @property
    def slot_to_stack(self) -> tuple[int, ...]:
        return tuple(i for i, stack in enumerate(self.stacks) for _ in stack)

    def __post_init__(self):
        if self.arrangement == "domino":
            assert len(self.stacks) == 1, "domino uses a single row"
            assert all(p.form == "domino" for p in self.stacks[0])
            assert self.domino_spacing < form_height("domino")


def _mono(form: Form, color: str, n: int) -> tuple[PieceCfg, ...]:
    return tuple(PieceCfg(form=form, color=color) for _ in range(n))


CUBE_VARIANTS: dict[str, CubeVariantCfg] = {
    v.name: v
    for v in (
        # Stack 3 same-colored cubes; only the needed pieces exist.
        CubeVariantCfg(
            name="Stack3",
            arrangement="stack",
            stacks=(_mono("cube", "red", 3),),
        ),
        # Stack 3 cubes in the commanded color order (bottom: red, middle:
        # green, top: blue).
        CubeVariantCfg(
            name="Stack3Ordered",
            arrangement="stack",
            stacks=(
                (
                    PieceCfg(color="red"),
                    PieceCfg(color="green"),
                    PieceCfg(color="blue"),
                ),
            ),
        ),
        # Stack 3 red cubes among distractors of other colors and forms.
        CubeVariantCfg(
            name="Stack3Distract",
            arrangement="stack",
            stacks=(_mono("cube", "red", 3),),
            distractors=(
                PieceCfg(form="cube", color="blue"),
                PieceCfg(form="cylinder", color="green"),
            ),
        ),
        # Two single-color stacks: sort pieces by color.
        CubeVariantCfg(
            name="TwoStacks",
            arrangement="stack",
            stacks=(_mono("cube", "red", 2), _mono("cube", "blue", 2)),
        ),
        # Other forms.
        CubeVariantCfg(
            name="Stack3Cylinders",
            arrangement="stack",
            stacks=(_mono("cylinder", "yellow", 3),),
        ),
        CubeVariantCfg(
            name="Stack3Rects",
            arrangement="stack",
            stacks=(_mono("rect", "purple", 3),),
        ),
        # Mixed forms in fixed order: rect base, cube middle, cylinder top.
        CubeVariantCfg(
            name="StackMixedForms",
            arrangement="stack",
            stacks=(
                (
                    PieceCfg(form="rect", color="red"),
                    PieceCfg(form="cube", color="red"),
                    PieceCfg(form="cylinder", color="red"),
                ),
            ),
        ),
        # Domino rows.
        CubeVariantCfg(
            name="Domino5",
            arrangement="domino",
            stacks=(_mono("domino", "orange", 5),),
        ),
        CubeVariantCfg(
            name="Domino5Distract",
            arrangement="domino",
            stacks=(_mono("domino", "orange", 5),),
            distractors=(
                PieceCfg(form="domino", color="cyan"),
                PieceCfg(form="cube", color="red"),
            ),
        ),
    )
}
