"""Variant definitions for the ball task category.

Two task families, each with its own set of layouts:

- ``minigolf``: a raised platform with a pit (a physical hole) at the target.
  The four layouts (``direct``/``using_wall``/``obstacle``/``ramp``) change
  only what sits between the ball's spawn point and the pit.
- ``throw``: a flat field with a wall between the ball's spawn point and the
  target (a marked disk or a U-shaped pocket), so the ball must clear the
  wall. ``ThrowWallVariation`` additionally moves the wall (distance and
  sink depth, i.e. effective height) every reset instead of baking a fixed
  wall into the field.

Both families share a single target x (``TARGET_X``); only the target's
local y (minigolf's ``pit_y``) and z (minigolf's ``target_z``, raised for
``ramp``) vary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BALL_RADIUS = 0.025
BALL_MASS = 0.03
BALL_SPAWN_X = 0.27
HOLE_RADIUS = 0.045
PLATFORM_Z = 0.03
RAMP_RISE = 0.03
"""Extra height of the raised (target-side) platform section, ``ramp`` only."""
TARGET_X = 0.85
WALL_X = 0.42
WALL_HEIGHT = 0.20
WALL_HALF_X = 0.02
WALL_HALF_Y = 0.25

MinigolfLayout = Literal["direct", "using_wall", "obstacle", "ramp"]
ThrowTarget = Literal["u", "marked"]


@dataclass(frozen=True)
class BallVariantCfg:
    name: str
    """Variant identifier used in the task id (e.g. "MinigolfDirect")."""
    task: Literal["minigolf", "throw"]
    margin: float = 1.0
    """Scale on the throw target radius (pit success is physical, unscaled)."""
    minigolf_layout: MinigolfLayout = "direct"
    """Minigolf-only: which platform layout to build."""
    throw_target: ThrowTarget = "marked"
    """Throw-only: pocket shape."""
    movable_wall: bool = False
    """Throw-only (WallVariation): the wall is a separate fixed entity that
    moves (distance, sink depth) every reset, instead of baked into the
    field."""
    target_radius: float = 0.04
    """Throw success radius around the target centre, before ``margin``."""
    at_rest_speed: float = 0.06
    """Ball speed below which it counts as at rest."""

    def __post_init__(self):
        if self.movable_wall:
            assert self.task == "throw"

    @property
    def pit_y(self) -> float:
        """Local y of the minigolf pit centre (UsingWall offsets it)."""
        return 0.12 if self.minigolf_layout == "using_wall" else 0.0

    @property
    def spawn_z(self) -> float:
        """Height the ball starts/rolls on at the spawn side of the field."""
        return PLATFORM_Z if self.task == "minigolf" else 0.0

    @property
    def target_z(self) -> float:
        """Height of the surface the target sits on."""
        if self.task != "minigolf":
            return 0.0
        return PLATFORM_Z + RAMP_RISE if self.minigolf_layout == "ramp" else PLATFORM_Z

    @property
    def has_wall(self) -> bool:
        return self.task == "throw"


BALL_VARIANTS: dict[str, BallVariantCfg] = {
    v.name: v
    for v in (
        BallVariantCfg(
            name="MinigolfDirect", task="minigolf", minigolf_layout="direct"
        ),
        BallVariantCfg(
            name="MinigolfUsingWall", task="minigolf", minigolf_layout="using_wall"
        ),
        BallVariantCfg(
            name="MinigolfObstacle", task="minigolf", minigolf_layout="obstacle"
        ),
        BallVariantCfg(name="MinigolfRamp", task="minigolf", minigolf_layout="ramp"),
        BallVariantCfg(name="ThrowU", task="throw", throw_target="u"),
        BallVariantCfg(name="ThrowMarkedArea", task="throw", throw_target="marked"),
        BallVariantCfg(
            name="ThrowWallVariation",
            task="throw",
            throw_target="marked",
            movable_wall=True,
        ),
    )
}
