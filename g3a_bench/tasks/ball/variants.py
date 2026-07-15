"""Variant definitions for the ball task category.

Move a ball into a target area. Variants combine an *action* (how the ball is
moved) with a *target* (where it must end up):

- actions: ``golf`` (one closed-gripper strike, do not lead the ball),
  ``push`` (lead the ball along with a closed gripper), ``throw`` (fling the
  ball over a wall).
- targets: ``marked`` (a marked disk on the ground), ``u_boundary`` (a U-shaped
  wall pocket), ``hole`` (a real pit in a raised play platform).

The action sets how far the target sits from the ball: ``push`` keeps it within
reach so the robot can guide the ball all the way; ``golf`` puts it out of reach
so the ball must roll there after a strike; ``throw`` puts it behind a wall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BallAction = Literal["golf", "push", "throw"]
TargetType = Literal["marked", "u_boundary", "hole"]

BALL_RADIUS = 0.025
BALL_MASS = 0.03

# Target centre x by action (env-local metres, robot base at x=0). The DK1's
# measured max horizontal reach at ball height is ~0.74m (0.9 soft-joint-limit
# factor): push sits near that border (still reachable end-to-end), golf/throw
# sit clear of it (genuinely unreachable by the gripper at any height, so the
# ball must actually be struck/thrown there, not carried).
TARGET_X = {"golf": 0.85, "push": 0.68, "throw": 0.85}
# Ball spawn centre.
BALL_SPAWN_X = 0.27
# Throw wall (between ball and target). Half-extents shared by the wall's geom
# (assets.py) and its collision-box termination check (mdp/terminations.py) so
# they can't drift apart.
WALL_X = 0.42
WALL_HEIGHT = 0.20
WALL_HALF_X = 0.02
WALL_HALF_Y = 0.25
# Raised platform top for hole variants (the pit drops to the floor below).
PLATFORM_Z = 0.03
HOLE_RADIUS = 0.045


@dataclass(frozen=True)
class BallVariantCfg:
    name: str
    """Variant identifier used in the task id (e.g. "GolfMarked")."""
    action: BallAction
    target: TargetType
    target_radius: float = 0.04
    """Success radius around the target centre (ignored extent for hole)."""
    at_rest_speed: float = 0.06
    """Ball speed below which it counts as at rest."""

    @property
    def target_x(self) -> float:
        return TARGET_X[self.action]

    @property
    def has_wall(self) -> bool:
        return self.action == "throw"

    @property
    def has_platform(self) -> bool:
        return self.target == "hole"

    @property
    def surface_z(self) -> float:
        """Height of the surface the ball starts/rolls on."""
        return PLATFORM_Z if self.has_platform else 0.0

    def __post_init__(self):
        if self.action == "throw":
            # A pit behind a wall is reached by chance only; keep throw targets
            # on a flat surface.
            assert self.target in ("marked", "u_boundary")


BALL_VARIANTS: dict[str, BallVariantCfg] = {
    v.name: v
    for v in (
        BallVariantCfg(name="GolfMarked", action="golf", target="marked"),
        BallVariantCfg(name="GolfU", action="golf", target="u_boundary"),
        BallVariantCfg(name="GolfHole", action="golf", target="hole"),
        BallVariantCfg(name="PushMarked", action="push", target="marked"),
        BallVariantCfg(name="PushU", action="push", target="u_boundary"),
        BallVariantCfg(name="PushHole", action="push", target="hole"),
        BallVariantCfg(name="ThrowMarked", action="throw", target="marked"),
        BallVariantCfg(name="ThrowU", action="throw", target="u_boundary"),
    )
}
