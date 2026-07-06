"""MjSpec builders for the arrangement pieces and the "push" maze wall.

The maze is a *fixed-base* entity (a body with no free joint, auto-wrapped by
mjlab as a mocap body pinned to the env origin), like the insertion boards.
Its wall constants below also define the spawn and goal bands the command term
samples from, so wall geometry and goal sampling derive from one source and
cannot drift apart.
"""
# pyright: reportAttributeAccessIssue=false
# (mujoco ships no type stubs for the compiled MjSpec module)

import mujoco

from g3a_bench.tasks.cube.variants import FORM_HALF_EXTENTS, CubeVariantCfg, PieceCfg

PALETTE: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.15, 0.15, 1.0),
    "green": (0.15, 0.7, 0.2, 1.0),
    "blue": (0.15, 0.3, 0.85, 1.0),
    "yellow": (0.9, 0.8, 0.1, 1.0),
    "purple": (0.6, 0.2, 0.8, 1.0),
    "orange": (0.95, 0.55, 0.1, 1.0),
    "cyan": (0.1, 0.75, 0.8, 1.0),
    "white": (0.95, 0.95, 0.95, 1.0),
}

PIECE_MASS = 0.04

WALL_RGBA = (0.55, 0.55, 0.6, 1.0)

# --- "push" maze layout (env-local metres). --------------------------------
# A single barrier wall runs along y at PUSH_WALL_X with ``variant.maze_gates``
# gate openings. Pieces spawn in the near (robot-side) band and their goal
# zones are sampled in the far band, so every piece has to travel through a
# gate (or over the wall) to reach its zone.
PUSH_WALL_X = 0.31
PUSH_WALL_HALF_T = 0.008
PUSH_WALL_HEIGHT = 0.06  # taller than any piece so a straight place is blocked
PUSH_WALL_HALF_SPAN = 0.20  # wall half length along y
PUSH_GATE_HALF_W = 0.05  # half width of each gate opening
PUSH_GATE_SPACING = 0.20  # centre distance of neighboring gates

# Spawn/goal bands, inset from the wall faces so the gripper fits between a
# piece and the wall on either side.
_PUSH_BAND_MARGIN = 0.05
PUSH_SPAWN_X = (0.19, PUSH_WALL_X - _PUSH_BAND_MARGIN)
PUSH_GOAL_X = (PUSH_WALL_X + _PUSH_BAND_MARGIN, 0.44)
PUSH_BAND_HALF_Y = 0.16
# Goal zones sit on fixed y lanes (shuffled + jittered per reset); the spacing
# keeps zones of neighboring lanes from overlapping.
PUSH_GOAL_LANE_SPACING = 0.11


def push_gate_centers(num_gates: int) -> list[float]:
    """Gate centre y coordinates, evenly spread around the wall middle."""
    reach = (num_gates - 1) / 2 * PUSH_GATE_SPACING + PUSH_GATE_HALF_W
    assert reach < PUSH_WALL_HALF_SPAN, "gates must open inside the wall span"
    return [(k - (num_gates - 1) / 2) * PUSH_GATE_SPACING for k in range(num_gates)]


def get_piece_spec(piece: PieceCfg, name: str = "piece") -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_joint")
    half = FORM_HALF_EXTENTS[piece.form]
    geom_type = (
        mujoco.mjtGeom.mjGEOM_CYLINDER
        if piece.form == "cylinder"
        else mujoco.mjtGeom.mjGEOM_BOX
    )
    size = (half + (0.0,) * 3)[:3]
    body.add_geom(
        name=f"{name}_geom",
        type=geom_type,
        size=size,
        mass=PIECE_MASS,
        rgba=PALETTE[piece.color],
    )
    return spec


def get_maze_spec(variant: CubeVariantCfg, name: str = "maze") -> mujoco.MjSpec:
    """Fixed-base barrier wall with gate openings for the "push" arrangement."""
    spec = mujoco.MjSpec()
    # Body stays at the origin; the geoms carry the env-local workspace pose
    # (mjlab overwrites a fixed-base root body's pose from init_state).
    maze = spec.worldbody.add_body(name=name)
    edges = [-PUSH_WALL_HALF_SPAN]
    for c in push_gate_centers(variant.maze_gates):
        edges += [c - PUSH_GATE_HALF_W, c + PUSH_GATE_HALF_W]
    edges.append(PUSH_WALL_HALF_SPAN)
    hz = PUSH_WALL_HEIGHT / 2
    for i in range(0, len(edges), 2):
        y0, y1 = edges[i], edges[i + 1]
        maze.add_geom(
            name=f"{name}_seg_{i // 2}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(PUSH_WALL_X, (y0 + y1) / 2, hz),
            size=(PUSH_WALL_HALF_T, (y1 - y0) / 2, hz),
            rgba=WALL_RGBA,
        )
    return spec
