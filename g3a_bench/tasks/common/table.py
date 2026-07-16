"""Shared workspace table, reused as fixed-base scene dressing by every task
category.

Purely visual (no collision): the ground ``plane`` terrain remains the actual
physical resting/contact surface for every object, so this box never needs to
line up with per-task spawn heights. It only needs to *look* like the bench
those objects sit on.

One fixed size is shared unchanged across categories rather than sized per
task, so it must cover the union of every category's workspace:
- stacking/domino: object scatter region, x in (0.22, 0.42), y in
  (-0.18, 0.18).
- shape sorter: sorter cube + piece scatter, x in (0.08, 0.60), y in
  (-0.34, 0.34).
- ball: farthest target (the "hole" platform reaches x <= 0.975 now that
  golf/throw sit beyond the arm's ~0.74m reach), y in (-0.25, 0.25).
plus a margin and the robot base itself (env origin) so the arm reads as
mounted on the table rather than floating above bare floor.
"""
# pyright: reportAttributeAccessIssue=false
# (mujoco ships no type stubs for the compiled MjSpec module)

import mujoco
from mjlab.entity import EntityCfg

TABLE_RGBA = (0.5, 0.5, 0.5, 1.0)  # flat neutral gray

# Body placement (env-local); geoms below are relative to this. Off-centre
# (rather than symmetric around the robot base) so the footprint below can
# grow to the right, toward the ball task's far targets, without needing to
# also extend left past the robot base.
TABLE_POS = (0.48, 0.0, 0.0)

# Half-extents of the tabletop footprint.
TABLE_HALF_X = 0.53
TABLE_HALF_Y = 0.35

# Top surface sits a hair above the ground plane to avoid z-fighting. The
# block extends straight down from there, so it's a solid cuboid, not a
# modeled table+legs.
TABLE_TOP_Z = 0.002
TABLE_HEIGHT = 0.35


def table_entity_cfg() -> EntityCfg:
    """The table as a fixed-base entity, positioned via ``init_state`` (mocap
    prop convention); add a ``reset_table`` pin event via the workbench's
    ``fixed_props``."""
    return EntityCfg(
        spec_fn=get_table_spec,
        init_state=EntityCfg.InitialStateCfg(pos=TABLE_POS),
    )


def get_table_spec(name: str = "table") -> mujoco.MjSpec:
    """Fixed-base, visual-only workspace table shared by every task category."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    hz = TABLE_HEIGHT / 2
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, TABLE_TOP_Z - hz),
        size=(TABLE_HALF_X, TABLE_HALF_Y, hz),
        rgba=TABLE_RGBA,
        contype=0,
        conaffinity=0,
    )
    return spec
