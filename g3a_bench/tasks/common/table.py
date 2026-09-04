"""Visual-only workspace table shared by every task category."""
# pyright: reportAttributeAccessIssue=false
# (mujoco ships no type stubs for the compiled MjSpec module)

import mujoco
from mjlab.entity import EntityCfg

TABLE_RGBA = (0.5, 0.5, 0.5, 1.0)  # flat neutral gray

# Body placement, env-local. Off-centre toward ball far targets.
TABLE_POS = (0.48, 0.0, 0.0)

# Half-extents of the tabletop footprint.
TABLE_HALF_X = 0.53
TABLE_HALF_Y = 0.35

# Top surface slightly above ground to avoid z-fighting.
TABLE_TOP_Z = 0.002
TABLE_HEIGHT = 0.35


def table_entity_cfg() -> EntityCfg:
    """Fixed-base table entity. Pin via workbench fixed_props."""
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
