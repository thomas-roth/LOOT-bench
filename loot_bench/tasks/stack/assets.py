"""MjSpec builders for the stack blocks."""
# pyright: reportAttributeAccessIssue=false
# (mujoco ships no type stubs for the compiled MjSpec module)

import mujoco

from loot_bench.tasks.common.colors import COLOR_VALUES
from loot_bench.tasks.stack.variants import BlockCfg


def get_block_spec(block: BlockCfg, name: str = "block") -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=block.half_extents,
        mass=block.mass,
        rgba=COLOR_VALUES[block.color],
        condim=6,
        friction=(1.0, 0.01, 0.0001),
    )
    return spec
