"""MjSpec builders for the stack blocks."""
# pyright: reportAttributeAccessIssue=false
# (mujoco ships no type stubs for the compiled MjSpec module)

import mujoco

from g3a_bench.tasks.common.colors import PALETTE
from g3a_bench.tasks.stack.variants import FORM_HALF_EXTENTS, FORM_MASS, BlockCfg


def get_block_spec(block: BlockCfg, name: str = "block") -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_joint")
    half = FORM_HALF_EXTENTS[block.form]
    geom_type = (
        mujoco.mjtGeom.mjGEOM_CYLINDER
        if block.form == "cylinder"
        else mujoco.mjtGeom.mjGEOM_BOX
    )
    size = (half[0], half[2], 0.0) if block.form == "cylinder" else half
    body.add_geom(
        name=f"{name}_geom",
        type=geom_type,
        size=size,
        mass=FORM_MASS[block.form],
        rgba=PALETTE[block.color],
        condim=6,
        friction=(1.0, 0.01, 0.0001),
    )
    return spec
