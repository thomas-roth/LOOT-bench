"""MjSpec builder for the place cube."""
# pyright: reportAttributeAccessIssue=false
# (mujoco ships no type stubs for the compiled MjSpec module)

import mujoco

from g3a_bench.tasks.common.colors import PALETTE
from g3a_bench.tasks.place.variants import CUBE_HALF_EXTENTS, CUBE_MASS


def get_cube_spec(color: str = "red", name: str = "cube") -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=CUBE_HALF_EXTENTS,
        mass=CUBE_MASS,
        rgba=PALETTE[color],
        condim=6,
        friction=(1.0, 0.01, 0.0001),
    )
    return spec
