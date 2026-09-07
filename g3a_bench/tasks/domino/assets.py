"""MjSpec builders for the domino pieces."""
# pyright: reportAttributeAccessIssue=false
# (mujoco ships no type stubs for the compiled MjSpec module)

import mujoco

from g3a_bench.tasks.common.colors import COLOR_VALUES
from g3a_bench.tasks.domino.variants import domino_half_extents, domino_mass


def get_domino_spec(
    index: int, different: bool, color: str, name: str = "domino"
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=domino_half_extents(index, different),
        mass=domino_mass(index, different),
        rgba=COLOR_VALUES[color],
        condim=6,
        friction=(1.0, 0.01, 0.0001),
    )
    return spec
