"""MjSpec builders for the arrangement pieces."""
# pyright: reportAttributeAccessIssue=false
# (mujoco ships no type stubs for the compiled MjSpec module)

import mujoco

from g3a_bench.tasks.cube.variants import FORM_HALF_EXTENTS, PieceCfg

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
