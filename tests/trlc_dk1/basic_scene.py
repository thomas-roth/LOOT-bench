"""Launch the DK1 robot in the MuJoCo viewer with a floor and skybox.

See tests/trlc_dk1/README.md.
"""

import mujoco
import mujoco.viewer
from mjlab.entity.entity import Entity

from loot_bench.robots.trlc_dk1 import get_unimanual_dk1_robot_cfg

entity = Entity(get_unimanual_dk1_robot_cfg())

# add floor
entity.spec.add_texture(
    name="grid",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=(0.2, 0.3, 0.4),
    rgb2=(0.9, 0.9, 0.9),
    width=300,
    height=300,
)
# textures is indexed by texture role; the visible color is the RGB role, not
# slot 0 (USER). Binding to slot 0 leaves the plane flat grey.
_tex = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
_tex[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "grid"
entity.spec.add_material(
    name="grid", textures=_tex, texrepeat=(5, 5), texuniform=True, reflectance=0.2
)
entity.spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(2, 2, 0.1),
    material="grid",
)

# add skybox
entity.spec.add_texture(
    name="skybox",
    type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    rgb1=(0.6, 0.7, 0.9),
    rgb2=(0.9, 0.9, 0.95),
    width=64,
    height=64,
)

mujoco.viewer.launch(entity.spec.compile())
