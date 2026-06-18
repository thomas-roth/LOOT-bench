"""TRLC-DK1 collision configuration.

Finger-pad geom pattern (single source of truth) and collision sets for
full-contact and gripper-only training modes.
"""

# TODO?: fix small self-collision of dark part near joint2 end of link2-3 with joint4 end of link3-4
# TODO?: fix too-large collision box of link2-3 (seen when trying to hit joint2 end of link2-3 with lower black part of joint5)

from mjlab.utils.spec_config import CollisionCfg

# Single source of truth for the finger-pad geoms. The ManipulatorInterface
# (task-side matching) uses the bare pattern; the collision config anchors it
# with `.*` so the bimanual `left_`/`right_` body prefixes still match.
DK1_PAD_GEOM_PATTERN = r"finger_(left|right)_pad_collision"
_PAD_GEOMS = rf".*{DK1_PAD_GEOM_PATTERN}"

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={
        _PAD_GEOMS: 6,
        ".*_collision": 3,
    },
    friction={
        _PAD_GEOMS: (1, 5e-3, 5e-4),
        ".*_collision": (0.6,),
    },
    solref={
        _PAD_GEOMS: (0.01, 1),
    },
    priority={
        _PAD_GEOMS: 1,
    },
)

# Collisions limited to the wrist link and fingers: cheaper, and the arm body
# cannot get stuck on scene objects during training.
#
# contype/conaffinity use two bits instead of one: bit 0 (SCENE) lets the
# wrist/fingers contact scene objects, bit 1 (SELF) keeps the whole arm
# self-colliding. Without the SELF bit the upper arm has contype=conaffinity=0,
# so e.g. the gripper can swing straight through link2-3/link4-5 when folded.
_SCENE = 1
_SELF = 2
GRIPPER_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype={
        r".*(link6-7|finger_(left|right))(_pad)?_collision": _SCENE | _SELF,
        ".*_collision": _SELF,
    },
    conaffinity={
        r".*(link6-7|finger_(left|right))(_pad)?_collision": _SCENE | _SELF,
        ".*_collision": _SELF,
    },
    condim={
        _PAD_GEOMS: 6,
        ".*_collision": 3,
    },
    friction={
        _PAD_GEOMS: (1, 5e-3, 5e-4),
        ".*_collision": (0.6,),
    },
    solref={
        _PAD_GEOMS: (0.01, 1),
    },
    priority={
        _PAD_GEOMS: 1,
    },
)
