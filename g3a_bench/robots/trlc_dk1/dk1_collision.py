"""DK1 collision configs: full, gripper-only, gripper-only with self-collision."""

# TODOs affect self-colliding modes only; GRIPPER_ONLY_COLLISION disables
# affected geoms.
# TODO: fix self-collision of dark part near link2-3/link3-4.
# TODO: fix oversized collision box of link2-3.

from mjlab.utils.spec_config import CollisionCfg

# Bare finger-pad pattern; anchored below with `.*` so the bimanual
# `left_`/`right_` body prefixes still match.
DK1_PAD_GEOM_PATTERN = r"finger_(left|right)_pad_collision"
DK1_PAD_GEOMS_PATTERN = rf".*{DK1_PAD_GEOM_PATTERN}"

# The gripper geoms: wrist link, fingers, and finger pads. Anchored with `.*`
# so the bimanual `left_`/`right_` body prefixes still match.
DK1_GRIPPER_GEOMS_PATTERN = r".*(link6-7|finger_(left|right))(_pad)?_collision"


def _collision_with_pad_tuning(
    geom_names_expr: tuple[str, ...],
    contype: int | dict[str, int] = 1,
    conaffinity: int | dict[str, int] = 1,
) -> CollisionCfg:
    """CollisionCfg with shared contact tuning (grippy pads, plain rest)."""
    return CollisionCfg(
        geom_names_expr=geom_names_expr,
        contype=contype,
        conaffinity=conaffinity,
        condim={
            DK1_PAD_GEOMS_PATTERN: 6,
            ".*_collision": 3,
        },
        friction={
            DK1_PAD_GEOMS_PATTERN: (1, 5e-3, 5e-4),
            ".*_collision": (0.6,),
        },
        solref={
            DK1_PAD_GEOMS_PATTERN: (0.01, 1),
        },
        priority={
            DK1_PAD_GEOMS_PATTERN: 1,
        },
    )


FULL_COLLISION = _collision_with_pad_tuning((".*_collision",))

# Gripper-only: wrist/fingers collide with scene and each other; arm disabled.
# Wrist/finger pairs removed by MuJoCo parent-child filter.
GRIPPER_ONLY_COLLISION = _collision_with_pad_tuning((DK1_GRIPPER_GEOMS_PATTERN,))

# Gripper-only scene contact; whole arm keeps self-collision.
# Bit 0 SCENE for wrist/fingers, bit 1 SELF for the arm.
_SCENE = 1
_SELF = 2
GRIPPER_ONLY_WITH_SELF_COLLISION = _collision_with_pad_tuning(
    (".*_collision",),
    contype={
        DK1_GRIPPER_GEOMS_PATTERN: _SCENE | _SELF,
        ".*_collision": _SELF,
    },
    conaffinity={
        DK1_GRIPPER_GEOMS_PATTERN: _SCENE | _SELF,
        ".*_collision": _SELF,
    },
)
