"""TRLC-DK1 collision configuration.

Finger-pad geom pattern (single source of truth) and collision sets for the
three modes: full contact, gripper-only (no self-collision, the unimanual
training default), and gripper-only scene contact with whole-arm
self-collision (the bimanual default).
"""

# The two TODOs below only matter for the self-colliding modes
# (FULL_COLLISION, GRIPPER_ONLY_WITH_SELF_COLLISION); GRIPPER_ONLY_COLLISION
# disables the affected geoms entirely.
# TODO: fix small self-collision of dark part near joint2 end of link2-3 with
#   joint4 end of link3-4.
# TODO: fix too-large collision box of link2-3 (seen when trying to hit joint2
#   end of link2-3 with lower black part of joint5).

from mjlab.utils.spec_config import CollisionCfg

# Bare finger-pad pattern; anchored below with `.*` so the bimanual
# `left_`/`right_` body prefixes still match.
DK1_PAD_GEOM_PATTERN = r"finger_(left|right)_pad_collision"
_PAD_GEOMS = rf".*{DK1_PAD_GEOM_PATTERN}"

# The gripper geoms: wrist link, fingers, and finger pads. Anchored with `.*`
# so the bimanual `left_`/`right_` body prefixes still match.
DK1_GRIPPER_GEOMS_PATTERN = r".*(link6-7|finger_(left|right))(_pad)?_collision"


def _collision_with_pad_tuning(
    geom_names_expr: tuple[str, ...],
    contype: int | dict[str, int] = 1,
    conaffinity: int | dict[str, int] = 1,
) -> CollisionCfg:
    """CollisionCfg with the contact tuning shared by all three modes.

    Grippy high-condim finger pads, plain condim-3 contacts everywhere else —
    grasp physics are identical across modes.
    """
    return CollisionCfg(
        geom_names_expr=geom_names_expr,
        contype=contype,
        conaffinity=conaffinity,
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


FULL_COLLISION = _collision_with_pad_tuning((".*_collision",))

# Only the wrist link, fingers, and pads collide — with the scene, and (the
# sibling finger pair only, deliberately: fingers must not pass through each
# other when closing empty) with each other. Every other robot geom has
# collision disabled via disable_other_geoms, so no arm self-collision remains;
# the arm can fold through itself. Wrist<->finger pairs are removed by MuJoCo's
# parent-child filter.
GRIPPER_ONLY_COLLISION = _collision_with_pad_tuning((DK1_GRIPPER_GEOMS_PATTERN,))

# Scene contact limited to the wrist link and fingers — cheaper than full, and
# the arm body cannot get stuck on scene objects during training — but the
# whole arm keeps self-colliding.
#
# contype/conaffinity use two bits instead of one: bit 0 (SCENE) lets the
# wrist/fingers contact scene objects, bit 1 (SELF) keeps the whole arm
# self-colliding. Without the SELF bit the upper arm has contype=conaffinity=0,
# so e.g. the gripper can swing straight through link2-3/link4-5 when folded.
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
