"""Observation terms shared by every task category."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def rot6d_from_matrix(mat: torch.Tensor) -> torch.Tensor:
    """The first two columns of a rotation matrix, flattened (..., 6).

    A continuous, unique encoding of a rotation: the local x and y axes in
    world coordinates. The third column carries no extra information, since it
    is their cross product.

    Prefer this over a quaternion for anything a network reads. A quaternion
    covers each rotation twice (``q`` and ``-q`` are the same rotation), so
    which of the two a converter returns is a free choice -- and mjlab's
    ``quat_from_matrix`` makes that choice with an ``argmax`` over four
    candidates that are equal in exact arithmetic. Wherever two candidates tie,
    float noise decides, and one smooth motion comes back as a sequence that
    jumps by up to 2.0 between steps. The 6D form has no sign freedom, so it
    cannot jump.

    Args:
        mat: (..., 3, 3) rotation matrices, row-major.
    """
    assert mat.shape[-2:] == (3, 3), f"expected (..., 3, 3), got {tuple(mat.shape)}"
    return mat[..., :, :2].mT.reshape(*mat.shape[:-2], 6)


def ee_pose_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector site pose in the env-local world frame (E, 9).

    Position (xyz) then the 6D rotation of :func:`rot6d_from_matrix`.
    ``asset_cfg.site_names`` is set to the robot's grasp site by
    ``manipulator_wiring.wire_manipulator_into_env``.

    Publishing a quaternion here would be a bug, not a style choice. Site
    orientation reaches the entity as a rotation matrix, and ``site_quat_w``
    converts it with a sign-unstable branch (see :func:`rot6d_from_matrix`).
    The DK1 grasp site sits right on the unstable region: measured on
    ``G3A-Stack-Low-Dk1_uni``, the sign flipped on 14 of 120 consecutive steps,
    a jump of 2.0 against a median step change of 0.005. The same physical
    state would reach the policy as two different observations.

    Going back through ``matrix_from_quat`` removes the flip rather than
    hiding it: a quaternion and its negation name the same rotation, so the
    conversion to a matrix drops the sign the branch invented. Canonicalizing
    the quaternion instead would not work -- the DK1 grasp site passes through
    ``w == 0``, where the sign of ``w`` is noise.

    Body poses do not share the problem. Their quaternion comes from
    ``data.xquat``, which MuJoCo integrates continuously, so
    :func:`flatten_object_poses` keeps a quaternion.
    """
    robot: Entity = env.scene[asset_cfg.name]
    pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    quat = robot.data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
    pos = pos - env.scene.env_origins
    return torch.cat([pos, rot6d_from_matrix(matrix_from_quat(quat))], dim=-1)


def flatten_pose(
    pos_w: torch.Tensor, quat_w: torch.Tensor, env_origins: torch.Tensor
) -> torch.Tensor:
    """A single object's pose in the env-local world frame (E, 7).

    Position (env-relative xyz) then quaternion (wxyz).
    """
    pos = pos_w - env_origins
    return torch.cat([pos, quat_w], dim=-1)


def flatten_object_poses(
    pos_w: torch.Tensor, quat_w: torch.Tensor, env_origins: torch.Tensor
) -> torch.Tensor:
    """Per-env object poses in the env-local world frame, flattened (E, N*7).

    Object-major: each object contributes (pos xyz, quat wxyz). Dropping the
    trailing objects' 7-dim slices adapts the observation to an easier task
    (demo cutoff).

    Args:
        pos_w: (E, N, 3) object positions.
        quat_w: (E, N, 4) object orientations.
        env_origins: (E, 3).
    """
    num_envs = pos_w.shape[0]
    pos = pos_w - env_origins.unsqueeze(1)
    return torch.cat([pos, quat_w], dim=-1).reshape(num_envs, -1)
