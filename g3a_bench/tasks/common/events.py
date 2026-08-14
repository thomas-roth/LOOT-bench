"""Reset events shared by all task categories.

mjlab's :func:`~mjlab.envs.mdp.events.reset_joints_by_offset` resets *relative*
to the entity's default joint pose, which for the DK1 is the MJCF keyframe.
Teleop starts from a different pose (the rest pose the leader arm hangs in), so
a keyframe reset makes the recorded demos and the evaluated episodes start from
two different states.

:func:`reset_joints_to_pose` resets to an **absolute** per-joint pose instead,
plus a uniform sample around it. The pose is a robot fact, injected by
``manipulator_wiring.wire_manipulator_into_env`` from
``ManipulatorInterface.reset_joint_pos``; the sample range is a task fact, set
by ``env_cfg.make_manipulation_env_cfg``'s ``start_pose_noise``.

Set the range to ``(0.0, 0.0)`` to get a bit-identical start every episode,
which is what replay verification needs.
"""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform


def reset_joints_to_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pose: dict[str, float],
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Reset the named joints to ``pose`` plus a uniform sample.

    Joints absent from ``pose`` are not written at all, so they keep the
    default position the scene reset just put them in. That is also how a joint
    opts out of the noise: ``position_range`` applies to exactly the joints
    ``pose`` names. Sampled positions
    are clamped to the **hard** joint limits, so a wide range saturates instead
    of starting the episode in a limit violation.

    The clamp deliberately skips the soft limits (``soft_joint_pos_limit_factor``
    shrinks each range toward its midpoint). The DK1's rest pose sits outside
    the soft band on joint2 and joint3 -- soft-clamping it would silently move
    the start pose ~0.07 and ~0.15 rad away from where the leader arm actually
    rests, which is the exact recording/eval mismatch this event exists to
    close.

    Args:
        pose: absolute target position per joint name, in the entity's own
            joint frame. Names not present on the entity raise.
        position_range: uniform range added to every joint in ``pose``.
        velocity_range: uniform range for the start joint velocities.
        asset_cfg: the entity to reset; its ``joint_ids`` are ignored, the
            joints come from ``pose``.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    hard_limits = asset.data.joint_pos_limits
    assert hard_limits is not None

    joint_names = list(asset.joint_names)
    missing = [name for name in pose if name not in joint_names]
    assert not missing, f"{asset_cfg.name} has no joints {missing}"
    joint_ids = torch.tensor(
        [joint_names.index(name) for name in pose], device=env.device
    )
    target = torch.tensor(
        [pose[name] for name in pose], device=env.device, dtype=default_joint_pos.dtype
    )

    joint_pos = target.unsqueeze(0).expand(len(env_ids), -1).clone()
    joint_pos += sample_uniform(*position_range, joint_pos.shape, env.device)
    limits = hard_limits[env_ids][:, joint_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

    joint_vel = sample_uniform(*velocity_range, joint_pos.shape, env.device)

    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids
    )
