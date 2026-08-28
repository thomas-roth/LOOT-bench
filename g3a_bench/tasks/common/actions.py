"""The action convention shared by every task category.

**An action is a relative end-effector delta, commanded at 20 Hz.** The first
six channels are a pose delta on the robot's grasp site -- ``±1`` is
:data:`DELTA_POS_SCALE` metres per axis and :data:`DELTA_ORI_SCALE` radians per
axis -- solved into joint targets by mjlab's differential IK. The seventh is
the gripper, an absolute normalized stroke where ``-1`` is the closed limit and
``+1`` the open one. Total width 7.

This is the control envelope both QC-validated reference domains use
(robomimic ``can``, OGBench ``cube-*``): ±5 cm at 20 Hz. See
``docs/adr/016-control-at-20hz-through-a-5cm-end-effector-delta.md`` for the
measurements. What it replaces -- absolute normalized joint positions at
50 Hz -- made a fixed action error a *pose-dependent* Cartesian error, so the
policy had to invert the arm's kinematics itself: an action error of 0.0766
moved the end-effector 0.0719 m against a 0.02 m success tolerance.

**The box is per-channel, not a norm.** ``±1`` on every position channel at
once is 0.087 m of travel, not 0.05 m, and the rotation corner is 0.87 rad.
Both reference domains bound their channels the same way, so this matches
them; the number to quote is "±5 cm per axis", never "≤5 cm per step".

**The clamp is the convention, not a safety net.** mjlab's
:class:`~mjlab.envs.mdp.actions.differential_ik.DifferentialIKAction` applies
``delta_pos_scale`` to whatever it is handed, so without the clamp in
:class:`EndEffectorDeltaAction` the scale would be a *unit* rather than a
*bound*, and an action of 3.0 would honestly ask for 15 cm. Two things depend
on the bound being real: every stored action has to be one a tanh-squashed
policy can emit, and the recorder's mapping has to clamp identically to the
term or the two disagree on exactly the fast frames. (mjlab's
``ActionTermCfg.clip`` is not this: it is implemented only in ``JointAction``,
and it clips the *processed joint target* rather than the raw action.)

:func:`delta_action_from_poses` is the other half of that agreement. It is the
exact inverse of what :class:`EndEffectorDeltaAction` applies -- mjlab's
``compute_pose_error`` and ``apply_delta_pose`` invert each other -- and it is
the only place a delta action is built anywhere in this repo. Teleoperation
calls it through ``live_bridge/mapper.py``, so a recorded action and a policy
action cannot mean different things.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.envs.mdp.actions.differential_ik import (
    DifferentialIKAction,
    DifferentialIKActionCfg,
)
from mjlab.utils.lab_api.math import compute_pose_error

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_Array = TypeVar("_Array", np.ndarray, torch.Tensor)

EE_DELTA_ACTION_NAME = "ee_delta"
"""Key of the end-effector delta term in every task's action manager."""

GRIPPER_ACTION_NAME = "gripper"
"""Key of the gripper term in every task's action manager."""

DELTA_POS_SCALE = 0.05
"""[m] end-effector travel one action unit asks for, per position axis.

Matches robomimic ``can``'s ``OSC_POSE`` ``output_max`` and OGBench
``cube-*``'s ``action_range`` exactly. ADR-016 measured the 99 recorded
``place`` demos at 20 Hz: p99 per-step displacement is 2.7 cm, so the box is
fitted rather than lucky, and it also sets a *floor* on the control rate --
at 10 Hz 3 % of demo steps no longer fit inside it.
"""

DELTA_ORI_SCALE = 0.5
"""[rad] end-effector rotation one action unit asks for, per axis.

The demos use only ~26-38 % of this. A fitted ±0.25 was rejected: 0.5 is
robomimic's value, inside the validated envelope, and the point is to stop
introducing free variables nobody has swept.
"""

IK_DAMPING = 0.05
"""Damped-least-squares lambda. Singularity behaviour is this alone's job."""

IK_JOINT_LIMIT_WEIGHT = 1.0
"""Weight of differential IK's joint-limit residual.

Positive so the solver pushes back out of a limit instead of grinding into it.
Note what this does and does not buy: the residual is exactly zero strictly
inside the limits, so it acts *at* the wall rather than decelerating before
it. Preemptive avoidance would need the soft limits shrunk below the hard
ones, which is what ADR-005 refused -- the outer slice of every joint is
teleoperated through, and shrinking it would make demonstrated poses
uncommandable. The robots therefore set ``soft_joint_pos_limit_factor`` to
1.0; differential IK is the only path in this repo that reads the soft limits.
"""

IK_POSTURE_WEIGHT = 0.0
"""Off. A 6-DOF arm under a 6D task objective is exactly determined, so there
is no null space for a posture term to act in."""


def normalized_scale_offset(lower: _Array, upper: _Array) -> tuple[_Array, _Array]:
    """The ``(scale, offset)`` mapping ``[-1, 1]`` onto ``[lower, upper]``.

    Kept apart from the action term so the mapping can be tested, and read,
    without a simulator. Takes torch tensors on the task side and numpy arrays
    on the teleop side (``live_bridge/mapper.py``) -- one implementation, so
    the recording and the simulator cannot drift apart. Only the gripper is
    driven this way now; the arm moves through :class:`EndEffectorDeltaAction`.
    """
    return (upper - lower) / 2.0, (upper + lower) / 2.0


def unclamped_delta_from_poses(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> torch.Tensor:
    """The 6-D delta asking for ``target`` from ``current``, in action units.

    Not itself an action -- it can leave the box. Two callers need it before
    the clamp: :func:`delta_action_from_poses`, which clamps it, and
    :func:`pose_within_one_delta`, which asks whether the clamp would bite.

    **Both poses must be in one frame.** Any two frames related by a pure
    translation give the same answer -- the position term is a difference and
    the rotation term is translation-invariant -- so world and env-local both
    work, but mixing them is wrong by one ``env_origin``.

    Args:
        current_pos: (N, 3) current frame position.
        current_quat: (N, 4) current frame orientation, ``wxyz``.
        target_pos: (N, 3) requested position, in the same frame.
        target_quat: (N, 4) requested orientation, ``wxyz``.

    Returns:
        (N, 6): position, then axis-angle rotation.
    """
    pos_error, rot_error = compute_pose_error(
        current_pos, current_quat, target_pos, target_quat, rot_error_type="axis_angle"
    )
    return torch.cat([pos_error / DELTA_POS_SCALE, rot_error / DELTA_ORI_SCALE], dim=-1)


def delta_action_from_poses(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> torch.Tensor:
    """The clamped 6-D delta action that asks for ``target`` from ``current``.

    The inverse of what :class:`EndEffectorDeltaAction` applies, and the only
    place a delta action is built. Feeding the result to the term reproduces
    ``target`` exactly whenever it is within one box, and asks for the nearest
    point on the box when it is not. Arguments as
    :func:`unclamped_delta_from_poses`.

    Returns:
        (N, 6) action inside ``[-1, 1]``: position, then axis-angle rotation.
    """
    return unclamped_delta_from_poses(
        current_pos, current_quat, target_pos, target_quat
    ).clamp(-1.0, 1.0)


def pose_within_one_delta(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> torch.Tensor:
    """Whether one action reaches ``target`` from ``current`` without clamping.

    The recorder's episode-start gate. ADR-016 measured that 66 of the 99
    recorded ``place`` demos open with the sim arm snapping toward an
    already-parked leader -- a jump that clamps, and is therefore
    unreproducible by any bounded policy. Gating on this instead of clamping
    the action or dropping the row is what keeps "the stored action is what
    ``env.step()`` took" true for *every* row (ADR-011).

    Arguments as :func:`unclamped_delta_from_poses`.

    Returns:
        (N,) bool.
    """
    delta = unclamped_delta_from_poses(
        current_pos, current_quat, target_pos, target_quat
    )
    return delta.abs().amax(dim=-1) <= 1.0


class EndEffectorDeltaAction(DifferentialIKAction):
    """Differential IK on the grasp site, with the action box made real.

    See the module docstring for why the clamp is the convention rather than a
    guard. Clamping the raw action rather than the solved joint target is what
    makes ``raw_action`` -- what every diagnostic and the dataset check read --
    hold the value the simulator actually acted on.
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions.clamp(-1.0, 1.0))

    @property
    def frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The IK frame's ``(position, wxyz quaternion)``, in the world frame.

        Exactly the buffers the solver itself reads, published so
        teleoperation can build its delta from the same numbers rather than
        from the ``ee_pose`` observation, which publishes a 6-D rotation and
        subtracts the env origin. Whoever reads this and pairs it with a pose
        in another frame is wrong by one ``env_origin``.
        """
        return self._get_frame_pose()


@dataclass(kw_only=True)
class EndEffectorDeltaActionCfg(DifferentialIKActionCfg):
    """Configuration for :class:`EndEffectorDeltaAction`.

    ``actuator_names``, ``frame_name`` and ``max_dq`` are placeholders filled
    per-robot by ``manipulator_wiring.wire_manipulator_into_env`` -- the arm's
    actuator split, its grasp site, and its velocity limit over one physics
    step. A task file never names any of them, and a robot that declares no
    split fails here rather than reaching a training run.
    """

    actuator_names: tuple[str, ...] = ()
    frame_name: str = ""
    frame_type: Literal["body", "site", "geom"] = "site"
    use_relative_mode: bool = True
    delta_pos_scale: float = DELTA_POS_SCALE
    delta_ori_scale: float = DELTA_ORI_SCALE
    damping: float = IK_DAMPING
    orientation_weight: float = 1.0
    joint_limit_weight: float = IK_JOINT_LIMIT_WEIGHT
    posture_weight: float = IK_POSTURE_WEIGHT
    max_dq: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> EndEffectorDeltaAction:
        if not self.actuator_names:
            raise ValueError(
                "no arm actuators declared for the end-effector delta term; "
                "wire_manipulator_into_env fills them from the robot's "
                "ManipulatorInterface.arm_actuators"
            )
        if not self.frame_name:
            raise ValueError(
                "no IK frame declared for the end-effector delta term; "
                "wire_manipulator_into_env fills it from the robot's grasp site"
            )
        if self.max_dq <= 0.0:
            raise ValueError(
                f"max_dq must be positive, got {self.max_dq}; "
                "wire_manipulator_into_env derives it from the robot's arm "
                "velocity limit and the physics timestep"
            )
        return EndEffectorDeltaAction(self, env)


class NormalizedJointPositionAction(JointPositionAction):
    """Joint-position control whose action box is the joint limits.

    The gripper's term: ``-1`` commands the closed limit, ``+1`` the open one,
    and the limits come from the compiled model so this file holds no numbers.
    The arm no longer moves this way (ADR-016), but the reasoning that keeps
    the gripper here survives from ADR-005: mjlab's plain
    ``JointPositionActionCfg`` uses a *residual* scale, which suits an
    unbounded Gaussian policy and collapses a tanh-squashed one's reach to a
    band around the keyframe.

    One rough edge, accepted on purpose: the DK1 gripper's mechanical range is
    wider than the range its real controller commands, so a small slice of the
    box asks for a stroke the hardware never uses. The joint limit clamps it.
    Special-casing the gripper would put a robot-specific number back into a
    robot-agnostic rule.
    """

    def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg=cfg, env=env)
        limits = self._entity.data.joint_pos_limits
        assert limits is not None
        # The hard limits, deliberately, not the soft ones. Soft limits shrink
        # each range toward its midpoint, which would leave the outer slice of
        # the stroke uncommandable.
        joint_limits = limits[:, self._target_ids]
        # An unlimited joint has limits of -inf/+inf, which would give a scale
        # of inf and an offset of NaN -- every commanded target NaN, and
        # nothing raised. Neither wired robot has one, but this rule is the
        # robot-agnostic contract, so the next robot should fail at build time
        # rather than at the first step.
        if not bool(torch.isfinite(joint_limits).all()):
            unlimited = [
                name
                for index, name in enumerate(self._target_names)
                if not bool(torch.isfinite(joint_limits[:, index]).all())
            ]
            raise ValueError(
                f"joints {unlimited} have no position limit, so [-1, 1] cannot "
                "be mapped onto their travel; give them a range in the MJCF"
            )
        self._scale, self._offset = normalized_scale_offset(
            joint_limits[..., 0], joint_limits[..., 1]
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        """Clamp to the action box, then apply the scale and offset.

        The clamp makes the box authoritative for *every* caller, not only for
        a policy that already squashes its output, and it is what
        ``raw_action`` then holds -- so a recorded action is the one the
        simulator applied.
        """
        super().process_actions(actions.clamp(-1.0, 1.0))


@dataclass(kw_only=True)
class NormalizedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`NormalizedJointPositionAction`.

    ``scale`` and ``offset`` are ignored: both are derived from the compiled
    model. ``use_default_offset`` defaults to ``False`` for the same reason --
    the offset is the midpoint of the joint range, not the keyframe.
    """

    actuator_names: tuple[str, ...] = ()
    use_default_offset: bool = False

    def build(self, env: ManagerBasedRlEnv) -> NormalizedJointPositionAction:
        if not self.actuator_names:
            raise ValueError(
                "no gripper actuators declared for the gripper term; "
                "wire_manipulator_into_env fills them from the robot's "
                "ManipulatorInterface.gripper_actuators"
            )
        return NormalizedJointPositionAction(self, env)
