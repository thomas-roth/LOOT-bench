"""The action convention shared by every task category.

**One action unit is one half of a joint's travel.** ``-1`` commands a joint's
lower limit, ``+1`` its upper limit, ``0`` the midpoint. Every task and every
robot uses the same rule, so a recorded action means the same thing wherever
it is replayed.

Why not mjlab's plain :class:`~mjlab.envs.mdp.actions.JointPositionActionCfg`:
it computes ``target = action * scale + default_joint_pos``, where ``scale``
is a *residual* step size. The DK1's is 0.07 rad, chosen as the position error
that produces a quarter of the motor's torque. That suits an unbounded
Gaussian policy, which emits whatever magnitude it needs -- PPO reaches the
table with actions of ±30.

It does not suit a **bounded** policy. QC-FQL squashes its output through a
tanh, so the action is inside ``[-1, 1]`` by construction and the reachable
band collapses to ``keyframe ± 0.07 rad``. The table, the blocks and the
demonstrated trajectories all sit outside it. No policy in that action space
can reproduce a single recorded frame, and the failure is silent: training
runs, and evaluation scores zero.

The limits come from the compiled model, so this file holds no numbers. The
converter reads ``scale``/``offset`` back off the built term rather than
recomputing them, which is why a recording and the simulator cannot disagree
about what an action means.

One rough edge, accepted on purpose: the DK1 gripper's *mechanical* range is
wider than the range its real controller commands, so a small slice of the
gripper's action box asks for a stroke the hardware never uses. The joint
limit clamps it. Special-casing the gripper would put a robot-specific number
back into a robot-agnostic rule, which costs more than the slice is worth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_Array = TypeVar("_Array", np.ndarray, torch.Tensor)


def normalized_scale_offset(lower: _Array, upper: _Array) -> tuple[_Array, _Array]:
    """The ``(scale, offset)`` mapping ``[-1, 1]`` onto ``[lower, upper]``.

    Kept apart from the action term so the mapping can be tested, and read,
    without a simulator. Takes torch tensors on the task side and numpy arrays
    on the teleop side (``live_bridge/mapper.py``) -- one implementation, so
    the recording and the simulator cannot drift apart.
    """
    return (upper - lower) / 2.0, (upper + lower) / 2.0


class NormalizedJointPositionAction(JointPositionAction):
    """Joint-position control whose action box is the joint limits."""

    def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg=cfg, env=env)
        limits = self._entity.data.joint_pos_limits
        assert limits is not None
        # The hard limits, deliberately, not the soft ones. Soft limits shrink
        # each range toward its midpoint, which would leave the outer slice of
        # every joint uncommandable -- including poses the operator actually
        # teleoperates through.
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
        a policy that already squashes its output. Teleoperation is the case
        that needs it: the recorder pushes its target past gravity sag, which
        near a joint limit lands slightly outside the box (measured at 1.014
        on joint5). Without the clamp the recorder would fly a trajectory
        keyed to an action no bounded policy can emit, and the demonstration
        would be unreproducible by construction.

        Clamping the raw action rather than the joint target is deliberate:
        it is also what this term's ``raw_action`` then holds, which is what
        the ``actions`` observation reads (``env_cfg`` names the term for
        exactly this reason), so the observation agrees with what was applied.
        """
        super().process_actions(actions.clamp(-1.0, 1.0))


@dataclass(kw_only=True)
class NormalizedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`NormalizedJointPositionAction`.

    ``scale`` and ``offset`` are ignored: both are derived from the compiled
    model. ``use_default_offset`` defaults to ``False`` for the same reason --
    the offset is the midpoint of the joint range, not the keyframe.
    """

    use_default_offset: bool = False

    def build(self, env: ManagerBasedRlEnv) -> NormalizedJointPositionAction:
        return NormalizedJointPositionAction(self, env)
