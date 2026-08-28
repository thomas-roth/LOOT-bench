# One action unit is half a joint's travel

status: superseded by ADR-016, which replaced joint-position actions with a
5 cm end-effector delta at 20 Hz.

Two things below outlive the decision and still bind. **A bounded policy needs
a bounded, reachable action space** — the residual-scale argument in "Why" is
the reasoning ADR-016 inherits, and it is why the delta term clamps to
`[-1, 1]` rather than treating the scale as a unit. **Hard joint limits stay
teleoperable** — the outer slice of every joint is flown through during
recording, which is why the robots set `soft_joint_pos_limit_factor = 1.0`
instead of shrinking the soft limits under differential IK.

The rule itself is gone: `NormalizedJointPositionAction` survives only as the
gripper term, one joint wide. Everything below describes the arm as it was
until ADR-016.

Every task uses `tasks/common/actions.NormalizedJointPositionAction`: `-1`
commands a joint's lower limit, `+1` its upper limit, `0` the midpoint. The
term reads the limits off the compiled model, so no file holds a copy of the
numbers. It also clamps the raw action into `[-1, 1]`.

## Why

mjlab's `JointPositionActionCfg` computes `target = action * scale +
default_joint_pos`, where `scale` is a residual step size. The DK1 used
0.07 rad, the position error that gives a quarter of the motor's torque. That
suits an unbounded Gaussian policy: PPO reaches the table with actions near
±30. It does not suit a bounded policy. QC-FQL squashes its output through a
tanh, so the reachable band collapsed to the keyframe ±0.07 rad. The table,
the blocks and every recorded trajectory sat outside that band. No policy in
that action space could reproduce one demonstrated frame, and nothing raised an
error: training ran, and evaluation scored zero.

## Consequences

- `ManipulatorInterface.action_scale`, and the DK1 and YAM values behind it,
  are gone. The action term reads the model.
- The teleop bridge inverts the same mapping
  (`live_bridge/mapper.action_scale_offset`), and `tests/tasks/test_actions.py`
  asserts that the two agree. That seam is where a recording and an evaluation
  silently stop describing the same task.
- The clamp binds teleoperation as well as the policy. Gravity compensation
  pushes the target past the sag, which near a joint limit lands outside the
  box (measured 1.014 on joint5). Without the clamp the operator would fly a
  trajectory keyed to an action no bounded policy can emit. The simulator
  clamps, and the converter stores the clamped value.
- A joint without a position limit would give `scale = inf` and
  `offset = NaN`, so the term fails at build time instead. Neither wired robot
  has such a joint, but the rule is robot-agnostic.
- Accepted cost: the DK1 gripper's mechanical range is wider than the range its
  real controller commands, so a small slice of the gripper's action box asks
  for a stroke the hardware never uses. The joint limit clamps it. A special
  case for the gripper would put a robot-specific number back into a
  robot-agnostic rule.
