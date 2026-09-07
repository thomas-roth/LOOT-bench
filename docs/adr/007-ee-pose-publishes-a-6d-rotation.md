# `ee_pose` publishes a 6D rotation, not a quaternion

The end-effector observation publishes the position and then the first two
columns of the rotation matrix, six numbers. The observation is 9 wide, where
it was 7. Body pose observations keep their quaternion.

## Why

mjlab builds a site quaternion with `quat_from_matrix`, which picks its branch
by an `argmax` over four candidates that are equal in exact arithmetic. The
DK1 grasp site sits on that tie, so float noise decides the sign. Measured on
`LOOT-Stack-Low-Dk1`, the sign flipped on 14 of 120 consecutive steps of one
smooth motion: a jump of 2.0 against a median step change of 0.005. The same
physical state reached the policy as two different observations.

The 6D form covers each rotation once, so it has no sign freedom and cannot
jump.

## Considered options

Canonicalizing the quaternion was rejected. The DK1 grasp site passes through
`w == 0`, where the sign of `w` is itself noise.

Changing body poses too was rejected, and is not needed. A body orientation
comes from `data.xquat`, which MuJoCo integrates continuously, so the unstable
branch never runs.

## Consequence

The apparent spread of the start orientation was nearly all sign flip. At
±0.02 rad of start-pose noise the quaternion range read 1.088; the real spread
is 0.081.
