# A held block earns no stack progress

Stack's progress is the tallest block's fraction of the target height, and the
gripper can carry a block to that height without placing it. Progress now
ignores the block currently in the gripper (`stack_progress(exclude=)`, fed by
`StackCommand._held_block_mask`), so a lift pays nothing until the block is
released.

Without the exclusion the shaped reward peaks while the arm merely holds a
block aloft, and drops when it lets go and the tower settles a few millimetres
low. That is a gradient toward hovering, which is the same failure ADR-006
closes for the success reward: a policy paid for approaching a goal it never
has to reach.

A block counts as held when the gripper pads report force and it is the block
nearest the grasp site, within `GRIP_REACH` (0.08 m). Both conditions are
needed: the contact alone cannot say *which* block, and proximity alone would
drop a block the arm is only reaching past.

## Consequences

- Success is unaffected. `stack_reached` still asks only whether some block's
  top face is at the target height, and the release half of the success gate
  already required an open, force-free gripper. Only the shaping moved.
- The pad reading has to mean "holding something". The DK1's pads overlapped at
  full close and reported force on an empty gripper, which would have marked a
  block held whenever the arm passed near one; `dk1.xml` excludes the pair.
  A new robot needs the same exclusion.
- `scripts/split_demos.py` reproduces the exclusion offline to relabel a
  recording, approximating "pads in contact" as "gripper closed" because the
  contact force is not recorded. Its per-episode self-check bounds the error
  against the recorded reward; `GRIP_REACH` is shared rather than copied so the
  two cannot drift.
