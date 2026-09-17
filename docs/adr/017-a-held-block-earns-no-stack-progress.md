# A held block earns no stack progress

> **Amended 2026-09-11.** Stack became the pi family (ADR-020): progress
> is the fraction of levels complete, and a level is a horizontal block at a
> fixed height. The mechanism here is unchanged; "no stack progress" now
> means "no level credit". The symbol names, the success sentence and the
> splitter bullet were updated in place. The file keeps its name because
> other records cite it as ADR-017.

Stack's progress is the fraction of pi levels complete, and a level is
credited to any block lying horizontal at its target height (ADR-020), which
the gripper can carry a block to without placing it. Progress ignores the
block currently in the gripper (`pi_levels_complete(..., exclude=)`, fed
by `PiCommand._held_block_mask`), so a lift pays nothing until the block
is released.

Without the exclusion the shaped reward peaks while the arm merely holds a
block aloft, and drops when it lets go and the lintel settles a few
millimetres low. That is a gradient toward hovering, which is the same failure ADR-006
closes for the success reward: a policy paid for approaching a goal it never
has to reach.

A block counts as held when the gripper pads report force and it is the block
nearest the grasp site, within `GRIP_REACH` (0.08 m). Both conditions are
needed: the contact alone cannot say *which* block, and proximity alone would
drop a block the arm is only reaching past.

## Consequences

- Success is unaffected. The lintel test still asks only whether some unheld
  block lies horizontal with its centre at a level's target height, and the
  release half of the success gate already requires force-free pads
  (ADR-018). Only the shaping moved.
- The pad reading has to mean "holding something". The DK1's pads overlapped at
  full close and reported force on an empty gripper, which would have marked a
  block held whenever the arm passed near one; `dk1.xml` excludes the pair.
  A new robot needs the same exclusion.
- `scripts/split_demos.py` reproduces the exclusion offline to relabel a
  recording. Its held-block mask reads the recorded pad force, which is an
  observation term since ADR-018, so the offline test is the environment's
  test and no proxy for "pads in contact" is needed. Its per-episode
  self-check bounds the error against the recorded reward; `GRIP_REACH` is
  shared rather than copied so the two cannot drift.
