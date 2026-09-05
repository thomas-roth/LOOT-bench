# Rebuild the QC integration instead of repairing it

The first QC-FQL integration, on branch `qc-fql`, trained without an error and
scored exactly 0 % success at every evaluation point. An 18-ticket diagnosis
cleared the recording, the simulator, the reward, the discount, the reset
coverage and the adapter wiring, and found no cause. We deleted the
integration and built it again from commit `09e5dfa`, the merge-base of
`qc-fql` and `main`, where no part of the QC work exists.

## Considered options

Continuing the diagnosis was rejected. 15 closed tickets had removed every
cheap suspect, and each remaining candidate cost a training run or teleop
time.

Repairing the branch was rejected too. It held 28 commits from 3 authors, a
3636-line in-repo copy of the trainer, about 230 MB of binaries, and
uncommitted work in two worktrees and a sibling repository.

## Consequences

- A rewrite that makes the same choices will reproduce the flat-zero curve.
  The rebuild therefore kept only the constraints that crash, corrupt or
  mislead. Every other old choice was open again: the reward sign, the action
  mapping, the chunk length, the debug task.
- Rebuilding found three faults, and all three were present in the old
  pipeline and in every task before it. A sign-unstable end-effector rotation
  (ADR-007), a start pose that did not match the demonstrations (ADR-008), and
  an action space that could not reach the table (ADR-005). The last one alone
  explains a curve flat at zero.
- The archived recordings under `dataset/demos/` do not have to keep working.
  No decision here is shaped by them.
