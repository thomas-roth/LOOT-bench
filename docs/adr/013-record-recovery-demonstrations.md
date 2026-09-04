---
status: rejected for now
---

# Record recovery demonstrations, not more nominal ones

The next demonstration effort records attempts that start off the ideal path,
near the states a rollout really visits. Do not record more nominal
demonstrations of the same kind.

**Rejected for now.** A cheaper strategy was taken instead: pin the cube spawn
and the target, and let the four `Place` variants (`RandS`/`FixedS`,
`RandT`/`FixedT`) shrink the state space the demonstrations have to cover
rather than widen the data that covers it. That needs no teleoperation time.
299 more nominal demonstrations were recorded for it (99 more `RandS-RandT`,
then 100 for each of the three pinned variants), and `start_pose_noise` went
back to `0.0`.

Nothing here is refuted. The measurements below still stand, and no recovery
demonstration has been recorded. This record stays parked until a fixed-variant
run reports: a rising curve there retires it, a flat one makes it the standing
proposal again.

## Why

The 100 archived demonstrations are 100 separate threads through the state
space, and they are too far apart to interpolate between. Measured:

- the nearest frame in any other episode is a median 11 times farther than the
  next frame in the same episode, and beyond 10 times for 56 % of frames;
- predicting a frame's action from other episodes stops at 0.149 (5 nearest
  neighbours) against 0.0203 for the temporal neighbour in the same episode;
- a network fitted to 0.0013 on 90 episodes reads 0.118 on the other 10, and is
  flat from 25k steps. More fitting buys no generalization.

A policy drifts 1 to 2 mm of gripper motion per step, leaves a 2 cm tube around
the demonstrated path within 10 to 16 steps, and then finds no data that shows
how to come back. A 101st nominal demonstration would be predicted at about
0.149, which is roughly 29 times the step-to-step change of a human action, so
it adds nothing the fit does not already have.

## Consequences

- This is the only option that attacks the measured cause, it is the only
  expensive one, because it needs teleoperation time, and it is unproven.
  Nothing has yet shown that recovery data produces a rising curve.
- Start-state variety (ADR-008) serves the same purpose at the reset state,
  where the old data has one single start.
- Improving the actor's fit is cheap and worth having, and it is not enough by
  itself: a 400 times better fit tracked 6 times longer and still scored 0 of
  20. Do not spend a whole effort on it alone.
