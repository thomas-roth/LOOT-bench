# Contact-only release gate

A task counted as done when its objects were placed, at rest, and the
gripper joints sat past a robot-specific open threshold — a proxy for
"the robot let go". `released` is now exactly "the finger pads exert no
grip force": the max over the pads of the `grip_contact` sensor at or
below `MIN_GRIP_FORCE` (0.1 N). The joint gate, its command cfg, the
wiring hook, and every threshold constant are deleted, and the pad force
joins the proprioceptive observation (actor and critic alike), widening it
by one.

## Why the joint gate was a proxy

Joint position answers "are the fingers apart", not "is anything between
them". A wedged or diagonal grasp crosses the open threshold while the
pads are still loaded, so a task scored while the robot held the object.
The threshold was also a per-robot constant (`_GRIPPER_OPEN_FRACTION` of
the joint stroke) that every robot had to declare and every downstream
tool had to re-derive: `loot-split-demos` reconstructed release offline
from `gripper_left` in the recorded `qpos` because pad force was not
recorded. The finger-pad contact sensor already existed and already
answered the question directly for telemetry — but only as a second
opinion, so the joint gate's failures still leaked through.

## Why deleted rather than kept as a second opinion

ANDing the proxy with the measurement can only veto true releases while
adding no true accept the measurement misses: pads force-free with the
joints below threshold (a nudge after release, a closed empty gripper at
the goal) is released, and the joint gate would say otherwise. Keeping it
would also keep everything the proxy costs — the threshold per robot, the
wiring hook, the splitter's qpos index — for a signal that carries no
information the force reading lacks. `gripper_actuators` and `ik_max_dq`
stay: the action still drives the gripper, only the success gate stopped
reading its joints.

## Why the pad force joins the observation

Termination depended on a number the policy could not see. An episode
that ends for an invisible reason is unlearnable on that margin, so the
gate's own number is appended to the proprioceptive head after `ee_pose`
(width 1, no noise, actor and critic identical per ADR-015). The observed
value, the gate value and the `pad_contact_force` telemetry are one
reduction (`tasks.common.commands.pad_contact_force`).

`CONTACT_THRESHOLD` (`qc/episode.py`) and `MIN_GRIP_FORCE` stay two
constants: the first gates a trainer-side metric across a repo boundary,
the second is the env's release gate. Aliasing them would let a retune of
one silently move the other.

## Consequences

- Every committed dataset is stale: the observation is one wider, so an
  archived `.npz` loaded against a rebuilt env raises on the width check
  (ADR-011). The archives are renamed with a `-gripper-open-gate` tag and
  treated as archive-only; `Place` must be re-recorded.
- An empty gripper closed all the way reads zero here: `dk1.xml` excludes
  the pad pair, and a new robot needs the same exclusion.
- `ball` is untouched: it has no release gate and declares no
  `grip_contact` sensor, so its observation is unchanged.
