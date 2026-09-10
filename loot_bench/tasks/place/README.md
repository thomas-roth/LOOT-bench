# Place task category

Carry one cube to a sampled target spot on the table and let go of it there.

This is the simplest complete manipulation task in the suite, and it exists
for one reason: it is the sanity check the QC-FQL pipeline is brought up on
(see `docs/adr/001-rebuild-the-qc-integration-from-09e5dfa.md`). Every
recorded demonstration set and every training run to date is a `Place` run.
Bring a harder category to QC only after this one moves.

There is no difficulty axis. One cube to one target already is the whole
task, so a *variant* (`variants.py`) instead switches the two **randomness
axes** independently: the cube spawn (`RandS`/`FixedS`) and the target
(`RandT`/`FixedT`). Pinning an axis removes a source of generalization the
policy would otherwise have to cover, which makes `FixedS-FixedT` the
smallest learnable version of the task and `RandS-RandT` the real one.

The `PlaceCommand` draws the cube spawn and the target uniformly over the
shared envelope when the variant axis says random (`x ~ U[0.20, 0.45]`,
`y ~ U[-Y(x), +Y(x)]`, two draws per object in index order — no rejection
loop, so the seed-prefix rule holds), and pins them when it says fixed
(spawn at `x = 0.25`, target at `x = 0.40`, both on the centreline). A
single deterministic push-apart keeps the cube and target at least 0.10 m
apart without moving a pinned axis, so an episode can never start already
solved.

Success holds on the first step where the cube centre is within
`pos_threshold * margin` (0.02 m) of the target in 3D, the cube moves slowly,
and the gripper has released it. The criterion is **position only** — no
orientation term at all, unlike domino's row-aligned upright check: this is
"move to a location", not "move and orient". Release is the shared
contact-only gate: the pad force is at or below `MIN_GRIP_FORCE`.

Episodes are 15 s, which is 300 control steps at the shared 20 Hz
(`docs/adr/016-control-at-20hz-through-a-5cm-end-effector-delta.md`). The arm
starts at the DK1 home pose with no sample around it
(`place_env_cfg.START_POSE_NOISE = 0.0`, see
`docs/adr/008-episodes-start-at-the-home-pose.md`); the value lives on the
task because the demo recorder and the training run must read one number.

## Observations (state-based)

36 wide, in order:

- joint positions/velocities (8 each: 6 arm + 2 gripper)
- EE pose (9): position, then the 6D rotation, never a quaternion
  (`docs/adr/007-ee-pose-publishes-a-6d-rotation.md`)
- pad contact force (1): the same reduction the release gate reads
  (`docs/adr/018-contact-only-release-gate.md`)
- cube pose (7): env-local position + quaternion
- the command (3): the env-local target point

No term carries noise, and `actor` and `critic` read the same terms
(`docs/adr/015-observations-carry-no-noise.md`).

Two contact sensors are declared. `grip_contact` is the unfiltered gripper
pads, and the success gate reads it: `ContactReleaseGate` requires the pad
force at or below `MIN_GRIP_FORCE` before the cube counts as released.
`object_contact` is the same pads filtered to the cube, and it is
**telemetry only** — no gate reads it. The unfiltered sensor alone cannot
answer "did the policy touch the cube", because a gripper closed on empty air
reports its own two pads pressing together, which is what a failing policy
does all episode. That is why the telemetry uses the filtered sensor.

## Rewards

One term, in `[-1, 0]`, exactly `0` only at success
(`docs/adr/006-one-task-reward-in-minus-one-to-zero.md`). It is shaped by the
fraction of the spawn→target travel completed, normalized by the fixed
`MAX_TRAVEL_DIST` rather than by anything per-episode, and floored at
`-MIN_SHORTFALL` so that no non-success step can read a true zero.

## Variants

| Task id (`-Dk1`)        | Cube spawn        | Target            |
| --------------------------- | ----------------- | ----------------- |
| `LOOT-Place-RandS-RandT`     | sampled per reset | sampled per reset |
| `LOOT-Place-RandS-FixedT`    | sampled per reset | pinned            |
| `LOOT-Place-FixedS-RandT`    | pinned            | sampled per reset |
| `LOOT-Place-FixedS-FixedT`   | pinned            | pinned            |

Play e.g. with:

```bash
uv run loot-play LOOT-Place-RandS-RandT-Dk1
```
