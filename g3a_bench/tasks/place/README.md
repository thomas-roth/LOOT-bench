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

The `PlaceCommand` scatters the cube upright at a fresh spawn pose through
the shared `ScatterGrid` (the same API every other category uses, even with
one object and no collision to avoid), then draws a target point uniformly in
its own region. The two regions are **disjoint in x** — spawn 0.20–0.28 m,
target 0.36–0.44 m, both spanning y ±0.18 m — so the cube can never spawn
already at the target. A pinned axis writes its region's midpoint
(`FIXED_SPAWN_XY` / `FIXED_TARGET_XY`, derived from the ranges rather than
written out) every reset instead, at yaw 0.

Success holds on the first step where the cube centre is within
`pos_threshold * margin` (0.02 m) of the target in 3D, the cube moves slowly,
and the gripper has released it. The criterion is **position only** — no
orientation term at all, unlike domino's row-aligned upright check: this is
"move to a location", not "move and orient". Release is the shared gate: the
gripper joint is open *and* the pad contact is gone.

Episodes are 10 s, which is 500 control steps at the shared 50 Hz. The arm
starts at the DK1 home pose with no sample around it
(`place_env_cfg.START_POSE_NOISE = 0.0`, see
`docs/adr/008-episodes-start-at-the-home-pose.md`); the value lives on the
task because the demo recorder and the training run must read one number.

## Observations (state-based)

42 wide, in order:

- joint positions/velocities (8 each: 6 arm + 2 gripper)
- EE pose (9): position, then the 6D rotation, never a quaternion
  (`docs/adr/007-ee-pose-publishes-a-6d-rotation.md`)
- cube pose (7): env-local position + quaternion
- the command (3): the env-local target point
- last action (7)

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

| Task id (`-Dk1_uni`)        | Cube spawn        | Target            |
| --------------------------- | ----------------- | ----------------- |
| `G3A-Place-RandS-RandT`     | sampled per reset | sampled per reset |
| `G3A-Place-RandS-FixedT`    | sampled per reset | pinned            |
| `G3A-Place-FixedS-RandT`    | pinned            | sampled per reset |
| `G3A-Place-FixedS-FixedT`   | pinned            | pinned            |

Play e.g. with:

```bash
uv run g3a-play G3A-Place-RandS-RandT-Dk1_uni
```
