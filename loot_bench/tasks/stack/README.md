# Stack task category (henge building)

Build a henge: one, two or three *levels* stacked on top of each other. A
level is one block lying flat (the **lintel**) carried at a fixed height by
whatever stands under it. Two builds satisfy a level, and the task encodes
neither:

```
  trilithon              T
   +-----+            +-----+
   |     |            |     |      lintel: one block lying flat
   #     #               #         upright(s): block(s) standing on end
  ---------          ---------     table
```

The criterion tests only the lintel, never the supports, so both forms are
legal. The trilithon costs a third block and stands; the T is faster and tips
when the next level is built on it. The physics decides, which is the point of
the multi-level tasks. See
`docs/adr/020-a-henge-level-is-a-horizontal-block-at-a-fixed-height.md` for
the decision and the arithmetic behind the two tolerances.

## Tasks

A *variant* (`variants.py`, `HengeVariantCfg`) fixes the level count. There
is no difficulty axis. The category segment stays `Stack`; the level count is
the task segment.

| Task id (`-Dk1`)          | Levels | Blocks | Lintel centre heights (m) | Episode           |
| ------------------------- | ------ | ------ | ------------------------- | ----------------- |
| `LOOT-Stack-1Henge`       | 1      | 3      | 0.075                     | 75 s (1500 steps)  |
| `LOOT-Stack-2Henges`      | 2      | 6      | 0.075, 0.160              | 150 s (3000 steps) |
| `LOOT-Stack-3Henges`      | 3      | 9      | 0.075, 0.160, 0.245       | 225 s (4500 steps) |

Heights are the lintel's *centre* above the table, measured at 20 Hz.

## The block

One block type, a square prism, serves as upright and as lintel. The
cross-section is square because blocks spawn at a random yaw and must grasp
the same standing and lying. Blocks spawn standing.

| Property       | Value                        |
| -------------- | ---------------------------- |
| Full size      | 0.065 x 0.020 x 0.020 m      |
| `half_extents` | (0.010, 0.010, 0.0325)       |
| Long axis      | local z                      |
| Mass           | 40.6 g at `BLOCK_DENSITY` 1562.5 kg/m^3 |
| Level height   | 0.085 m (length plus width)  |

## Success criterion

Level `k` is complete when some block, all at once:

- is horizontal: its long axis is within `TILT_COS_MAX` = 0.10 (5.7 deg) of
  level (`mdp/success.block_is_horizontal`);
- has its centre within `HEIGHT_TOL` = 3 mm of level `k`'s height above the
  table;
- is not held in the gripper (`docs/adr/017-a-held-block-earns-no-stack-progress.md`).

Progress is the fraction of levels standing right now. It is instantaneous,
not latched: a collapse drops it, because the work really was undone. Success
holds on the first step where every level is complete, all blocks move slowly,
and the gripper has released them (contact-only release gate,
`docs/adr/018-contact-only-release-gate.md`). One predicate feeds both the
reward and the termination.

The two constants are coupled to the block geometry and checked against it at
import (`variants.assert_no_alternative_build`). Do not tune either one on its
own; ADR-020 explains why.

## Rewards

One term, in `[-1, 0]`, exactly `0` only at success
(`docs/adr/006-one-task-reward-in-minus-one-to-zero.md`). It is shaped by the
level fraction (`progress - 1`) and floored at `-MIN_SHORTFALL` so that no
non-success step can read a true zero. Under `--sparse` a `1Henge` run is
literally sparse; `3Henges` keeps three coarse rungs that the flag flattens.

## Colour tiers and the split

Block lists tail-extend across the level axis: `1Henge`'s 3 blocks are blocks
0-2 of `2Henges`, which are blocks 0-5 of `3Henges`. Resets draw per-block
randomness in index order, so `loot-split-demos` can cut `1Henge` and
`2Henges` demo sets out of a single `3Henges` recording. `order_color(i, 3, 6)`
paints those cut boundaries onto the blocks, one tier per level:

| Blocks | Colour | Level |
| ------ | ------ | ----- |
| 0-2    | navy   | 1     |
| 3-5    | teal   | 2     |
| 6-8    | white  | 3     |

## Recording protocol

The code cannot enforce any of this. A teleoperator has to follow it, or the
recording is not splittable and the trilithon does not fit.

1. **Build bottom-up, one colour tier per level.** Level 1 from the navy
   blocks, level 2 from teal, level 3 from white. A recording that builds
   level 1 from white blocks cannot be relabelled into `1Henge`, and the
   splitter fails loudly on it.
2. **Grip lintels end-to-end.** Put the pads on the two 20 x 20 mm end faces.
   Gripping across the width scrapes the uprights, for any block size,
   because the pads and the upright faces both sit at half the block width
   from the centre line.
3. **Space trilithon uprights about 36 mm apart, centre to centre.** 32 mm
   is the minimum to lower the second upright past the first; 45 mm is the
   maximum before the lintel stops overhanging both.
4. **Make a lintel by knocking a standing block over, then re-grasping it
   lying.** Tip it onto the table with the closed gripper, then pick it up
   end-to-end. `joint4` has only 22 deg of pitch from the home pose and cannot
   roll a standing block flat in the wrist. Budget both motions.
5. **The T is legal and faster**, but it tips when the next level is built on
   it. Use it for `1Henge` if you like; prefer the trilithon below any level
   that has to carry another.

## Observations (state-based)

- joint positions/velocities, EE pose
- all block poses (env-local position + quaternion, object-major)

There is no command observation: the level heights are per-task constants
(`command_obs=False`).

## Spawns

`HengeCommand` stages all blocks standing from the fixed spawn table at every
reset: object `i` on cell `i`, +/-1 cm jitter (`docs/adr/019-fixed-spawn-points.md`).
The build zone, x in [0.24, 0.32] and |y| <= 0.07, is kept clear by
construction, so a session starts with the work area empty.

## Playing

```bash
uv run loot-play LOOT-Stack-2Henges-Dk1
```

`loot-play` and `loot-record-demos` draw a ghost trilithon per level
directly right of the robot as a build guide (uprights at 36 mm spacing).
Success is position-free: each level's ghost greens when that level is
accepted anywhere, and the ghost itself is a legal build to copy. Markers
are viewer-only; they never enter observations or the dataset. The recorder
adds a gripper laser with a footprint preview under the held block and a
wrist-camera PiP: operator aids, viewer-only like the ghost.
