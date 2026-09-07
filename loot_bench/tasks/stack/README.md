# Stack task category

Stack blocks to reach a target height. Only the final height matters; block
xy position (and which blocks form the stack) does not.

A *variant* (`variants.py`) fixes two axes: the target height in blocks
(`Stack{3,4,5}`) and the block's height:width ratio (`{Easy,Medium,Hard}`).
The `StackCommand` scatters all blocks upright on a jittered grid at every
reset (no goal poses — the target height is a per-task constant, so
`command_obs=False`). Success holds on the first step where some block's top
face is within `HEIGHT_TOL * margin` of the target height, all blocks move
slowly, and the gripper is open.

All blocks of a variant are identical, so the target height is reachable by
exactly `target_blocks` blocks and by no other count (the tolerance stays
below half a block height). Each variant carries `2n` blocks — strictly more
than needed, so height, not block count, is the binding constraint and
support structures (staircase, buttress, pyramid) stay viable.

The difficulty axis is the block's aspect ratio; the footprint is square
because blocks spawn at a random yaw. Mass follows the shape from
`BLOCK_DENSITY = 1562.5 kg/m^3`, so the slenderest block is also the
lightest:

| Difficulty | Size (m)               | Ratio | Mass  |
| ---------- | ---------------------- | ----- | ----- |
| Easy       | 0.04 x 0.04 x 0.04     | 1.0   | 100 g |
| Medium     | 0.03 x 0.03 x 0.05     | 1.67  | 70 g  |
| Hard       | 0.025 x 0.025 x 0.06   | 2.4   | 59 g  |

Block lists tail-extend across the target-height axis within one difficulty
(Stack3's blocks are blocks 0..5 of Stack4) and resets draw per-block
randomness in index order, so demos for easier tasks can be cut from the
hardest task's recordings (see `docs/task_categories.md`). The blocks are
colored by that cut boundary (`order_color`): Stack5's ten blocks read six
navy, two teal, two white.

## Observations (state-based)

- joint positions/velocities, EE pose
- all block poses (env-local position + quaternion, object-major)

## Rewards

One term, in `[-1, 0]`, exactly `0` only at success
(`docs/adr/006-one-task-reward-in-minus-one-to-zero.md`). It is shaped by the
fraction of the target height that the tallest **resting** block reaches —
the block currently held in the gripper is excluded, so merely lifting one to
the target height earns nothing
(`docs/adr/017-a-held-block-earns-no-stack-progress.md`) — and floored at
`-MIN_SHORTFALL` so that no
non-success step can read a true zero.

## Variants

| Task id (`-Dk1`)              | Target height    | Blocks |
| --------------------------------- | ---------------- | ------ |
| `LOOT-Stack3-{Easy,Medium,Hard}`   | 0.12/0.15/0.18 m | 6      |
| `LOOT-Stack4-{Easy,Medium,Hard}`   | 0.16/0.20/0.24 m | 8      |
| `LOOT-Stack5-{Easy,Medium,Hard}`   | 0.20/0.25/0.30 m | 10     |

Play e.g. with:

```bash
uv run loot-play LOOT-Stack3-Easy-Dk1
```

`loot-play` also draws a ghost target stack beside the workspace as a build
guide (`SHOW_GOAL_STACK`); teleop recording keeps the plain target-height
plane.
