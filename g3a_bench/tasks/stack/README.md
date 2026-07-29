# Stack task category

Stack blocks to reach a target height. Only the final height matters; block
xy position (and which blocks form the stack) does not.

A *variant* (`variants.py`) fixes two independent axes: the target height in
blocks (`Stack3`/`Stack4`/`Stack5`) and the block's height:width ratio
(`Easy`/`Medium`/`Hard`). The `StackCommand` scatters all blocks upright on a
jittered grid at every reset (no goal poses — the target height is a per-task
constant, so `command_obs=False`). Success holds on the first step where some
block's top face is within `HEIGHT_TOL * margin` of the target height, all
blocks move slowly, and the gripper is open.

All blocks of a variant are identical, so the target height is reachable by
exactly `target_blocks` blocks and by no other count (the tolerance stays
below half a block height). Each variant still carries `2n` blocks — strictly
more than needed — so block count is never the constraint and support
structures (staircase, buttress, pyramid) stay viable.

The footprint is square (blocks spawn at a random yaw, so both horizontal axes
must be graspable), which makes the difficulty exactly the height:width ratio.
Easy is the standard cube; every step from there is narrower *and* taller.
Mass follows the shape at a constant density of 1562.5 kg/m³ (the cube's
100 g), so the slender blocks are also the lightest:

| Difficulty | Size (m)             | h:w  | Mass  | Stack5 height |
| ---------- | -------------------- | ---- | ----- | ------------- |
| `Easy`     | 0.04 x 0.04 x 0.04   | 1.0  | 100 g | 0.20 m        |
| `Medium`   | 0.03 x 0.03 x 0.05   | 1.67 | 70 g  | 0.25 m        |
| `Hard`     | 0.025 x 0.025 x 0.06 | 2.4  | 59 g  | 0.30 m        |

Narrower and taller means a smaller landing area and grasp target, a higher
center of mass per placement, and a tower that reaches further above the
workspace — Hard's 0.30 m Stack5 tower is the highest the arm can still place
a block on.

Block lists tail-extend across the target height *within one difficulty*
(Stack3's blocks are blocks 0..5 of Stack4) and resets draw per-block
randomness in index order, so demos for easier tasks can be cut from the
hardest task's recordings (see `docs/task_categories.md`). Across difficulties
the block shape differs, so those recordings are never mixed.

## Debug visualization

A translucent plane over the workspace marks the target height (green once
reached, brighter green on full success). The play entry point
(`g3a_bench/scripts/play.py`, i.e. `g3a-play`) additionally switches on a
**ghost stack** beside the workspace — `target_blocks` outlined ghost blocks
showing how many blocks of that shape reach the target, each level turning
green as `progress` passes it. It stands next to the scatter region because
the real stack's xy is free; it is off during training and teleop recording
(`SHOW_GOAL_STACK` in `mdp/commands.py`).

## Observations (state-based)

- joint positions/velocities, EE pose, last action
- all block poses (env-local position + quaternion, object-major)

## Rewards

- `progress`: fraction of the target height reached by the tallest block,
  in [0, 1]
- `success`: bonus while the success criterion holds

## Variants

| Task id (`-Dk1_uni`)             | Description                              |
| -------------------------------- | ---------------------------------------- |
| `G3A-Stack{3,4,5}-Easy`          | 3/4/5 standard cubes high (6/8/10 blocks) |
| `G3A-Stack{3,4,5}-Medium`        | 3/4/5 narrower, taller blocks high (6/8/10 blocks) |
| `G3A-Stack{3,4,5}-Hard`          | 3/4/5 narrowest, tallest blocks high (6/8/10 blocks) |

Play e.g. with:

```bash
uv run g3a-play G3A-Stack3-Easy-Dk1_uni
```
