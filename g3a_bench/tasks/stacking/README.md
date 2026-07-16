# Stacking task category

Stack blocks to reach a target height. Only the final height matters; block
xy position (and which blocks form the stack) does not.

A *variant* (`variants.py`) fixes the block set and the target height
(`target_blocks * 0.04 m`). The `StackingCommand` scatters all blocks upright
on a jittered grid at every reset (no goal poses — the target height is a
per-task constant, so `command_obs=False`). Success holds on the first step
where some block's top face is within `HEIGHT_TOL * margin` of the target
height, all blocks move slowly, and the gripper is open. TwoStacks variants
need two such blocks at least 0.06 m apart in xy.

Block lists tail-extend across the difficulty axis (Stack3's blocks are
blocks 0..2 of Stack4) and resets draw per-block randomness in index order,
so demos for easier tasks can be cut from the hardest task's recordings
(see `docs/TaskCategories.md`).

## Observations (state-based)

- joint positions/velocities, last action
- all block positions relative to the grasp site (object-major)

## Rewards

- `progress`: fraction of the target height reached by the tallest block
  (mean over the two best-separated leaders for TwoStacks), in [0, 1]
- `success`: bonus while the success criterion holds

## Variants

| Task id (`-Dk1_uni`)                    | Description                                  |
| --------------------------------------- | -------------------------------------------- |
| `G3a-Stacking-Stack{3,4,5}`              | stack n standard cubes                       |
| `G3a-Stacking-Stack{n}DifferentBlocks`   | mixed forms (rect base, tall block) + cubes  |
| `G3a-Stacking-Stack{n}TwoStacks`         | two separate stacks, each at target height   |

Play e.g. with:

```bash
uv run g3a-play G3a-Stacking-Stack3-Dk1_uni
```
