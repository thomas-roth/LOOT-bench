# Stack task category

Stack blocks to reach a target height. Only the final height matters; block
xy position (and which blocks form the stack) does not.

A *variant* (`variants.py`) fixes the block set and the target height
(`target_blocks * 0.04 m`). The `StackCommand` scatters all blocks upright
on a jittered grid at every reset (no goal poses — the target height is a
per-task constant, so `command_obs=False`). Success holds on the first step
where some block's top face is within `HEIGHT_TOL * margin` of the target
height, all blocks move slowly, and the gripper is open. TwoStacks variants
need two such blocks at least 0.06 m apart in xy.

Block sets carry strictly more material than the minimum needed to reach the
target height: base/TwoStacksDifferentBlocks sets are `2n` blocks, TwoStacks
`2n+2`, DifferentBlocks a mixed prefix plus cubes — so height, not block
count, is the binding constraint, and support structures (staircase,
buttress, pyramid) become viable strategies. Four forms, all heights
multiples of 0.02 m so mixed sets can always hit the target height (a
multiple of 0.04 m) exactly:

- `cube` 0.04x0.04x0.04, 100 g — balanced baseline.
- `plate` 0.06x0.04x0.02, 90 g — wide stabilizing cap/base, big landing
  area, barely any height.
- `tall` 0.03x0.03x0.08, 60 g — most height per placement; slender, high
  center of mass, tiny top face.
- `cylinder` r 0.02 x h 0.06, 80 g — good height, yaw-invariant grasp;
  rolls away when tipped, round top edge.

Block lists tail-extend across the difficulty axis within each family
(Low's blocks are blocks 0..2 of Medium) and resets draw per-block
randomness in index order, so demos for easier tasks can be cut from the
hardest task's recordings (see `docs/TaskCategories.md`).

## Observations (state-based)

- joint positions/velocities, EE pose
- all block poses (env-local position + quaternion, object-major)

## Rewards

- `progress`: fraction of the target height reached by the tallest block
  (mean over the two best-separated leaders for TwoStacks), in [0, 1]
- `success`: bonus while the success criterion holds

## Variants

| Task id (`-Dk1_uni`)                     | Description                                  |
| ----------------------------------------- | -------------------------------------------- |
| `G3A-Stack-{Low,Medium,High}`              | stack 3/4/5 standard cubes (2n blocks)       |
| `G3A-Stack-{Low,Medium,High}-DifferentBlocks` | mixed forms (tall, cylinder, plate x2) + cubes |
| `G3A-Stack-{Low,Medium,High}-TwoStacks`    | two separate stacks, each at target height (2n+2 cubes) |
| `G3A-Stack-{Low,Medium,High}-TwoStacksDifferentBlocks` | two separate stacks, each buildable from a mix of forms |

Play e.g. with:

```bash
uv run g3a-play G3A-Stack-Low-Dk1_uni
```
