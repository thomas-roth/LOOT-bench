# Task Categories

## General
Goal:
- only collect Data for hardest Task in all variations
- use a script to create demonstrations for smaller tasks
  - split at progress rewards
  - adapt rewards (scale progress rewards to [0,1], add episode success)
  - adapt observations to fit task (e.g. only 3 blocks for 1Henge (remove the last 6 blocks from 3Henges))

Pay attention to the drawing of random numbers from a seed. First draw all randomness for the easiest task (e.g. 1Henge), and then continue step-wise to the hardest task (e.g. then 2Henges and after that 3Henges). This allows the cutoff to work.

### Global Modifiers:
- Margin
- Ordered

-> always collect data in small Margins & Ordered (needed so cutoff works)

### Randomization:
always randomize objects & goal poses (position and orientation) a little if not stated otherwise to prevent overfitting. 

## Category: Stack

Build a henge: a *level* is one horizontal block (a lintel) held at a fixed
height by whatever stands under it, and the task is to stack one, two or
three levels on top of each other. Only the lintel is scored, never the
supports (`docs/adr/020-a-henge-level-is-a-horizontal-block-at-a-fixed-height.md`).

| **Evaluated Capabilities** | Description |
| --- | --- |
| Long-horizon precision | Place up to nine blocks in sequence within millimetre tolerances. |
| Reorientation | Turn a standing block into a lying one: knock it over, then re-grasp it. |
| Robustness | Keep building on a base that may be unstable. |

| **Task** | Levels | Blocks | Lintel centres (m) | Episode |
| --- | --- | --- | --- | --- |
| 1Henge | 1 | 3 | 0.075 | 75 s |
| 2Henges | 2 | 6 | 0.075, 0.160 | 150 s |
| 3Henges | 3 | 9 | 0.075, 0.160, 0.245 | 225 s |

There is no variation axis; the task ids are `LOOT-Stack-1Henge-Dk1`,
`LOOT-Stack-2Henges-Dk1` and `LOOT-Stack-3Henges-Dk1`.

Two builds complete a level and the criterion prefers neither:

```
  trilithon              T
   +-----+            +-----+
   |     |            |     |      lintel: one block lying flat
   #     #               #         upright(s): block(s) standing on end
  ---------          ---------     table
```

The trilithon costs a third block and stands. The T is faster and tips.
Nothing rewards or forbids either; the physics decides, and its verdict only
matters from 2Henges up, where a level has to carry the next one.

One block type serves as upright and as lintel: a square prism of
0.065 x 0.020 x 0.020 m, 40.6 g. The cross-section is square because a block
spawns at a random yaw and must grasp identically standing and lying. Blocks
are coloured in tiers of three, navy, teal and white, one tier per level:
1Henge's blocks are the first three of 2Henges, which are the first six of
3Henges.

Operator protocol, which the code cannot enforce: build level 1 from the navy
blocks, level 2 from teal, level 3 from white, bottom-up, or a 3Henges
recording will not split into the easier tasks. Grip lintels end-to-end (a
grip across the width scrapes both uprights at zero clearance). Space
trilithon uprights about 36 mm apart. Blocks spawn standing, so make a lintel
by knocking the block over with the closed gripper and re-grasping it; the
wrist cannot roll it flat in one motion.

Spawns stage from the fixed table (object `i` on cell `i`, ±1 cm jitter),
with the build site kept clear; see `docs/adr/019-fixed-spawn-points.md`.

## Category: Domino

Build a domino line that can topple sequentially.

| **Evaluated Capabilities** | Description |
| --- | --- |
| Precision | Place dominoes accurately without knocking over those already placed. |

| **Task** | Description |
| --- | --- |
| 2Pieces | Place two dominoes in a line. |
| 3Pieces | Place three dominoes in a line. |
| 4Pieces | Place four dominoes in a line. |
| 5Pieces | Place five dominoes in a line. |
| 6Pieces | Place six dominoes in a line. |

| **Variation** | Description |
| --- | --- |
| *(none)* | Use identical basic dominoes only. The base variant carries no variation segment: `LOOT-Domino-2Pieces-Dk1`. |
| `RandomRow` | Row centre y and yaw redrawn every reset; row centre x stays pinned. |
| `DifferentDominoes` | Dominoes become progressively thinner and lighter. |
| `KnockOver` | Topple the completed line as a chain: every domino falls along the row, same direction, in row order; phase observation build (`0`) / topple (`1`). |

Spawns stage from the fixed table (object `i` on cell `i`, ±1 cm jitter),
left of the pinned row lane by construction; see `docs/adr/019-fixed-spawn-points.md`.

## Category: Sorter

Insert every peg into its matching hole in a shape sorter. Each peg fits only its corresponding hole.

| **Evaluated Capabilities**  | Description |
| --- | --- |
| Combinatorial complexity | Reposition the cube or perform more difficult insertions. |
| Strategy | Choose an effective insertion order. |
| Precision | Handle tight insertion tolerances. |

| **Task** | Description |
| --- | --- |
| 2Shapes | Insert two shapes into the sorter. |
| 3Shapes | Insert three shapes into the sorter. |
| 4Shapes | Insert four shapes into the sorter. |
| 5Shapes | Insert five shapes into the sorter. |

| **Variation** | Description |
| --- | --- |
| `FixedOrientation` | The sorter orientation is fixed; its position varies. |
| `NoRotation` | The sorter orientation varies, but no required hole faces the ground. |
| `Full` | The sorter position and orientation are random; rotating it may be required. |

Pieces stage from the fixed five-cell table in front of the cube (object `i`
on cell `i`, ±1 cm jitter) with the insertion corridor kept clear; see
`docs/adr/019-fixed-spawn-points.md`.

## Category: Ball

Bring a ball to its goal by hitting or throwing it.

| **Evaluated Capabilities** | Description |
| --- | --- |
| Dynamics | Apply sufficient force to reach the goal. |
| Long-horizon credit assignment | Receive the reward substantially after the action. |

### Task: Minigolf

Hit the ball once to reach a goal outside the robot's direct reach. Every
layout has a perimeter railing around the platform edges, and the layouts
are strictly additive: `UsingWall` ⊂ `Obstacle` ⊂ `Ramp`, each adding one
more element on top of the previous layout.

| **Variation** | Description |
| --- | --- |
| `Direct` | Reach the goal with a direct hit; the perimeter railing alone. |
| `UsingWall` | Adds a blocker across the direct line; the perimeter railing serves as the bank for a redirect shot. |
| `Obstacle` | `UsingWall` plus an obstacle in the lane the blocker leaves open. |
| `Ramp` | `Obstacle` plus the goal raised onto an elevated section, reached via an incline. |

### Task: Throw

Throw the ball over a wall to reach the target. Evaluate whether reduced gravity is needed.

| **Variation** | Description |
| --- | --- |
| `U` | U-shaped walls surround the goal. |
| `MarkedArea` | A marked area defines the goal. |
| `WallVariation` | Vary wall height and the robot-to-wall distance. |

## Category: Place

Carry one cube to a target spot on the table and release it there.

**This is a debug task.** It is the sanity check the QC-FQL training pipeline is brought up
on: the smallest complete manipulation task, used to prove the pipeline can learn *something*
before a real category is attempted.

| **Variation** | Description |
| --- | --- |
| `RandS-RandT` | Cube spawn and target both sampled every reset; the real task. |
| `RandS-FixedT` | Cube spawn sampled, target pinned. |
| `FixedS-RandT` | Cube spawn pinned, target sampled. |
| `FixedS-FixedT` | Both pinned; the smallest learnable version. |

Place is exempt from the spawn tables: spawn and target draw uniformly over
the shared envelope when random (with one deterministic push-apart keeping
them 0.10 m apart) and pin exactly when fixed; see
`docs/adr/019-fixed-spawn-points.md`.

## Future Work (don't implement for now):

### Extra Variation:
place objects in bowl at beginning -> has to first get objects out of bowl

### D3IL: Inserting
Dont implement for now. Replaces lead ball (remove). Also remove Push Cube (is not D3IL Inserting task).

### Peg Insertion
Current Variations (keep):
- Form: circle, rect, hexagon
- Orientation: horizontal, vertical
- 2Sheets: through 2 Holes
