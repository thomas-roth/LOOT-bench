# Task Categories

## General
Goal:
- only collect Data for hardest Task in all variations
- use a script to create demonstrations for smaller tasks
  - split at progress rewards
  - adapt rewards (scale progress rewards to [0,1], add episode success)
  - adapt observations to fit task (e.g. only 2 cubes for Stack2 (remove last 2 cubes from Stack4))

Pay attention to the drawing of random numbers from a seed. First draw all randomness for the easiest task (e.g. Stack2), and then continue step-wise to the hardest task (e.g. then Stack3 and after that Stack4). This allows the cutoff to work.

### Global Modifiers:
- Margin
- Ordered

-> always collect data in small Margins & Ordered (needed so cutoff works)

### Randomization:
always randomize objects & goal poses (position and orientation) a little if not stated otherwise to prevent overfitting. 

## Category: Stack

Stack blocks to reach a target height. Only the final height matters; block
position does not. Every variant carries `2n` identical blocks, more than the
minimum needed, so support structures (staircase, pyramid) are viable
strategies — height, not block count, is the constraint.

| **Evaluated Capabilities** | Description |
| --- | --- |
| Combinatorial complexity and transfer | Use different stacking strategies for differently shaped blocks. |
| Robustness | Continue building even with an unstable base. |

| **Task** | Description |
| --- | --- |
| Stack3 | Build a stack three blocks high. |
| Stack4 | Build a stack four blocks high. |
| Stack5 | Build a stack five blocks high. |

| **Variation** | Description |
| --- | --- |
| `Easy` | Flat, wide blocks (0.05 x 0.05 x 0.025 m, h:w 0.5): large landing area, low center of mass per placement, short tower. |
| `Medium` | Standard cubes (0.04 m, h:w 1.0). |
| `Hard` | Tall, slender blocks (0.03 x 0.03 x 0.06 m, h:w 2.0): small top face, tippy stack, and a Stack5 tower reaching 0.30 m. |

All blocks of a variant are identical, so the target height is reachable by
exactly `n` blocks and by no other count. Blocks tail-extend across the target
height *within one difficulty* only — the block shape differs between Easy,
Medium and Hard, so their recordings are never mixed when splitting demos.

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

| **Variation** | Description |
| --- | --- |
| `base` | Use identical basic dominoes only. |
| `differentDominoes` | Dominoes become progressively thinner and lighter. |
| `knockOver` | Topple the completed line as a chain: every domino falls along the row, same direction, in row order; phase observation build (`0`) / topple (`1`). |

## Category: Sorter

Insert every peg into its matching hole in a shape sorter. Each peg fits only its corresponding hole.

| **Evaluated Capabilities**  | Description |
| --- | --- |
| Combinatorial complexity | Reposition the cube or perform more difficult insertions. |
| Strategy | Choose an effective insertion order. |
| Precision | Handle tight insertion tolerances. |

| **Task** | Description |
| --- | --- |
| 3Shapes | Insert three shapes into the sorter. |
| 4Shapes | Insert four shapes into the sorter. |
| 5Shapes | Insert five shapes into the sorter. |

| **Variation** | Description |
| --- | --- |
| `fixedOrientation` | The sorter orientation is fixed; its position varies. |
| `noRotation` | The sorter orientation varies, but no required hole faces the ground. |
| `full` | The sorter position and orientation are random; rotating it may be required. |
| `randomPositions` | Pegs start at random positions around the sorter. |

## Category: Ball

Bring a ball to its goal by hitting or throwing it.

| **Evaluated Capabilities** | Description |
| --- | --- |
| Dynamics | Apply sufficient force to reach the goal. |
| Long-horizon credit assignment | Receive the reward substantially after the action. |

### Task: Minigolf

Hit the ball once to reach a goal outside the robot's direct reach. Every
layout has a perimeter railing around the platform edges, and the layouts
are strictly additive: `usingWall` ⊂ `obstacle` ⊂ `ramp`, each adding one
more element on top of the previous layout.

| **Variation** | Description |
| --- | --- |
| `direct` | Reach the goal with a direct hit; the perimeter railing alone. |
| `usingWall` | Adds a blocker across the direct line; the perimeter railing serves as the bank for a redirect shot. |
| `obstacle` | `usingWall` plus an obstacle in the lane the blocker leaves open. |
| `ramp` | `obstacle` plus the goal raised onto an elevated section, reached via an incline. |

### Task: Throw

Throw the ball over a wall to reach the target. Evaluate whether reduced gravity is needed.

| **Variation** | Description |
| --- | --- |
| `U` | U-shaped walls surround the goal. |
| `markedArea` | A marked area defines the goal. |
| `wallVariation` | Vary wall height and the robot-to-wall distance. |


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
