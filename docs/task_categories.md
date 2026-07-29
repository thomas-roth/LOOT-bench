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
position does not. Block sets carry extra material beyond the minimum
needed, so support structures (staircase, pyramid) are viable strategies —
height, not block count, is the constraint.

| **Evaluated Capabilities** | Description |
| --- | --- |
| Combinatorial complexity and transfer | Use different block sets and stacking strategies. |
| Robustness | Continue building even with an unstable base. |

| **Task** | Description |
| --- | --- |
| Low | Build a stack three standard blocks high. |
| Medium | Build a stack four standard blocks high. |
| High | Build a stack five standard blocks high. |

| **Variation** | Description |
| --- | --- |
| `base` | Identical cubes only, `2n` of them for a target height of `n` blocks. |
| `differentBlocks` | A mixed prefix of four forms plus cubes: `cube` (balanced baseline), `plate` (wide, stabilizing, barely any height), `tall` (most height per placement, slender and tippy), `cylinder` (good height, yaw-invariant grasp, rolls when tipped). All heights are multiples of 0.02 m, so every set has an exact-height combination for the target. |
| `2Stacks` | Build two separate stacks that each reach the target height, `2n+2` cubes. |
| `2StacksDifferentBlocks` | `2Stacks` with a doubled mixed prefix of the four forms plus cubes. |

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
