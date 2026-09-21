# Task Categories

## General
Goal:
- record demonstrations on the hardest task of a category
- cut easier tasks out with `loot-split-demos` (prefix rule below)

Seed-prefix rule: randomness for the easiest task is drawn first, harder
tasks extend the draws, so an easier reset is a prefix of a harder one.
Shapes/blocks tail-extend the same way (2 of 5 = first 2 of 5).

### Modifiers
- Margin (all categories): scales the position/tilt success tolerances.
  Default 1.0 = tight, for data collection. Ball pit is physical and
  unscaled; only the throw radius scales. Stack tilt gate is fixed.
- Ordered (domino only): identity slot assignment. `DifferentDominoes`
  requires it; all registered domino variants set it.

Collect data at margin 1.0 & ordered.

Every category gates success on at-rest + released on top of its placement
test (`common/success.py`); see `rl_settings.md`.

## Category: Stack

Build a pi: a *level* is one horizontal block (a lintel) held at a fixed
height by whatever stands under it, and the task is to stack one, two or
three levels on top of each other. Only the lintel is scored, never the
supports (`docs/adr/020-a-pi-level-is-a-horizontal-block-at-a-fixed-height.md`).

| **Evaluated Capabilities** | Description |
| --- | --- |
| Long-horizon sequencing | Chain up to 9 block placements over 4500 steps; progress is instantaneous, so a collapse undoes it. Scales with the task (1/2/3 Pis). |
| Precision placement | Seat each lintel within 3 mm height and 5.7° tilt gates; a held block earns nothing. |
| Reorientation | Turn a standing spawn into a horizontal lintel; knock-over-then-regrasp is the operator method, the code checks endpoints only. |
| Load-bearing base | Upper levels need a stable base; T vs trilithon is open, decided by physics. Matters from 2Pis up. |

`FreePos` isolates no capability: success is position-free both ways, it only moves the ghost out of the build zone.

| **Task** | Levels | Blocks | Lintel centres (m) | Episode |
| --- | --- | --- | --- | --- |
| 1Pi | 1 | 3 | 0.075 | 75 s |
| 2Pis | 2 | 6 | 0.075, 0.160 | 150 s |
| 3Pis | 3 | 9 | 0.075, 0.160, 0.245 | 225 s |

| **Variation** | Description |
| --- | --- |
| *(none)* | Ghost trilithon in the build-zone middle as the build guide: `LOOT-Stack-1Pi-Dk1`. |
| `FreePos` | Ghost out of the build zone, operator builds anywhere: `LOOT-Stack-1Pi-FreePos-Dk1`. Success is position-free both ways. |

Two builds complete a level and the criterion prefers neither:

```
  trilithon              T
   +-----+            +-----+      lintel: one block lying flat
   #     #               #         upright(s): block(s) standing on end 
   #     #               #         
```

The trilithon costs a third block, but stands more stable. The T is faster but 
tips easily. Nothing rewards or forbids either; the physics decides, and its 
verdict only matters from 2Pis up, where a level has to carry the next one.

One block type serves as upright and as lintel: a square prism of
0.065 x 0.020 x 0.020 m, 40.6 g. Blocks are coloured in tiers of three, navy, 
teal and white, one tier per level: 1Pi's blocks are the first three of 2Pis, 
which are the first six of 3Pis. 
Operator protocol, which the code cannot enforce: build level 1 from the navy
blocks, level 2 from teal, level 3 from white, bottom-up, or a 3Pis
recording will not split into the easier tasks.

Spawns stage from the fixed table (object `i` on cell `i`, ±1 cm jitter),
with the build site kept clear; see `docs/adr/019-fixed-spawn-points.md`.
Margin scales the height tolerance; the tilt gate is fixed.

## Category: Domino

Build a domino line that can topple sequentially.

| **Evaluated Capabilities** | Description |
| --- | --- |
| Narrow-tolerance placement | Seat each domino within 15 mm × margin and 15° × margin beside standing dominoes; knocking one over loses its slot, no explicit penalty. Scales with the task (2–6 pieces). |
| Row-pose variation | Build at a resampled row pose (`RandomRow`: y ±0.08 m, yaw ±0.35 rad; x pinned). Per-variant eval, no held-out protocol. |
| Property variation | Handle dominoes whose thickness and mass vary along the row (`DifferentDominoes`: 0.8^i, ordered slots). Train and test distributions match. |
| Ordered toppling | Topple the finished line same-direction, in row order (`KnockOver`; phase-observed). |

| **Task** | Description | Episode |
| --- | --- | --- |
| 2Pieces | Place two dominoes in a line. | 40 s |
| 3Pieces | Place three dominoes in a line. | 60 s |
| 4Pieces | Place four dominoes in a line. | 80 s |
| 5Pieces | Place five dominoes in a line. | 100 s |
| 6Pieces | Place six dominoes in a line. | 120 s |

`KnockOver` adds 30 s on top. Task ids read
`LOOT-Domino-{n}Pieces[-{RandomRow,DifferentDominoes,KnockOver}]-Dk1`.

| **Variation** | Description |
| --- | --- |
| *(none)* | Use identical basic dominoes only. The base variant carries no variation segment: `LOOT-Domino-2Pieces-Dk1`. |
| `RandomRow` | Row centre y (±0.08 m) and yaw (±0.35 rad about ±π/2) redrawn every reset; row centre x stays pinned at 0.40. |
| `DifferentDominoes` | Domino thickness and mass scale by 0.8^i along the row. Requires `ordered`. |
| `KnockOver` | Topple the completed line as a chain: every domino falls along the row, same direction, in row order; phase observation build (`0`) / topple (`1`). |

Spawns stage from the fixed table (object `i` on cell `i`, ±1 cm jitter),
left of the pinned row lane by construction; see `docs/adr/019-fixed-spawn-points.md`.

## Category: Sorter

Insert every peg into its matching hole in a shape sorter. Each peg fits only its corresponding hole.

| **Evaluated Capabilities**  | Description |
| --- | --- |
| Tight-tolerance insertion | Pass each piece through its hole past the crossing plane at 5 mm × margin slack. Kinematic latch, no force gate. Scales with the task (2–5 shapes; circle/rect core first). |
| Multi-piece completion | Complete up to 5 insertions in any order; latches hold. |
| Pose inference | Recover every hole pose from the cube pose alone; holes are fixed offsets, the command is unobserved. |
| Cube reorientation | Turn the cube when a required hole faces the table (`Full` only; `NoRotation` never needs it, `FixedOrientation` pins the dead star face down). |

| **Task** | Description | Episode |
| --- | --- | --- |
| 2Shapes | Insert two shapes into the sorter. | 40 s |
| 3Shapes | Insert three shapes into the sorter. | 60 s |
| 4Shapes | Insert four shapes into the sorter. | 80 s |
| 5Shapes | Insert five shapes into the sorter. | 100 s |

Shapes tail-extend (2 = first 2 of the 5-shape list), so easier tasks
split out of the hardest recording. Task ids read
`LOOT-Sorter-{n}Shapes-{FixedOrientation,NoRotation,Full}-Dk1`.

| **Variation** | Description |
| --- | --- |
| `FixedOrientation` | Cube face-down on the dead star face, position jittered ±1 cm. |
| `NoRotation` | Cube face-down on an allowed face (no required hole faces the table), plus random yaw. |
| `Full` | Cube face-down on any of the six faces (plus random yaw); rotating it may be required. |

Cube position is always (0.375, 0) ±1 cm; hole slack is 5 mm × margin.

Pieces stage from the fixed five-cell table in front of the cube (object `i`
on cell `i`, ±1 cm jitter) with the insertion corridor kept clear; see
`docs/adr/019-fixed-spawn-points.md`.

## Category: Ball

Bring a ball to its goal by hitting or throwing it. The reward is fully
sparse (`-1` until it scores), so only the scored state pays; repeated attempts allowed.

| **Evaluated Capabilities** | Description |
| --- | --- |
| Dynamic striking | Score by hitting (minigolf) or throwing (throw); the ball must end at the target, at rest and released. |
| Sparse-reward credit assignment | No progress signal: `-1` every step until the ball scores. The only fully sparse task. |
| Cluttered lane | Play a lane with blocker and obstacle present; contact with them is unpenalized (`UsingWall` affords a bank shot, not enforced). |
| Elevated goal via incline | Climb the incline onto the raised goal section (`Ramp`). |
| Release-gated throw | Clear the wall with the ball past 0.08 m from the grasp site; distance-only gate, no timing condition (throw). |
| Spawn and wall variation | Cover resampled ball spawns and field poses; moving wall in `WallVariation` only. No held-out protocol. |

### Task: Minigolf

Hit the ball once to reach a goal outside the robot's direct reach. Every
layout has a perimeter railing around the platform edges, and the layouts
are strictly additive: `Direct` ⊂ `UsingWall` ⊂ `Obstacle` ⊂ `Ramp`, each adding one
more element on top of the previous layout.

| **Variation** | Description | Episode |
| --- | --- | --- |
| `Direct` | Reach the goal with a direct hit; the perimeter railing alone. | 10 s |
| `UsingWall` | Adds a blocker across the direct line; the perimeter railing serves as the bank for a redirect shot. | 12 s |
| `Obstacle` | `UsingWall` plus an obstacle in the lane the blocker leaves open. | 16 s |
| `Ramp` | `Obstacle` plus the goal raised onto an elevated section, reached via an incline. | 20 s |

### Task: Throw

Throw the ball over a wall to reach the target (marked disk or U-shaped
pocket). The wall is baked into the field, except in `WallVariation`, where
a separate wall moves (x ±0.06 m, sink 0..0.08 m) every reset.

| **Variation** | Description | Episode |
| --- | --- | --- |
| `U` | U-shaped walls surround the goal. | 12 s |
| `MarkedArea` | A marked area defines the goal. | 12 s |
| `WallVariation` | Marked area plus randomized wall pose. | 16 s |

## Category: Place

Carry one cube to a target spot on the table and release it there.

**This is a debug task.** It is the sanity check the QC-FQL training pipeline is brought up
on: the smallest complete manipulation task, used to prove the pipeline can learn *something*
before a real category is attempted.

| **Evaluated Capabilities** | Description |
| --- | --- |
| Pick-and-place proficiency | Grasp one cube, carry it, release inside 0.02 m × margin. Baseline before any real category. |
| Goal-conditioned coverage | Cover the sampled spawn/target envelope; each axis pins independently down to `FixedS-FixedT`. Interpolation only, no held-out protocol. |

| **Variation** | Description |
| --- | --- |
| `RandS-RandT` | Cube spawn and target both sampled every reset; the real task. |
| `RandS-FixedT` | Cube spawn sampled, target pinned. |
| `FixedS-RandT` | Cube spawn pinned, target sampled. |
| `FixedS-FixedT` | Both pinned; the smallest learnable version. |

Place is exempt from the spawn tables: spawn and target draw uniformly over
the shared envelope when random (with one deterministic push-apart keeping
them 0.10 m apart) and pin exactly when fixed (`FIXED_SPAWN_XY=(0.25, 0)`,
`FIXED_TARGET_XY=(0.40, 0)`, yaw 0); see
`docs/adr/019-fixed-spawn-points.md`. Success radius is 0.02 m × margin.

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
