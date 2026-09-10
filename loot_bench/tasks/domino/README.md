# Domino task category

Build a row of dominoes standing upright, close enough together that they
could knock each other over.

A *variant* (`variants.py`) fixes the domino count. The row's centre x is
pinned at `DOMINO_ROW_X` every reset — so a teleoperator recording demos
always builds in the same spot; `RandomRow` is the one variant that instead
draws the row's centre y and yaw (one shared draw, sized for the hardest
count so its distribution never shifts with the variant's n). The dominoes
themselves stage upright from the fixed spawn table at every reset in both
cases (object `i` on cell `i`, ±1 cm jitter). Spawns sit left of
`DOMINO_LANE_EDGE` while every row slot sits right of it, so no domino ever
starts on or right next to a target — aligning one out of its own goal
position is needlessly fiddly, and a domino that close could even count as
placed straight out of the reset; the separation holds by construction, with
no keep-out mask. Success holds on
the first step where every domino stands within `pos_threshold * margin` of
its row slot and within `upright_threshold_deg * margin` of upright and
row-aligned, all dominoes move slowly, and the gripper has released them
(contact-only release gate). `KnockOver`
variants latch a `phase` (0 = build, 1 = topple) the first time the row is
complete, and then additionally require the row to topple as a chain: every
domino falls along the row axis (tilted past 45°, leaning toward the row
direction rather than sideways), all leaning the same way, with first-fall
times monotonic along the row in that direction — sideways or out-of-order
toppling doesn't count.

## Piece identity colors

Every registered variant uses `ordered=True` (fixed domino-index -> slot-index
assignment), but dominoes previously all rendered identically and spawned at
a uniformly random position independent of index -- so a teleoperator had no
way to tell which physical domino belonged in which slot. Each domino now
gets a distinct identity color from the shared color palette, indexed 0 through
`HARDEST_COUNT - 1` (5).

The matching row-slot ghost marker (`DominoCommand._debug_vis_impl`,
`mdp/commands.py`) is drawn as two overlaid boxes, since mjlab's debug
visualizer has no separate outline primitive: a slightly larger, near-opaque
**red** box (`tasks/common/debug_markers.OUTLINE_RGBA`) marking it as a
virtual target, and a smaller fill box in the slot's assigned domino's own
identity color while pending (at 75% opacity -- higher than the shared
default, since the red outline already carries the "this is a ghost, not a
real object" signal, freeing the fill to closely match the real piece color).
The fill switches to green when that slot is individually placed, and a
brighter green once the whole task succeeds (unchanged from the shared
`tasks/common/debug_markers.marker_color` convention every category uses).

## Observations (state-based)

- joint positions/velocities, EE pose
- all domino poses (env-local position + quaternion, object-major)
- per-slot placed indicator
- the command: per-slot target pose (env-local position + upright/row-aligned
  quaternion), object-major
- `phase` (build/topple), KnockOver variants only

## Rewards

One term, in `[-1, 0]`, exactly `0` only at success
(`docs/adr/006-one-task-reward-in-minus-one-to-zero.md`). It is shaped by the
placement fraction while building. For KnockOver it is shaped by half the
placement fraction during the build phase, then by half the placement
fraction plus half the toppled fraction during the topple phase. It is
floored at `-MIN_SHORTFALL` so that no non-success step can read a true
zero.

## Variants

| Task id (`-Dk1`)                              | Description                                        |
| -------------------------------------------------- | -------------------------------------------------- |
| `LOOT-Domino-{2,3,4,5,6}Pieces`                         | place n standard dominoes in the fixed row          |
| `LOOT-Domino-{2,3,4,5,6}Pieces-RandomRow`               | row start pose and yaw redrawn every reset          |
| `LOOT-Domino-{2,3,4,5,6}Pieces-DifferentDominoes`       | dominoes progressively thinner and lighter          |
| `LOOT-Domino-{2,3,4,5,6}Pieces-KnockOver`               | topple the completed line after building it         |

Play e.g. with:

```bash
uv run loot-play LOOT-Domino-3Pieces-Dk1
```
