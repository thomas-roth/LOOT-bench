# Domino task category

Build a row of dominoes standing upright, close enough together that they
could knock each other over.

A *variant* (`variants.py`) fixes the domino count. The `DominoCommand` draws
a row start pose and direction (one shared draw, sized for the hardest count
so its distribution never shifts with the variant's n) and scatters all
dominoes upright elsewhere in the workspace at every reset. Success holds on
the first step where every domino stands within `pos_threshold * margin` of
its row slot and within `upright_threshold_deg * margin` of upright and
row-aligned, all dominoes move slowly, and the gripper is open. `KnockOver`
variants latch a `phase` (0 = build, 1 = topple) the first time the row is
complete, and then additionally require the row to topple as a chain: every
domino falls along the row axis (tilted past 45°, leaning toward the row
direction rather than sideways), all leaning the same way, with first-fall
times monotonic along the row in that direction — sideways or out-of-order
toppling doesn't count.

## Observations (state-based)

- joint positions/velocities, EE pose
- all domino poses (env-local position + quaternion, object-major)
- per-slot placed indicator
- the command: per-slot target pose (env-local position + upright/row-aligned
  quaternion), object-major
- `phase` (build/topple), KnockOver variants only

## Rewards

- `progress`: placement fraction while building; for KnockOver, half the
  placement fraction during the build phase plus half the toppled fraction
  during the topple phase
- `success`: bonus while the success criterion holds

## Variants

| Task id (`-Dk1_uni`)                            | Description                                    |
| ------------------------------------------------ | ----------------------------------------------- |
| `G3A-Domino-{2,3,4}Pieces`                        | place n standard dominoes in a row               |
| `G3A-Domino-{2,3,4}Pieces-DifferentDominoes`      | dominoes progressively thinner and lighter       |
| `G3A-Domino-{2,3,4}Pieces-KnockOver`              | topple the completed line after building it      |

Play e.g. with:

```bash
uv run g3a-play G3A-Domino-3Pieces-Dk1_uni
```
