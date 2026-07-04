# Cube task category

Long-horizon piece arrangement, in the spirit of the OGBench cube tasks but
procedurally generated and with more variation axes:

- **forms**: cube, cylinder, rect (flat cuboid), domino (tall thin cuboid)
- **colors**: single color per stack (one or several stacks), or a specific
  color order within one stack
- **arrangement**: `stack` (pieces on top of each other) or `domino` (pieces
  standing upright in a row, spaced closely enough for the domino effect)
- **available pieces**: exactly the needed pieces, or extra distractor pieces
  with other colors/forms

## Concepts

A *variant* (`variants.py`) fixes the piece set and the goal arrangement.
A *slot* is one position in the goal arrangement (bottom-to-top within a
stack; row order for dominos). The `PieceArrangementCommand` samples goal
poses and scatters the pieces at every episode reset:

1. Stack base positions (or the domino row pose) are sampled inside the
   workspace.
2. All pieces are scattered over distinct cells of a jittered grid, which is
   collision-free by construction and fully vectorized.
3. Goal slots are randomly re-assigned among interchangeable pieces (same
   form and color), so identical pieces can fill each other's slots.

A slot counts as *placed* when its piece is within `pos_threshold` (1.5 cm) of
the slot goal; domino slots additionally require the piece to be upright and
rotated so its thin axis lies along the row (otherwise toppling would not
propagate). The episode terminates successfully on the first step where every
slot is placed, all goal pieces are moving slowly, and the gripper is open.

## Observations (state-based)

- joint positions/velocities, last action
- all piece positions relative to the grasp site
- goal deltas per slot (goal minus assigned piece position)
- per-slot placed flags
- the command: slot goal positions (env-origin relative), plus the row
  direction (cos/sin) for domino variants

## Rewards

- `placed_slots`: fraction of placed slots (progress + keep-placed incentive)
- `success`: bonus while the full arrangement is complete
- penalties: action rate, joint position limits, joint velocity hinge
  (with the standard curriculum)

## Variants

| Task id (`-Dk1_uni`)        | Description                                       |
| --------------------------- | ------------------------------------------------- |
| `G3a-Cube-Stack3`           | stack 3 red cubes                                 |
| `G3a-Cube-Stack3Ordered`    | stack red, green, blue cubes in this order        |
| `G3a-Cube-Stack3Distract`   | stack 3 red cubes; blue cube + green cylinder distractors |
| `G3a-Cube-TwoStacks`        | two single-color stacks (2 red, 2 blue)           |
| `G3a-Cube-Stack3Cylinders`  | stack 3 cylinders                                 |
| `G3a-Cube-Stack3Rects`      | stack 3 flat cuboids                              |
| `G3a-Cube-StackMixedForms`  | rect base, cube middle, cylinder top              |
| `G3a-Cube-Domino5`          | stand 5 dominos in a row (domino-effect spacing)  |
| `G3a-Cube-Domino5Distract`  | as above, plus a wrong-color domino and a cube    |

Train e.g. with:

```bash
uv run g3a-train G3a-Cube-Stack3-Dk1_uni --env.scene.num-envs 1024
uv run g3a-play G3a-Cube-Stack3-Dk1_uni
```
