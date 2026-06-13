# Object-insertion task category

Insert one or more pegs into tight-fitting bores, covering the spec's three
families:

- **peg insertion**: a single peg into a single bore on a fixed board
  - **forms**: cylinder (`circle`), cuboid (`rect`), hex key (`hexagon`)
  - **orientation**: `vertical` (mouth up) or `horizontal` (mouth facing the
    robot, on a cantilevered tube)
- **two-sheet alignment**: the bore only lines up after a movable top sheet is
  slid over the fixed bottom sheet, then the peg goes through both
- **shape sorter**: a board with several differently shaped bores where every
  peg only fits (and only scores in) its matching bore

## Geometry

Pegs and bores share one cross-section description (`SHAPE_GEOMS` in
`variants.py`). A peg is a convex prism — `circle` is a cylinder, `square`/
`rect` are boxes, `triangle`/`hexagon` are convex-hull meshes of their corner
vertices. The matching bore is a ring of wall boxes with apothem
`peg_inradius + clearance` (1.5 mm), so the peg is a tight fit and a wrong peg
does not drop in. Boards are *fixed-base* entities (mjlab auto-wraps them as
mocap props pinned to the env origin); pegs and the movable sheet are free
bodies. `socket_layout` (`assets.py`) produces both the wall geometry and the
goal poses used by the command, so they can never drift apart.

## Insertion success

For each socket the assigned peg counts as *inserted* when

1. its deepest tip is at least `insert_depth` (2.6 cm) below the bore mouth
   along the bore axis,
2. it is laterally centred on the axis (within `lateral_tol`, 1 cm), and
3. its long axis is aligned with the bore axis (within `tilt_tol_deg`).

Two-sheet variants additionally require the top sheet hole to be aligned over
the bottom hole. An episode terminates successfully when all pegs stay
inserted for `hold_time` (0.5 s).

## Observations (state-based)

- joint positions/velocities, last action
- all peg positions relative to the grasp site
- bore-mouth deltas per socket (mouth minus assigned peg position)
- per-socket inserted flags
- the command: bore mouth positions (env-origin relative) and bore axes
- two-sheet only: top-sheet position relative to grasp and to the bore

## Rewards

- `staged_next_socket`: `reaching * (1 + approaching)` focused on the next
  un-inserted socket (reach the peg, bring it to a pre-insertion waypoint)
- `insertion_depth`: per-socket depth fraction, gated by lateral alignment
- `inserted_count`: fraction of pegs correctly inserted
- `success`: bonus while every peg is inserted
- two-sheet only: `sheet_alignment` shaping (reach the sheet, align its hole)
- penalties: action rate, joint position limits, joint velocity hinge
  (with the standard curriculum)

## Variants

| Task id (`-Dk1` / `-Yam`)      | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| `G3a-Insertion-PegCylinder`    | cylinder peg into a vertical round bore           |
| `G3a-Insertion-PegCuboid`      | rectangular peg into a vertical rectangular bore  |
| `G3a-Insertion-PegHex`         | hexagonal peg into a vertical hex bore            |
| `G3a-Insertion-PegCylinderH`   | cylinder peg into a horizontal bore facing robot  |
| `G3a-Insertion-TwoSheet`       | align the top sheet, then insert through both     |
| `G3a-Insertion-Sorter3`        | shape sorter: circle, square, triangle            |
| `G3a-Insertion-Sorter4`        | shape sorter: circle, square, triangle, hexagon   |

Train e.g. with:

```bash
uv run g3a-train G3a-Insertion-Sorter4-Dk1 --env.scene.num-envs 2048
```
