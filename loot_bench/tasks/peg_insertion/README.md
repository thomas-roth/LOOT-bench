# Peg-insertion task category (archived)

**Archived, unregistered.** This category predates the `tasks/task_categories`
revamp; its sorter family was carried forward and rebuilt as
`tasks/sorter` (every piece must latch, not just one commanded target;
STL assets moved to `sorter/xmls/`). What's left here — a single peg
into a fixed bore, and the two-sheet alignment puzzle — has no registered
task ids, no `config/`, and is not covered by the test suite. It is kept
importable and pyright-clean (in case a future category wants to borrow the
bore/peg geometry) but nothing here runs in CI beyond `import`.

Two task kinds share one command/observation/reward stack:

- **peg insertion**: a single peg into a single bore on a fixed board
  - **forms**: cylinder (`circle`), cuboid (`rect`), hex key (`hexagon`)
  - **orientation**: `vertical` (mouth up) or `horizontal` (mouth facing the
    robot, on a cantilevered tube)
- **two-sheet alignment**: the bore only lines up after a movable top sheet is
  slid over the fixed bottom sheet, then the peg goes through both

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
the bottom hole. Success also requires every peg to be at rest and released
(contact-only release gate). The episode terminates successfully on the first step all of
that holds; the `success` reward and `episode_success` metric use the same
condition.

## Observations (state-based)

- joint positions/velocities, EE pose
- the board's own pose (env-local position + quaternion) — a single pose for
  the whole rigid board/socket assembly (fixed and constant per task, but
  included for completeness/parity with sorter's cube pose)
- all peg poses (env-local position + quaternion, object-major)
- per-socket inserted flags
- two-sheet only: top-sheet pose (env-local position + quaternion)

## Rewards

Sparse by design, to pair with a demonstration pipeline (GELLO teleop →
offline-to-online RL) that supplies the exploration dense shaping would
otherwise provide:

- `inserted_count`: fraction of pegs correctly inserted (position only)
- `success`: bonus while every peg is inserted, at rest, and released

## Variants (archived, no task ids registered)

| Variant           | Description                                       |
| ------------------ | -------------------------------------------------- |
| `PegCylinder`       | cylinder peg into a vertical round bore           |
| `PegCuboid`         | rectangular peg into a vertical rectangular bore  |
| `PegHex`            | hexagonal peg into a vertical hex bore            |
| `PegCylinderH`      | cylinder peg into a horizontal bore facing robot  |
| `TwoSheet`          | align the top sheet, then insert through both     |
