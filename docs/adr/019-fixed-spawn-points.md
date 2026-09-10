# Fixed spawn points from the arm's measured envelope

Objects used to scatter over whole workspace rectangles, landing in the work
area and at the edge of reach, with a keep-out mask to repair the overlap.
Spawns are now hand-written tables laid out inside the arm's measured
envelope, clear of the work zone by construction: object ``i`` takes cell
``i`` with +/-1 cm jitter, so an easier task's reset is a prefix of a harder
one's.

## Measured envelope

Reach at table height is a **disc of radius 0.51 m**, constant from 0 to 70
deg on the compiled DK1. The shared envelope keeps margin inside it:

- ``x >= 0.20`` (near bound),
- ``hypot(x, y) <= 0.49`` at any jittered pose (0.45 stays the far reach on
  the centreline; the radial bound is what the old far corner already
  implied, and it frees the near band's lateral room),
- ``abs(y) + object_radius <= 0.35`` (the table's half-width).

Every cell is checked at all four corners of its jitter box. The reachable
volume is a **dome**: grasp-site ``z`` ceiling 0.28 at x = 0.25-0.30, 0.26 at
x = 0.35, 0.22 at x = 0.40, 0.20 at x = 0.45; z = 0.30 is unreachable
everywhere.

## Pitch from the gripper

Neighbours must clear the descending gripper, so pitch is
``2 * jitter + r_a + r_b + CLEARANCE`` with ``CLEARANCE = 0.04``. That is the
**closed** gripper's swept half-span (measured 39 mm); the gripper descends
open (152 mm span, 80 mm swept radius), and sizing on the open span is not
required because yaw is free and OGBench uses a 0.10-0.15 m band for a
same-size cube under the same +/-1 cm jitter. The formula yields 0.117 m for
stack blocks and 0.094 m for dominoes, inside that band. Layouts win by
zoning, not density.

## Zoning rule

Each category stages its objects near the robot and keeps its work area
empty by construction, so the runtime keep-out mask, ``blocked_near`` and
``keep_free`` go away: stack keeps the tower site
``x in [0.24, 0.32], |y| <= 0.07`` plus a halo of object radius + jitter +
0.02 free; domino pins the row at ``x = 0.40`` so spawns and row separate in
``x`` for any row ``y`` and tilt; sorter keeps the rect hole's centreline
approach corridor free. Jitter is +/-1 cm everywhere, including the sorter
cube's reset. ``Place`` is exempt: its random/fixed axis stays fully uniform
over the envelope when random and fully pinned when fixed.

Stack's zoning is inverted relative to the others (tower **near**, staging
around it) because the dome forces it: placing Hard Stack5's top block puts
the grasp site at 0.27 m, which IK confirms at x = 0.24-0.32 and fails at
x = 0.35 (19.6 mm residual). A far tower would make the hardest variant
unwinnable.
