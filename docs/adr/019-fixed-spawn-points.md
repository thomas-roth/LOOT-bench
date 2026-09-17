# Fixed spawn points from the arm's measured envelope

> **Amended 2026-09-11.** Stack became the pi family (ADR-020) and its
> block changed from the 0.025 x 0.025 x 0.06 m `Hard` block to a 0.065 x
> 0.020 x 0.020 m prism. The envelope, the pitch formula and the zoning
> conclusion are unchanged. Two cited numbers were replaced below and are
> marked *Amended*: the pitch for stack blocks, and the top-block reach datum
> that justified building near the robot. Amended in place, not overturned.

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
same-size cube under the same +/-1 cm jitter. The formula yields 0.088 m for
the pi block (*Amended*: an upright 20 mm-square prism, half-diagonal
0.01414 m, so ``2*0.01 + 2*0.01414 + 0.04``; it was 0.117 m for the old
stack cube) and 0.094 m for dominoes, inside that band. Stack's table is the
first 9 cells of the old one, whose minimum pairwise separation is 0.12 m, so
it now carries margin over the pitch rather than sitting on it. Layouts win
by zoning, not density.

## Zoning rule

Each category stages its objects near the robot and keeps its work area
empty by construction, so the runtime keep-out mask, ``blocked_near`` and
``keep_free`` go away: stack keeps the build site
``x in [0.24, 0.32], |y| <= 0.07`` plus a halo of object radius + jitter +
0.02 free; domino pins the row at ``x = 0.40`` so spawns and row separate in
``x`` for any row ``y`` and tilt; sorter keeps the rect hole's centreline
approach corridor free. Jitter is +/-1 cm everywhere, including the sorter
cube's reset. ``Place`` is exempt: its random/fixed axis stays fully uniform
over the envelope when random and fully pinned when fixed.

Stack's zoning is inverted relative to the others (build site **near**,
staging around it) because the dome forces it. *Amended*: the datum is now
the 3-level pi's top lintel, centre 0.245 m on upright tops at 0.235 m.
That puts the grasp site in a **band of 0.239-0.264 m**, not at one height,
because the site is not the block's centre: the finger pad spans site-frame
``z`` in ``[-0.029, +0.041]`` (``dk1.xml:266-268``), so the site height
depends on where along that 70 mm band the lintel is gripped (0.239 with the
pad centred on it, 0.264 with the pad's bottom flush with its underside).
The whole band is under the 0.28 m ceiling at x = 0.25-0.30 and the ~0.272 m
the dome interpolates to at the build zone's far edge x = 0.32. The x = 0.35
failure (19.6 mm residual) was measured on the old Hard Stack5 top block at
0.27 m; it established the dome's fall-off and still holds, since it lies
inside the pi band. A far build site would make ``3Pis`` unwinnable.
