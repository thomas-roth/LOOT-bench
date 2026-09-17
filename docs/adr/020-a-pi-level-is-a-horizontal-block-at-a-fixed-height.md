# A pi level is a horizontal block at a fixed height

Stack scored a tower by height alone, nine tasks over `Stack{3,4,5}` x
`{Easy,Medium,Hard}`. It is replaced by three pi tasks
(`LOOT-Stack-1Pi-Dk1`, `-2Pis-`, `-3Pis-`): level `k` is complete
when some block lies horizontal with its centre at `k(L+w) - w/2` above the
table, and the task encodes nothing about what holds it there. The criterion
tests the lintel and never the supports, because any structural test either
forbids a legitimate build or admits one it meant to exclude, so the supports
are left to the physics.

## The criterion

Level `k` is complete when one block satisfies all of these at once:

1. horizontal: `|cos(long axis, world z)| <= TILT_COS_MAX` (`0.10`, 5.7 deg),
2. at height: `|centre_z - table_z - (k(L+w) - w/2)| <= HEIGHT_TOL` (`0.003`),
3. not held: excluded by the ADR-017 mask.

Progress is `completed_levels / N`, instantaneous, so a collapse takes the
credit back. Success is progress `1` plus the shared at-rest and released
gates (`SuccessGate`, ADR-018). One predicate feeds both the reward and the
termination so the two cannot drift.

| Task | Levels | Blocks | Lintel centres (m) | Episode |
| --- | --- | --- | --- | --- |
| `1Pi` | 1 | 3 | 0.075 | 75 s |
| `2Pis` | 2 | 6 | 0.075, 0.160 | 150 s |
| `3Pis` | 3 | 9 | 0.075, 0.160, 0.245 | 225 s |

The block is one square prism, `0.065 x 0.020 x 0.020 m` (40.6 g), serving
as upright and as lintel. The cross-section is square because a block spawns
at a random yaw and must grasp the same standing and lying. `L = 0.065`,
`w = 0.020`, level height `L + w = 0.085`.

## Why the T and the trilithon are both legal

```
  trilithon              T
   +-----+            +-----+
   |     |            |     |      lintel: one block lying flat
   #     #               #         upright(s): standing on end
  ---------          ---------
```

The trilithon costs a third block and stands. The T is one block cheaper and
tips. Neither is rewarded or forbidden by rule. The instability is emergent,
and it only bites at `2Pis` and `3Pis`, where a level has to survive
the next one being built on top of it. That trade is the reason the
multi-level tasks exist.

No dwell gate is needed to keep a falling T from scoring.
`SuccessGate.update` ANDs `task_done & at_rest & released` in the same step,
and a toppling block exceeds the 0.01 m/s rest threshold inside one 50 ms
control step, so it cannot latch a success on the way down.

## Why `(L, w, HEIGHT_TOL, TILT_COS_MAX)` are one coupled system

The worry is a pile of upright and flat blocks that puts *some* block
horizontal at a target height with no pi under it.

**One pile.** `a` uprights and `b` flats carry a lintel's centre to
`aL + bw + w/2`. Level `k` wants `kL + (k-1)w + w/2`. With `L = 13w/4` this
is `13a + 4b = 17k - 4`, so `a = k (mod 4)`:

- `a = k` gives `b = k - 1`, the intended build;
- `a = k + 4` gives `b = k - 14 < 0` for `k <= 3`, impossible;
- `a = k - 4 < 0` for `k <= 3`, impossible.

The reachable centres form a lattice of pitch `w/4 = 5 mm`, and
`HEIGHT_TOL = 3 mm` leaves 2 mm of margin. On a single pile the height test
alone is enough.

**Two piles are what fixes the gate.** A lintel spanning two piles of unequal
height rests at their mean, which halves the pitch to `w/8 = 2.5 mm`, inside
`HEIGHT_TOL`. Enumerated at this geometry:

| level | pile A | pile B | lintel centre | miss | tilt | blocks |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 flats (60 mm) | 1 upright (65 mm) | 72.5 mm | 2.5 mm | 6.34 deg | 5 |
| 2 | 1U+4F (145 mm) | 2U+1F (150 mm) | 157.5 mm | 2.5 mm | 6.34 deg | 9 |

Both are inside the block budget. No reachable `(L, w)` removes them: the
pair pitch grows with the absolute block size, and the ADR-019 reach dome
caps the block near this size, so the best reachable pitch is about 5 mm
against the 6 mm that `2 * HEIGHT_TOL` needs.

The variable that does exclude them is the horizontality gate, which bounds
the end-height difference of a lintel directly: a pair mismatch `dz` tilts it
by `atan(dz / (L - w))`.

| `TILT_COS_MAX` | angle | max end-height delta at `L - w` = 45 mm | lattice cheats at `tol` = 3 mm |
| --- | --- | --- | --- |
| 0.26 | 15.1 deg | 12.1 mm | 4 |
| **0.10** | **5.7 deg** | **4.5 mm** | **0** |
| 0.05 | 2.9 deg | 2.3 mm | 0 |

`TILT_COS_MAX` is `0.10`, not the `0.26` the other categories use. It leaves
4.5 mm of end-height slack for a real build and removes every lattice pair.

The four numbers are therefore one system. `HEIGHT_TOL` cannot be raised
after a settle measurement without tightening `TILT_COS_MAX` to match, and
the check is an import-time assertion in `variants.py`
(`assert_no_alternative_build` over the pure enumeration
`alternative_builds`), not prose and not a unit test, so the package refuses
to import with a pair that admits a cheat. `TILT_COS_MAX` is defined in one
place, and `loot-split-demos` imports the same kernel, because a drifted copy
would relabel a failed take a success offline.

The trade has a ceiling: `HEIGHT_TOL < w/4 = 5 mm`. At 5 mm a zero-tilt
build opens: one pile of 3 flats (top 60 mm) puts a lintel's centre at 70 mm
against level 1's 75 mm, exactly level, 4 blocks, in budget from `2Pis`
up. No gate value excludes a level lintel. At or above 5 mm the geometry has
to change, not the constants.

## Why the centre and not the AABB top

The old `block_top_heights` (`pos_z + half_h`) is gone, and the axis-aligned
bounding-box top was rejected as its replacement. A lintel at the gate's
5.7 deg limit reads `0.0325 * sin(5.7 deg) + 0.010 * cos(5.7 deg) = 0.0132`
against `0.010` when level, 3.2 mm high, the size of `HEIGHT_TOL` itself. The
centre of a lintel on two uneven uprights sits at their mean, so its height
is exact geometry to first order, and it keeps the rotation matrix out of the
height kernel. The quaternion is read only by the horizontality test.

## The limit of the criterion, stated plainly

Builds using only upright and flat blocks are enumerated and excluded at this
geometry. A support held at an intermediate lean presents a top corner
anywhere in a continuous band, `L cos φ + w sin φ`, peaking at 68.0 mm at
17.1 deg for a 65 mm block, and is **not** excluded. One upright at 65 mm plus
a leaner presenting 65.5 mm puts the lintel centre at 75.25 mm, 0.25 mm from
target, at 0.6 deg of tilt. The tilt gate does not catch this and must not be
claimed to.

It is accepted for three reasons:

- it needs a block held stable at a specific intermediate angle, which a
  teleoperator does not produce by accident, so the dataset-contamination
  risk (a failed take relabelled as a success by `loot-split-demos`) lives in
  the lattice family, which is excluded;
- a BC-bootstrapped sparse-reward policy has no gradient toward it;
- excluding it would need a universal "every load-bearing support is
  upright" test. An existence test is satisfied by the counterexample, which
  contains a genuine upright, and a universal test needs a definition of
  "load-bearing" that gives false negatives on legitimate builds whose upright
  drifted.

## The teleop protocol the geometry forces

Code cannot enforce any of this. It lives in the task README.

- **Lintels are gripped end-to-end.** The pads sit at `±w/2` from a held
  block's centre line, and an upright's side faces are also at `±w/2`, so an
  across-the-width grip scrapes both uprights at zero clearance for any
  `(L, w)`. Shrinking `w` moves the pads in with it. End-to-end, the pads
  hang outside the lintel's ends (open gap 78 mm > `L` = 65 mm) and the
  clearance is `L/2 - s/2 - w/2`.
- **Trilithon uprights about 36 mm apart.** Placing the second upright needs
  `s >= w/2 + 0.022 = 32 mm` (pad half-width 22 mm, `dk1.xml:266-268`,
  pad geom 70 x 3 x 44 mm); the uprights must stay under the lintel's ends,
  `s <= L - w = 45 mm`. At 36 mm both operations have about 4 mm of
  clearance.
- **Lintels are made by knock-over, then re-grasp.** `joint4`
  (`dk1.xml:173`, range `-1.95` to `1.5708`, home `-1.5708`) has 22 deg of
  pitch one way from home, so the wrist cannot roll a standing block flat in
  one motion. Tip it over with the closed gripper, then grasp the lying block
  end-to-end.

Blocks spawn standing (`write_upright_spawn`), so uprights are free and each
lintel costs one reorientation.

## The gamma-horizon departure

ADR-016 records the `Place` demos at median 5.9 s, p90 11.8 s, max 17.5 s
for one carry-and-place. A lintel adds a knock-over and a re-grasp, so the
budget is 25 s per block: 75 / 150 / 225 s, or 1500 / 3000 / 4500 steps.

At 4500 steps QC's 100-step horizon covers 2.2 % of an episode, against the
33-50 % band ADR-016 validates. This is a known departure, accepted because
the recorder truncates at the episode length (ADR-016, "The episode budget"),
so an under-budgeted episode silently yields truncated demos rather than
failed ones, and because ADR-014 has closed the discount experiment.

## Reach

| | height (m) |
| --- | --- |
| level-3 upright tops (`3L + 2w`) | 0.235 |
| lintel-3 centre | 0.245 |
| grasp site, pad centred on the lintel | 0.239 |
| grasp site, pad bottom flush with the lintel bottom | 0.264 |

The grasp site is a band and not a point because the pad spans site-frame
`z` in `[-0.029, +0.041]`. The whole band is under ADR-019's 0.28 m ceiling
at `x = 0.25-0.30` and its ~0.272 m at the build zone's far edge `x = 0.32`.
Carry height equals placement height, so there is no extra carry penalty.

## Consequences

- `3N` blocks per task is a choice, not a minimum. A single pile
  `U, F, U, F, U, F` completes all three levels in six blocks (upright tops
  65 / 150 / 235 mm). `3N` is the trilithon budget, it lands the colour tiers
  (navy, teal, white) on level boundaries, and it keeps `1Pi` a prefix of
  `2Pis` a prefix of `3Pis` for the demo splitter.
- Supersedes Stack's height-only criterion and all nine
  `LOOT-Stack{3,4,5}-{Easy,Medium,Hard}-Dk1` ids. No demonstrations existed
  for them, so nothing recorded is lost.
- ADR-017 and ADR-019 are amended in place; their conclusions hold, their
  cited numbers change.
