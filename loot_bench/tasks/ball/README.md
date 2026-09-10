# Ball task category

Bring a ball to its goal by hitting or throwing it. Two task families, each
with its own set of layouts:

- **minigolf**: a raised platform with a pit (a physical hole) at the
  target, out of the arm's direct reach — the ball must roll there after a
  strike. Every layout has a perimeter railing around the platform edges
  (like real minigolf), and the layouts are strictly additive — each one
  contains everything the previous layout has, plus one more element. The
  pit sits off-centre (local y +0.12) for every layout except `Direct`, so
  the target position stays consistent across `UsingWall`/`Obstacle`/`Ramp`.
  - `Direct`: a clear line from spawn to the pit; the railing alone.
  - `UsingWall`: adds a blocker wall across the direct line; the perimeter
    railing itself serves as the bank for a redirect shot.
  - `Obstacle`: `UsingWall` plus a box in the lane the blocker leaves open.
  - `Ramp`: `Obstacle` plus the pit raised onto an elevated platform
    section, reached via an inclined connector positioned past the blocker
    and obstacle.
- **throw**: a flat field with a wall between the spawn area and the target
  (a marked disk or a U-shaped pocket), so the ball must clear the wall.
  - `U`: a U-shaped wall pocket open toward the incoming ball.
  - `MarkedArea`: a marked disk on the ground.
  - `WallVariation`: like `MarkedArea`, but the wall is a separate entity
    that moves (distance from the robot, and sink depth i.e. effective
    height) every reset instead of sitting at a fixed pose.

The target/goal itself also moves a little every reset (the field's pose is
randomized), so the command tracks the target relative to the field's live
pose rather than a constant offset.

## Success

The ball is *at target* when it is within the target radius and on the
surface (for minigolf, when it has dropped below the pit's local surface).
Success also requires the ball to be at rest and released from the gripper —
the termination, the `task` reward, and the `episode_success` metric all
use this same condition. The episode also terminates if the ball leaves the
workspace, falls off the platform (minigolf — the perimeter railing normally
prevents this; the termination is a safety net for balls that bounce over
it), or hits the wall instead of clearing it (throw).

## Observations (state-based)

- joint positions/velocities, EE pose
- ball position (env-local; no orientation — a sphere's quaternion carries no
  task-relevant signal)
- field pose (env-local position + quaternion), and wall pose (only for
  variants with an independently-moving wall entity, e.g. `WallVariation`)
- ball linear velocity
- the command: raw env-local target position (the field's pose is
  per-episode randomized, so this is the only source of goal observability)

## Rewards

One term, in `[-1, 0]`, exactly `0` only at success
(`docs/adr/006-one-task-reward-in-minus-one-to-zero.md`). Ball is the sparse
member of the convention: it has no progress signal, so the term is `-1`
every step until the ball scores (at the target, at rest, and released).
There is no shaping, no penalty and no regularization.

## Variants

| Task id (`-Dk1`)                       | Description                                       |
| --------------------------------------------- | -------------------------------------------------- |
| `LOOT-Ball-Minigolf-Direct`                    | hit the ball straight into the pit                |
| `LOOT-Ball-Minigolf-UsingWall`                  | bank the ball off a wall around a blocker          |
| `LOOT-Ball-Minigolf-Obstacle`                  | hit the ball past an obstacle into the pit         |
| `LOOT-Ball-Minigolf-Ramp`                       | hit the ball up a ramp into the pit                |
| `LOOT-Ball-Throw-U`                             | throw the ball over a wall into a U pocket         |
| `LOOT-Ball-Throw-MarkedArea`                    | throw the ball over a wall onto a marked disk      |
| `LOOT-Ball-Throw-WallVariation`                 | as MarkedArea, but the wall moves every reset      |

Play e.g. with:

```bash
uv run loot-play LOOT-Ball-Minigolf-Direct-Dk1
```
