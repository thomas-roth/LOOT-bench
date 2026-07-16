# Ball task category

Bring a ball to its goal by hitting or throwing it. Two task families, each
with its own set of layouts:

- **minigolf**: a raised platform with a pit (a physical hole) at the
  target, out of the arm's direct reach — the ball must roll there after a
  strike.
  - `direct`: a clear line from spawn to the pit.
  - `usingWall`: a blocker wall sits across the direct line; a bank wall
    along the platform's near edge allows a redirect shot.
  - `obstacle`: a box sits between the spawn area and the pit.
  - `ramp`: the pit sits on a raised platform section, reached via an
    inclined connector.
- **throw**: a flat field with a wall between the spawn area and the target
  (a marked disk or a U-shaped pocket), so the ball must clear the wall.
  - `U`: a U-shaped wall pocket open toward the incoming ball.
  - `markedArea`: a marked disk on the ground.
  - `wallVariation`: like `markedArea`, but the wall is a separate entity
    that moves (distance from the robot, and sink depth i.e. effective
    height) every reset instead of sitting at a fixed pose.

The target/goal itself also moves a little every reset (the field's pose is
randomized), so the command tracks the target relative to the field's live
pose rather than a constant offset.

## Success

The ball is *at target* when it is within the target radius and on the
surface (for minigolf, when it has dropped below the pit's local surface).
Success also requires the ball to be at rest and released from the gripper —
the termination, the `success` reward, and the `episode_success` metric all
use this same condition. The episode also terminates if the ball leaves the
workspace, falls off the platform (minigolf), or hits the wall instead of
clearing it (throw).

## Observations (state-based)

- joint positions/velocities, last action
- ball position relative to the grasp site
- target position relative to the ball
- ball linear velocity

## Rewards

Sparse by design: the only task reward term is `success` (the ball is at the
target, at rest, and released) — no shaping, penalties, or regularization.

## Variants

| Task id (`-Dk1_uni`)                | Description                                       |
| ------------------------------------ | -------------------------------------------------- |
| `G3a-Ball-MinigolfDirect`            | hit the ball straight into the pit                |
| `G3a-Ball-MinigolfUsingWall`         | bank the ball off a wall around a blocker          |
| `G3a-Ball-MinigolfObstacle`          | hit the ball past an obstacle into the pit         |
| `G3a-Ball-MinigolfRamp`              | hit the ball up a ramp into the pit                |
| `G3a-Ball-ThrowU`                    | throw the ball over a wall into a U pocket         |
| `G3a-Ball-ThrowMarkedArea`           | throw the ball over a wall onto a marked disk      |
| `G3a-Ball-ThrowWallVariation`        | as MarkedArea, but the wall moves every reset      |

Train e.g. with:

```bash
uv run g3a-train G3a-Ball-MinigolfDirect-Dk1_uni --env.scene.num-envs 2048
```
