# Ball task category

Move a ball into a target area. Each variant combines an **action** (how the
ball is moved) with a **target** (where it must end up).

- **actions**
  - `golf`: a single closed-gripper strike — impart velocity and let go, do
    not lead the ball. The target sits out of reach so the ball must roll
    there after the hit.
  - `push`: lead the ball with a closed gripper. The target is within reach so
    the robot can guide the ball all the way.
  - `throw`: fling the ball over a wall into a target behind it.
- **targets**
  - `marked`: a marked disk on the ground (the ball must stop on it).
  - `u_boundary`: a U-shaped wall pocket open toward the incoming ball.
  - `hole`: a real pit — the play area is a raised platform with a square
    opening, and the ball drops through it to the floor below.

## Hit vs. push (soft)

The golf/push distinction is **not** in the reward (the task is sparse — see
Rewards): both share the same `success` signal. What differs is the field
layout — `golf` places the target out of reach so the ball must roll there
after a strike, while `push` keeps it within reach so the robot can guide the
ball all the way — together with the demonstrations that bootstrap each
behaviour. Success in both cases only requires the ball to come to rest in the
target.

## Success

The ball is *at target* when it is within the target radius and on the surface
(for `hole`, when it has dropped below the platform surface). An episode
terminates successfully when the ball stays at target and at rest for
`hold_time` (0.3 s); it also terminates if the ball leaves the workspace.

## Observations (state-based)

- joint positions/velocities, last action
- ball position relative to the grasp site
- target position relative to the ball
- ball linear velocity
- the command: target position relative to the ball

## Rewards

Sparse by design, to pair with a demonstration pipeline (GELLO teleop →
offline-to-online RL) that supplies the exploration dense shaping would
otherwise provide:

- task signal: `success` (the ball is at the target) — the only positive term
- regularization: the standard action-rate / joint-limit / joint-velocity-hinge
  penalties

## Variants

| Task id (`-Dk1_uni`)      | Description                                       |
| ------------------------- | ------------------------------------------------- |
| `G3a-Ball-GolfMarked`     | hit the ball onto a marked disk                   |
| `G3a-Ball-GolfU`          | hit the ball into a U pocket                      |
| `G3a-Ball-GolfHole`       | hit the ball into a pit                           |
| `G3a-Ball-PushMarked`     | lead the ball onto a marked disk                  |
| `G3a-Ball-PushU`          | lead the ball into a U pocket                     |
| `G3a-Ball-PushHole`       | lead the ball into a pit                          |
| `G3a-Ball-ThrowMarked`    | throw the ball over a wall onto a marked disk     |
| `G3a-Ball-ThrowU`         | throw the ball over a wall into a U pocket        |

Train e.g. with:

```bash
uv run g3a-train G3a-Ball-PushU-Dk1_uni --env.scene.num-envs 2048
```
