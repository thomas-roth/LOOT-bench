# live_bridge

Record DK1 demonstrations into a LeRobot dataset for offline RL. A GELLO leader
(and optional DK1 follower) drives the sim; episodes are saved with success/failure
labels.

## Setup

Find USB ports with `uv run lerobot-find-port`. On Linux, grant access with
`sudo chmod 666 /dev/ttyACM*`. The MuJoCo viewer needs a display.

## live_record.py

Records episodes at ~50 Hz from live teleop. Requires hardware.

```bash
uv run python g3a_bench/scripts/live_bridge/live_record.py \
    [--follower] [--leader-port PORT] [--follower-port PORT] \
    [--task TASK] [--num-episodes N] [--dataset-dir PATH]
```

Ports are auto-detected if omitted. Asks for operator ID (1–4) at startup. **Letter-key controls** (an on-screen HUD shows state, saved counts and the currently valid keys):
- `V`: **Save** episode as **success** → reset
- `F`: **Save** episode as **failure** → reset
- `D`: **Discard** → reset
- `U`: **Undo** the last saved episode (single-level)
- `Q`: Finalize dataset and quit
- `C`: Toggle wrist camera ↔ orbit view

On task success an episode auto-saves; any other termination (out of bounds,
collision, timeout) pauses the sim (orange ghost) and auto-saves as failure
after 2s unless `F` (save now), `D` (discard) or `V` (override to success) is
pressed first. After each save the arm ghosts back to rest, then recording
resumes.

Episodes go to `out/demos/success/<timestamp>/` and `out/demos/failure/<timestamp>/`
at the repo root (override with `--dataset-dir`). Nothing is ever overwritten, and a
directory is created only once an episode of that kind exists.

Set `USE_FOLLOWER_STATE_FOR_SIM = True` in the script to drive the sim from follower
joint states instead of leader actions.

## split_demos.py

Turn one recording of the *hardest* task into demos for the easier tasks in the
same family, so you only teleoperate the hard task. **Stack** and **Domino**
(base, RandomRow + DifferentDominoes; not KnockOver) variants tail-extend (an
easier task's objects are the first `k` of the harder one) and resets are
seed-prefix safe, so an easier task's rollout is embedded in the hard recording. Each episode is
relabeled **offline** (no re-sim): trailing objects are truncated from every
object-major slice of `observation.state` and from `qpos`/`qvel`, the target's
`progress`/`success` reward is recomputed from the stored state
(`reward = step_dt * (progress + gate)`, reusing the env's own success kernels),
and the episode is cut at the first frame the target success holds. A per-episode
self-check asserts the recomputed *source* reward matches the recording, so a
wrong layout/threshold can't pass silently (it also validates each category the
first time real demos of it are split). No hardware, no GPU.

```bash
uv run python g3a_bench/scripts/live_bridge/split_demos.py \
    --source-dir dataset/demos/G3A-Stack-High-Dk1_uni/<stamp> \
    --source-task G3A-Stack-High-Dk1_uni \
    --output-dir dataset/demos_split            # all easier same-family tasks
# or a subset: --target-task G3A-Stack-Low-Dk1_uni [--target-task ...]
```

Output is a LeRobot dataset per target task (same format as `live_record.py`,
so `lerobot_to_qc.py` and `verify_replay.py` work on it), with each source
episode's seed carried over unchanged (seed-prefix reset reproduces the kept
objects in the target env). Splits only within a family, down the object-count
axis (Stack Low/Medium/High, Domino 2/3/4 Pieces — the fixed-row base family and
`RandomRow` split separately, never into each other), and only when demos are
collected **Ordered** (use the low-index objects first) and **settled at each
intermediate stage**: an episode with no settled+released frame at the target
stage is skipped, and movement of a dropped (non-prefix) object is flagged (see
`docs/TaskCategories.md` "General"). Ball is not splittable (its layouts are
additive scene geometry, not an object-count reduction).

## record_replay.py

Replays a LeRobot dataset through the sim and records sim observations plus
reward/done. No hardware.

```bash
uv run python g3a_bench/scripts/live_bridge/record_replay.py
```

First run downloads an example dataset from HuggingFace (~655 MB, cached). Output
goes to `logs/sim_replay_cube_stack3/success/`.

## verify_replay.py

Numerically verifies that a recorded episode replays *exactly* — no viewer, no
eyeballing. Re-simulates the episode headlessly (seeded reset + recorded
actions, stepping exactly like live_record) and compares every frame's full sim
qpos against the ground truth stored in `observation.qpos`, printing deviation
stats, the first diverging frame, the worst DOFs, and a PASS/FAIL verdict
(exit code 0/1). No hardware.

```bash
uv run python g3a_bench/scripts/live_bridge/verify_replay.py \
    --dataset-dir out/demos/success/<timestamp> [--episode-index N] \
    [--tol 1e-5] [--sync-start] [--csv drift.csv]
```

Frames are bitwise-compared: any nonzero deviation is real. Expect a bitwise
match until the gripper first contacts an object; contact impulses on the GPU
are where replay determinism is decided (see the script docstring for the
comparison semantics and known caveats — done-frame qpos, unrecorded qvel,
initial live arm pose). `--sim-replay-dir` instead diffs an existing
record_replay.py output dataset offline against the original recording.
