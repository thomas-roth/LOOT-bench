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

Ports are auto-detected if omitted. Enter an operator ID (1–4) at startup, then
control recording with the numpad:

- `1` — save as success, reset
- `2` — save as failure, reset
- `3` — discard, reset
- `4` — finalize and quit

On task success an episode auto-saves; on a ground collision the sim freezes for
you to label. After each save the arm ghosts back to rest, then recording resumes.

Episodes go to `out/demos/success/<timestamp>/` and `out/demos/failure/<timestamp>/`
at the repo root (override with `--dataset-dir`). Nothing is ever overwritten, and a
directory is created only once an episode of that kind exists.

Set `USE_FOLLOWER_STATE_FOR_SIM = True` in the script to drive the sim from follower
joint states instead of leader actions.

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
