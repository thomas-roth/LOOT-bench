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

Episodes go to `out/demos/success/<task>/<timestamp>/` and
`out/demos/failure/<task>/<timestamp>/` at the repo root (override with
`--dataset-dir`). Nothing is ever overwritten, and a directory is created only once
an episode of that kind exists.

`out/` is scratch and is not committed. A session that is worth keeping is moved
into `dataset/demos/<task>/<stamp>/`, which *is* committed, with the failure half
of the same session as the sibling `dataset/demos/<task>/<stamp>_failure/`. Read
demos from there; write new ones to `out/demos/`.

Set `USE_FOLLOWER_STATE_FOR_SIM = True` in the script to drive the sim from follower
joint states instead of leader actions.

A live sanity-check panel (`debug_panel.py`) is always on: a per-frame line —
frame/step/fps/speed/rtf, EEF/goal/object position, reward, done — replaces
mjlab's own status print, redrawing in place by default so it stays at one
spot in the terminal instead of scrolling. When an episode ends, an
aggregated min/mean/max breakdown of everything saved to the dataset
(`observation.state`, action, robot_state, qpos, qvel) prints once, along
with any non-finite-value anomalies seen that episode. Pass `--debug-obs` to
switch the per-frame line to append-only instead — nothing is ever
overwritten, so the full per-frame history for every demonstration stays in
the scrollback or a piped log file (`--debug-obs > demo_debug.log`).

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
    --dataset-dir dataset/demos/<task>/<stamp> [--episode-index N] \
    [--tol 1e-5] [--sync-start] [--csv drift.csv]
```

Frames are bitwise-compared: any nonzero deviation is real. Expect a bitwise
match until the gripper first contacts an object; contact impulses on the GPU
are where replay determinism is decided (see the script docstring for the
comparison semantics and known caveats — done-frame qpos, unrecorded qvel,
initial live arm pose). `--sim-replay-dir` instead diffs an existing
record_replay.py output dataset offline against the original recording.
