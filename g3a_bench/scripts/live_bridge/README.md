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
