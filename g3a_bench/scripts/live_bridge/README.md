# live_bridge

Teleop → sim mirroring → dataset recording pipeline for the G3A benchmark.

Connects a real TRLC-DK1 robot (GELLO leader + DK1 follower) to a MuJoCo
simulation, mirrors every movement in real time, and records both sim
observations and real robot state into a LeRobot dataset for offline
behaviour-cloning training.

---

## Prerequisites

### Always required

```bash
# from repo root — install the package and all dependencies
uv sync --extra cu128   # GPU rig (recommended for live recording)
uv sync --extra cpu     # CPU-only (fine for replay/testing)
```

### Only for `live_record.py` (real hardware)

```bash
# install the trlc-dk1 hardware drivers
uv pip install -e /path/to/trlc-dk1
```

Physical connections:
- **Leader** (GELLO controller): plugged in via U2D2 USB-serial adapter (FTDI)
- **Follower** (DK1 arm): plugged in via CAN/CDC-ACM USB adapter

---

## Scripts

### `record_replay.py` — replay a dataset through the sim and record results

No hardware needed. Reads an existing LeRobot dataset, feeds each frame's
action through the sim, and saves a new dataset with sim observations +
reward/done.

```bash
uv run --extra cpu python g3a_bench/scripts/live_bridge/record_replay.py
```

**What happens:**
1. Downloads `FabianKerj/corkinbox100-fullrel` from HuggingFace (~655 MB, cached after first run)
2. Opens a MuJoCo viewer window showing the replay
3. Records sim obs + reward + done to `g3a_bench/scripts/live_bridge/logs/sim_replay_cube_stack3/`
4. When you close the window, finalizes and saves the dataset

**On Windows (first run):** HuggingFace may print a symlink warning — harmless, download completes.

---

### `live_record.py` — live teleop + sim mirroring + recording

Requires real hardware. Reads the GELLO leader at ~200 Hz, mirrors it in the
sim at ~50 Hz, and records everything.

```bash
# ports auto-detected (reliable on Linux)
uv run python g3a_bench/scripts/live_bridge/live_record.py

# explicit ports (recommended on Windows, or if auto-detect warns)
uv run python g3a_bench/scripts/live_bridge/live_record.py \
    --follower-port /dev/ttyACM0 \
    --leader-port /dev/ttyUSB0

# different task
uv run python g3a_bench/scripts/live_bridge/live_record.py \
    --task G3a-Cube-Stack3-Dk1_uni
```

**What happens:**
1. Loads the task in play mode (no noise, infinite episode)
2. Starts a background thread reading the leader at ~200 Hz and commanding the follower
3. Opens a MuJoCo viewer; move the leader arm to drive the sim and the follower
4. Records sim obs (`observation.state`, 53-dim) + real robot state (`observation.robot_state`, 7-dim) + leader action + reward + done
5. On close: finalizes the dataset, prints frame/episode count

**Output:** `g3a_bench/scripts/live_bridge/logs/live_record/`

**Warning:** the dataset directory is **wiped on each run**. Move or copy it before starting a new session if you want to keep the previous recording.

**Port auto-detection:** on Linux, `/dev/ttyUSB*` = leader, `/dev/ttyACM*` = follower — unambiguous. On Windows, ports are `COMx` and detection uses the USB vendor id (FTDI = leader). If detection prints `[WARN]`, pass ports explicitly.

---

## Module overview

| File | Role |
|------|------|
| `reader.py` | `LiveReader` (real robot, 200 Hz), `DatasetReader` (offline replay). Both return `ReaderFrame(joint_pos, leader_action, timestamp)`. |
| `mapper.py` | Converts real-robot joint/gripper values to MuJoCo raw action format. `lerobot_to_mjlab_raw_action(leader_action)` is the main entry point. |
| `recorder.py` | Wraps `LeRobotDataset`. `add_frame()` buffers; `end_episode()` flushes to disk; `close()` finalizes metadata. |
| `live_record.py` | Main live teleop script. HW thread (~200 Hz) + sim viewer thread (50 Hz), decoupled via single-element lists. |
| `record_replay.py` | Offline replay + recording. No hardware. |

---

## Data layout

Each recorded frame contains:

| Column | Shape | Contents |
|--------|-------|----------|
| `observation.state` | (53,) | Sim actor obs: joint_pos(8) + joint_vel(8) + pieces_rel_grasp(9) + slot_goal_deltas(9) + slot_placed(3) + command(9) + last_action(7) |
| `observation.robot_state` | (7,) | Real robot: follower joint pos in radians, gripper in [0=open, 1=closed] |
| `action` | (7,) | Leader command: same format as robot_state |
| `next.reward` | (1,) | Reward returned by env.step() for this frame |
| `next.done` | (1,) | Whether the episode terminated after this step |
| `task` | str | Task id, e.g. `G3a-Cube-Stack3-Dk1_uni` |

---

## Tests

Unit and integration tests live in `tests/live_bridge/`. Run with:

```bash
uv run pytest tests/live_bridge/ -v
```

See `tests/live_bridge/README.md` for details on what each test covers.
