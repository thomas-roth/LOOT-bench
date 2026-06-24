# live_bridge

Teleop → sim mirroring → dataset recording pipeline for the G3A benchmark.

Connects a real TRLC-DK1 robot (GELLO leader + DK1 follower) to a MuJoCo
simulation, mirrors every movement in real time, and records multiple labeled
demonstrations per session into a LeRobot dataset for offline behaviour-cloning
training.

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

A display is required — `live_record.py` opens a native MuJoCo viewer window.

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
sim at ~50 Hz, and records multiple labeled demonstrations in one session.

```bash
# ports auto-detected (reliable on Linux)
uv run python g3a_bench/scripts/live_bridge/live_record.py

# explicit ports (recommended on Windows, or if auto-detect warns)
uv run python g3a_bench/scripts/live_bridge/live_record.py \
    --follower-port /dev/ttyACM0 \
    --leader-port /dev/ttyUSB0

# stop automatically after N saved episodes
uv run python g3a_bench/scripts/live_bridge/live_record.py --num-episodes 10

# write to a specific directory (must not exist)
uv run python g3a_bench/scripts/live_bridge/live_record.py \
    --dataset-dir path/to/my_dataset
```

**Startup:** the script asks for an operator number (1–4) before doing anything
else. The number is stored in `meta/info.json` of the recorded dataset.

**What happens:**
1. Asks `Operator (1 / 2 / 3 / 4):` on stdin
2. Loads the task in play mode (no noise, infinite episode length)
3. Adds a ghost arm to the scene (visual-only DK1 fixed at home pose, initially invisible)
4. Starts a background thread reading the leader at ~200 Hz and commanding the follower
5. Opens a MuJoCo viewer; move the leader arm to drive the sim and the follower
6. Records sim obs + real robot state + leader action + reward + done, **buffered in memory** per episode
7. Handles episode end automatically (see state machine below)

**Key bindings:**

| Key | Action |
|-----|--------|
| `S` | Save current episode as **success** → enter reset phase |
| `F` | Save current episode as **failure** → enter reset phase (only valid during collision prompt) |
| `D` | **Discard** current episode → enter reset phase → redo |
| `Q` | Stop session and finalize dataset |

---

### State machine

Each episode moves through three states:

```
RECORDING ──── task complete (all cubes placed) ──────► auto-save SUCCESS
          └─── ground collision (ee_ground_collision) ──► AWAITING_LABEL
          └─── S key ───────────────────────────────────► save SUCCESS

AWAITING_LABEL (sim frozen, collision prompt on screen)
          ├─── F key ──► save FAILURE ──► RESET_PHASE
          ├─── D key ──► discard       ──► RESET_PHASE
          └─── S key ──► save SUCCESS  ──► RESET_PHASE

RESET_PHASE (sim live, not recorded, ghost arm visible at home)
          └─── sim arm ≈ ghost (all 6 joints within 0.2 rad of home) ──► RECORDING
```

**RECORDING:** Normal operation. Each sim step gets recorded. When `done=True`
fires, the script checks whether all cubes are placed — if yes, it auto-saves;
if a ground collision is detected, it shows the collision prompt.

**AWAITING_LABEL:** Sim is frozen. The operator decides what to do with the
episode (save as failure or discard). S can also override and save as success.

**RESET_PHASE:** After any episode ends, the ghost arm (semi-transparent grey
DK1) becomes visible in the sim at the home pose. The operator moves the leader
arm back toward home — the live sim arm follows. When the sim arm is within
0.2 rad of home on all 6 joints, the next episode begins automatically and the
ghost disappears. Nothing is recorded during this phase.

---

**Output:** `g3a_bench/scripts/live_bridge/logs/live_record/<YYYY-MM-DD_HH-MM-SS>/`

Each session creates a new timestamped directory. Data is **never overwritten**.
If `--dataset-dir` is given and already exists, the script errors rather than
wiping it.

**Port auto-detection:** on Linux, `/dev/ttyUSB*` = leader, `/dev/ttyACM*` =
follower — unambiguous. On Windows, ports are `COMx` and detection uses the USB
vendor id (FTDI = leader). If detection prints `[WARN]`, pass ports explicitly.

**`USE_FOLLOWER_STATE_FOR_SIM` toggle:** at the top of `live_record.py`, set
`USE_FOLLOWER_STATE_FOR_SIM = True` to drive the sim from the follower's joint
observations instead of the leader's action command. Default is `False` (leader
action drives the sim).

---

## Module overview

| File | Role |
|------|------|
| `reader.py` | `LiveReader` (real robot, 200 Hz), `DatasetReader` (offline replay). Both return `ReaderFrame(joint_pos, leader_action, timestamp)`. |
| `mapper.py` | Converts real-robot joint/gripper values to MuJoCo raw action format. `lerobot_to_mjlab_raw_action(leader_action)` is the main entry point. |
| `recorder.py` | Wraps `LeRobotDataset`. `add_frame()` buffers frames in memory; `save_episode(success)` flushes with a label; `discard_episode()` drops the buffer; `close(operator_id)` writes operator id to `meta/info.json` and finalizes. |
| `live_record.py` | Main live teleop script. HW thread (~200 Hz) + sim viewer thread (~50 Hz), decoupled via single-element lists. State machine handles multi-episode recording with ghost-arm reset phase. |
| `record_replay.py` | Offline replay + recording. No hardware. |

---

## Data layout

Each recorded frame contains:

| Column | Shape | Contents |
|--------|-------|----------|
| `observation.state` | (53,) | Sim actor obs: joint_pos_rel(6) + joint_vel_rel(6) + gripper(2) + pieces_rel_grasp(9) + slot_goal_deltas(9) + slot_placed(3) + command(9) + last_action(7) (exact size depends on task config) |
| `observation.robot_state` | (7,) | Real robot: follower joint pos in radians, gripper in [0=open, 1=closed] |
| `action` | (7,) | Leader command: same format as robot_state |
| `next.reward` | (1,) | Reward returned by env.step() for this frame |
| `next.done` | (1,) | Whether the episode terminated after this step |
| `success` | (1,) | Operator label: `True` = success, `False` = failure (constant within an episode) |
| `task` | str | Task id, e.g. `G3a-Cube-Stack3-Dk1_uni` |

Dataset `meta/info.json` also includes `operator_id` (int 1–4) for the session.

---

## Architecture notes

Two threads run in parallel:

- **HW thread (~200 Hz, daemon):** reads leader → sends to follower → writes
  latest action to `action_list[0]` (last-writer-wins, GIL-safe single-element
  list). If this thread crashes (serial disconnect), it sets `hw_failed[0]=True`
  and the viewer aborts cleanly.

- **Main thread (viewer-paced ~50 Hz):** `_LivePolicy` reads `action_list[0]` →
  maps to MuJoCo raw action → `env.step()` → records frame. The state machine
  lives here.

Termination detection: both `ee_ground_collision` and `success` terminations are
**kept active** (not popped in play mode). When `done=True` fires, the script
queries `env.env.command_manager.get_term("arrangement").slot_at_goal` to
distinguish success (all cubes placed) from collision.

---

## Tests

Unit and integration tests live in `tests/live_bridge/`. Run with:

```bash
uv run pytest tests/live_bridge/ -v
```

Expected: 41 tests pass, 2 skipped (hardware/network). See
`tests/live_bridge/README.md` for details on what each test covers.
