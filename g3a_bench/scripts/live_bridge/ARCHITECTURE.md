# Live Bridge Architecture — Complete Review

**Date**: 2026-06-21
**Audience**: Claude Pro (AI assistant context for future development)
**Repo root**: `/home/minhtrung/Documents/UNI/G3a/g3a-bench/`
**Package**: `g3a_bench.scripts.live_bridge`

## 1. What Is This?

The **Live Bridge** is a modular pipeline that connects a real TRLC-DK1 robot arm to an MJLab physics simulation. It serves three simultaneous functions:

1. **Teleoperation** — reads the human-operated leader arm → sends commands to the real follower arm (the arm moves)
2. **Simulation mirroring** — sends the same commands to an MJLab simulation (a virtual arm mirrors the real one)
3. **Recording** — saves both sim observations (53-dim) and real robot observations (7-dim) to a LeRobot-format dataset

The purpose is to collect **demonstration data for difficult environments where only a simulation exists** — the human teleoperates the real robot in a simple task, and the sim mirrors the movement in a visually or physically different environment. This enables future behavior cloning (BC) training on sim observations while also keeping real robot data as a backup.

---

## 2. Overall Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        live_record.py                            │
│                                                                  │
│  ┌─────────────────────────┐     ┌────────────────────────────┐ │
│  │  Hardware Thread (daemon) │     │  Main Thread (viewer)     │ │
│  │  ~200 Hz real-time       │     │  Variable rate (sim-paced) │ │
│  │                          │     │                            │ │
│  │  LiveReader              │     │  _LivePolicy               │ │
│  │    .read() ───→ frame    │     │    reads action_list[0]    │ │
│  │    .send_action()        │     │    → mapper                │ │
│  │                          │     │    → env.step()            │ │
│  │  writes to:              │     │    → recorder.add_frame()  │ │
│  │    action_list[0]  ◀─────┼─────┼── (last-writer-wins)       │ │
│  │    robot_obs_list[0]     │     │    → _LiveRecordingViewer   │ │
│  └─────────────────────────┘     └────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘

External repos referenced:
  trlc-dk1/    → DK1Follower, DK1Leader (hardware I/O)
  MJLab/       → ManagerBasedRlEnv, NativeMujocoViewer (sim + viewer)
  lerobot/     → LeRobotDataset (dataset format for recording)
  g3a_bench/   → Task configs (Cube-Stack3-Dk1_uni), MJLab constants
```

### Threading model

- **Hardware thread** (daemon, started first): Runs a tight loop at ~200 Hz. Reads leader position → sends to follower → writes latest (7,) action and (7,) robot observation to shared single-element lists. If it falls behind 200 Hz, it skips sleep but never accumulates lag.
- **Main thread**: Creates the env, viewer, and recorder. The viewer's `run()` method blocks on the main thread, calling `_execute_step()` at the sim's pace. Within each step, `_LivePolicy.__call__()` reads the latest action from the shared list.

**Why daemon thread?** The hardware loop must stay responsive (safety-critical for real robot control). The viewer's `run()` handles Ctrl+C natively; when it exits, the `finally` block sets the shutdown flag, and the daemon thread is joined (or auto-exits on process exit).

### Thread-safe shared state

Three single-element Python lists serve as lock-free communication:

```python
action_list: list = [None]     # latest (7,) np.ndarray from leader
robot_obs_list: list = [None]  # latest (7,) np.ndarray from follower
shutdown_flag: list = [False]  # set True to stop the HW loop
```

Python's GIL guarantees atomic reads/writes of list elements, so no `threading.Lock` is needed. The HW thread **overwrites** `action_list[0]` every cycle; the sim thread **reads** it when it's ready to step. If the sim is slower than the HW (which it always is on CPU: ~0.2-0.4x RTF), intermediate actions are silently discarded — the sim always gets the **latest** human command.

---

## 3. End-to-End Data Flow (One Control Cycle)

Here's the exact sequence for one sim step during live operation:

```
Step 1: HW thread calls LiveReader.read()
        └── DK1Leader.get_action()       → {"joint_1.pos": 0.1, ..., "gripper.pos": 0.5}
        └── DK1Follower.get_observation() → {"joint_1.pos": 0.095, ..., "gripper.pos": 0.48}
        └── Returns ReaderFrame(leader_action=np(7,), joint_pos=np(7,))

Step 2: HW thread calls LiveReader.send_action(frame.leader_action)
        └── converts np(7,) → dict → DK1Follower.send_action(dict)
        └── Real arm starts moving toward commanded position

Step 3: HW thread writes to shared lists
        └── action_list[0] = frame.leader_action.copy()
        └── robot_obs_list[0] = frame.joint_pos

── (HW thread continues to next 200Hz cycle, overwriting action_list) ──

Step 4: Main thread (viewer) calls _execute_step()
        └── env.get_observations()  → obs dict with "actor" tensor (1, 53)

Step 5: _LivePolicy.__call__(obs)
        └── reads action_list[0] → gets latest leader action
        └── lerobot_to_mjlab_raw_action(action) → raw_action (7,)
        └── returns torch tensor (1, 7)

Step 6: env.step(raw_action)
        └── ManagerBasedRlEnv applies action to MJLab simulation
        └── Returns (obs_next, reward, done, extras)

Step 7: _LiveRecordingViewer records the frame
        └── Recorder.add_frame(
                obs_buf={"actor": obs["actor"]},    # 53-dim sim state
                action=leader_action,               # 7-dim commanded action
                reward=rew, done=done,
                robot_obs=robot_obs_list[0]         # 7-dim real joint positions
            )
        └── LeRobotDataset.add_frame() stores in episode buffer

Step 8: Viewer renders the updated sim state
        └── sync_env_to_viewer() copies MuJoCo state → window
        └── OpenGL render at target FPS
```

---

## 4. File-by-File Deep Dive

### 4.1 `reader.py` (229 lines)

**Role**: Data source abstraction. Provides a uniform `ReaderFrame` regardless of whether data comes from hardware or a saved file.

**Key classes and functions**:

| Symbol | Type | Purpose |
|--------|------|---------|
| `ReaderFrame` | dataclass | Uniform envelope: `joint_pos(7,)`, `leader_action(7,)`, `cameras{}`, `timestamp` |
| `Reader` | ABC | `read() → ReaderFrame`, `close()`. Two subclasses. |
| `DatasetReader` | class | Reads from `LeRobotDataset`. Finds episode boundaries by scanning `episode_index` column. Used for offline testing/replay. |
| `LiveReader` | class | Connects to real DK1Follower + DK1Leader via serial. Guards imports with try/except. |
| `_find_ports()` | function | Auto-detects `ttyUSB*` (leader/GELLO) and `ttyACM*` (follower/DM motors). |
| `LiveReader.send_action()` | method | Converts (7,) numpy → dict → `DK1Follower.send_action()`. Added specifically for live_record.py. |

**Hardware import guard** (lines 13-26):
```python
try:
    from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig
    from lerobot_robot_trlc_dk1.leader import DK1Leader, DK1LeaderConfig
except ImportError:
    DK1Follower: Any = None
```

This allows importing `reader.py` on machines without the trlc-dk1 package (e.g., CI servers, other dev machines). The hardware-specific classes become `None`, and `LiveReader` raises a clear `ImportError` at construction time if someone tries to use it.

**`LiveReader.read()` ordering** (line 167-198):
1. `follower.get_observation()` — reads where the arm IS
2. `leader.get_action()` — reads what the human WANTS

This ordering is intentional: we read the current robot state BEFORE sending the next action. The `send_action()` call is separate (not within `read()`) so the caller controls timing.

**`DatasetReader._load_episode()`** (lines 79-95):
LeRobot datasets store all episodes contiguously in a single HuggingFace dataset table. The function finds boundaries by detecting where `episode_index` changes:
```python
boundaries = np.where(ep_idxs[:-1] != ep_idxs[1:])[0] + 1
```

**External references**:
- `lerobot_robot_trlc_dk1.follower.DK1Follower` → `send_action()`, `get_observation()`, `connect()`, `disconnect()`
- `lerobot_robot_trlc_dk1.leader.DK1Leader` → `get_action()`, `connect()`, `disconnect()`
- `lerobot.datasets.lerobot_dataset.LeRobotDataset` → for `DatasetReader`

---

### 4.2 `mapper.py` (53 lines)

**Role**: Converts between the real-robot coordinate system (LeRobot convention) and the MJLab simulation action space.

**Key functions**:

| Function | Input | Output | Math |
|----------|-------|--------|------|
| `mjlab_gripper_meters(norm)` | gripper [0,1] | prismatic meters [0.0, -0.045] | linear interpolation |
| `lerobot_to_mjlab_ctrl(pos)` | (7,) joint_pos | (7,) MJLab ctrl | joints pass through, gripper mapped |
| `lerobot_to_mjlab_raw_action(pos)` | (7,) joint_pos | (7,) raw action for MJLab | `(ctrl - HOME) / ACTION_SCALE` |

**Constants** (lines 6-26):

```python
HOME_JOINT_POS = np.array([0.0, 1.7, 1.7, -pi/2, 0.0, 0.0, 0.0])  # + gripper
ACTION_SCALE = np.array([...])  # per-joint scales from DK1_UNIMANUAL_ACTION_SCALE
```

These come from MJLab's source of truth (`g3a_bench.robots.trlc_dk1.dk1_actuators`, `g3a_bench.robots.trlc_dk1.dk1_constants`) — they define the DK1 robot model's home position and action bounds.

**Why two conversion functions?** MJLab's `JointPositionActionCfg(use_default_offset=True)` computes:
```
target_pos = HOME_JOINT_POS + raw_action * ACTION_SCALE
```
So `lerobot_to_mjlab_ctrl` converts real-robot units → target position, and `lerobot_to_mjlab_raw_action` inverts the formula to get the raw action that produces that target.

**Gripper mapping detail**:
- Real robot gripper: 0 = open, 1 = closed (normalized)
- MJLab gripper: 0.0m = open, -0.045m = closed (prismatic joint)
- `np.interp(val, [0, 1], [0.0, -0.045])` handles this

**External references**:
- `g3a_bench.robots.trlc_dk1.dk1_actuators.DK1_UNIMANUAL_ACTION_SCALE`
- `g3a_bench.robots.trlc_dk1.dk1_constants._HOME_JOINT_POS`

---

### 4.3 `simulator.py` (34 lines)

**Role**: Thin wrapper around MJLab's `ManagerBasedRlEnv` providing a stable `step()` interface.

**Key details**:

```python
class Simulator:
    def step(self, action):
        # Handles np.ndarray → torch.tensor, 1D → 2D
        # Returns (obs, reward, done, extras)
        # done = terminated | time_out (bool)
```

This wrapper exists because `ManagerBasedRlEnv.step()` returns `(obs, reward, terminated, time_out, extras)` — a different signature from gymnasium's `(obs, reward, done, info)`. The wrapper combines `terminated | time_out` into a single `done` flag.

**Important note**: This `Simulator` class is used by `orchestrator.py` (offline replay). The `live_record.py` and `record_replay.py` scripts use `ManagerBasedRlEnv` + `RslRlVecEnvWrapper` directly instead, because the viewer requires the wrapper. This creates an inconsistency: two different APIs for the same underlying env.

**External references**:
- `mjlab.envs.ManagerBasedRlEnv`, `ManagerBasedRlEnvCfg`

---

### 4.4 `recorder.py` (65 lines)

**Role**: Saves sim observations + real robot observations + actions + reward/done to LeRobot-format dataset.

**Dataset features**:

| Key | Shape | Dtype | Source |
|-----|-------|-------|--------|
| `observation.state` | (53,) | float32 | MJLab sim obs tensor (`obs_buf["actor"][0]`) |
| `observation.robot_state` | (7,) | float32 | Real robot joint positions |
| `action` | (7,) | float32 | Leader commanded action (same as `joint_pos` format) |
| `next.reward` | (1,) | float32 | Reward from env after step |
| `next.done` | (1,) | bool | Done flag from env after step |

**Key method**: `add_frame(obs_buf, action, reward, done, task, robot_obs=None)`

The `obs_buf` parameter expects a dict with key `"actor"` containing a (1, 53) tensor. This matches the output format of `ManagerBasedRlEnv.get_observations()` and `RslRlVecEnvWrapper.get_observations()`.

`robot_obs` is optional — if None, zeros are stored (for backward compatibility with offline replay scripts that have no real robot data).

**LeRobot dataset format compliance**: The frame dict includes all required keys (`observation.state`, `observation.robot_state`, `action`, `next.reward`, `next.done`, `task`). The `task` key is special: it's required by LeRobot but not part of `features`. Timestamps and frame indices are auto-generated by `DatasetWriter`.

**Important: `LeRobotDataset.create(root=...)`** — the `root` parameter is the **final dataset directory** (not `root/repo_id`). The directory must not already exist (it creates it with `exist_ok=False`).

**External references**:
- `lerobot.datasets.lerobot_dataset.LeRobotDataset` (create, add_frame, save_episode, finalize)
- `lerobot.datasets.dataset_writer.DatasetWriter.add_frame` (validates features)

---

### 4.5 `orchestrator.py` (24 lines)

**Role**: First attempt at a recording pipeline. Takes a pre-built list of actions and replays them through the sim while recording.

**Limitations** (vs newer scripts):
- Uses the old `Simulator` wrapper (not `RslRlVecEnvWrapper`)
- Requires pre-built action list (cannot do live teleop)
- No viewer
- Single episode only
- No `robot_obs` passed (stores zeros)

This file is effectively superseded by `record_replay.py` and `live_record.py` but kept for backward compatibility with the original 5-module design (Reader, Mapper, Simulator, Recorder, Orchestrator).

---

### 4.6 `record_replay.py` (210 lines)

**Role**: Replays a saved LeRobot dataset through the sim, shows the sim in a native MuJoCo window, and saves the resulting sim observations to a new dataset.

**Flow**:
1. Load task config (`G3a-Cube-Stack3-Dk1_uni` in play mode)
2. Create env → wrap with `RslRlVecEnvWrapper`
3. Open source dataset (`FabianKerj/corkinbox100-fullrel`)
4. Create output recorder
5. Create `_ReplayAndRecord` policy that reads dataset frames → maps to raw_action
6. Launch `_RecordingViewer` (subclass of `NativeMujocoViewer`)

**Key innovation: `_RecordingViewer`** (lines 168-207):
The base `NativeMujocoViewer._execute_step()` calls `env.step()` but discards the return values. This subclass overrides that method to capture `reward` and `done`, then calls `policy.record_step_result(rew, done)` to feed them back to the recording policy.

```python
class _RecordingViewer(NativeMujocoViewer):
    def _execute_step(self) -> bool:
        obs = self.env.get_observations()
        actions = self.policy(obs)
        _obs_next, rew, dones, _extras = self.env.step(actions)
        record_fn = getattr(self.policy, "record_step_result", None)
        if record_fn is not None:
            record_fn(float(rew[0]), bool(dones[0]))
```

This pattern (viewer subclass that captures step results) is reused in `live_record.py`.

**The viewer's `_step_physics` loop** (from MJLab `base.py:304`):
The viewer accumulates real-time budget (`_sim_budget += dt * _time_multiplier`) and calls `_execute_step()` repeatedly until the budget is consumed. This means a single viewer frame may call `_execute_step()` multiple times, each time reading a new action from the policy. The `_time_multiplier` can be adjusted at runtime (1.0 = real-time, 2.0 = 2x speed, etc.).

**External references**:
- `mjlab.viewer.NativeMujocoViewer` — MuJoCo native window viewer
- `mjlab.viewer.base.BaseViewer._execute_step` — the method being overridden
- `mjlab.viewer.base.BaseViewer._step_physics` — the budget-accumulating loop
- `mjlab.rl.RslRlVecEnvWrapper` — wraps env for viewer compatibility

---

### 4.7 `live_record.py` (303 lines)

**Role**: The main live operation script. Connects to real hardware, mirrors in sim, records both real and sim observations, shows sim in viewer.

**This is the file that ties everything together.**

**Threading architecture**:

The `_hw_loop` function (lines 48-94) runs in a daemon thread:
```python
def _hw_loop(action_list, robot_obs_list, shutdown_flag, follower_port, leader_port):
    reader = LiveReader(follower_port, leader_port, connect_on_init=True)
    while not shutdown_flag[0]:
        t0 = time.perf_counter()
        frame = reader.read()
        robot_obs_list[0] = frame.joint_pos
        if frame.leader_action is not None:
            reader.send_action(frame.leader_action)
            action_list[0] = frame.leader_action.copy()
        sleep(1/200 - elapsed)
```

The main thread runs the viewer:
```python
viewer = _LiveRecordingViewer(env, live_policy, recorder, robot_obs_list)
viewer.run()  # blocks until window closes or Ctrl+C
```

**`_LivePolicy`** (lines 97-118):
Acts as the viewer's policy. Returns the latest leader action from the shared list. Stores the action on `self.latest_leader_action` so `_LiveRecordingViewer._execute_step()` can access it for recording.

**`_LiveRecordingViewer`** (lines 121-175):
Similar to `_RecordingViewer` from `record_replay.py`, but instead of calling `policy.record_step_result()`, it directly accesses `policy.latest_leader_action` and `robot_obs_list[0]` and calls `recorder.add_frame()`.

**Cleanup** (lines 292-299):
```python
finally:
    shutdown_flag[0] = True        # signal HW thread
    hw_thread.join(timeout=2.0)    # wait for it
    recorder.end_episode()         # save episode
    recorder.close()               # finalize dataset
    env.close()                    # close sim
```

**CLI arguments**:

| Argument | Default | Purpose |
|----------|---------|---------|
| `--follower-port` | auto-detect | Serial port for follower (e.g. `/dev/ttyACM0`) |
| `--leader-port` | auto-detect | Serial port for leader (e.g. `/dev/ttyUSB0`) |
| `--task` | `G3a-Cube-Stack3-Dk1_uni` | MJLab task name |
| `--dataset-dir` | `logs/live_record/` | Output directory |
| `--num-episodes` | 5 | Episode count (stub — not yet implemented) |
| `--framerate-mode` | `decoupled` | `decoupled` / `every-n` / `as-fast` |
| `--headless` | False | Skip opening the viewer window |

---

## 5. External Dependencies Detail

### 5.1 `trlc-dk1` (`/home/minhtrung/Documents/UNI/G3a/trlc-dk1/`)

| File | Key Symbols | Used By |
|------|-------------|---------|
| `lerobot_robot_trlc_dk1/follower.py` | `DK1Follower`, `DK1FollowerConfig`, `JOINT_NAMES` | `reader.py` |
| `lerobot_robot_trlc_dk1/leader.py` | `DK1Leader`, `DK1LeaderConfig` | `reader.py` |
| `examples/teleop.py` | Reference implementation of leader→follower loop | Inspiration for `live_record.py` |

**`DK1Follower.send_action(action: dict)`** (follower.py:258):
Takes a dict like `{"joint_1.pos": 0.1, ..., "gripper.pos": 0.5}`. Sends to the real arm via either impedance mode (non-blocking, background thread) or POS_VEL mode (blocking serial I/O).

**`DK1Follower.get_observation()`** (follower.py:223):
Returns dict of joint positions + camera images. Keys: `"joint_1.pos"` through `"joint_6.pos"`, `"gripper.pos"`.

**`DK1Leader.get_action()`** (leader.py:104):
Returns dict of leader positions. Same key format. Gripper is normalized to [0, 1].

**Installation**: `uv pip install -e /path/to/trlc-dk1`
**Registration**: The package is discovered automatically by lerobot's `register_third_party_plugins()` via `importlib.metadata` (searches for packages starting with `lerobot_robot_`).

### 5.2 `MJLab` (`/home/minhtrung/Documents/UNI/G3a/MJLab/`)

| File | Key Symbols | Used By |
|------|-------------|---------|
| `mjlab/envs/manager_based_rl_env.py` | `ManagerBasedRlEnv` | `simulator.py`, `record_replay.py`, `live_record.py` |
| `mjlab/rl/vecenv_wrapper.py` | `RslRlVecEnvWrapper` | `record_replay.py`, `live_record.py` |
| `mjlab/viewer/base.py` | `BaseViewer`, `_execute_step`, `_step_physics`, `run`, `tick` | Viewer subclasses |
| `mjlab/viewer/native/viewer.py` | `NativeMujocoViewer` | `record_replay.py`, `live_record.py` |

**`ManagerBasedRlEnv.step(action)` return**: `(obs, reward, terminated, time_out, extras)` — note the split terminated/time_out.
**`RslRlVecEnvWrapper`**: Wraps the env to present an RLlib-compatible interface. Required by the viewer's `env` parameter.
**`BaseViewer._step_physics(dt)`** (base.py:304): Accumulates sim budget, calls `_execute_step()` in a while loop. This is why the viewer can call the policy multiple times per frame.

### 5.3 `lerobot` (`/home/minhtrung/Documents/UNI/G3a/lerobot/`)

| File | Key Symbols | Used By |
|------|-------------|---------|
| `lerobot/datasets/lerobot_dataset.py` | `LeRobotDataset` (create, add_frame, save_episode, finalize) | `recorder.py`, `reader.py` |
| `lerobot/datasets/dataset_writer.py` | `DatasetWriter.add_frame`, `validate_frame` | Behind `add_frame()` |
| `lerobot/scripts/lerobot_record.py` | Official recording script (no sim) | Reference |
| `lerobot/scripts/lerobot_teleoperate.py` | Official teleop-only script (no recording) | Reference |

**`LeRobotDataset.create(repo_id, fps, features, root)`**: Creates a new dataset directory. `root` is the full path (not `root/repo_id`). Must not already exist.
**`LeRobotDataset.add_frame(frame)`**: Validates all features are present (via `validate_frame`), appends to episode buffer. Images go to temporary directory; video encoding happens at `save_episode()`.

### 5.4 `g3a_bench` (this project)

| File | Key Symbols | Used By |
|------|-------------|---------|
| `g3a_bench/tasks/cube_stack3/` | `G3a-Cube-Stack3-Dk1_uni` task config | `live_record.py`, `record_replay.py` |
| `g3a_bench/robots/trlc_dk1/dk1_constants.py` | `_HOME_JOINT_POS` | `mapper.py` |
| `g3a_bench/robots/trlc_dk1/dk1_actuators.py` | `DK1_UNIMANUAL_ACTION_SCALE` | `mapper.py` |

---

## 6. Design Decisions and Tradeoffs

### 6.1 Decoupled threads vs. single-threaded

**Chosen**: Two threads (HW daemon + main viewer thread)
**Alternative**: Single thread with time-multiplexed I/O and sim stepping
**Why**: The hardware loop must run at ~200 Hz to keep the robot responsive. A single sim step on CPU takes ~250ms (RTF ~0.2x). Blocking the hardware loop for 250ms would make the robot feel laggy and could cause safety issues.

### 6.2 Last-writer-wins vs. queue

**Chosen**: Single-element list with overwrite
**Alternative**: `queue.Queue(maxsize=N)` with producer-consumer
**Why**: The sim should always receive the **most recent** human command, not a backlog of old ones. A queue would either overflow (if bounded) or accumulate stale actions (if unbounded). Overwrite semantics naturally handle the rate mismatch: the sim always gets the latest command.

### 6.3 Viewer subclass vs. manual render loop

**Chosen**: Subclass `NativeMujocoViewer` and override `_execute_step()`
**Alternative**: Manual loop calling `env.step()` + MuJoCo render functions
**Why**: The viewer provides interactive controls (pause, speed adjustment, reset) and handles the real-time budget management (`_step_physics` with `_time_multiplier`). Writing all that from scratch would be error-prone.

### 6.4 `RslRlVecEnvWrapper` vs. raw `ManagerBasedRlEnv`

**Chosen**: Wrap with `RslRlVecEnvWrapper` for viewer compatibility
**Why**: `NativeMujocoViewer.__init__` expects the env to be wrapped. The wrapper provides the `get_observations()` method interface that the viewer's `_execute_step` calls.

**Downside**: There are now two different env APIs in the codebase:
- `Simulator.step()` (used by `orchestrator.py`) → raw `ManagerBasedRlEnv.step()`
- `RslRlVecEnvWrapper` env → viewer-compatible interface

### 6.5 Both sim and real observations saved

**Chosen**: Store both `observation.state` (53-dim) and `observation.robot_state` (7-dim)
**Why**: Insurance policy. If sim-based BC training works, use `observation.state`. If there's a sim-to-real gap, fall back to `observation.robot_state`. Having both in the same dataset avoids needing to collect data twice.

### 6.6 Imports guarded by try/except

**Chosen**: Wrap `lerobot_robot_trlc_dk1` imports in try/except
**Why**: The bridge modules can be imported on machines without the trlc-dk1 package (e.g., CI, laptops without robot hardware). Only `LiveReader` construction raises an error if the package is missing.

### 6.7 The Mapper lives in the bridge, not in MJLab

**Chosen**: `mapper.py` is part of `g3a_bench.scripts.live_bridge`, not in MJLab or trlc-dk1
**Why**: The mapper converts between two independent coordinate systems (LeRobot convention and MJLab convention). It doesn't belong to either project. It's a bridge-specific concern.

---

## 7. Known Gaps and Future Work

### 7.1 Episode management
`live_record.py` records a single episode (everything between start and Ctrl+C). The `--num-episodes` flag is declared but not implemented. Future work: detect `done=True` → save episode → reset env → start new episode → repeat until count reached.

### 7.2 Framerate modes
Only `decoupled` mode is implemented. `every-n` and `as-fast` are stubs. These would be useful for:
- `every-n`: deterministic replay at fixed sub-rate (easier to compare results)
- `as-fast`: one sim step per HW cycle (simplest, but blocks HW)

### 7.3 Headless mode
`--headless` flag is declared but `_LiveRecordingViewer` is always used. A headless mode would skip the viewer and run a manual `env.step()` loop in the main thread.

### 7.4 Camera recording
The `ReaderFrame.cameras` dict exists but is not yet saved by `Recorder`. Adding camera features (e.g., `observation.images.top`, `observation.images.wrist`) would require:
1. Adding image features to `Recorder.__init__` features dict
2. Grabbing camera images from `env.render()` or from `LiveReader`
3. LeRobot handles image encoding automatically (PNG for stills, MP4 for video)

### 7.5 Two env APIs
`Simulator` (used by orchestrator.py) wraps `ManagerBasedRlEnv` directly. `record_replay.py` and `live_record.py` use `RslRlVecEnvWrapper`. These should be unified. The `Simulator` class could be updated to use the wrapper, or the scripts could use `Simulator` for consistency.

### 7.6 Performance on CPU
RTF is 0.20x–0.40x on CPU. The sim runs ~4x slower than real-time. GPU acceleration (CUDA) is required for 1.0x real-time. The `device` is auto-detected (`cuda:0` if available, else `cpu`).

### 7.7 Real robot observation timing
`robot_obs_list[0]` contains the follower position from the hardware thread's most recent `read()` call. This is slightly out of sync with the sim observation (the sim step happens ~250ms later). The real robot observation and sim observation are correlated but not perfectly temporally aligned.

---

## 8. Testing Strategy

| Test File | What it covers | Count |
|-----------|----------------|-------|
| `tests/trlc_dk1/test_reader.py` | ReaderFrame, DatasetReader (mock), LiveReader (mock), send_action | 12 tests (2 skipped = integration) |
| `tests/trlc_dk1/test_mapper.py` | Gripper mapping, ctrl conversion, raw action roundtrip | 6 tests |
| `tests/trlc_dk1/test_simulator.py` | Env creation, reset, step, 1D→2D, close | 5 tests |
| `tests/trlc_dk1/test_recorder.py` | Dataset creation, add_frame, episode lifecycle | 3 tests |
| `tests/trlc_dk1/test_orchestrator.py` | record_episode with mock sim/recorder | 2 tests |

**Total: 26 tests (24 pass, 2 skipped require network)**

**What's NOT tested**:
- `record_replay.py` — integration test requires downloading a real dataset
- `live_record.py` — integration test requires real DK1 hardware
- Thread safety of shared lists (assumed safe for GIL, no stress tests)
- Viewer subclass behavior (requires display)

**Testing pattern**: Hardware/network-dependent tests are guarded by `@pytest.mark.skipif` with `HARDWARE_TEST` env var. All other tests use `unittest.mock` to isolate the module under test.

---

## 9. File Map

```
g3a_bench/g3a_bench/scripts/live_bridge/
├── __init__.py          # empty
├── reader.py            # Reader ABC, ReaderFrame, DatasetReader, LiveReader
├── mapper.py            # lerobot→mjlab coordinate conversion
├── simulator.py         # ManagerBasedRlEnv wrapper
├── recorder.py          # LeRobot dataset recorder
├── orchestrator.py      # record_episode() — OLD API, pre-built actions
├── record_replay.py     # Dataset replay → sim → viewer → recording
├── live_record.py       # Live teleop → sim → viewer → recording
├── test_replay.py       # Same as record_replay but with ViserPlayViewer (deprecated)
├── ARCHITECTURE.md      # This file
└── logs/                # gitignored output directory
```

---

## 10. Quick Start Commands

```bash
# Install dependencies
cd /home/minhtrung/Documents/UNI/G3a/g3a-bench
uv sync --extra cpu
uv pip install -e /home/minhtrung/Documents/UNI/G3a/trlc-dk1

# Test offline replay (no hardware needed)
uv run --extra cpu python g3a_bench/scripts/live_bridge/record_replay.py

# Live operation (hardware required)
uv run --extra cpu python g3a_bench/scripts/live_bridge/live_record.py

# With explicit ports
uv run --extra cpu python g3a_bench/scripts/live_bridge/live_record.py \
    --follower-port /dev/ttyACM0 --leader-port /dev/ttyUSB0

# Run tests
uv run --extra cpu python -m pytest tests/trlc_dk1/ -v

# Pre-commit checks
uv run ruff check --config ruff.toml g3a_bench/ tests/
uv run ruff format --check --config ruff.toml g3a_bench/ tests/
uv run pyright --project pyrightconfig.json
```
