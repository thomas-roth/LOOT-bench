# live_bridge tests

Unit and integration tests for the live-bridge pipeline
(`g3a_bench/scripts/live_bridge/`). No real hardware or GPU needed.

---

## Prerequisites

```bash
# from repo root
uv sync --extra cpu
```

---

## Run all tests

```bash
uv run pytest tests/live_bridge/ -v
```

Expected output: all green, ~10 tests, a few seconds.

---

## What each file tests

### `test_mapper.py`

Pure math — no sim, no hardware, no I/O.

Tests `mapper.py` which converts real-robot joint positions into the MuJoCo
action format the sim expects.

| Test | What it checks |
|------|----------------|
| `test_open_to_meters` | gripper 0.0 (open) → 0.0 m in sim |
| `test_closed_to_meters` | gripper 1.0 (closed) → −0.045 m in sim |
| `test_joints_pass_through` | arm joints pass unchanged to ctrl |
| `test_gripper_maps_to_meters` | gripper channel maps correctly via ctrl |
| `test_home_position_returns_near_zero` | home pose → raw action ≈ 0 |
| `test_roundtrip_with_ctrl` | `raw_action → HOME + raw·SCALE` recovers `ctrl` |

```bash
uv run pytest tests/live_bridge/test_mapper.py -v
```

---

### `test_reader.py`

Tests `reader.py`'s serial-port auto-detection logic (`_find_ports`).
Uses monkeypatching — no real serial ports needed.

| Test | What it checks |
|------|----------------|
| `test_find_ports_uses_pyserial` | basic enumeration: two ports with clear hints |
| `test_find_ports_prefers_device_path` | Linux `/dev/ttyACM0` / `ttyUSB0` paths take priority over description text |
| `test_find_ports_ftdi_vid_identifies_leader` | Windows COM ports: FTDI vendor id `0x0403` pins the leader even when both ports have identical description text |

The FTDI test is the important one for Windows — it covers the scenario where
both ports show "USB Serial Device" and only the vendor id distinguishes them.

```bash
uv run pytest tests/live_bridge/test_reader.py -v
```

---

### `test_recorder.py`

Tests `recorder.py`. Split into two parts:

**Mocked tests** (`TestRecorder` class) — fast, no disk I/O, patch `LeRobotDataset`.

| Test | What it checks |
|------|----------------|
| `test_create_dataset` | `Recorder.__init__` calls `LeRobotDataset.create` |
| `test_add_frame` | frame dict has correct keys and shapes |
| `test_end_episode_and_close` | `save_episode()` then `finalize()` called in order |
| `test_end_episode_skips_when_empty` | `save_episode()` not called if no frames added |

**Real round-trip test** (`test_recorder_real_roundtrip`) — writes an actual
LeRobot dataset to a `tmp_path`, reads it back, checks shapes.

This is the most valuable test: if a future `lerobot` version changes the schema
or the `add_frame` API, this test will catch it.

Expected output:
```
tests/live_bridge/test_recorder.py::TestRecorder::test_create_dataset PASSED
tests/live_bridge/test_recorder.py::TestRecorder::test_add_frame PASSED
tests/live_bridge/test_recorder.py::TestRecorder::test_end_episode_and_close PASSED
tests/live_bridge/test_recorder.py::TestRecorder::test_end_episode_skips_when_empty PASSED
tests/live_bridge/test_recorder.py::test_recorder_real_roundtrip PASSED
```

```bash
uv run pytest tests/live_bridge/test_recorder.py -v
```

---

### `test_replay.py`

**Not a pytest test — an interactive script.** Opens a MuJoCo viewer window
and replays `FabianKerj/corkinbox100-fullrel` through the sim. No recording,
no hardware.

Use it to visually verify the dataset→sim pipeline end to end.

```bash
uv run --extra cpu python tests/live_bridge/test_replay.py
```

**First run:** downloads ~655 MB from HuggingFace (cached after).

**On Windows:** HuggingFace will print a symlink warning — this is harmless,
the download completes anyway. If it fails on the first attempt, run it a
second time (partial cache resolves the issue).

**What to expect:** a MuJoCo viewer window opens showing the cube-stacking
scene. The arm replays the recorded trajectory. When the episode ends it loops
back to the start automatically. Close the window or press `Ctrl+C` to exit.
Runs at ~8 FPS on CPU (RTF ~0.18x) — normal, sim is not GPU-accelerated here.
