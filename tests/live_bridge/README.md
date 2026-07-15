# live_bridge tests

Unit and integration tests for the live-bridge pipeline
(`g3a_bench/scripts/live_bridge/`). No real hardware or GPU needed.

---

## Run all tests

```bash
uv run pytest tests/live_bridge/ -v
```

Expected output: all green, 41 tests (43 collected, 2 skipped = hardware/network), a few seconds.

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

Tests `reader.py` — the `ReaderFrame` dataclass, `DatasetReader` (offline replay),
`LiveReader` (real robot), and `_find_ports` auto-detection.
Uses mocks and monkeypatching — no real serial ports or network needed.

**`TestReaderFrame`** — dataclass construction:

| Test | What it checks |
|------|----------------|
| `test_creation_with_all_fields` | all four fields (joint_pos, leader_action, cameras, timestamp) accepted |
| `test_default_fields` | only joint_pos required; leader_action=None, cameras={}, timestamp=0.0 |

**`TestDatasetReader`** — offline replay from a mocked LeRobot dataset:

| Test | What it checks |
|------|----------------|
| `test_read_returns_readerframe` | successive reads return correctly shaped ReaderFrame objects |
| `test_read_raises_stopiteration_when_exhausted` | StopIteration raised after last frame |

**`TestDatasetReaderIntegration`** *(skipped unless `HARDWARE_TEST=1`)* — downloads the real `FabianKerj/corkinbox100-fullrel` dataset:

| Test | What it checks |
|------|----------------|
| `test_read_first_frame` | joint_pos/leader_action shape (7,), joints in radian range, gripper in [0,1] |
| `test_read_multiple_frames` | 10 consecutive frames, timestamps non-decreasing |

**`TestLiveReader`** — real-robot reader with mocked DK1 hardware:

| Test | What it checks |
|------|----------------|
| `test_lazy_init` | `connect_on_init=False` skips port connection |
| `test_connect_on_init` | both follower and leader `.connect()` called on init |
| `test_read_returns_readerframe` | follower obs + leader action unpacked correctly into ReaderFrame |
| `test_close_disconnects` | `.close()` calls `.disconnect()` on both devices |
| `test_send_action` | (7,) numpy → dict with correct joint keys sent to follower |
| `test_read_leader_action_dict_is_unpacked_in_order` | get_action() dict → leader_action array in joint_1..6, gripper order |
| `test_send_action_raises_without_connect` | RuntimeError if send_action called before connect |

**Port auto-detection** (module-level):

| Test | What it checks |
|------|----------------|
| `test_find_ports_uses_pyserial` | basic enumeration: two ports with clear description hints |
| `test_find_ports_prefers_device_path` | Linux `/dev/ttyACM0`/`ttyUSB0` paths win over description text regardless of list order |
| `test_find_ports_ftdi_vid_identifies_leader` | Windows COM ports: FTDI vendor id `0x0403` pins the leader when both descriptions are identical |

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
| `test_create_dataset_lazily` | no dataset on init — `LeRobotDataset.create` happens on the first `save_episode()` of that kind |
| `test_add_frame_buffers_until_save` | `add_frame()` holds frames in memory; dataset NOT written yet |
| `test_save_episode_labels_frames_success` | `save_episode(True)` stamps `success=True` on every frame, then flushes |
| `test_save_episode_labels_frames_failure` | `save_episode(False)` stamps `success=False` |
| `test_save_episode_flushes_all_frames` | all buffered frames (not just the first) are written on save |
| `test_discard_episode_drops_buffer` | `discard_episode()` clears buffer without writing anything |
| `test_save_episode_empty_is_noop` | `save_episode()` with no frames buffered writes nothing and creates no dataset |
| `test_close_finalizes_dataset` | `close()` calls `finalize()` on every dataset that was created |
| `test_discard_then_save_new_episode` | after discard, a fresh episode can be buffered and saved normally |

**Real round-trip tests** — write actual LeRobot datasets to `tmp_path`, read back, check shapes.

| Test | What it checks |
|------|----------------|
| `test_recorder_real_roundtrip` | obs/action shapes, success column persists after finalize; no failure dataset for a success-only session |
| `test_close_stores_operator_id` | `close(operator_id=3)` → `meta.info["operator_id"] == 3` after re-open |
| `test_close_without_operator_id_still_finalizes` | `close()` with no argument does not raise |

```bash
uv run pytest tests/live_bridge/test_recorder.py -v
```

---

### `test_live_record_helpers.py`

Unit tests for `live_record.py` helper functions (`ask_operator`).
No sim, no hardware, no I/O.

| Test | What it checks |
|------|----------------|
| `test_ask_operator_valid` | Returns int for valid input "2" |
| `test_ask_operator_retries_on_bad_input` | Loops until valid input, ignores "0"/"5"/"abc"
```bash
uv run pytest tests/live_bridge/test_live_record_helpers.py -v
```

---

### `test_replay.py`

**Not a pytest test — an interactive script.** Opens a MuJoCo viewer window
and replays `FabianKerj/corkinbox100-fullrel` through the sim. No recording,
no hardware.

Use it to visually verify the dataset→sim pipeline end to end.

```bash
uv run python tests/live_bridge/test_replay.py
```

**First run:** downloads ~655 MB from HuggingFace (cached after).

**On Windows:** HuggingFace will print a symlink warning — this is harmless,
the download completes anyway. If it fails on the first attempt, run it a
second time (partial cache resolves the issue).

**What to expect:** a MuJoCo viewer window opens showing the cube-stacking
scene. The arm replays the recorded trajectory. When the episode ends it loops
back to the start automatically. Close the window or press `Ctrl+C` to exit.
Runs at ~8 FPS on CPU (RTF ~0.18x) — normal, sim is not GPU-accelerated here.
