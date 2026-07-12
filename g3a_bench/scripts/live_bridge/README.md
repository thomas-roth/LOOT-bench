# live_bridge

Teleop → sim mirroring → dataset recording for G3A benchmark. Records demonstrations from a real DK1 robot (GELLO leader + optional DK1 follower) into a LeRobot dataset for offline RL training.

**Setup:** See `pyproject.toml` for dependencies. Hardware setup: find USB ports with `uv run lerobot-find-port`, grant access on Linux with `sudo chmod 666 /dev/ttyACM*`. A display is required for the MuJoCo viewer.

## Scripts

### `live_record.py` — record demonstrations via live teleop

**Requires hardware.** Records episodes at ~50 Hz from real robot actions (~200 Hz GELLO leader, optional DK1 follower).

```bash
uv run python g3a_bench/scripts/live_bridge/live_record.py [--follower] [--leader-port PORT] [--follower-port PORT] [--num-episodes N] [--dataset-dir PATH]
```

Asks for operator ID (1–4) at startup. **Letter-key controls** (an on-screen HUD shows state, saved counts and the currently valid keys):
- `E`: **End** episode manually → save-or-discard prompt
- `S`: **Save** the ended episode as failure (only in the prompt)
- `D`: **Discard** → reset phase → redo
- `U`: **Undo** the last saved episode (single-level; the last save is staged and only committed when the next episode ends or at finalize)
- `Q`: Finalize dataset and quit
- `W`: Toggle wrist camera ↔ orbit view

Episode labels come from the env termination: the success termination auto-saves as **success** (wonky-but-successful included); any other termination — or `E` — freezes the sim into a no-timeout prompt (orange ghost) where the episode is saved as **failure** or discarded.

**State machine:** RECORDING → [auto-save on task success | prompt (AWAITING_DECISION) → `S`/`D`] → RESET_PHASE (ghost arm at rest pose, no recording) → back to RECORDING when sim arm near rest.

Output: `out/demos/success/<YYYY-MM-DD_HH-MM-SS>/` and `out/demos/failure/<YYYY-MM-DD_HH-MM-SS>/` at the repo root (timestamped, never overwrites). A directory is only created if at least one episode of that kind was saved.

**Optional:** set `USE_FOLLOWER_STATE_FOR_SIM = True` in `live_record.py` to drive from follower joint observations instead of leader action.

### `record_replay.py` — offline dataset replay

**No hardware.** Replays a LeRobot dataset through the sim, records sim obs + reward/done.

```bash
uv run python g3a_bench/scripts/live_bridge/record_replay.py
```

Downloads example dataset from HuggingFace on first run (~655 MB cached). Output: `logs/sim_replay_cube_stack3/success/`
