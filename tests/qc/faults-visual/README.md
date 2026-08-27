# Before-and-after pictures of the three faults

Three scripts, one question each. Every one rebuilds the old behaviour from
constants that still live in the tree, measures it and the current behaviour
on the same machine, and writes pictures you can put side by side.
They change nothing: no task code, no dataset, no run.

Run everything from the repo root. Each script takes about a minute, most of
it the first environment build.

```bash
uv run python tests/qc/faults-visual/run_all.py            # all three, plus index.html
uv run python tests/qc/faults-visual/run_all.py --cpu      # no GPU needed
uv run python tests/qc/faults-visual/fault01_action_reach.py       # one at a time
uv run python tests/qc/faults-visual/fault02_ee_pose_signflip.py
uv run python tests/qc/faults-visual/fault03_start_pose.py
```

Output goes to `out/render/faults-visual/`. Open `out/render/faults-visual/index.html`: it
carries all the pictures and the measured numbers on one page.

Flags shared by all three: `--cpu` forces CPU, `--out DIR` moves the output,
`--no-render` writes the plots only (use it where there is no GL context).

## What each script shows

### `fault01_action_reach.py`: could the policy command the demonstrated pose?

Takes the recorded actions and asks which of them each action convention can
express. Before is mjlab's residual rule, `target = action * scale +
keyframe`, with the scale recomputed from the actuator configs exactly as the
deleted `_dk1_action_scale` did (0.07 rad on joints 1 to 3). A bounded policy
lives in `[-1, 1]`, so it could only command a narrow band around the keyframe.
After is the rule we use now, where `-1` is a joint's lower limit and `+1` its
upper limit.

| Picture | What is in it |
|---|---|
| `fault01_bands.png` | One demonstration per joint, against both reachable bands |
| `fault01_poses.png` | The demonstrated grasp, and the closest pose the old convention could command, with the cube where the operator saw it |
| `fault01_legacy.png` | The session recorded *before* the rebuild: the actions the operator flew, against the box a bounded policy can emit |

The third picture is the strongest: 0 of 24 548 frames of the pre-rebuild
session sit inside `[-1, 1]`. That session is the one the first training runs
used.

### `fault02_ee_pose_signflip.py`: does the pose jump while the arm is still?

Sweeps one joint smoothly and reads the grasp site both ways: the site
quaternion, which mjlab builds with an `argmax` over four candidates that tie
at this pose, and the 6D rotation the observation publishes today.

| Picture | What is in it |
|---|---|
| `fault02_trace.png` | Both forms along the sweep, and the change per step |
| `fault02_frames.png` | The two frames on either side of the first jump, the same arm, twice |

### `fault03_start_pose.py`: do the demonstrations and the tests start together?

Reads the first frame of every episode of both recorded sessions and compares
it with where an evaluation episode really begins, over 20 live resets.

| Picture | What is in it |
|---|---|
| `fault03_before.png` | Where a demonstration began, and where a test episode began, before the rebuild |
| `fault03_after.png` | The same two poses today |
| `fault03_gap.png` | Per joint: the distance between the two starts, and the variety of the demo start |

Start-state variety is read peak to peak, never as a standard deviation: the
observations are float32, so samples that never vary still return about 1e-6
(ADR-008).

## What these pictures do and do not prove

They prove that each fault was real, and that it is gone. They do not
measure how much of the flat-zero training result each fault caused: the old
pipeline was deleted rather than run again, so nothing compares the two action
spaces under equal conditions. Fault 1 is the strongest of the three, because an
action outside the reachable band is a fact of arithmetic on the compiled model.
Faults 2 and 3 are confirmed defects of unmeasured size.

## Data these scripts read

| Path | Recorded |
|---|---|
| `dataset/demos/G3A-Place-RandS-RandT-Dk1_uni/G3A-Place-RandS-RandT-Dk1_uni_qc_physical.npz` | Before the rebuild, kept as it was |
| `out/qc/place-rands-randt.npz` | After, the 2026-08-15 session |

Point a script at other data with `--dataset` / `--legacy-dataset`. A missing
file is reported and skipped, not raised.
