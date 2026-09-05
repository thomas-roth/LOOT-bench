# A dataset carries the action convention, and loading checks it

The converter writes the action `scale` and `offset` in force at record time,
the `start_pose_noise` the episodes were recorded under, and the task id
itself into the `.npz`. `load_dataset(path, env)` compares those stamps
against the environment a training run builds, and raises on a mismatch.
A training run should read its dataset through `load_dataset`, not through
`np.load`.

## Why

A dataset recorded under a different action convention is the failure mode
this exists for. Nothing raises, the loss falls, and evaluation scores zero,
which is exactly what ADR-005 found. A stale dataset must fail at startup
instead.

The width and convention checks cannot see a **task swap**: variants of one
category (the four Place variants in particular) share the robot, the
observation terms, the action width, and the start-pose noise, so a dataset
recorded for `FixedS-FixedT` passes every width check against `RandS-RandT`
and trains silently on the wrong spawn and target distribution. The task
stamp is what distinguishes the two: the recorder writes `task` per frame,
the converter resolves it (inferred from the recording or given with
`--task`) and stamps it into the `.npz`, and the check compares it against
`env.task_id` and raises on a disagreement.

The task check is **warn-gated on old files**: a `.npz` without the stamp
predates it, so it loads with a `UserWarning` rather than an error. A
stamped dataset always gets the check. Rebuild a dataset with
`g3a-convert-demos` to upgrade it.

The converter also repeats the record-time action conversion exactly, using
the same function and the same gravity compensation as the recorder's live
policy. Anything else records one trajectory and trains on another.

## Consequences

- The stamp keys, and the `episode_index` column, are metadata. Filter the
  dataset to `QC_KEYS` before `Dataset.create`, because `qc`'s replay buffer
  demands every offline key from the online transition dicts too, and those
  carry only the six.
- The check converts through the device. A CUDA tensor cannot be read with
  `np.asarray`, which made the first version of this guard raise `TypeError`
  the first time it ran for real.
- `live_bridge/lerobot_to_qc.py` is deleted, because it wrote the old
  convention. `live_bridge/merge_datasets.py` is deleted too: `convert` takes
  several session roots at once, and a separate merge tool would have to know
  the file layout as well as the converter does, for one concatenation.
