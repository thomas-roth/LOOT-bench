# A dataset carries the action convention, and loading checks it

The converter writes the action `scale` and `offset` in force at record time
into the `.npz`. `load_dataset(path, env)` compares that stamp against the
environment a training run builds, and raises on a mismatch. A training run
should read its dataset through `load_dataset`, not through `np.load`.

## Why

A dataset recorded under a different action convention is the failure mode this
exists for. Nothing raises, the loss falls, and evaluation scores zero, which is
exactly what ADR-005 found. A stale dataset must fail at startup instead.

The converter also repeats the record-time action conversion exactly, using the
same function and the same gravity compensation as the recorder's live policy.
Anything else records one trajectory and trains on another.

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
