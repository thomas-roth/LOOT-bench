# A dataset carries the action convention, and loading checks it

The converter writes the **convention record** in force at record time into the
`.npz`. `load_dataset(path, env)` compares that stamp against the environment a
training run builds, and raises on a mismatch, naming the field that differs. A
training run should read its dataset through `load_dataset`, not through
`np.load`.

The record is `qc/dataset.STAMP_KEYS`: a `convention` name checked first (so a
dataset from an older MDP says *that* rather than reporting whichever number
happens to differ), the two delta scales, the gripper stroke mapping, the
`control_step_s`, and the `task_id`. *(Widened by ADR-016. It was the action
`scale`/`offset` pair, which the delta term does not have; the control rate had
never been stamped at all, so changing it was silent, and the task id closes an
already-realised hole — every `place` variant shares its widths, so a
`RandS-RandT` dataset trained against a `FixedS-FixedT` env for all fourteen
runs of the α ladder.)*

## Why

A dataset recorded under a different action convention is the failure mode this
exists for. Nothing raises, the loss falls, and evaluation scores zero, which is
exactly what ADR-005 found. A stale dataset must fail at startup instead.

The converter also stores the action the recording was flown with, byte for
byte. Anything else records one trajectory and trains on another. *(How it does
that changed with ADR-016. It used to re-run the record-time leader-pose
conversion, which only worked while an action was absolute; a delta means "move
this far from where you are", and the pose it was measured against is not in the
file. The recorder now writes the raw action `env.step()` was given and the
converter copies it through — which makes this paragraph's claim literally true
rather than true by reconstruction. `build_dataset` raises if any stored action
lies outside the action box, since that would mean the recorder's mapping and
the action term disagreed.)*

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
