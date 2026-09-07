# Training lives in the upstream `qc` checkout, not in this repository

QC-FQL training runs from a separate checkout of upstream `qc`
(`ColinQiyangLi/qc`). This repository keeps only the seam to it: the gym
adapter, the demo converter and the dataset contract in `loot_bench/qc/`.
Commit `84cf304` deleted the in-repo reimplementation of QC-FQL, which was 39
files, 3636 lines and an 81.7 MB `.npz`.

## Why

A local copy of a research trainer has to be kept equal to the original by
hand. Ours had drifted, so every measurement had two possible causes: the
algorithm, or our copy of it. Upstream `qc` is the reference the results are
compared against, so we run the reference.

## Consequences

- mjlab's registry needs an `rl_cfg` for each task, so every task still
  carries an RSL-RL PPO config. That config satisfies registration. It is not
  the training path.
- The seam stays small on purpose. The adapter changes no part of the MDP
  (ADR-009), because the recorder has to see the same MDP as the trainer.
- Anything the two repositories must agree on is checked, not assumed. The
  action convention is stamped into each dataset and validated at load
  (ADR-011).
