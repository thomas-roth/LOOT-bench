# Gate a demonstration set with a behaviour-cloning fit before training on it

Before a QC-FQL run is started on a new demonstration set, fit a plain MSE
network to it and read the held-out error. The fit takes about two minutes for
300k steps on one GPU. A training run takes hours.

## Why

Two very different problems give the same flat-zero training curve: the learner
is weak and the data is good, or the data cannot work and no learner will
succeed. No training curve separates them. The fit does, because it is the
simplest thing that can possibly work: observation in, the next 5 actions out.
It uses no reward and no critic, and it is deterministic.

On the archived 100 demonstrations the fit reached an action error of 0.0013
along the path, against 0.033 for QC-FQL's own actor. That is 19 times more
accurate, and 400 times more accurate at the start state. So the data is easy
to fit and the algorithm's actor under-fits it.

## The bar, and why there is no longer one

Read the held-out error, not the training error, and state which number is
being used. Two reference points existed on the old 100 demonstrations, and
they are not interchangeable: 0.118 when the 530-frame first episode is
skipped, and 0.166 when all 10 held-out episodes are read. Both read 0.0013 on
trained episodes, and both were flat from 25k steps.

**Amended (2026-09-11).** The gate is written again, as `tests/qc/bc_gate.py`,
and the bar is 0.0766. Neither 0.118 nor 0.166 is the bar any more, and neither
can be compared with the new number, because the three were read differently.
The old two took the holdout from the front of the session, stopped the read at
frame 500, and used a constant learning rate over 25k steps. The gate holds out
at random, because the first attempts of a session are the clumsy ones, and it
reads the whole body of every held-out episode.

0.0766 is measured, not inherited. It is what this gate reads on the same
legacy file the old numbers came from, the pre-rebuild export
`LOOT-Place-Easy-Dk1_qc_physical.npz`, with ten episodes held out. The same
fit read 0.0075 on the episodes it saw. That file is a single legacy export,
**not** the committed sessions under
`dataset/demos/LOOT-Place-RandS-RandT-Dk1`: that directory has grown past
the set the number was read on, so the gate will not reproduce 0.0766 there.

**Amended again (2026-09-12): the bar is withdrawn, and the gate no longer
passes or fails anything.** 0.0766 was never read on a set whose training
outcome was known, and when it was checked against two sets whose outcome *is*
known, it ordered them backwards. Both readings below use the CLI defaults over
`--holdout-seed 0 1 2`, so the spread is the split-to-split spread:

| demonstration set | eval success | fitted | held out | own \|a[t+1] - a[t]\| |
|---|---|---|---|---|
| `place_rand-s_rand-t_100-demos_2026-09-04...` | **0.56** | 0.0089 | 0.1946 / 0.2262 / 0.2382 | 0.0608 median |
| `dominos_4-base_2026-09-11` | 0.00 | 0.0256 | 0.1327 / 0.1330 / 0.1413 | 0.0451 median |

The 0.56 figure is `eval/success` in wandb group
`place_rand-s_rand-t_100k-600k_sparse-True`; the domino figure is the whole
domino sweep, including the `critic_off` pure-BC arm, which placed no piece at
all. The set that works reads a held-out error 1.5 to 1.8 times *worse* than
the set that does not, and both are far above 0.0766. So held-out interior
error does not order these sets by outcome, and a bar drawn anywhere between
them would reject the only set known to train.

The three-way verdict is withdrawn with the bar. Its middle branch,
`train_interior < 0.5 * held_interior`, fired on both sets and told the reader
that the demonstrations were "memorized, not learned" and that "more of the
same will not help" — advice that, on the Place set, would have thrown away the
only working data in the repository.

What the numbers are still good for is a *scale*, so the gate now prints both
readings next to any new one and leaves the judgement to the reader. A set that
reads much worse than both, or whose fitted error stays near its held-out
error, is saying something. A set that lands between them is saying nothing yet.

## Consequences

- The gate is a measurement instrument, not pipeline code. Its whole shape:
  observation in, the `horizon_length` action chunk out, the same body,
  learning rate and batch size as the flow actor, standardized observations,
  and a cosine learning-rate schedule, because a constant rate makes the tail
  oscillate. An earlier implementation shipped as `loot_bench.bc` and the
  `loot-fit-bc` command, and it was lost with its branch.
- **Amended: the gate now lives at `tests/qc/bc_gate.py`.** It sits in `tests/`
  because it measures, and it is not part of the package. It imports qc's own
  `MLP` body, so the regressor and the flow actor cannot drift apart, and it
  therefore runs from the **qc venv** with both checkouts on `PYTHONPATH`, not
  under `uv run` and never under pytest:

  ```bash
  PYTHONPATH=<loot-bench>:<qc> <qc>/.venv/bin/python tests/qc/bc_gate.py \
      out/qc/demos.npz
  ```

  **Amended (2026-09-12): there is no `--target` and no verdict exit status.**
  The gate prints the fitted error, the held-out error, the demonstrations' own
  `|a[t+1] - a[t]|` on the same statistic and scale, and the two reference
  readings in the table above; the exit status is `0` unless the read itself
  fails. The default fit is 50k steps, not the 300k quoted above. Everything
  the gate decides before the fit is plain array work: the episode split, the
  chunk windows, the standardizer. Each of those is silent when it is wrong, so
  `tests/qc/test_bc_gate.py` unit-tests them without JAX.
- The gate cannot tell you whether a policy will succeed. A fit 400 times
  better than the actor still scored 0 successes in 20 rollouts, and the 2026-
  09-12 readings show the relation running backwards across two sets. It tells
  you whether the demonstrations are fittable and whether they carry to an
  episode that was not fitted, and nothing beyond that.
- **A failing gate read is not a reason to re-record.** It was used that way,
  and on the domino set it pointed at the data while the actual finding lay
  elsewhere: the critic extrapolates at the actor's own out-of-distribution
  action chunks, which is what drives Q positive on a reward that is bounded
  above by zero. Rule the learner out before blaming the recording.
