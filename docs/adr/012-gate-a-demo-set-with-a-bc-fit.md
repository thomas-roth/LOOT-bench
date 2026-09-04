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

## The bar

Read the held-out error, not the training error, and state which number is
being used. Two reference points existed on the old 100 demonstrations, and
they are not interchangeable: 0.118 when the 530-frame first episode is
skipped, and 0.166 when all 10 held-out episodes are read. Both read 0.0013 on
trained episodes, and both were flat from 25k steps.

**Amended.** The gate is written again, as `tests/qc/bc_gate.py`, and **the bar
is now 0.0766**. Neither 0.118 nor 0.166 is the bar any more, and neither can
be compared with the new number, because the three were read differently. The
old two took the holdout from the front of the session, stopped the read at
frame 500, and used a constant learning rate over 25k steps. The gate holds out
at random, because the first attempts of a session are the clumsy ones, and it
reads the whole body of every held-out episode.

0.0766 is measured, not inherited. It is what this gate reads on the same
legacy file the old numbers came from, the pre-rebuild export
`G3A-Place-Easy-Dk1_uni_qc_physical.npz`, with ten episodes held out. The same
fit read 0.0075 on the episodes it saw. That file is a single legacy export,
**not** the committed sessions under
`dataset/demos/G3A-Place-RandS-RandT-Dk1_uni`: that directory has grown past
the set the number was read on, so the gate will not reproduce 0.0766 there.

## Consequences

- The gate is a measurement instrument, not pipeline code. Its whole shape:
  observation in, the `horizon_length` action chunk out, the same body,
  learning rate and batch size as the flow actor, standardized observations,
  and a cosine learning-rate schedule, because a constant rate makes the tail
  oscillate. An earlier implementation shipped as `g3a_bench.bc` and the
  `g3a-fit-bc` command, and it was lost with its branch.
- **Amended: the gate now lives at `tests/qc/bc_gate.py`.** It sits in `tests/`
  because it measures, and it is not part of the package. It imports qc's own
  `MLP` body, so the regressor and the flow actor cannot drift apart, and it
  therefore runs from the **qc venv** with both checkouts on `PYTHONPATH`, not
  under `uv run` and never under pytest:

  ```bash
  PYTHONPATH=<g3a-bench>:<qc> <qc>/.venv/bin/python tests/qc/bc_gate.py \
      out/qc/demos.npz
  ```

  The exit status is the verdict: `0` when the held-out interior error beats
  `--target` (default 0.0766), `1` when it does not. The default fit is 50k
  steps, not the 300k quoted above. Everything the gate decides before the fit
  is plain array work: the episode split, the chunk windows, the standardizer.
  Each of those is silent when it is wrong, so `tests/qc/test_bc_gate.py`
  unit-tests them without JAX.
- The gate cannot tell you whether a policy will succeed. A fit 400 times
  better than the actor still scored 0 successes in 20 rollouts. It tells you
  whether the demonstrations are fittable and whether they carry to an episode
  that was not fitted.
