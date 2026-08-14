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
being used. Two reference points exist on the old 100 demonstrations, and they
are not interchangeable: 0.118 when the 530-frame first episode is skipped, and
0.166 when all 10 held-out episodes are read. Both read 0.0013 on trained
episodes, and both were flat from 25k steps.

## Consequences

- The gate is a measurement instrument, not pipeline code. Its whole shape:
  observation in, the `horizon_length` action chunk out, the same body,
  learning rate and batch size as the flow actor, standardized observations,
  and a cosine learning-rate schedule, because a constant rate makes the tail
  oscillate. An earlier implementation shipped as `g3a_bench.bc` and the
  `g3a-fit-bc` command; it is lost with its branch and has to be written
  again.
- The gate cannot tell you whether a policy will succeed. A fit 400 times
  better than the actor still scored 0 successes in 20 rollouts. It tells you
  whether the demonstrations are fittable and whether they carry to an episode
  that was not fitted.
