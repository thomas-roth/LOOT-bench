# The episode boundary and the bootstrap mask are separate columns

In a converted dataset, `terminals` marks an episode boundary and `masks` marks
whether to bootstrap past a transition. Every episode end is a boundary; only a
success is absorbing. The old converter wrote `masks = 1 - terminals`, which
collapses the two.

## Why

`qc`'s `sample_sequence` keys its `valid` mask off `terminals` alone, so an
unmarked boundary lets a sampled action chunk walk out of one episode into the
next. And a zero mask at a timeout tells the learner that running out of time is
worth exactly as much as reaching the goal, which an offline critic cannot
recover from.

## The timeout row

A timed-out episode ends on a boundary that is not absorbing, so its final row
needs a next state to bootstrap into, and the recorder never stored one. The
old converter zero-filled that observation *and* set `masks = 0`, so the two
faults hid each other: the zeros were never read because the mask was wrong.
The converter now drops the unusable row. The row before it bootstraps into the
dropped row's observation, which is a real state.

## Consequences

- A successful episode's final row keeps `masks = 0` and points at its own
  observation as its successor. `qc` still feeds that successor through the
  critic before multiplying by the mask, so pointing at the row's own
  observation costs the same as zeros and puts nothing out of distribution into
  the network.
- The converter refuses a recording that predates ADR-006: a reward above 0,
  below -1, or exactly 0 anywhere but an episode's final row raises at
  conversion time.
- Episode structure is read off row order, so the converter sorts rows by
  `(episode, frame)` before anything looks at them.
