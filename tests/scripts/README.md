# Script tests

Tests for the standalone tools in `loot_bench/scripts/` that are not the play
loop. Right now that is the demo splitter, `split_demos.py`, and the session
fuse, `fuse_demos.py` (`test_fuse_demos.py`: sidecar merge rules, episode
reindexing, fused-root converts). Run from the **repo root**.

```bash
uv run pytest tests/scripts/ -v
```

Expected: all green, 27 passed, about 15 seconds. No simulator is built and no
recorded demo is read.

---

## What `test_split_demos.py` covers

Every episode is synthetic. No demo of a splittable category is committed, and
a fabricated one pins the contract more sharply than a recording would: it can
put the state exactly on a threshold, hold everything else fixed, and assert a
known answer.

| Group | What it pins |
| --- | --- |
| `TestTaskIds` | Which tasks a recording can be split into. Stack is one family (`3Pis` -> `2Pis`, `1Pi`), Domino one per row mode, Sorter one per spawn mode; `KnockOver` is not splittable at all. |
| `TestPlanValidation` | The plan refuses a harder target, a different family, and an unsplittable source. |
| `TestPiRecompute` | Progress is the fraction of levels with a horizontal, unheld lintel at that level's height (through the env's own kernels, so the tilt gate is never copied), a lintel in the closed gripper earns no level, a tier-order build cuts one frame past each target's first success, a white-first build relabels to nothing, and the truncation keeps the proprioceptive head plus the first-k blocks of obs/qpos/qvel. |
| `TestSkipsAreFatal` | A source episode that never pays a target's success fails the run *after* the good episodes are written, names the episode and the blocks that formed its level 1, exits non-zero from `main`, and `--allow-skips` demotes it to a warning. Proven for Domino too. |
| `test_domino_recompute_*` | The row yaw is recovered from the slot quaternion across the whole sampled range, including the negative angles `RandomRow` produces. |
| `test_sorter_*` | The cube's fixed-width pose rides in the head (not a truncatable section) while remaining a leading free body in qpos. |
| `test_lagged_rewards_*` | A success is paid one frame late, on the frame that also terminates (ADR-006). |
| `test_write_roundtrip` | The written LeRobot dataset has the truncated widths, one `done`, and the session's sidecar metadata carried over (`split_from` is `Stack-3Pis`). |

## A never-successful relabel is fatal, by default

An easier task's rollout is only embedded in a harder recording if the
operator followed the collection order: a pi's level 1 from the navy blocks
(0-2), level 2 from teal (3-5), level 3 from white (6-8), bottom-up. Build
level 1 from the white blocks and the `1Pi` target cuts those blocks out of
the observation, so its success is never paid and the episode is not a demo of
it. The splitter used to count such episodes silently; a whole session could
yield a near-empty dataset.

Now `split_dataset` writes every relabelable episode of every target first,
then raises one `ValueError` that lists, per target, the skipped source
episodes and a per-episode diagnostic (for pi: which block was the lintel
of the first level 1 and which blocks stood under it), and `main` exits
non-zero. `--allow-skips` keeps the old warning-plus-count behaviour for a
session whose loss is understood; it has to be typed. The `disturbed` path (a
dropped object moved before the cut) stays a warning.

## The self-check is the real test

`relabel_episode` recomputes the *source* episode's reward and asserts it
matches the recording. Every fixture here builds its reward column to satisfy
it, so the check runs on every relabel above — and
`test_self_check_rejects_a_mismatched_reward_column` proves it actually bites.

That check is what will catch a real drift: it compares against data the
simulator produced, so a changed observation layout, gripper qpos index, open
threshold or success kernel fails on the first real recording rather than
silently writing a corrupt dataset. None of these synthetic tests can do that
on their own.
