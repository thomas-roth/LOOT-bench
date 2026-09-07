# Script tests

Tests for the standalone tools in `loot_bench/scripts/` that are not the play
loop. Right now that is the demo splitter, `split_demos.py`. Run from the
**repo root**.

```bash
uv run pytest tests/scripts/ -v
```

Expected: all green, 21 passed, about 15 seconds. No simulator is built and no
recorded demo is read.

---

## What `test_split_demos.py` covers

Every episode is synthetic. No demo of a splittable category is committed, and
a fabricated one pins the contract more sharply than a recording would: it can
put the state exactly on a threshold, hold everything else fixed, and assert a
known answer.

| Group | What it pins |
| --- | --- |
| `TestTaskIds` | Which tasks a recording can be split into. A family is one Stack difficulty, one Domino row mode, one Sorter spawn mode; `KnockOver` is not splittable at all. |
| `TestPlanValidation` | The plan refuses a harder target, a different family, and an unsplittable source. |
| `TestStackRecompute` | Progress is the tallest block's height fraction, a block in the closed gripper is excluded from it, and the truncation keeps the proprioceptive head plus the first-k blocks of obs/qpos/qvel. |
| `test_domino_recompute_*` | The row yaw is recovered from the slot quaternion across the whole sampled range, including the negative angles `RandomRow` produces. |
| `test_sorter_*` | The cube's fixed-width pose rides in the head (not a truncatable section) while remaining a leading free body in qpos. |
| `test_lagged_rewards_*` | A success is paid one frame late, on the frame that also terminates (ADR-006). |
| `test_write_roundtrip` | The written LeRobot dataset has the truncated widths, one `done`, and the session's sidecar metadata carried over. |

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
