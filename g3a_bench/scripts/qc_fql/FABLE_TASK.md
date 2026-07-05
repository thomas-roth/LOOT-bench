# Task for Claude Fable: audit, fix, re-document, and productionize the QC-FQL port

## Who you are and what this is

You're being handed a JAX→PyTorch port of an offline-RL algorithm
(**QC-FQL**, Action-Chunked Flow Q-Learning) that a previous Claude session
built incrementally, one module at a time, over a long conversation. That
session is over — you have no memory of it, and you should not take its
claims on faith. Everything in this file is written so you can execute cold.

**Your mandate, in one sentence:** verify this port is actually correct by
running it yourself (not by reading comments or trusting a README), fix
whatever you find wrong, throw out every existing comment/docstring and
write new ones from your own understanding, wire up a real CLI entry point
so this is as easy to invoke as the repo's existing PPO workflow, and report
back on what you found and how to use it.

You are authorized to make real code changes — this is not a read-only
audit. "Improve the code" is a literal instruction, not a figure of speech.

## Where things live

- **Reference implementation (ground truth):** `qc/`, a sibling repository
  at `../../../qc/` relative to this file's directory
  (`g3a_bench/scripts/qc_fql/`). This is the original JAX/Flax implementation
  the port is derived from. **Never modify anything under `qc/`** — treat it
  as read-only ground truth for comparison.
- **The port itself:** `g3a_bench/scripts/qc_fql/` (this directory) — `networks/`,
  `data/`, `agent/` (singular — see "traps" below), `scripts/`, `tests/`.
- **The target project:** `g3a-bench`, a suite of MuJoCo/mjlab robot-manipulation
  benchmark tasks. It already has a working PPO training pipeline (`g3a-train`/
  `g3a-play` console scripts, see `pyproject.toml` and `g3a_bench/scripts/train.py`/
  `play.py`). Your job includes making QC-FQL feel like a sibling of that
  workflow, not a script only someone who read the source would know how to run.
- **`g3a_bench/scripts/qc_fql/README.md`**: the previous session's own account
  of what it built and why, phase by phase. Read it for orientation, but
  **verify its claims rather than assume them** — if you find it's stale or
  wrong anywhere, fix it (don't just silently leave it lying).

### Dev-only dependencies

`jax`, `flax`, `distrax`, and `ml_collections` are already installed in
`g3a-bench/.venv`, added via `uv pip install` (**not** via `uv add`, so they
are deliberately absent from `pyproject.toml`/`uv.lock`). They exist purely
so the `tests/gen_gold_*.py` scripts can import the real JAX reference code
and regenerate gold reference data — they are never imported by the actual
PyTorch runtime code. If you hit a missing import while regenerating gold
data, install it the same way (`uv pip install <package>`); do **not** add
it to `pyproject.toml`.

## Task 1 — Confirm completeness

Don't assume the file tree under `g3a_bench/scripts/qc_fql/` is exhaustive or
current just because the README claims a status of "done" for every phase.
Check for yourself:

- Walk `qc/agents/acfql.py` (the reference agent) and everything it imports
  transitively (`qc/utils/networks.py`, `qc/utils/datasets.py`,
  `qc/utils/flax_utils.py`) and confirm every piece of *actually-exercised*
  logic has a corresponding ported file here. `qc/` contains other agents
  (`acrlpd.py`), other environments (d4rl/ogbench/robomimic support), and
  visual-encoder machinery that QC-FQL on this project does not use and are
  correctly out of scope — don't flag those as "missing," but do verify for
  yourself that they're genuinely unused by `acfql.py`'s actual code paths
  rather than trusting that a prior session made that determination correctly.
- Check for anything imported or referenced by the ported code that doesn't
  actually exist, or exists as an empty/stub file.
- Check for anything present that's genuinely dead weight (unused, never
  imported by anything).

Report and fix any gaps you find.

## Task 2 — Verify correctness for real, not by assumption

This is the core of your job. A previous session's tests passing in a
transcript is not evidence they pass now, and is never evidence they were
sound tests in the first place.

1. **Run the whole suite fresh:**
   ```bash
   cd <g3a-bench repo root>
   PYTHONPATH="g3a_bench/scripts/qc_fql:$PYTHONPATH" uv run python -m pytest g3a_bench/scripts/qc_fql/tests/ -v
   ```
   Do not assume this passes. Watch the actual output.

2. **Regenerate every gold-data file from scratch and re-run its test.**
   Each `tests/gen_gold_*.py` script imports the real JAX reference code and
   dumps a `.npz` into `tests/gold_data/`. Delete (or move aside) the
   existing `.npz` files, regenerate them by running the `gen_gold_*.py`
   scripts, and confirm the corresponding `test_*.py` still passes against
   the freshly-generated data — not stale files that might silently no
   longer reflect the current source on either side of the comparison.

3. **Scrutinize whether each test is actually testing something meaningful.**
   Specifically look for:
   - Tolerances loose enough that a subtly wrong implementation would still
     pass.
   - Gold data generated by a hand-written reimplementation of the JAX
     module instead of importing the real class. This exists at least once
     already and is flagged as a pattern **not** to repeat:
     `tests/gen_gold_mlp.py` defines its own `TestMLP` class inline instead
     of importing `qc/utils/networks.py::MLP` directly. It happens to match,
     according to the previous session, but "happens to match a
     hand-transcription" is a materially weaker guarantee than "imports the
     real thing." Verify it still matches; consider fixing it to import the
     real class instead, matching the pattern used everywhere else (e.g.
     `tests/gen_gold_dataset.py` correctly imports `qc/utils/datasets.py::Dataset`
     directly).
   - Mocks or fakes that are never actually exercised by the code path under test.
   - Any test that would pass even if you deliberately broke the
     implementation it's supposedly testing — try this on at least one or
     two tests as a sanity check (temporarily introduce a wrong value, see
     if the test actually catches it, then revert).

4. **Independently re-derive the trickiest claim in this codebase rather
   than trusting it blindly:** `agent/qcfql_agent.py`'s `actor_loss` scores
   the actor's chosen action through the critic, but the critic's own
   weights must receive **zero gradient** from that term (only the actor
   should be trained by it — the critic is trained solely by its own TD
   regression in `critic_loss`). This is implemented via a
   `requires_grad_(False)`/`requires_grad_(True)` toggle around the one
   critic call inside `actor_loss`. There is a test for this,
   `test_critic_only_gets_gradient_from_its_own_loss` in
   `tests/test_qcfql_agent.py`. Read `qc/utils/flax_utils.py`'s
   `TrainState.__call__` docstring yourself (it explains JAX's own
   stop-gradient convention: passing `params=None` uses stored/frozen
   parameters and gradients don't flow) and confirm independently that (a)
   the PyTorch mechanism genuinely achieves the same effect JAX's
   `params=None` convention achieves in `qc/agents/acfql.py::actor_loss`,
   and (b) the test actually would fail if this mechanism were removed (try
   it: comment out the freeze/unfreeze lines, re-run the test, confirm it
   now fails, then restore it). Do not just read this paragraph and accept
   it — reproduce the check yourself.

## Task 3 — Use your tools and skills properly

Don't just read code and declare it correct. Use whatever debugging,
verification, and code-review skills/tools you have available: run things,
watch them fail when they're going to fail, form a hypothesis, test it, fix
the root cause, then verify the fix with fresh output (not by re-reading the
code and reasoning that it should now work). Before claiming anything is
"fixed" or "passing," you must have just run the command that proves it and
read its actual output.

## Task 4 — Fix whatever you find broken

You have real authority to edit code here, not just produce a findings list.
Guardrails while doing so:

- **Any change to `networks/`, `data/`, or `agent/` that could affect
  numerical behavior must be re-verified against the real `qc/` source**
  (regenerate gold data if relevant, re-run the test) before you consider it
  fixed. Don't "fix" something that turns out to be a legitimate design
  choice rather than a bug — check against the reference first.
- **Don't break what currently passes.** Run the full test suite before you
  start changing things and again after, and diff the results.
- Be especially careful with `agent/qcfql_agent.py`'s gradient-isolation
  logic (see Task 2, point 4) — it looks like it could be simplified or is
  "obviously" over-engineered if you don't understand why it's there. It
  isn't. Re-derive the reasoning yourself before touching it.
- The port intentionally drops things that are dead code or unused by
  QC-FQL specifically even though they exist in `qc/` (visual encoders,
  other agents, dataset-format-specific reward adjustments for d4rl/ogbench/
  robomimic, etc.) — don't "restore" these as if their absence were a bug;
  confirm they're actually unused before treating a difference from `qc/` as
  a defect.

## Task 5 — Full re-documentation pass

Delete every existing comment and docstring in every `.py` file under
`g3a_bench/scripts/qc_fql/` — `networks/`, `data/`, `agent/`, `scripts/`,
and `tests/` (including the `gen_gold_*.py` and `test_*.py` files, not just
the port's own source) — and write entirely new ones, from your own
understanding of the code after you've verified it, not copied or lightly
reworded from what's currently there.

For every file, add a top-of-file docstring that explains:
- What this file is.
- Why it exists — what problem it solves in the context of the larger
  QC-FQL port (not just "this is the X module" — say why X needed to exist
  as its own file/class here).
- What it depends on and what depends on it, where that's non-obvious.

For functions and classes, write comments that explain the non-obvious
*why* — a hidden constraint, the reason a particular JAX construct was
translated a particular way, a subtlety that would genuinely surprise a
future reader — not a restatement of what the code already visibly does.
Match this codebase's existing tone: terse, no restating the code, no
multi-paragraph docstring bloat, no comment where the code is already
self-explanatory.

**This pass must not change behavior.** After rewriting comments in a file,
re-run its test(s) and confirm nothing broke. If you find yourself wanting
to change actual logic while rewriting a comment because you've spotted a
real issue, that's Task 4's job — do it deliberately and re-verify, don't
let it slip in accidentally alongside a comment change.

## Task 6 — Wire up a real CLI entry point

Right now, running QC-FQL means invoking `train_qcfql.py` directly with a
`PYTHONPATH` prefix and knowing its internal module layout. The repo's
existing PPO workflow is much friendlier — `uv run g3a-train <task_id>` and
`uv run g3a-play <task_id>` — via console scripts registered in
`pyproject.toml`:

```toml
[project.scripts]
g3a-train = "g3a_bench.scripts.train:main"
g3a-play = "g3a_bench.scripts.play:main"
```

with each pointing at a thin wrapper, e.g. `g3a_bench/scripts/train.py`:

```python
"""Train entry point: mjlab's trainer with the g3a-bench tasks registered."""
from mjlab.scripts.train import main
import g3a_bench  # noqa: F401  (registers benchmark tasks)
if __name__ == "__main__":
    main()
```

Add a matching entry for QC-FQL, pointing at its own `main()` (already
implemented in `g3a_bench/scripts/qc_fql/scripts/train_qcfql.py` — this one
doesn't wrap an mjlab entry point, it's the port's own training loop):

```toml
g3a-train-qcfql = "g3a_bench.scripts.qc_fql.scripts.train_qcfql:main"
```

Then actually verify it: `uv sync`, then `uv run g3a-train-qcfql --help`
should print the argparse help from `train_qcfql.py`'s `build_arg_parser()`.
If the current CLI's argument names, defaults, or required arguments are
awkward for someone actually invoking this from the command line (as
opposed to calling the Python functions directly in a test), improve them —
the bar is "this genuinely feels like a sibling of `g3a-train`," not "this
technically works if you already know the internals."

## Task 7 — Report on correctness and usage

When you're done, produce a clear report covering:

1. **Is the port actually correct?** What did you personally, independently
   verify (re-derived, re-ran, regenerated gold data for), versus what you
   are still relying on prior tests for because you didn't find a
   independent way to check it? Be honest about the difference — "I ran the
   existing test and it passed" is weaker evidence than "I regenerated gold
   data from the real JAX source and independently confirmed the match,"
   and both are weaker than "I additionally tried breaking it and confirmed
   the test catches the break."
2. **What's genuinely still missing** for training on the actual target
   task, `G3a-Cube-Stack3-Dk1_uni` (DK1 unimanual robot, cube-stacking). As
   of this writing, no converted demonstration dataset (`.npz`) exists yet
   for this task — recording demos via the project's `live_bridge` tooling
   and converting them via `g3a_bench/scripts/live_bridge/lerobot_to_qc.py`
   is a separate, manual prerequisite this task does not cover. Confirm this
   is still true (or note if it's changed) rather than assuming.
3. **How to actually use this now**, end to end, in terms comparable to the
   existing PPO workflow: what `g3a-train-qcfql` needs as arguments, what a
   valid dataset path/format looks like, what a successful offline
   pretraining run looks like, and what's explicitly *not* supported yet
   (online fine-tuning requires a recorded dataset to seed the replay
   buffer from; multi-env vectorized training is out of scope by design —
   this is a single-env algorithm, not a limitation to "fix").
4. Update `g3a_bench/scripts/qc_fql/README.md` if you find its claims are
   stale or wrong anywhere — it should remain an accurate account after
   your pass, not a frozen snapshot of the previous session's beliefs.

## What NOT to do

- **Never modify anything under `qc/`** — it's read-only ground truth.
- **Don't add multi-env/vectorized support** or otherwise change the
  single-env design decision. That's a deliberate scope boundary, not an
  oversight.
- **Don't expand scope to other robots or task categories** (insertion,
  ball, bimanual). This pass is about correctness and usability of what
  already exists for the cube task / DK1 unimanual robot, not growing scope.
- **Don't add `jax`/`flax`/`distrax`/`ml_collections` to `pyproject.toml`** —
  they must stay dev-only, installed manually into the venv, exactly as
  described above.
- **Don't rename `agent/` (singular) to `agents/` (plural).** This looks like
  an inconsistency or typo at first glance — it is deliberate. The reference
  repo `qc/` has its own top-level `agents/` package
  (`qc/agents/acfql.py`, etc.), and Python caches imports by top-level
  package name across a whole process — if this port's own agent package
  were also named `agents`, whichever one gets imported first would
  silently shadow the other for the rest of that process (this was an
  actual bug a previous session hit and fixed by renaming to the singular
  `agent/`). Verify this reasoning yourself if you doubt it — trying to
  rename it back and running the full test suite in one process will
  reproduce the failure.
