# Domino Task — Detailed Analysis

This note explains how the `Domino` task category actually works, mechanism
by mechanism, and root-causes a specific inconsistency observed during
manual testing: placing a domino "correctly" sometimes registers as success
(marker turns green) and sometimes does nothing, even when it looks the same
to the eye.

## 1. What the task is

Build a row of `n` dominoes standing upright, close enough together
(`SPACING = 0.05` m, less than a domino's height) that toppling one could
knock over the next. Every registered variant additionally requires a
`KnockOver` follow-through (see §3) only if the `KnockOver` modifier is set.

Files: `g3a_bench/tasks/domino/{variants.py, domino_env_cfg.py, assets.py}`,
`g3a_bench/tasks/domino/mdp/{commands.py, success.py, rewards.py,
observations.py, terminations.py}`.

## 2. Variants and modifiers

A variant (`DominoVariantCfg`, `variants.py:49-77`) is `num_dominoes` (2, 3,
or 4) plus four independent knobs:

| Field | Default | Effect |
| --- | --- | --- |
| `different` | `False` | Dominoes get progressively **thinner and lighter** along the row (`_THICKNESS_DECAY = 0.8` per index step, applied to `hx` and mass). Height/width and **color are unchanged** — see §5. |
| `knock_over` | `False` | After the row is fully built, the row must additionally be **toppled as a chain** (§3). |
| `ordered` | `True` | Whether slot assignment is by fixed domino identity or by proximity. **No registered variant sets this to `False`** — see §4. |
| `margin` | `1.0` | Scales the position/tilt success tolerances. Not varied by any registered variant either. |

Currently registered task IDs (`config/__init__.py`, `_task_id_parts`):

| Task id (`-Dk1_uni` suffix) | `num_dominoes` | `different` | `knock_over` | `ordered` |
| --- | --- | --- | --- | --- |
| `G3A-Domino-{2,3,4}Pieces` | 2/3/4 | No | No | **Yes (fixed)** |
| `G3A-Domino-{2,3,4}Pieces-DifferentDominoes` | 2/3/4 | Yes | No | **Yes (fixed)** |
| `G3A-Domino-{2,3,4}Pieces-KnockOver` | 2/3/4 | No | Yes | **Yes (fixed)** |

`different` and `ordered` interact: `DominoVariantCfg.__post_init__`
(`variants.py:74-77`) asserts `not (different and not ordered)` — different
(i.e. visually/physically distinguishable) dominoes may never be combined
with proximity-based (`ordered=False`) matching, since that combination
wouldn't make sense (see §4). Every one of the 9 registered variants uses
the default `ordered=True`, `different` and `knock_over` independently
toggled, `margin=1.0`.

## 3. How success is computed (build phase)

At every reset (`DominoCommand._resample_command`, `commands.py:239-287`):

1. One shared draw picks a row **start position and yaw** — sized for
   `HARDEST_COUNT = 4` regardless of the variant's actual `n`, so a
   shorter row's start-position distribution never shifts with `n` (a
   deliberate "seed-prefix" reproducibility rule used throughout this
   codebase).
2. Row **slot goal poses** are laid out from that start, one per domino,
   spaced `SPACING` apart along the row direction.
3. All dominoes are **scattered upright** elsewhere in the workspace via
   `ScatterGrid.sample_poses` (`tasks/common/scatter.py:81-109`) — a true
   Fisher-Yates shuffle over grid cells, drawn **in domino-index order**
   (again for reproducibility) but landing on a **uniformly random** cell
   for each index. There is no relationship whatsoever between a domino's
   spawn position and its index or its target slot.

Every step (`DominoCommand._update_metrics`, `commands.py:175-237`):

- Each domino's position/orientation is compared against **the slot
  assigned to it** via `piece_for_slot` (see §4 for what "assigned" means).
- `domino_upright_aligned` (`success.py:18-46`) checks the domino is
  within `upright_threshold_deg` of vertical **and** within the same
  tolerance of the row direction (mod π, so either end can face either way).
- A slot counts as `slot_at_goal` (`placed`) when its assigned domino is
  within `pos_threshold * margin` of the goal position **and** passes the
  upright/alignment check.
- The task-level `task_done` is `all_placed` (every slot placed
  simultaneously) for non-`KnockOver` variants.
- `SuccessGate` (`tasks/common/success.py:64-106`) additionally requires:
  all dominoes' speed ≤ `OBJECT_VEL_THRESHOLD` (0.01 m/s) **and** the
  gripper released (open + no contact) before `success` goes true.
  `episode_success` latches `success` until the next reset.

## 4. The `ordered` flag — identity vs. proximity matching

This is the crux of the reported inconsistency.

`piece_for_slot` (`commands.py:116-118`, `183-190`) is the array mapping
"slot index → which domino is checked against this slot's goal."

- **`ordered=True` (every registered variant, no exceptions):**
  `piece_for_slot` is set once, at reset, to the **identity** mapping
  (`[0, 1, 2, ..., n-1]`) and **never changes for the rest of the
  episode** (`_resample_command`, line 252-254). Slot `i` is checked
  **specifically against domino `i`** — not "whichever domino happens to
  be sitting there."
- **`ordered=False` (implemented, but not exposed by any variant):**
  every step, `piece_for_slot` is recomputed via greedy nearest-neighbor
  matching between the current domino positions and the slot goals
  (`_greedy_match`, `commands.py:292-313`) — **any** domino may fill
  **any** slot. The code comment is explicit: *"No registered variant
  uses this yet; the knob exists for manual experiments"*
  (`commands.py:187-188`).

**So: is there an unordered version? At the code level, yes — the
mechanism is implemented and presumably tested. At the task-selection
level, no — every task id you can actually pick (`G3A-Domino-*`) uses the
fixed identity mapping.** If you want the proximity-matching behavior,
someone would need to register a new variant with `ordered=False` (and,
per the `__post_init__` assertion, it can only be combined with
`different=False`).

## 5. Root cause of "sometimes nothing happens, sometimes it turns green"

Given `ordered=True` is the only mode ever exercised, here is the actual
failure mode:

**Dominoes are visually identical, but the success check is keyed to a
specific, invisible domino identity.**

- `get_domino_spec` (`assets.py:11-26`) always renders a domino with
  `color="orange"` — the call site in `domino_env_cfg.py:32-35` never
  passes a different `color=` per index. Even the `DifferentDominoes`
  modifier only changes thickness/mass, not color (`variants.py:35-46`) —
  and the thickness taper (`0.8^index`) is subtle enough not to be a
  reliable visual index cue during teleop anyway.
- Spawn position gives no clue either (§3): each domino's starting
  position is drawn from a uniformly random grid cell, independent of its
  index or its target slot.
- Therefore: **a human operator has no way to tell, by looking at the
  scene, which physical domino is "domino 0" (must go to slot 0) versus
  "domino 2" (must go to slot 2).**

Concretely, this produces exactly the symptom you saw: if you pick up
whichever domino is most convenient and place it at what looks like the
next open slot, you will satisfy the identity mapping only if you happen
to grab dominoes in the exact index order the task silently expects.
Swap the order — put domino 2 where domino 0 "should" go, and vice versa —
and:

- **Neither slot registers `placed`.** Slot 0's check looks for domino 0
  specifically; domino 2 sitting there doesn't count, no matter how
  precisely it's positioned. Slot 2's check looks for domino 2; domino 0
  sitting there doesn't count either. Two dominoes can be sitting
  exactly on two goal positions, upright and aligned, and **both markers
  stay at their pending/orange color** (see §6) — which reads to a human
  as "I placed it correctly and nothing happened."
- This is the same mechanism behind "so when it was ordered the color
  was still wrong" — placing dominoes in *some* self-consistent order
  isn't the same as placing them in the *specific* index order the task
  fixed at reset. There is no bug in the color logic itself (§6); the
  color is correctly reporting that the wrong domino is at that slot.

This is a genuine task/UX gap, not a training or observation bug: nothing
in the simulation or the viewer currently gives a teleoperator the
information needed to satisfy `ordered=True` reliably, short of reading
this source code before every session.

## 6. Visual feedback (debug markers)

`marker_color` (`tasks/common/debug_markers.py:20-32`) is a simple,
three-state function per slot marker:

| Condition | Color |
| --- | --- |
| Overall command `success` (all slots placed simultaneously + at rest + gripper released) | bright/opaque green (`SUCCESS_RGBA`) |
| This slot's assigned domino is at its goal (`placed`), but overall success not yet reached | medium green (`PLACED_RGBA`) |
| Otherwise | the slot's own base color (orange) at low, "pending" alpha |

`DominoCommand._debug_vis_impl` (`commands.py:315-334`) draws one such
marker per slot, at the goal pose, colored via this function. This logic
is correct and behaves as designed — the confusion in §5 comes entirely
from *which* domino counts toward *which* slot's `placed` state, not from
how the color itself is computed.

## 7. `KnockOver` chain-topple logic (build phase's follow-through)

Only relevant for `*-KnockOver` variants. Once every slot is placed
simultaneously, `phase` latches from 0 (build) to 1 (topple) and never
reverts (`commands.py:210-211`, `torch.maximum`). From then on:

- `domino_fallen_along_row` (`success.py:49-78`) flags a domino as
  "chain-fallen" once it's tilted past `fallen_tilt_deg` (45°) **and**
  leaning along the row axis (within 45° of it) rather than sideways.
- The step at which each domino first counts as chain-fallen is latched
  into `fall_order` — but **only while `phase > 0.5`** (`commands.py:215-221`),
  so a domino that gets knocked over by accident during the *build* phase
  does not count toward the chain.
- `chain_topple_success` (`success.py:81-101`) requires: every domino
  fallen, all leaning the **same direction**, and their first-fall steps
  **monotonically increasing along the row** in that direction (ties
  allowed — neighbors may fall within the same simulation step).
  Sideways or out-of-order toppling does not count.
- Reward during the topple phase is `0.5 + 0.5 * fallen_frac`
  (`commands.py:226-228`) — half credit is already banked for having
  built the row.

## 8. Observations, rewards, terminations (summary)

- **Observations** (`observations.py`): all domino poses (env-local
  position + quaternion, object-major, `(E, G*7)`), a per-slot `placed`
  indicator `(E, G)`, and — `KnockOver` variants only — the build/topple
  `phase` scalar.
- **Rewards**: `progress` (placement fraction while building; for
  `KnockOver`, half placement fraction + half toppled fraction) and a
  sparse `success` bonus while the full success criterion holds — both
  weight 1.0, same pattern as every other task category.
- **Termination**: `domino_task_complete` fires exactly when
  `SuccessGate.success` (§3) is true.

## 9. Open items / recommendations

1. **Before recording live demos on the domino task**, decide how to
   handle §5: either (a) give each domino a distinct, teleop-visible color
   per index (a small `assets.py`/`domino_env_cfg.py` change — pass a
   different `color=` per index into `get_domino_spec`), or (b) register
   an `ordered=False` variant so any domino may fill any slot (already
   implemented, just unused — note the `different=False` restriction).
   Recording demos against the current `ordered=True` + identical-color
   setup risks exactly the same "sometimes nothing happens" confusion
   during teleop that was already observed manually, which would show up
   as inconsistent/mislabeled success signals in the resulting dataset.
2. This is unrelated to the QC-FQL training-pipeline work done on the
   stack task — the domino task's `mdp`/`commands.py` structure otherwise
   follows the exact same shared patterns (`SuccessGate`, `ScatterGrid`,
   `progress`/`success` reward pair) already validated there.
