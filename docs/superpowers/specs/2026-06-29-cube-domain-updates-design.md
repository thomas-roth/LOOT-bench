# Design: Cube Domain Updates

**Date:** 2026-06-29
**Branch:** task/cube
**Scope:** Three improvements to the cube task domain, with reward-mode infrastructure extended to all three task domains.

---

## 1. Reward Mode CLI Wrapper

### Problem
All reward terms are always active during training. There is no way to select sparse vs. dense shaping without editing source files.

### Solution
A thin `--reward-mode` wrapper in `g3a_bench/scripts/train.py` that mutates the registered task config before forwarding to mjlab's training runner.

### Modes
| Mode | Description |
|------|-------------|
| `success` | Only the sparse success signal + regularization |
| `progress` | Per-substep sparse reward + success + regularization |
| `dense` | Full shaping including distance/staged terms (default) |

Default is `dense` so that the GELLO demo pipeline (which calls `load_env_cfg` directly, bypassing the wrapper) records the richest reward signal by default.

### Per-domain term mapping

**Regularization terms** (always active in every mode): `action_rate_l2`, `joint_pos_limits`, `joint_vel_hinge`.

#### Cube (`cube_env_cfg.py`)
All three tiers of reward terms are already in the registered config.

| Mode | Active task terms |
|------|------------------|
| `success` | `success` |
| `progress` | `placed_slots`, `success` |
| `dense` | `staged_next_slot`, `placed_slots`, `success` |

#### Ball (`ball_env_cfg.py`)
Action-specific terms (`strike`, `reach`, `contact_time`, `push_guide`, `throw_height`) are variant-specific; zeroing a term not present in the config is a no-op.

| Mode | Active task terms |
|------|------------------|
| `success` | `success` |
| `progress` | `ball_distance`, `success` |
| `dense` | `ball_distance`, `success`, all action terms |

#### Insertion (`insertion_env_cfg.py`)
The dense shaping terms (`staged_next_socket`, `insertion_depth`, `sheet_alignment`) currently exist in `mdp/rewards.py` but are **not** in the active config. To support `dense` mode and make the GELLO default richer:
- Add all three dense terms to `make_insertion_env_cfg`'s rewards dict at their design-time weights (mirroring the cube pattern: `staged_next_socket` weight ≈ 1.0, `insertion_depth` weight ≈ 1.0, `sheet_alignment` weight ≈ 1.0).
- `sheet_alignment` uses an `asset_cfg` placeholder (grasp site, filled by `wire_manipulator_into_env`). Since the function returns zeros when no sheet is present, it is safe to include it for all variants.

| Mode | Active task terms |
|------|------------------|
| `success` | `success` |
| `progress` | `inserted_count`, `success` |
| `dense` | `staged_next_socket`, `insertion_depth`, `sheet_alignment`, `inserted_count`, `success` |

### Convention (`REWARD_MODE_TERMS`)
Each domain's `<domain>_env_cfg.py` exports a module-level constant:
```python
REWARD_MODE_TERMS: dict[str, frozenset[str]] = {
    "success":  frozenset([...]),
    "progress": frozenset([...]),
    "dense":    frozenset([...]),
}
```
The wrapper imports this dict, then for every reward term in the registered config that is **not** in `REWARD_MODE_TERMS[mode]`, it sets `weight = 0.0`.

### Train wrapper implementation (`g3a_bench/scripts/train.py`)
```
1. Pre-parse --reward-mode (argparse, add_help=False) and task_id (first non-flag argv)
   before any g3a_bench/mjlab imports.
2. Strip --reward-mode from sys.argv.
3. import g3a_bench  (populates _REGISTRY)
4. If reward_mode != "dense": look up domain from task_id ("Cube"/"Ball"/"Insertion"),
   import its REWARD_MODE_TERMS, and zero out non-active weights in
   _REGISTRY[task_id].env_cfg.rewards.
5. from mjlab.scripts.train import main; main()
```

`_REGISTRY` stores the *original* config object (not a copy); `load_env_cfg` deep-copies on read. So mutating `_REGISTRY[task_id].env_cfg` before `main()` is called causes `main()`'s internal `load_env_cfg` call to see the mutated version.

### Non-goal
`--reward-mode` applies only to the training config. The play/GELLO config is unaffected (full dense rewards always available for demo recording).

---

## 2. Cube Manipulator Wiring

### Problem
`make_cube_env_cfg(variant, robot, play)` hardcodes the robot directly, unlike the insertion task which uses a clean placeholder + wiring pattern. This makes testing a robot-agnostic cube cfg impossible without a concrete robot.

### Solution
Mirror the insertion task's architecture exactly.

### Changes

#### `g3a_bench/tasks/cube/cube_env_cfg.py`
- Remove `robot: ManipulatorInterface` parameter from `make_cube_env_cfg`.
- Replace all direct `robot.*` accesses with placeholders:
  - `robot.entity_cfg_fn()` → omit (no `"robot"` entity added here)
  - `robot.grasp_site` → `site_names=()` in grasp `SceneEntityCfg`
  - `robot.action_scale` → placeholder scalar `0.5`
  - `robot.finger_pad_geoms` → placeholder `geom_names=""` in friction event
  - `robot.ee_subtree` → placeholder `pattern=""` in contact sensor
  - `robot.viewer_body` → placeholder `body_name=""`

#### `g3a_bench/tasks/cube/manipulator_wiring.py` (new file)
```python
def wire_manipulator_into_env(
    cfg: ManagerBasedRlEnvCfg,
    manipulator: ManipulatorInterface,
) -> None:
    """Fill robot-specific placeholders left by make_cube_env_cfg."""
```
Fills: robot entity, action scale (joint_pos), grasp site on `pieces_rel_grasp` obs and `staged_next_slot` reward, finger-pad geoms, ee subtree on contact sensor, viewer body. Mirrors `tasks/insertion/manipulator_wiring.py`.

Grasp-site terms to update:
- Observations: `pieces_rel_grasp` (in both actor and critic groups)
- Rewards: `staged_next_slot`

#### `g3a_bench/tasks/cube/config/__init__.py`
Add `from g3a_bench.tasks.cube.manipulator_wiring import wire_manipulator_into_env` and call it after each `make_cube_env_cfg(...)` call for both training and play configs.

---

## 3. Instant Reset with Stricter Success Condition

### Problem
The current success termination waits `hold_time=1.0 s` with all pieces at goal. The policy can hold pieces at the goal without truly placing them. The harder condition — pieces stably at rest, gripper released — is not checked.

### Solution
Replace the single position check + hold time with three simultaneous conditions checked each step; terminate on the first step all three are satisfied.

### Three conditions
1. **Position**: all goal-piece positions within `pos_threshold` (existing).
2. **Velocity**: max linear speed of any goal piece ≤ `piece_vel_threshold` (new, default `0.05 m/s`).
3. **Gripper open**: `gripper_left` joint position > `gripper_open_threshold` (new).

**Gripper range:** `gripper_left` is a slide joint, range `[-0.045, 0.001]` m. `0.0` = home/open; `-0.045` = fully closed. Default threshold: `-0.015 m` (gripper at least 2/3 open).

### Changes

#### `g3a_bench/tasks/cube/mdp/commands.py` (`PieceArrangementCommandCfg` + `PieceArrangementCommand`)

New fields on `PieceArrangementCommandCfg`:
```python
piece_vel_threshold: float = 0.05   # m/s — max goal-piece speed to count as placed
gripper_open_threshold: float = -0.015  # m — min gripper_left joint pos (open check)
```

In `PieceArrangementCommand.__init__`: resolve gripper joint index from `env.scene["robot"]` joint names and cache as `self._gripper_joint_idx`.

In `_update_metrics()`: after the existing position (and domino orientation) check:
```python
# Velocity gate: all placed pieces moving slowly.
goal_piece_vel = <root_link_vel_w of pieces at their slots>  # (E, G, 3)
pieces_still = goal_piece_vel.norm(dim=-1).max(dim=-1).values < cfg.piece_vel_threshold  # (E,)

# Gripper gate.
gripper_q = robot.data.joint_pos[:, self._gripper_joint_idx]  # (E,)
gripper_open = gripper_q > cfg.gripper_open_threshold  # (E,)

all_conditions = at_goal.all(dim=1) & pieces_still & gripper_open
self.success_steps = (self.success_steps + 1.0) * all_conditions.float()
```

`slot_at_goal` (per-slot position flag) is **not** modified — it stays a pure position metric used by observations and the `placed_slots` reward. Only `success_steps` uses the stricter combined condition.

#### `g3a_bench/tasks/cube/mdp/terminations.py`
```python
def arrangement_complete(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """All conditions met for at least one step."""
    return get_cube_command(env, command_name).success_steps >= 1
```
No `hold_time` parameter. The `TerminationTermCfg` in `cube_env_cfg.py` no longer passes `hold_time`.

---

## Files changed

| File | Change |
|------|--------|
| `g3a_bench/scripts/train.py` | Add `--reward-mode` wrapper |
| `g3a_bench/tasks/cube/cube_env_cfg.py` | Remove robot param, add placeholders, add `REWARD_MODE_TERMS` |
| `g3a_bench/tasks/cube/manipulator_wiring.py` | **New** — `wire_manipulator_into_env` |
| `g3a_bench/tasks/cube/config/__init__.py` | Call `wire_manipulator_into_env` |
| `g3a_bench/tasks/cube/mdp/commands.py` | Add velocity + gripper conditions to `_update_metrics` |
| `g3a_bench/tasks/cube/mdp/terminations.py` | Simplify `arrangement_complete`, remove `hold_time` |
| `g3a_bench/tasks/ball/ball_env_cfg.py` | Add `REWARD_MODE_TERMS` |
| `g3a_bench/tasks/insertion/insertion_env_cfg.py` | Add dense shaping terms to rewards, add `REWARD_MODE_TERMS` |

## Non-goals
- EEF action mode and camera observations: not in scope (deferred).
- Ball and insertion: point 2 (wiring) and point 3 (instant reset) are cube-only.
- No changes to `g3a_bench/scripts/play.py` or the GELLO live-record scripts.
