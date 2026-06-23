# Scene Task

A multi-task, long-horizon manipulation benchmark built on [OGBench](https://ogbench.github.io/) conventions. One environment contains five manipulable objects — a cube, two push-buttons, a drawer, and a sliding window — and the agent must achieve a task-specific goal configuration of all five.

## Task IDs

| ID | Description |
|----|-------------|
| `Mjlab-Scene-Yam` | Proprioceptive observations only |
| `Mjlab-Scene-Yam-Rgb` | + 32×32 RGB camera |
| `Mjlab-Scene-Yam-Depth` | + 32×32 depth camera |

## The Five Tasks

Each episode draws one task uniformly at random per environment. The task determines the initial and goal configuration of every object.

| Task | Goal |
|------|------|
| `task1_open` | Open drawer and window (both buttons already active) |
| `task2_unlock_and_lock` | Close drawer and window (unlock by pressing buttons, then push closed) |
| `task3_rearrange_medium` | Move cube, toggle button 1, open drawer, close window |
| `task4_put_in_drawer` | Place cube inside the open drawer; button 0 unlocks it |
| `task5_rearrange_hard` | Put cube in drawer and open the window (no button assistance) |

## Reward

OGBench sparse reward: `num_successes − 5`, where success is checked per object:

- **Cube**: within 4 cm of goal XYZ
- **Button 0 / 1**: logical state matches target (toggle-based, not physical position)
- **Drawer**: joint position within 4 cm of target (range `[−0.16, 0]` m)
- **Window**: joint position within 4 cm of target (range `[0, 0.2]` m)

Reward ranges from −5 (nothing correct) to 0 (everything correct). `episode_success` metric tracks whether the full task was achieved at any point in the episode.

## Button Toggle Logic

Buttons are **toggle switches**, not momentary contacts. Pressing a button (joint position ≤ −0.02 m) flips its logical state between 0 and 1. This is implemented via falling-edge detection in `SceneTaskCommand._update_command`:

```
prev_pos > threshold AND cur_pos <= threshold  →  toggle
```

The logical state persists in `scene_button_states` on the command term. The physical button joint springs back to its rest position after release.

**Lock mechanism**: when a button's logical state is 0, the corresponding joint (drawer or window) is clamped to its initial position each step via `write_joint_position_to_sim`. This simulates a physical lock.

## Entities

All entities are loaded from XML assets in `assets/` and positioned relative to each environment's origin:

| Entity | Asset | Key joints / sites |
|--------|-------|--------------------|
| Robot (YAM) | mjlab asset zoo | `joint1`–`joint6`, `left_finger`, `right_finger`, `grasp_site` |
| Cube | `assets/cube.xml` | freejoint (6-DOF) |
| Buttons | `assets/buttons.xml` | `buttonbox_joint_0`, `buttonbox_joint_1` (slide, spring-loaded) |
| Drawer | `assets/drawer.xml` | `drawer_slide` (Y-axis, range `[−0.16, 0]`) |
| Window | `assets/window.xml` | `window_slide` (Y-axis, range `[0, 0.2]`) |

Hardcoded offsets from env origin (set in XML):

- Buttons: (0.58, ∓0.05, 0.048)
- Drawer: (0.33, −0.42, 0.084)
- Window: (0.3, 0.3, 0.202)

## Observation Space

The `actor` and `critic` groups share the same terms (critic has no added noise):

| Term | Shape | Description |
|------|-------|-------------|
| `joint_pos` | (7,) | Robot joint positions relative to default |
| `joint_vel` | (7,) | Robot joint velocities relative to default |
| `actions` | (8,) | Last action (7 arm + 1 gripper) |
| `ee_pos` | (3,) | EE site position in robot base frame |
| `ee_yaw` | (2,) | cos/sin of EE yaw |
| `gripper_opening` | (1,) | Sum of finger joint positions |
| `ee_to_cube` | (3,) | Vector from EE to cube in base frame |
| `cube_to_goal` | (3,) | Vector from cube to cube goal in base frame |
| `cube_yaw` | (2,) | cos/sin of cube yaw |
| `button_0_pos` | (1,) | Physical joint position of button 0 |
| `button_0_vel` | (1,) | Physical joint velocity of button 0 |
| `button_0_pressed` | (1,) | 1 if physically pressed, else 0 |
| `button_1_pos` | (1,) | Physical joint position of button 1 |
| `button_1_vel` | (1,) | Physical joint velocity of button 1 |
| `button_1_pressed` | (1,) | 1 if physically pressed, else 0 |
| `logical_button_states` | (2,) | Persistent toggle state of both buttons |
| `drawer_pos` | (1,) | Drawer slide joint position |
| `drawer_vel` | (1,) | Drawer slide joint velocity |
| `ee_to_drawer_handle` | (3,) | Vector from EE to drawer handle in base frame |
| `drawer_handle_yaw` | (2,) | cos/sin of drawer handle yaw |
| `drawer_to_goal` | (1,) | `target_drawer_pos − current_drawer_pos` |
| `window_pos` | (1,) | Window slide joint position |
| `window_vel` | (1,) | Window slide joint velocity |
| `ee_to_window_handle` | (3,) | Vector from EE to window handle in base frame |
| `window_handle_yaw` | (2,) | cos/sin of window handle yaw |
| `window_to_goal` | (1,) | `target_window_pos − current_window_pos` |
| `target_button_states` | (2,) | Goal logical button states |

Total actor observation dimension: **≈ 58** (before any actor noise).

The command vector (shape `(E, 7)`) concatenates: `target_cube_pos(3) ‖ target_button_states(2) ‖ target_drawer_pos(1) ‖ target_window_pos(1)`. It is stored on the command term but not directly included as an observation term (goals are conveyed via the per-object delta observations and `target_button_states`).

## Sim Parameters

| Parameter | Value |
|-----------|-------|
| Timestep | 0.005 s |
| Decimation | 4 (20 Hz control) |
| Episode length | 20 s |
| `nconmax` | 55 |
| `njmax` | 600 |

## Structure vs Cube Task

| Aspect | Cube task | Scene task |
|--------|-----------|------------|
| Task ID prefix | `G3a-` | `Mjlab-` |
| Robot | DK1 | YAM |
| Task variants | `variants.py` dataclass, registered per robot | 5 tasks encoded in `_SCENE_TASKS` list inside `commands.py` |
| Asset building | Procedural `MjSpec` builders in `assets.py` | XML files in `assets/` loaded via `MjSpec.from_file()` |
| Fixed-base positioning | `EntityCfg(init_state=InitialStateCfg(pos=...))` | XML body positions relative to mocap root (automatic) |
| `config/__init__.py` | Loops over variants × robots | 3 direct `register_mjlab_task` calls |
| Camera support | No | Yes (Rgb / Depth variants) |
| Training entry point | `g3a-train` | `mjlab train` |

The scene task intentionally departs from the cube pattern because it is a **multi-task** benchmark rather than a single-goal task. All 5 sub-goals are tracked simultaneously and the reward signal is their aggregate count, following the OGBench specification.
