# Graph Report - LOOT-bench  (2026-09-09)

## Corpus Check
- 299 files · ~153,997 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2831 nodes · 5254 edges · 202 communities (145 shown, 33 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 274 edges (avg confidence: 0.89)
- Token cost: 444,558 input · 0 output

## Community Hubs (Navigation)
- Success Reward Check
- BC Gate Fit
- ADR Decision Records
- Sim Settings Comparison
- Ball Field Assets
- Velocity Example MDP Terms
- Velocity Rewards and Terrain
- QC Transition Assembly
- Teleop Live Recording
- Teleop Recorder
- Sorter Asset Generation
- Demo Splitter
- Tracking Example Commands
- Demo Converter
- QC Episode Bookkeeping
- Ball MDP Terms
- Peg Insertion MDP Terms
- Peg Insertion Assets and Cfg
- Teleop Delta Mapper
- Native Viewer Wrapper
- Leader to Delta Mapping
- Leader Arm Reader
- Teleop Debug Panel Tests
- Stack Variants and Assets
- Recording Mixin
- Sorter MDP Terms
- Domino Split Relabeling
- Sorter Split Relabeling
- Debug Markers and Goal Commands
- Gripper Gate Command
- Shared Observation Terms
- One Task Reward Kernel
- Scatter Grid Sampler
- Shared Success Gating
- Converter Convention Tests
- Replay Verification
- Action Term Tests
- Domino Registration and Variants
- Domino Success Kernel
- Sorter Assets and Cfg
- Action Terms and Workbench Cfg
- YAM Lift Cube Example
- Manipulation Example Commands
- LootEnv Gym Seam
- Domino MDP Terms
- Recording Mixin Tests
- Tracking Metrics and Eval
- DK1 Constants and Actuators
- Manipulation Example Observations
- Peg Insertion Commands
- Action Convention Tests
- Leader Reader Internals
- Recorder Dataset Extras
- Record Replay Reader
- State Replay
- Place Command and Scatter
- Stack Success and Commands
- Converter Row Sorting
- Manipulation Example Rewards
- Dataset Convention Load
- Domino Row Command
- Stack MDP Terms
- Repo Concept Glossary
- Robot Interface Wiring
- Debug Panel Rendering
- Live Bridge READMEs
- Test Suite READMEs
- Record Replay Script
- Archived Category READMEs
- Shared Command Terms
- Domino and Splitter READMEs
- Place Variants and Cfg
- RL Settings Glossary
- Debug Panel Layout Tests
- Sorter Insert Command
- Recording Mixin Flush Tests
- Gym Adapter Tests
- Sorter Success Kernel
- Alpha by Seed Sweep
- Env Reset Tests
- Scatter Grid Tests
- Manipulation Command Terms
- Go1 Velocity Config
- Dataset Stamp Mismatch
- LootEnv Step Loop
- Peg Variants and Commands
- Stack and Viewer READMEs
- Context Glossary
- Task Category Docs
- G1 Velocity Config
- Robot and Task Registration
- Ball Goal Command
- Place and Domino READMEs
- Debug Panel Widget Tests
- Repo Guidance Docs
- Cartpole Example Cfg
- Tracking Example Rewards
- Mapper Alignment Gate
- Repo Conventions and Gotchas
- Cartpole Scene Cfg
- G1 Tracking Config
- Tracking Terminations
- Tracking RL Runner
- Debug Panel State
- Ball and Tasks READMEs
- Domino Upright Tests
- Place Goal Command
- Stack Variant Tests
- Success Kernel Tests
- Dataset Loader
- DK1 Mesh Manifest
- Debug Panel Formatting
- Splitter Episode Tests
- Recording Mixin State
- Env Observation Tests
- Delta Action Tests
- Gripper Stroke Action
- Smoke Dataset Builder
- Tracking Motion Commands
- Peg Variant Definitions
- Place Success Criteria
- Run Scripts README
- Recording Mixin Paths
- Recording Mixin Errors
- DK1 Collision Spec
- Tracking Command Sampling
- Tracking Observations
- Dataset Stamp Keys
- Home Pose Reset Event
- Shared Termination Terms
- DK1 Collision Contract Tests
- Dataset Flow Glossary
- GLB Mesh Conversion
- Ball Success Kernel
- Debug Panel Render Tests
- Recording Mixin Teardown
- Success Test Helpers
- DK1 Grasp Tests
- Agent Workflow Docs
- Tracking Env Cfg
- Velocity Env Cfg
- Ball Command Helpers
- Recording Mixin Session
- QC Launch Script
- Pre-commit Hook Config
- Manipulation RL Runner
- Velocity RL Runner
- Recording Mixin Helpers
- Shared PPO Runner Cfg
- Converter Error Paths
- Success Kernel Helpers
- Tracking Command Helpers
- Task Listing Script
- Splitter Reward Recompute
- Critic Off 100k Run
- Critic Off 1M Run
- Critic Selection 100k Run
- Critic Selection 1M Run
- Critic Selection Steering 100k
- Critic Selection Steering 1M
- Critic Steering 100k Run
- Critic Steering 1M Run
- Fixed Spawn Target 1M Run
- Fixed Spawn Target 200k Offline
- Fixed Spawn Target 200k Split
- Fixed Pose Critic Off 600k
- Fixed Pose Critic Selection 600k
- Fixed Spawn Random Target 200k
- Fixed Spawn Random Target Split
- Fixed Spawn Random Target Online
- Random Spawn Fixed Target Online
- Random Spawn Random Target Online
- Splitter Test Helpers
- Tracking Example Package
- Velocity Example Package
- Unix Installer
- Package Entry Point
- Project Packaging Cfg

## God Nodes (most connected - your core abstractions)
1. `MotionCommand` - 63 edges
2. `Recorder` - 38 edges
3. `make_harness()` - 32 edges
4. `assemble_transitions()` - 31 edges
5. `StackCommand` - 27 edges
6. `LootEnv` - 26 edges
7. `ScatterGrid` - 26 edges
8. `SuccessGate` - 26 edges
9. `GripperOpenGate` - 25 edges
10. `DominoCommand` - 25 edges

## Surprising Connections (you probably didn't know these)
- `verify_replay.py (bitwise qpos verification that an episode replays exactly)` --semantically_similar_to--> `relabel_episode recomputes the source episode's own reward, and that self-check is what catches real drift`  [INFERRED] [semantically similar]
  loot_bench/scripts/live_bridge/README.md → tests/scripts/README.md
- `BC gate (tests/qc/bc_gate.py)` --semantically_similar_to--> `Splitter self-check on recomputed reward`  [INFERRED] [semantically similar]
  CONTEXT.md → CLAUDE.md
- `manual_replay_demo.py (interactive viewer replay of corkinbox100-fullrel through the sim)` --semantically_similar_to--> `record_replay.py (replay a LeRobot dataset through the sim, no hardware)`  [INFERRED] [semantically similar]
  tests/live_bridge/README.md → loot_bench/scripts/live_bridge/README.md
- `Replay determinism decided by GPU contact impulses (bitwise match until first gripper contact)` --semantically_similar_to--> `conftest.py imports lerobot before mjlab; the reverse native-library order segfaults during collection`  [INFERRED] [semantically similar]
  loot_bench/scripts/live_bridge/README.md → tests/qc/README.md
- `Seeds drawn once and reused for every alpha, so an alpha difference is not a seed difference` --semantically_similar_to--> `A fabricated episode pins the contract more sharply than a recording: state exactly on a threshold, known answer`  [INFERRED] [semantically similar]
  run_scripts/README.md → tests/scripts/README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **The three faults the rebuild found behind the flat-zero curve** — docs_adr_001_rebuild_the_qc_integration_from_09e5dfa_flat_zero_success_curve, docs_adr_005_one_action_unit_is_half_a_joint_travel_normalized_joint_position_action, docs_adr_007_ee_pose_publishes_a_6d_rotation_ee_pose_6d_rotation, docs_adr_008_episodes_start_at_the_home_pose_home_pose_reset [EXTRACTED 1.00]
- **The loot-bench ↔ qc cross-repo contract (no version pin, checked not assumed)** — docs_adr_002_training_lives_in_the_upstream_qc_checkout_training_in_qc, docs_adr_004_qc_venv_reaches_loot_bench_by_pythonpath_pythonpath_reach, docs_adr_009_the_gym_adapter_is_a_seam_only_gym_adapter_seam, docs_adr_009_the_gym_adapter_is_a_seam_only_contact_telemetry_key_contract, docs_adr_011_datasets_stamp_the_action_convention_convention_record, docs_adr_010_episode_boundary_and_bootstrap_mask_are_separate_terminals_vs_masks [INFERRED 0.85]
- **The 20 Hz / ±5 cm end-effector delta action convention** — docs_adr_016_control_at_20hz_through_a_5cm_end_effector_delta_delta_control, docs_adr_016_control_at_20hz_through_a_5cm_end_effector_delta_action_box, docs_adr_016_control_at_20hz_through_a_5cm_end_effector_delta_actuator_split, docs_adr_016_control_at_20hz_through_a_5cm_end_effector_delta_ik_max_dq_torque_cap, docs_adr_016_control_at_20hz_through_a_5cm_end_effector_delta_gravity_compensation, docs_adr_016_control_at_20hz_through_a_5cm_end_effector_delta_episode_budget_15s, docs_adr_011_datasets_stamp_the_action_convention_stamp_keys [EXTRACTED 1.00]
- **One task reward in [-1, 0], exactly 0 only at success (ADR-006), floored at -MIN_SHORTFALL** — loot_bench_tasks_ball_readme_ball_rewards, loot_bench_tasks_domino_readme_domino_rewards, loot_bench_tasks_place_readme_place_rewards, loot_bench_tasks_shape_sorter_readme_sorter_rewards, loot_bench_tasks_stack_readme_stack_rewards, loot_bench_tasks_peg_insertion_readme_peg_sparse_rewards [EXTRACTED 1.00]
- **Teleop recording pipeline: record, label, inspect, replay, verify, keep** — loot_bench_scripts_live_bridge_readme_live_record, loot_bench_scripts_live_bridge_readme_episode_labeling_keys, loot_bench_scripts_live_bridge_readme_debug_panel, loot_bench_scripts_live_bridge_readme_record_replay, loot_bench_scripts_live_bridge_readme_verify_replay, loot_bench_scripts_live_bridge_readme_out_scratch_dataset_kept [EXTRACTED 1.00]
- **Demonstration-to-training-run flow: record, split, gate, smoke, sweep** — loot_bench_scripts_live_bridge_readme_live_record, tests_scripts_readme_test_split_demos, tests_qc_readme_bc_gate, run_scripts_readme_sweep_sh, tests_qc_readme_make_smoke_dataset [INFERRED 0.85]
- **Action convention contract spanning recorder, dataset and trainer** — context_action_box, context_end_effector_delta, context_control_step, context_action_convention, context_convention_record [INFERRED 0.85]
- **Success gating pipeline: criterion to latched episode metric** — context_success_criterion, context_successgate, context_released, context_episode_success [EXTRACTED 1.00]
- **Sim parameters decided together from one measurement campaign** — docs_sim_settings_timestep_decimation, docs_sim_settings_solver_iterations, docs_sim_settings_cone_impratio, docs_sim_settings_nativeccd_disabled, docs_sim_settings_profile_comparison [EXTRACTED 1.00]

## Communities (202 total, 33 thin omitted)

### Community 0 - "Success Reward Check"
Cohesion: 0.06
Nodes (61): CommandTerm, Entity, Tensor, (E, 1) target height — constant per task (``command_obs=False``)., (E,) float: target height reached, blocks at rest, gripper open., (E,) float: success latched since the last reset., (E, N) bool marking the single block currently held in the gripper. Held = the…, StackCommand (+53 more)

### Community 1 - "BC Gate Fit"
Cohesion: 0.06
Nodes (51): float32, Namespace, build_chunks(), episode_bounds(), first_frames(), fit(), FitCfg, FitData (+43 more)

### Community 2 - "ADR Decision Records"
Cohesion: 0.05
Nodes (58): Flat-zero success curve (the qc-fql failure mode), ADR-001: Rebuild the QC integration from 09e5dfa, RSL-RL PPO cfg exists only to satisfy mjlab registration, ADR-002: Training lives in the upstream qc checkout, ADR-003: Pin mjlab to >=1.5.1,<1.6, lerobot-before-mjlab import order warning, ADR-004: qc venv reaches loot_bench by PYTHONPATH, never pip install, Keep run input/output off NFS $HOME and UV_CACHE_DIR off /dev/shm (+50 more)

### Community 3 - "Sim Settings Comparison"
Cohesion: 0.07
Nodes (38): Exception, acc_stats(), AccStats, apply_profile(), build_env(), _fmt_diverge(), GraspScenario, main() (+30 more)

### Community 4 - "Ball Field Assets"
Cohesion: 0.07
Nodes (43): _add_blocker(), _add_box(), _add_incline(), _add_marker(), _add_obstacle(), _add_platform_with_hole(), _add_railing(), _add_ramp_platform() (+35 more)

### Community 5 - "Velocity Example MDP Terms"
Cohesion: 0.06
Nodes (35): commands_vel(), ManagerBasedRlEnv, SceneEntityCfg, Tensor, terrain_levels_vel(), VelocityStage, foot_air_time(), foot_contact() (+27 more)

### Community 6 - "Velocity Rewards and Terrain"
Cohesion: 0.08
Nodes (38): angular_momentum_penalty(), body_angular_velocity_penalty(), feet_air_time(), feet_clearance(), feet_slip(), feet_swing_height, DebugVisualizer, ManagerBasedRlEnv (+30 more)

### Community 7 - "QC Transition Assembly"
Cohesion: 0.08
Nodes (28): assemble_transitions(), _check_rewards(), _episode_bounds(), floating, integer, intp, NDArray, Half-open ``[start, stop)`` row bounds of each episode, in row order. (+20 more)

### Community 8 - "Teleop Live Recording"
Cohesion: 0.06
Nodes (33): ask_operator(), _build_wrist_pip(), _cast_gripper_ray(), _find_camera_id(), _find_site_id(), _get_mj_model(), _get_pip_app(), _get_qimage_and_qpixmap() (+25 more)

### Community 9 - "Teleop Recorder"
Cohesion: 0.08
Nodes (22): ndarray, Tensor, Flush buffered frames with label; seed restores replay layout., Drop staged episode; returns its label or None., Write the staged episode to its dataset (point of no return)., Drop the current episode without saving (operator rejected the demo)., Recorder, _add_dummy_frame() (+14 more)

### Community 10 - "Sorter Asset Generation"
Cohesion: 0.12
Nodes (40): best_fit(), classify_hole(), cross2(), dist_to_boundary(), emit_xml(), extrude(), face_frame(), face_loops() (+32 more)

### Community 11 - "Demo Splitter"
Cohesion: 0.15
Nodes (29): adapter_for(), plan_for(), proprio_width(), Width of the proprioceptive head of a recorded observation. Nothing follows the…, Relabel one source episode into the target task. Returns None when the target's…, relabel_episode(), SplitPlan, variant_from_task_id() (+21 more)

### Community 12 - "Tracking Example Commands"
Cohesion: 0.11
Nodes (6): MotionCommand, CommandTerm, Tensor, Clip joint positions and write root + joint state to sim., Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``. Called after…, Reset to exact reference state at a specific frame. Like ``_resample_command``…

### Community 13 - "Demo Converter"
Cohesion: 0.11
Nodes (25): action_convention(), build_dataset(), convert(), _load_columns(), _load_many(), main(), Any, floating (+17 more)

### Community 14 - "QC Episode Bookkeeping"
Cohesion: 0.09
Nodes (15): EpisodeStats, Running totals for one episode, and the info dict qc reads at the end.…, Fold one step's reward, and its two contact forces, in., The per-step info dict. Published every step rather than only at the end: qc…, Unit tests for ``loot_bench/qc/episode.py``., A category with no single scored object publishes none of the three., The case that makes the latch worth having. A success gate can unlatch: a block…, qc averages flatten(info) across episodes, so no arrays or dicts. (+7 more)

### Community 15 - "Ball MDP Terms"
Cohesion: 0.14
Nodes (25): get_command(), Ball command. Samples spawns, tracks the ball, tests at-target success., ball_pose(), ball_velocity(), field_pose(), ManagerBasedRlEnv, Tensor, Observation terms for the ball task category. (+17 more)

### Community 16 - "Peg Insertion MDP Terms"
Cohesion: 0.13
Nodes (25): get_command(), Archived peg-insertion command. Unregistered, see README. Goal bores, peg…, board_pose(), peg_poses(), ManagerBasedRlEnv, Tensor, Observation terms for the peg-insertion task category (archived, unregistered)., Board pose, env-local (E, 7). Constant per task. (+17 more)

### Community 17 - "Peg Insertion Assets and Cfg"
Cohesion: 0.14
Nodes (27): _add_walls(), frame_axes(), get_board_spec(), get_peg_spec(), get_sheet_board_spec(), get_sheet_spec(), _mat2quat(), _normalize() (+19 more)

### Community 18 - "Teleop Delta Mapper"
Cohesion: 0.13
Nodes (18): gripper_scale_offset(), leader_grasp_pose(), leader_to_delta_action(), The grasp-site pose (position, ``wxyz`` quaternion) a leader pose asks for.…, The 7-wide action that drives the sim arm toward a leader pose. Six delta…, (scale, offset) matching the task-side gripper term. Same model, same helper,…, ndarray, The one channel that is still absolute. It has to invert the gripper term's own… (+10 more)

### Community 19 - "Native Viewer Wrapper"
Cohesion: 0.10
Nodes (17): Handle, main(), Play entry point: mjlab's player with the loot-bench tasks registered., NativeViewerWrapper, NativeMujocoViewer, Native viewer with safe teardown and uncapped FPS. close() waits for GLFW…, Wait for GLFW teardown; True if done within TEARDOWN_TIMEOUT., NativeMujocoViewer with safe teardown and throttled overlays. (+9 more)

### Community 20 - "Leader to Delta Mapping"
Cohesion: 0.11
Nodes (19): _dk1_scratch(), joint_pos_limits(), lerobot_to_mjlab_ctrl(), mjlab_gripper_meters(), MjModel, ndarray, The teleop half of the shared action convention. A leader-arm pose is joint…, Convert real-robot joint_pos (7,) → MJLab data.ctrl (7,). Joints 0-5 (radians)… (+11 more)

### Community 21 - "Leader Arm Reader"
Cohesion: 0.13
Nodes (13): main(), Print the leader's live joint positions so you can pick a new home pose. Move…, LiveReader, Any, ndarray, Reads frames from a real DK1 robot in real-time. Connects to DK1Follower (the…, Open serial connections; retry handshake on dropped packets., Send (7,) action to follower; joints rad, gripper in [0, 1]. (+5 more)

### Community 22 - "Teleop Debug Panel Tests"
Cohesion: 0.11
Nodes (10): Per-frame output: EEF/GOAL/OBJECT/REWARD/DONE/ANOMALIES + optional FPS line…, The min/mean/max breakdown must NOT appear every step. Only in the once-per-…, A NaN in joint_pos is checked live (cheap), but not shown until the episode-end…, capsys is not a terminal, so isatty() is False here. This is the…, The old overwrite style is the default when actually attached to a terminal:…, The FPS/step/speed/RTF line mjlab's base viewer normally prints on its own…, Nothing recorded yet -> nothing to summarize, no banner printed., The reported min/mean/max must span every frame, not just whatever the last… (+2 more)

### Community 23 - "Stack Variants and Assets"
Cohesion: 0.12
Nodes (17): Difficulty, order_color(), Color an object by which split tier first needs it. Objects tail-extend across…, get_block_spec(), MjSpec, MjSpec builders for the stack blocks., make_stack_env_cfg(), ManagerBasedRlEnvCfg (+9 more)

### Community 24 - "Recording Mixin"
Cohesion: 0.12
Nodes (11): Drop last staged episode; single-level undo., Extract the Episode_Termination/* sub-dict from env.step() extras., Whether the success termination fired; extras is a fresh snapshot taken during…, Names of the termination terms that fired, read from step() extras., Multi-episode recording with 2 s failure grace and reset phase., Success saves at once; other dones get a 2 s failure grace., Two-column HUD text: state, counts, and valid keys., Log stage breakdown for slow steps. (+3 more)

### Community 25 - "Sorter MDP Terms"
Cohesion: 0.15
Nodes (18): get_command(), ManagerBasedRlEnv, Shape-sorter command. Owns hole poses, spawns, and insertion latching., ShapeSorterCommandCfg, cube_pose(), piece_poses(), ManagerBasedRlEnv, Tensor (+10 more)

### Community 26 - "Domino Split Relabeling"
Cohesion: 0.12
Nodes (22): _column(), _count_disturbed(), _features(), load_source_episodes(), main(), _make_stack_adapter(), _print_summary(), DataFrame (+14 more)

### Community 27 - "Sorter Split Relabeling"
Cohesion: 0.13
Nodes (22): CategoryAdapter, _check_source_reward(), _domino_progress_and_gate(), frame_state(), FrameState, _held_block_mask(), _make_domino_adapter(), _make_sorter_adapter() (+14 more)

### Community 28 - "Debug Markers and Goal Commands"
Cohesion: 0.11
Nodes (16): DebugVisualizer, add_box_wireframe(), marker_color(), DebugVisualizer, ndarray, Shared status coloring for the command terms' debug-vis goal markers. Every…, Draw an oriented box's 12 edges as thin cylinders. Unlike a second, larger…, Marker RGBA for one goal slot. Args: base_rgba: The target object's own color… (+8 more)

### Community 29 - "Gripper Gate Command"
Cohesion: 0.15
Nodes (16): GripperGateCommandCfg, GripperOpenGate, CommandTermCfg, Command-term helpers shared by the task categories. The success metrics of the…, Base for command cfgs whose success gate checks the gripper is open. Both…, Per-step "all gripper joints open" check for a command's success gate. Joint…, # TODO: check with real robot, NoReturn (+8 more)

### Community 30 - "Shared Observation Terms"
Cohesion: 0.14
Nodes (17): ee_pose_w(), flatten_object_poses(), ManagerBasedRlEnv, SceneEntityCfg, Tensor, Observation terms shared by every task category., The first two columns of a rotation matrix, flattened (..., 6). A continuous,…, End-effector site pose in the env-local world frame (E, 9). Position (xyz) then… (+9 more)

### Community 31 - "One Task Reward Kernel"
Cohesion: 0.13
Nodes (15): command_sparse_task_reward(), command_task_reward(), CommandTerm, ManagerBasedRlEnv, Tensor, Reward-term factories shared by every task category. **The convention.** A…, The task reward: ``progress - 1``, floored, and exactly ``0`` at success. Args:…, The whole reward for a category that reports progress. (+7 more)

### Community 32 - "Scatter Grid Sampler"
Cohesion: 0.11
Nodes (15): device, Entity, Tensor, Grid-based spawn sampling with jittered, distinct cell centers.…, Draw ``count`` distinct jittered cells plus a yaw per object. Sequential…, Write each entity upright at its scattered ``(xy, spawn_yaw)`` cell. Sits each…, Sample distinct, jittered xy positions from a fixed set of cells., Args: cells: (C, 2) cell-centre xy positions. jitter: (2,) max |offset| per… (+7 more)

### Community 33 - "Shared Success Gating"
Cohesion: 0.12
Nodes (14): bore_axial_lateral(), bore_axis_cos_tilt(), device, Tensor, Success gating shared by all task categories Contains the :class:`SuccessGate`…, # TODO: check with real robot, Decompose object-to-mouth offsets along/across a bore/hole axis. Shared by the…, |cos| of the angle between an object's body z-axis and a bore/hole axis. Sign-… (+6 more)

### Community 34 - "Converter Convention Tests"
Cohesion: 0.12
Nodes (12): _build(), A bounded policy can only emit [-1, 1]. An action outside it is one the policy…, The recorder's mapping clamps and the action term clamps, so an out-of-box row…, Episode structure is read off row order, so an out-of-order file would silently…, A sort applied to the observations but not the actions would pair every state…, The sort is what makes transitions.py's contiguity rule satisfiable., Every ``place`` variant shares its widths, so without this a RandS-RandT…, Call the converter with named columns, not a splat. The splat reads better but… (+4 more)

### Community 35 - "Replay Verification"
Cohesion: 0.17
Nodes (22): build_env_like_live_record(), diff_existing_replay(), Episode, FrameStats, load_episode(), main(), MjModel, ndarray (+14 more)

### Community 36 - "Action Term Tests"
Cohesion: 0.13
Nodes (15): delta_action_from_poses(), pose_within_one_delta(), The one action convention: a 20 Hz end-effector delta, 7 wide (ADR-016). Six…, The clamped 6-D delta action that asks for ``target`` from ``current``. The…, Whether one action reaches ``target`` from ``current`` without clamping. The…, The 6-D delta asking for ``target`` from ``current``, in action units. Not…, unclamped_delta_from_poses(), built_env() (+7 more)

### Community 37 - "Domino Registration and Variants"
Cohesion: 0.15
Nodes (15): Shared object color palette for the task categories' asset builders., get_domino_spec(), MjSpec, MjSpec builders for the domino pieces., Task registration for all domino variants on all supported robots., The ``(task, variation)`` id segments; also read by ``split_demos``., task_id_parts(), make_domino_env_cfg() (+7 more)

### Community 38 - "Domino Success Kernel"
Cohesion: 0.18
Nodes (9): domino_fallen_along_row(), Per-domino "toppled along the row" test (chain semantics). A domino counts as…, parametrize, Tensor, _quat_axis_angle(), _quat_mul(), (4,) wxyz quaternion for a rotation of *angle* about *axis* (unit)., TestDominoFallenAlongRow (+1 more)

### Community 39 - "Sorter Assets and Cfg"
Cohesion: 0.13
Nodes (19): get_sorter_piece_spec(), get_sorter_spec(), load_sorter_layout(), MjSpec, Shape, MjSpec builders and layout for the shape-sorter task category. The cube and its…, A free-body STL sorter piece, tinted with its socket colour., One face hole of the sorter cube, in the cube body frame. (+11 more)

### Community 40 - "Action Terms and Workbench Cfg"
Cohesion: 0.11
Nodes (18): DifferentialIKActionCfg, JointPositionAction, JointPositionActionCfg, EndEffectorDeltaActionCfg, NormalizedJointPositionAction, NormalizedJointPositionActionCfg, ManagerBasedRlEnv, Configuration for :class:`EndEffectorDeltaAction`. ``actuator_names``,… (+10 more)

### Community 41 - "YAM Lift Cube Example"
Cohesion: 0.16
Nodes (18): _cube_color(), get_cube_spec(), ManagerBasedRlEnvCfg, MjSpec, # NOTE: No noise for goal position., Generate a distinct color for cube i of n using HSV hue rotation., Multi-cube task: depth + segmentation mask for goal conditioning., yam_lift_cube_env_cfg() (+10 more)

### Community 42 - "Manipulation Example Commands"
Cohesion: 0.13
Nodes (8): LiftingCommand, MultiCubeLiftingCommand, CommandTerm, DebugVisualizer, Tensor, Selects one of N cubes as the target at each reset., Geom IDs of the target cube per env. Shape: (B, K)., Position of the target cube per env. Cached per step — updated in…

### Community 43 - "LootEnv Gym Seam"
Cohesion: 0.14
Nodes (12): LootEnv, Any, ContactSensor, ManagerBasedRlEnv, ndarray, The mjlab env underneath, for tools that need the simulator. Read-only: the…, What an action means in this env, as the stamped record. Published so a dataset…, The actor group, as a fresh float32 row. The copy is deliberate. mjlab hands… (+4 more)

### Community 44 - "Domino MDP Terms"
Cohesion: 0.17
Nodes (15): DominoCommandCfg, get_command(), ManagerBasedRlEnv, Domino command. Scatters dominoes, checks slot placement, latches topple., domino_poses(), phase_obs(), ManagerBasedRlEnv, Tensor (+7 more)

### Community 45 - "Recording Mixin Tests"
Cohesion: 0.13
Nodes (8): make_harness(), V/F mid-episode end the recording manually with that label., The grace window gets its own ghost color, so an undecided episode can't be…, TestGhostColor, TestHud, TestManualEnd, TestSessionEnd, TestUndo

### Community 46 - "Tracking Metrics and Eval"
Cohesion: 0.15
Nodes (19): compute_ee_orientation_error(), compute_ee_position_error(), compute_joint_velocity_error(), compute_mpkpe(), compute_root_relative_mpkpe(), _get_body_indices(), Tensor, Compute Mean Per-Keybody Position Error (MPKPE). MPKPE measures the average… (+11 more)

### Community 47 - "DK1 Constants and Actuators"
Cohesion: 0.18
Nodes (17): BuiltinPositionActuatorCfg, InitialStateCfg, _ArmJointCfg, get_arm_actuators(), get_gripper_actuator(), DK1 actuator configs (DAMIAO DM4340/DM4310). Motor specs and arm gains match…, _compensate_arm_gravity(), get_bimanual_dk1_robot_cfg() (+9 more)

### Community 48 - "Manipulation Example Observations"
Cohesion: 0.19
Nodes (19): camera_depth(), camera_rgb(), camera_segmentation(), camera_target_cube_mask(), ee_to_object_distance(), ee_velocity(), object_to_goal_distance(), ManagerBasedRlEnv (+11 more)

### Community 49 - "Peg Insertion Commands"
Cohesion: 0.14
Nodes (11): InsertionCommand, CommandTerm, Tensor, (E,) float: task done, scored pegs at rest, gripper open., (E,) float: success latched since the last reset., (S, 3) outward bore axes in the board frame., Position of the peg assigned to each socket, (E, S, 3)., Task progress in [0, 1], (E,): fraction of sockets filled. (+3 more)

### Community 50 - "Action Convention Tests"
Cohesion: 0.12
Nodes (12): _grasp_pose(), Tensor, The grasp site's world-frame pose, as the solver reads it., The convention as the simulator actually sees it., QC's chunk of 5 spans 0.25 s here, which is both reference domains. At the old…, "Repeat the last action" was a trivially strong regressor: 0.0242 held-out…, The number a researcher reads off the action space has to be the number the arm…, The box has to be a bound, not a unit. mjlab's own DiffIK term applies its… (+4 more)

### Community 51 - "Leader Reader Internals"
Cohesion: 0.13
Nodes (12): ABC, _connect_with_retry(), Build device and connect, retrying transient ConnectionError., Uniform data envelope for one timestep from any source. Every Reader subclass…, Abstract base class for all data sources. Concrete subclasses: LiveReader (real…, Return the next frame, or raise StopIteration when exhausted., Release any resources (serial ports, file handles)., Reader (+4 more)

### Community 52 - "Recorder Dataset Extras"
Cohesion: 0.15
Nodes (15): LeRobotDataset, extra_info_path(), load_extra_info(), Path, Sidecar metadata for LeRobot datasets recorded by this bridge. lerobot 0.6.0…, write_extra_info(), The dataset's sidecar extra info (episode_seeds, seed_device, operator_id, ...…, Un-mocked round-trip against the installed lerobot. Verifies the staging+label… (+7 more)

### Community 53 - "Record Replay Reader"
Cohesion: 0.15
Nodes (9): DatasetReader, Find the start and end indices for the target episode. LeRobot datasets store…, Read frames from a saved LeRobotDataset (offline replay)., skipif, End-to-end test with a real LeRobot dataset from HuggingFace., DatasetReader wraps a LeRobotDataset to produce ReaderFrame objects., Build a mock LeRobotDataset with num_frames frames in episode 0., TestDatasetReader (+1 more)

### Community 54 - "State Replay"
Cohesion: 0.15
Nodes (14): build_env(), Episode, load_episode(), main(), _NullPolicy, NativeMujocoViewer, Path, RslRlVecEnvWrapper (+6 more)

### Community 55 - "Place Command and Scatter"
Cohesion: 0.18
Nodes (12): get_command(), _midpoint(), PlaceCommandCfg, ManagerBasedRlEnv, Place command. Scatters the cube, draws a target, tests placement., cube_pose(), ManagerBasedRlEnv, Tensor (+4 more)

### Community 56 - "Stack Success and Commands"
Cohesion: 0.18
Nodes (11): block_top_heights(), _heights(), Tensor, Pure success criteria for the stack category. Plain tensor math, no mjlab or…, Top-face heights above the table, (E, N)., Top-face z per block. Args: pos_z: (E, N) block-centre z (world). half_h: (N,)…, Per-block "top face at the target height" test. Args: top_z: (E, N) top-face z…, Fraction of the target height reached by the tallest block, (E,) in [0, 1].… (+3 more)

### Community 57 - "Converter Row Sorting"
Cohesion: 0.13
Nodes (14): _action(), _columns(), ndarray, Tests for the recording -> training-format converter. The episode-structure…, Two sessions are each numbered from episode zero., Why _load_many offsets. Two failed sessions, each numbered from zero,…, The one case that is loud rather than silent: session A ended in success, so…, Delegated to transitions.py, but it has to reach the caller. (+6 more)

### Community 58 - "Manipulation Example Rewards"
Cohesion: 0.18
Nodes (15): bring_object_reward(), joint_velocity_hinge_penalty(), multi_cube_bring_object_reward(), multi_cube_staged_position_reward(), ManagerBasedRlEnv, SceneEntityCfg, Tensor, Quadratic hinge penalty on joint velocities exceeding a symmetric limit.… (+7 more)

### Community 59 - "Dataset Convention Load"
Cohesion: 0.25
Nodes (9): check_dataset_matches_env(), Raise unless ``data`` was recorded under the environment's conventions. Every…, _env_from(), parametrize, An env agreeing with ``data`` except where told otherwise. ``overrides`` takes…, The failure this exists for: a dataset recorded under a different MDP trains…, A pre-ADR-016 file stamps ``action_scale``/``action_offset`` and no…, An env builds its stamp out of the action terms' buffers, which are torch… (+1 more)

### Community 60 - "Domino Row Command"
Cohesion: 0.18
Nodes (9): DominoCommand, CommandTerm, Tensor, (E, G*7): per-slot env-local target pose, in slot order., (E,) float: task done, dominoes at rest, gripper open., (E,) float: success latched since the last reset., Positions of the domino assigned to each slot, (E, G, 3)., Per-piece values (E, N, D) -> per-slot values via piece_for_slot. (+1 more)

### Community 61 - "Stack MDP Terms"
Cohesion: 0.18
Nodes (11): get_command(), ManagerBasedRlEnv, Task command for the stack category. There is no goal pose to sample — the…, StackCommandCfg, block_poses(), ManagerBasedRlEnv, Tensor, Observation terms for the stack task category. (+3 more)

### Community 62 - "Repo Concept Glossary"
Cohesion: 0.15
Nodes (17): delta_action_from_poses (single build site), TRLC-DK1 robot asset, Arm gravity compensation, mjlab pinned >=1.5.1,<1.6, out/ is scratch, dataset/ is kept, Action box ([-1, 1] per channel), BC gate (tests/qc/bc_gate.py), Closed experiment (+9 more)

### Community 63 - "Robot Interface Wiring"
Cohesion: 0.18
Nodes (12): gripper_open_threshold(), ManipulatorInterface, Robot-agnostic manipulator description for benchmark tasks. Tasks use this…, Open threshold from joint_range via _GRIPPER_OPEN_FRACTION., get_yam_interface(), YAM manipulator interface. Thin loot-bench adapter over the mjlab asset-zoo YAM…, YAM constants for get_yam_interface from mjlab's asset zoo., ManagerBasedRlEnvCfg (+4 more)

### Community 64 - "Debug Panel Rendering"
Cohesion: 0.15
Nodes (7): LiveDebugPanel, Per-frame update; aggregate breakdown prints at finish_episode., Status is precomputed by the viewer; reading it is free., Print aggregate breakdown and banner; no-op if no frames., Redraw in place; append-only when piped or append_only set., Plain append for the episode summary (never overwritten)., A task/env shape the panel can't introspect still renders a (less detailed)…

### Community 65 - "Live Bridge READMEs"
Cohesion: 0.18
Nodes (17): debug_panel.py (live per-frame sanity-check panel, --debug-obs append-only mode), Letter-key episode labeling (V success / F failure / D discard / U undo / Q quit / C view), Auto-save on success, orange-ghost pause and 2 s failure auto-save on any other termination, live_bridge (DK1 teleop -> LeRobot demonstration recording), live_record.py (record episodes at ~50 Hz from live GELLO teleop), out/demos is scratch, dataset/demos/<task>/<stamp> is kept and committed, record_replay.py (replay a LeRobot dataset through the sim, no hardware), USE_FOLLOWER_STATE_FOR_SIM (drive the sim from follower joint states instead of leader actions) (+9 more)

### Community 66 - "Test Suite READMEs"
Cohesion: 0.15
Nodes (17): Replay determinism decided by GPU contact impulses (bitwise match until first gripper contact), grip_contact (gate) and object_contact (telemetry only, filtered to the cube), Place observations, 42 wide (joint pos/vel, EE pose with 6D rotation, cube pose, target, last action), START_POSE_NOISE = 0.0 lives on the task so recorder and training run read one number, bc_gate.py (fit recorded actions with a plain regressor, exit status is the verdict; ADR-012), conftest.py imports lerobot before mjlab; the reverse native-library order segfaults during collection, pad_contact_force / object_contact_force: a cross-repo contract with no version pin (check eval/pad_contact_runs), QC seam tests (contract tests for loot_bench/qc/ plus the two standalone scripts) (+9 more)

### Community 67 - "Record Replay Script"
Cohesion: 0.17
Nodes (9): main(), ndarray, Replay a LeRobot dataset through sim and record to logs/. Viewer holds last…, Acts as the viewer's policy AND manages recording. Flow per step: 1.…, Called by the custom viewer after env.step()., NativeViewerWrapper that records obs/action/reward/done per step., Step env and pass reward/done to the recording policy., _RecordingViewer (+1 more)

### Community 68 - "Archived Category READMEs"
Cohesion: 0.18
Nodes (17): Ball reward: the sparse member of the one-task-reward convention (-1 every step until scored), Insertion success: insert_depth below the mouth, laterally centred, axis-aligned, at rest, gripper open, Peg-insertion task category (archived, unregistered, kept importable), Peg insertion (one peg into one bore, vertical or horizontal orientation), Peg observations (board pose for parity with shape_sorter's cube pose, peg poses, per-socket inserted flags), Sparse peg rewards paired with a teleop demonstration pipeline instead of dense shaping, SHAPE_GEOMS (one cross-section description shared by peg and bore, clearance 1.5 mm), socket_layout (one source for the bore wall geometry and the command's goal poses) (+9 more)

### Community 69 - "Shared Command Terms"
Cohesion: 0.17
Nodes (11): ContactReleaseGate, pad_contact_force(), ContactSensor, Entity, Tensor, Per-step "pads not exerting grip force" check for a command's success gate.…, (E,) strongest finger-pad contact force magnitude [N], per env. The one…, _fake_contact_sensor() (+3 more)

### Community 70 - "Domino and Splitter READMEs"
Cohesion: 0.17
Nodes (17): Domino observations (all domino poses object-major, per-slot placed indicator, per-slot target pose, phase), Domino reward shaped by placement fraction (half placement + half toppled in the topple phase), Domino task category (build a row of upright dominoes), Fixed row pose every reset so a teleoperator always builds in the same spot, KnockOver phase latch and chain-topple criterion (same lean, monotonic first-fall times along the row axis), Per-domino identity colors and the two-box row-slot ghost marker, RandomRow variant (row start pose and yaw redrawn every reset, one shared draw sized for the hardest count), Cube-spawn variants: FixedOrientation, NoRotation, Full, RandomPositions (+9 more)

### Community 71 - "Place Variants and Cfg"
Cohesion: 0.18
Nodes (13): get_cube_spec(), MjSpec, MjSpec builder for the place cube., Task registration for the place category on all supported robots., _task_id_parts(), make_place_env_cfg(), ManagerBasedRlEnvCfg, Environment configuration for the place task category. The QC pipeline's sanity… (+5 more)

### Community 72 - "RL Settings Glossary"
Cohesion: 0.15
Nodes (16): Procedural geometry by default, One source for geometry and success metric, Bootstrap mask (masks column), Boundary (terminals column), Crossing plane, Episode success (latched), Insertion channel, SorterLatch (+8 more)

### Community 73 - "Debug Panel Layout Tests"
Cohesion: 0.19
Nodes (8): _aggregate_buffered(), Terminal dashboard for live_record.py, on by default. Per-frame line plus…, Aggregate buffered per-frame tensors once at episode end., First 3 values as x/y/z (positions, goals, object poses)., _xyz(), Unit tests for the --debug-obs live dashboard (no viewer, no simulator)., TestAggregateBuffered, TestXyz

### Community 74 - "Sorter Insert Command"
Cohesion: 0.16
Nodes (8): CommandTerm, DebugVisualizer, Tensor, (E, N*6): per hole mouth xyz and axis xyz, in piece order., (E,) float: every piece latched, gripper open., (E,) float: success latched since the last reset., (E, N) bool: pieces latched inside the cube., ShapeSorterCommand

### Community 75 - "Recording Mixin Flush Tests"
Cohesion: 0.14
Nodes (7): FakeEnv, FakePolicy, Harness, State-machine tests for RecordingViewerMixin, driven with fake recorder/env.…, Minimal env: steps, returns zeros, and counts how often it was asked., The two attributes the recording path reads off the live policy.…, Concrete host standing in for the real viewer subclass.

### Community 76 - "Gym Adapter Tests"
Cohesion: 0.12
Nodes (7): r in [-1, 0]. qc's --sparse depends on nothing else touching 0., qc clips to [-1, 1] itself, but a stray value must not raise., qc counts grips out of this key; nested-only would be a KeyError. It is the…, A key that appears only on some steps is harder to trust., The negative control for :class:`TestObjectContact`. Stack scores three blocks,…, Guards against handing back a stale or copied-once buffer., TestStepContract

### Community 77 - "Sorter Success Kernel"
Cohesion: 0.22
Nodes (8): device, Tensor, Latched mouth-plane crossing for the shape sorter. A piece scores the moment it…, Advance one step; returns the (E, S) bool latch state., SorterLatch, _latch(), _step(), TestSorterLatch

### Community 78 - "Alpha by Seed Sweep"
Cohesion: 0.26
Nodes (12): cell_of, clamp(), die(), is_value(), note_flag(), reap_one(), repo_needs_ticket(), sanitize() (+4 more)

### Community 79 - "Env Reset Tests"
Cohesion: 0.15
Nodes (10): slice, _peak_to_peak(), Largest per-column range. Never the standard deviation. The observations are…, The promise qc's evaluation loop rests on. With auto-reset on, the observation…, Replay verification needs a bit-identical start., Map each actor-observation term to its slice of the flat vector. Derived rather…, Split observations into (robot columns, scene columns)., _split() (+2 more)

### Community 80 - "Scatter Grid Tests"
Cohesion: 0.22
Nodes (3): Tensor, The goal-position keep-out mask (``blocked_near`` + ``blocked``)., TestBlockedNear

### Community 81 - "Manipulation Command Terms"
Cohesion: 0.22
Nodes (9): LiftingCommandCfg, MultiCubeLiftingCommandCfg, ObjectPoseRangeCfg, CommandTermCfg, ManagerBasedRlEnv, Configuration for target position sampling in dynamic mode., Configuration for object pose sampling when resampling commands., TargetPositionRangeCfg (+1 more)

### Community 82 - "Go1 Velocity Config"
Cohesion: 0.21
Nodes (11): ManagerBasedRlEnvCfg, Unitree Go1 velocity environment configurations., Create Unitree Go1 flat terrain velocity configuration., Create Unitree Go1 rough terrain velocity configuration., unitree_go1_flat_env_cfg(), unitree_go1_rough_env_cfg(), RslRlOnPolicyRunnerCfg, RL configuration for Unitree Go1 velocity task. (+3 more)

### Community 83 - "Dataset Stamp Mismatch"
Cohesion: 0.18
Nodes (10): get_training_columns(), load_dataset(), NDArray, Path, Load a converted ``.npz``, checked against ``env`` when one is given. The way a…, Only the keys ``qc``'s ``Dataset.create`` may be given. The stamp and…, Including the two string fields. ``np.savez`` stores a ``str`` as a 0-d unicode…, The path a training run actually takes: the stamp arrives from the file, not… (+2 more)

### Community 84 - "LootEnv Step Loop"
Cohesion: 0.18
Nodes (8): is_loot_task(), Whether ``env_name`` names a task this adapter can build., env(), place_env(), fixture, Contract tests for the qc gym adapter, run against a real task. These build a…, TestSpaces, TestTaskPrefix

### Community 85 - "Peg Variants and Commands"
Cohesion: 0.16
Nodes (7): Mouth pose of every bore in the board body frame., socket_layout(), InsertionCommandCfg, ManagerBasedRlEnv, InsertionVariantCfg, Shape, Orientation

### Community 86 - "Stack and Viewer READMEs"
Cohesion: 0.15
Nodes (14): Difficulty is the block's height:width ratio at BLOCK_DENSITY 1562.5 kg/m^3 (slenderest block is lightest), SHOW_GOAL_STACK (ghost target stack drawn beside the workspace in loot-play only), Each variant carries 2n blocks so height, not block count, is the binding constraint, Stack observations (joint pos/vel, EE pose, all block poses object-major), Stack success: some block's top face within HEIGHT_TOL of the target height, all blocks slow, gripper open, Stack task category (stack blocks to reach a target height), make_smoke_dataset.py (synthetic .npz with real widths and action stamp; no number off it means anything), compare_sim_settings.py (sim-setting profile speed/accuracy benchmark; stale, predates the ADR-016 delta action) (+6 more)

### Community 87 - "Context Glossary"
Cohesion: 0.17
Nodes (13): pre-commit hook: pyright, Serena project config (LOOT-bench), loot_bench package, num_qs must stay at 2 or more, run_scripts/ experiment launchers, train_qc.sh (cross-repo launcher), Action convention (scale and offset), Adapter (LootEnv gym env) (+5 more)

### Community 88 - "Task Category Docs"
Cohesion: 0.19
Nodes (13): Splitter self-check on recomputed reward, Reward lag (one control step), Shortfall floor (MIN_SHORTFALL 1e-3), Task reward (one term, [-1, 0]), Ball sparse reward (-1 until it scores), Stack progress (tallest resting block), nativeccd disabled (analytic box-box collider), Ball category (+5 more)

### Community 89 - "G1 Velocity Config"
Cohesion: 0.22
Nodes (10): ManagerBasedRlEnvCfg, Unitree G1 velocity environment configurations., Create Unitree G1 flat terrain velocity configuration., Create Unitree G1 rough terrain velocity configuration., unitree_g1_flat_env_cfg(), unitree_g1_rough_env_cfg(), RslRlOnPolicyRunnerCfg, RL configuration for Unitree G1 velocity task. (+2 more)

### Community 90 - "Robot and Task Registration"
Cohesion: 0.19
Nodes (10): get_bimanual_dk1_interface(), get_unimanual_dk1_interface(), The two-armed DK1. Declared, not registered against any task. Its actuator…, ManagerBasedRlEnvCfg, Register all robot x variant tasks of one category. Args: category: capitalized…, register_category(), Task registration for all shape-sorter variants on all supported robots., Task registration for all stack variants on all supported robots. The target… (+2 more)

### Community 91 - "Ball Goal Command"
Cohesion: 0.22
Nodes (6): BallCommand, CommandTerm, Tensor, (E,) float: ball at target, at rest, and released., (E,) float: success latched since the last reset., (E, 3): env-local target position (was a ball-relative delta).

### Community 92 - "Place and Domino READMEs"
Cohesion: 0.22
Nodes (13): Domino success: every domino upright, row-aligned and within pos_threshold of its slot, at rest, gripper open, _SLOT_KEEP_OUT (spawn cells near a row slot are blocked), ContactReleaseGate (pad force at or below MIN_GRIP_FORCE before the cube counts as released), Spawn and target regions disjoint in x so the cube can never spawn at the target, PlaceCommand (scatter the cube through ScatterGrid, then draw a target point), Place success: cube centre within pos_threshold in 3D, moving slowly, released (position only, no orientation), Place reward shaped by the spawn->target travel fraction normalized by MAX_TRAVEL_DIST, Place task category (carry one cube to a sampled target and let go) (+5 more)

### Community 93 - "Debug Panel Widget Tests"
Cohesion: 0.21
Nodes (6): _fake_env(), Stands in for the real env.env.observation_manager shape used by…, --debug-obs (append_only=True) must keep every frame in the scrollback, even…, domino's task_obs_terms puts domino_poses (positions) first, then…, Any env shape this doesn't understand disables the named breakdown rather than…, TestObsLayout

### Community 94 - "Repo Guidance Docs"
Cohesion: 0.23
Nodes (12): Automatic registration via package discovery, Task category pattern, Category (task family), Keep-out (blocked_near), Recovery demonstration, Row slot (domino), Variant, Domino progress (build / topple phases) (+4 more)

### Community 95 - "Cartpole Example Cfg"
Cohesion: 0.26
Nodes (11): cartpole_balance_env_cfg(), cartpole_ppo_runner_cfg(), cartpole_swingup_env_cfg(), _get_cartpole_cfg(), _get_spec(), _make_env_cfg(), EntityCfg, ManagerBasedRlEnvCfg (+3 more)

### Community 96 - "Tracking Example Rewards"
Cohesion: 0.41
Nodes (11): _get_body_indexes(), motion_global_anchor_orientation_error_exp(), motion_global_anchor_position_error_exp(), motion_global_body_angular_velocity_error_exp(), motion_global_body_linear_velocity_error_exp(), motion_relative_body_orientation_error_exp(), motion_relative_body_position_error_exp(), ManagerBasedRlEnv (+3 more)

### Community 97 - "Mapper Alignment Gate"
Cohesion: 0.21
Nodes (7): leader_is_aligned(), Whether an episode may start with the leader held at this pose. True when one…, Viewer-agnostic recording state machine (record, fail-grace, reset)., The gate that replaced the opening snap. ADR-016 measured 66 of 99 demos…, It is an absolute channel with no jump to gate., The pose the leader hangs in when the operator lets go. Recording from there is…, TestAlignment

### Community 98 - "Repo Conventions and Gotchas"
Cohesion: 0.18
Nodes (11): Explicit actuator split (arm / gripper / ik_max_dq), ee_pose publishes a 6D rotation, ManipulatorInterface, Observations carry no noise, Fixed prop, Released (robot has let go), Wiring (robot into task cfg), Workbench (shared env skeleton) (+3 more)

### Community 99 - "Cartpole Scene Cfg"
Cohesion: 0.25
Nodes (11): cartpole_smooth_reward(), _gaussian_tolerance(), pole_angle_cos_sin(), ManagerBasedRlEnv, SceneEntityCfg, Tensor, _quadratic_tolerance(), Gaussian sigmoid tolerance: 1 at x=0, value_at_margin=0.1 at |x|=margin. (+3 more)

### Community 100 - "G1 Tracking Config"
Cohesion: 0.22
Nodes (8): ManagerBasedRlEnvCfg, Unitree G1 flat tracking environment configurations., Create Unitree G1 flat terrain tracking configuration., unitree_g1_flat_tracking_env_cfg(), RslRlOnPolicyRunnerCfg, RL configuration for Unitree G1 tracking task., Create RL runner configuration for Unitree G1 tracking task., unitree_g1_tracking_ppo_runner_cfg()

### Community 101 - "Tracking Terminations"
Cohesion: 0.35
Nodes (8): bad_anchor_ori(), bad_anchor_pos(), bad_anchor_pos_z_only(), bad_motion_body_pos(), bad_motion_body_pos_z_only(), ManagerBasedRlEnv, SceneEntityCfg, Tensor

### Community 102 - "Tracking RL Runner"
Cohesion: 0.24
Nodes (5): MotionTrackingOnPolicyRunner, _OnnxMotionModel, MjlabOnPolicyRunner, ONNX-exportable model that wraps the policy and bundles motion reference data., VecEnv

### Community 103 - "Debug Panel State"
Cohesion: 0.25
Nodes (4): Running min/max/mean over an episode, O(1) memory., _RunningStats, The whole point: mean/min/max are over every value seen across all update()…, TestRunningStats

### Community 104 - "Ball and Tasks READMEs"
Cohesion: 0.25
Nodes (11): Strictly additive minigolf layouts with a shared off-centre pit, Ball success: in target zone, at rest, released from the gripper, Ball observations (ball position without orientation, field pose, wall pose, ball velocity, raw target), Ball task category (bring a ball to its goal by hitting or throwing), Minigolf family (raised platform with a pit out of the arm's reach, perimeter railing), Throw family (flat field with a wall between spawn area and target), WallVariation (the wall is a separate entity re-posed every reset), build_sim_registered_tasks.py (build/reset/step every registered task, render one scene per task) (+3 more)

### Community 105 - "Domino Upright Tests"
Cohesion: 0.29
Nodes (4): chain_topple_success(), Tensor, All dominoes chain-fallen the same way, in row order. Args: fallen: (E, G) bool…, TestChainToppleSuccess

### Community 106 - "Place Goal Command"
Cohesion: 0.24
Nodes (6): PlaceCommand, CommandTerm, Tensor, (E, 3): env-local target position., (E,) float: cube at target, at rest, gripper released., (E,) float: success latched since the last reset.

### Community 107 - "Stack Variant Tests"
Cohesion: 0.25
Nodes (7): parametrize, Invariants of the stack variant catalog (spares, prefix rule, scatter fit)., test_aspect_ratio_orders_the_difficulties(), test_blocks_tail_extend_across_target_height(), test_more_blocks_than_needed(), test_only_the_target_block_count_reaches_the_target(), test_scatter_grid_fits_block_set()

### Community 109 - "Dataset Loader"
Cohesion: 0.29
Nodes (5): Reading a converted dataset back, and checking it against an environment.…, A single-environment gym adapter over one mjlab task. ``qc`` drives a plain…, Per-episode bookkeeping for the qc adapter. Kept apart from the environment so…, Seam between loot-bench and qc training code. qc reaches this package by…, Recorded rows -> the transition arrays ``qc`` trains on. A recording is one row…

### Community 110 - "DK1 Mesh Manifest"
Cohesion: 0.20
Nodes (9): base_link, finger_left, finger_right, link1-2, link2-3, link3-4, link4-5, link5-6 (+1 more)

### Community 111 - "Debug Panel Formatting"
Cohesion: 0.20
Nodes (6): _ObsLayout, ndarray, Name to offset map for one flat obs vector (matches saved layout)., Best-effort layout; unknown envs disable the breakdown., Fold one frame's array in; True if it held a non-finite value., _to_numpy()

### Community 112 - "Splitter Episode Tests"
Cohesion: 0.33
Nodes (5): easier_target_task_ids(), parse_task_id(), Split ``LOOT-<Category>-<Task>[-<Variation>]-<Robot>`` into its parts., All registered same-family, splittable tasks with fewer objects., TestTaskIds

### Community 113 - "Recording Mixin State"
Cohesion: 0.20
Nodes (5): F during the grace window: the episode is staged as failure now, without…, V overrides a termination-labeled failure to success (e.g. the task succeeded…, Undecided after the grace period: _execute_step auto-saves failure., Within the grace period nothing is saved. The window stays open., TestProvisionalFailureActions

### Community 114 - "Env Observation Tests"
Cohesion: 0.22
Nodes (8): _gripper_index(), _hold_action(), ndarray, The cube-filtered sensor: the only key that answers "touched the cube".…, Neither sensor may fire on an empty grasp. The arm holds its reset pose, which…, The action that parks the arm where it is and holds the gripper open. A zero…, Which action column drives the gripper: the last one, after the 6-D delta.…, TestObjectContact

### Community 115 - "Delta Action Tests"
Cohesion: 0.28
Nodes (6): _Array, normalized_scale_offset(), Map [-1, 1] onto [lower, upper]. Shared with the teleop mapper. Takes torch on…, The gripper's half of the convention., The whole point of the offset. A joint whose range does not straddle zero still…, TestNormalizedScaleOffset

### Community 116 - "Gripper Stroke Action"
Cohesion: 0.28
Nodes (6): DifferentialIKAction, EndEffectorDeltaAction, Tensor, Differential IK on the grasp site, with the action box made real. See the…, The IK frame's ``(position, wxyz quaternion)``, in the world frame. Exactly the…, Clamp to [-1, 1], then apply scale and offset. The clamp binds every caller,…

### Community 117 - "Smoke Dataset Builder"
Cohesion: 0.28
Nodes (8): Space, build(), main(), Any, Write a synthetic dataset that the training path will accept. **This is not a…, The single dimension of a flat ``Box``. The adapter emits nothing else., Synthetic rows, shaped like a recording of ``episodes`` attempts. The first…, _width()

### Community 118 - "Tracking Motion Commands"
Cohesion: 0.29
Nodes (5): MotionCommandCfg, MotionLoader, CommandTermCfg, ManagerBasedRlEnv, VizCfg

### Community 119 - "Peg Variant Definitions"
Cohesion: 0.25
Nodes (6): Archived peg-insertion variants. Unregistered, see README. Two kinds of task…, Cross-section description shared by a peg and its matching bore., Centre-to-edge distance (apothem) of the peg cross section., One bore on a board, plus the peg colour assigned to it., ShapeGeom, SocketCfg

### Community 120 - "Place Success Criteria"
Cohesion: 0.32
Nodes (6): place_progress(), place_reached(), Tensor, Pure success criteria for the place category. Plain tensor math, no mjlab or…, Cube centre within ``pos_threshold`` of the target (3D, position-only). Args:…, Continuous distance-based shaping, in [0, 1]. 1.0 exactly at the target,…

### Community 121 - "Run Scripts README"
Cohesion: 0.32
Nodes (8): Alpha-major run pool with no barrier between alphas (--max-parallel, a failed run does not stop the others), critic-variants/ (eight single-seed pre-sweep runs, the record of a closed experiment, ADR-014), The '--' split: typed flags are tagged and named, the wrapper's fixed configuration is not, One sweep is one wandb group; run named alpha-<a>_seed-<s>; tags written in exactly one place each, No timestamp in the name, so re-running a wrapper joins the previous group (how you add seeds), run_scripts (ready-made QC-FQL launches, Linux/cluster only), sweep.sh (generic alpha x seed sweep, owns the sweep and nothing about the experiment), Sweep wrapper (pins task, dataset and one arm's configuration; exports LOOT_RUN_SCRIPT)

### Community 122 - "Recording Mixin Paths"
Cohesion: 0.25
Nodes (4): Accepted keys must match what the HUD advertises per state., A termination before any frame was recorded has nothing to decide. It must go…, Q must not bypass an open grace window. The un-decided episode would be…, TestStateGuards

### Community 123 - "Recording Mixin Errors"
Cohesion: 0.25
Nodes (4): What ends the reset phase (ADR-016). The gate reads the *leader*, and it is one…, Before the hardware has reported once there is nothing to compare, and the gate…, Held open, the arm must keep being driven — a frozen sim is what the operator…, TestResetPhaseAlignmentGate

### Community 124 - "DK1 Collision Spec"
Cohesion: 0.29
Nodes (6): CollisionCfg, _collision_with_pad_tuning(), DK1 collision configs: full, gripper-only, gripper-only with self-collision., CollisionCfg with shared contact tuning (grippy pads, plain rest)., # TODO: fix self-collision of dark part near link2-3/link3-4., # TODO: fix oversized collision box of link2-3.

### Community 125 - "Tracking Command Sampling"
Cohesion: 0.29
Nodes (4): Any, ViserServer, Create motion scrubber controls in the Viser viewer., Enable or disable the motion scrubber GUI controls.

### Community 126 - "Tracking Observations"
Cohesion: 0.57
Nodes (6): motion_anchor_ori_b(), motion_anchor_pos_b(), ManagerBasedRlEnv, Tensor, robot_body_ori_b(), robot_body_pos_b()

### Community 127 - "Dataset Stamp Keys"
Cohesion: 0.29
Nodes (6): _as_scalar(), _Env, Any, What :func:`check_dataset_matches_env` needs of an environment., One stamp value, out of whatever container it arrived in. np.load gives 0-d…, Protocol

### Community 128 - "Home Pose Reset Event"
Cohesion: 0.29
Nodes (6): ManagerBasedRlEnv, SceneEntityCfg, Tensor, Reset events shared by all task categories. mjlab's…, Reset the named joints to ``pose`` plus a uniform sample. Joints absent from…, reset_joints_to_pose()

### Community 129 - "Shared Termination Terms"
Cohesion: 0.29
Nodes (6): command_success_termination(), CommandTerm, ManagerBasedRlEnv, Tensor, Termination-term factories shared by every task category., Task's success condition (task-done, at-rest, and released) all met.

### Community 130 - "DK1 Collision Contract Tests"
Cohesion: 0.52
Nodes (6): _compiled_gripper_only_model(), _geom_names(), MjModel, DK1 collision-mode contract tests on the compiled MuJoCo model. Applies a…, test_gripper_only_collides_exactly_the_gripper_geoms(), test_gripper_only_keeps_pad_contact_tuning()

### Community 131 - "Dataset Flow Glossary"
Cohesion: 0.40
Nodes (6): tasks/common shared helpers, Distractor object, Home pose (START_JOINT_POS), ScatterGrid (spawn sampler), Tail-extension and seed draw ordering, loot-split-demos

### Community 132 - "GLB Mesh Conversion"
Cohesion: 0.53
Nodes (5): convert_visual(), linear_to_srgb(), main(), Path, One-time conversion of the official TRLC-DK1 URDF meshes to MuJoCo-loadable…

### Community 133 - "Ball Success Kernel"
Cohesion: 0.33
Nodes (4): ball_in_target_zone(), Tensor, Pure success criteria for the ball category. Plain tensor math, no mjlab or…, xy-radius target test for the ball. Hole targets additionally require the ball…

### Community 137 - "DK1 Grasp Tests"
Cohesion: 0.40
Nodes (5): _build_grasp_model(), MjModel, DK1 grasp contact contract on the compiled MuJoCo model. The sim gripper is a…, Unimanual DK1 plus a free box, physics matching the play env., test_grasp_holds_piece_without_crushing()

### Community 138 - "Agent Workflow Docs"
Cohesion: 0.40
Nodes (5): Flag ADR conflicts explicitly, Use the glossary's vocabulary, Read CONTEXT.md and docs/adr before exploring, Local markdown issue tracker (.scratch/), Wayfinding map, claims and blocking

### Community 139 - "Tracking Env Cfg"
Cohesion: 0.40
Nodes (4): make_tracking_env_cfg(), ManagerBasedRlEnvCfg, Motion mimic task configuration. This module defines the base configuration for…, Create base tracking task configuration.

### Community 140 - "Velocity Env Cfg"
Cohesion: 0.40
Nodes (4): make_velocity_env_cfg(), ManagerBasedRlEnvCfg, Velocity task configuration. This module provides a factory function to create…, Create base velocity tracking task configuration.

### Community 141 - "Ball Command Helpers"
Cohesion: 0.50
Nodes (3): BallCommandCfg, CommandTermCfg, ManagerBasedRlEnv

### Community 142 - "Recording Mixin Session"
Cohesion: 0.40
Nodes (3): Success termination → episode staged as success, reset phase entered., Non-success termination saves nothing yet. The grace window opens so the…, TestTerminationHandling

### Community 143 - "QC Launch Script"
Cohesion: 0.60
Nodes (4): die(), train_qc.sh script, usage(), XLA_PYTHON_CLIENT_PREALLOCATE

### Community 144 - "Pre-commit Hook Config"
Cohesion: 0.50
Nodes (4): pre-commit hook: ruff check --fix, pre-commit hook: ruff format, Platform-pinned torch build, install.sh / install.ps1

### Community 147 - "Recording Mixin Helpers"
Cohesion: 0.50
Nodes (3): build_ghost_model_and_qpos(), Set up recording state; call from subclass __init__., Build ghost model: robot visual geoms only, colored, at the start pose.

### Community 148 - "Shared PPO Runner Cfg"
Cohesion: 0.50
Nodes (3): manipulation_ppo_runner_cfg(), RslRlOnPolicyRunnerCfg, Runner config required by mjlab's task registry. Training is QC-FQL in the…

### Community 153 - "Splitter Reward Recompute"
Cohesion: 0.67
Nodes (3): lagged_rewards(), Rewards for frames ``0..last_frame``, under the env's reward lag. mjlab runs…, test_lagged_rewards_pay_success_one_frame_late()

### Community 172 - "Splitter Test Helpers"
Cohesion: 0.67
Nodes (3): _episode_frame(), DataFrame, One relabeled episode as the DataFrame ``split_to_task`` consumes.

## Knowledge Gaps
- **87 isolated node(s):** `VizCfg`, `VizCfg`, `Ranges`, `VizCfg`, `install.sh script` (+82 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1059 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **33 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `make_velocity_env_cfg()` connect `Velocity Env Cfg` to `Velocity Example MDP Terms`, `Action Terms and Workbench Cfg`, `YAM Lift Cube Example`, `Go1 Velocity Config`, `G1 Velocity Config`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Why does `UniformVelocityCommandCfg` connect `Velocity Example MDP Terms` to `Velocity Env Cfg`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Why does `make_tracking_env_cfg()` connect `Tracking Env Cfg` to `G1 Tracking Config`, `Action Terms and Workbench Cfg`, `YAM Lift Cube Example`, `Go1 Velocity Config`, `Tracking Motion Commands`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `MotionCommand` (e.g. with `motion_anchor_ori_b()` and `motion_anchor_pos_b()`) actually correct?**
  _`MotionCommand` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `Recorder` (e.g. with `_LiveRecordingViewer` and `_ReplayAndRecord`) actually correct?**
  _`Recorder` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `VizCfg`, `VizCfg`, `Ranges` to the rest of the system?**
  _87 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Success Reward Check` be split into smaller, more focused modules?**
  _Cohesion score 0.06128364389233954 - nodes in this community are weakly interconnected._