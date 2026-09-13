# Graph Report - LOOT-bench  (2026-09-13)

## Corpus Check
- 125 files · ~187,020 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2980 nodes · 5173 edges · 266 communities (161 shown, 80 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 210 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- BC Gate Type Tests
- Transitions Test Helpers
- Sim Task Build Checks
- Sim Settings Comparison
- Stack Command Term
- Velocity Command Tracking
- Velocity Reward Terms
- QC Training ADRs
- Action Convention Tests
- Peg Insertion Assets
- Sorter Asset Generator
- Place Spawn Tests
- Live Recording Guide
- Tracking Motion Command
- Color Palette Tiers
- Recorder Episode Tests
- QC Episode Bookkeeping
- Graphify Rebuild Flows
- Tracking Env Config
- Peg Insertion Command
- Split Demos Frame State
- Shared Observation Terms
- Dataset Converter
- Shared Task Workbench
- Success Gate Kernels
- Debug Panel Output Tests
- Live Record Script
- Leader Mapper Tests
- Recording Mixin Logic
- Place Command Term
- Graphify Subcommand Reference
- Split Demos Adapter
- Reward Convention Tests
- Converter Guard Tests
- Cartpole Example Config
- Dataset Sidecar Metadata
- QC Dataset Reader
- Peg Insertion MDP
- Converter Contract Tests
- Lift Cube Example
- Manipulator Interface Wiring
- Live Record Viewer
- Replay Verification
- Ball Task Assets
- Recording Mixin Ghost Tests
- YAM Cube Example
- Tracking Error Metrics
- Split Plan Validation
- Project Context Glossary
- Camera Observation Example
- Dataset Reader Tests
- Domino Command Term
- Go1 Velocity Config
- Play Viewer Wrapper
- Leader Pose Printer
- State Replay Script
- DK1 Actuator Config
- Lift Reward Example
- Dataset Env Agreement
- Record Replay Script
- Sorter Command Term
- Domino Observation Terms
- Reader Data Envelope
- Action Convention Record
- Debug Panel Widget
- Spawn Table Validation
- Debug Panel Env Tests
- Recording Mixin Fake Env
- QC Env Contract Tests
- Mapper Grasp Tests
- Ball Env Config
- Rotation Observation Tests
- Domino Asset Builders
- Sweep Run Script
- Lift Command Config
- QC Gym Adapter Env
- Spawn Reachability Check
- Split Demos IO
- Ball Observation Terms
- Release Gate Tests
- Fixed Spawn Envelope
- Delta Action Terms
- G1 Velocity Config
- Debug Panel Env Shapes
- DK1 Collision Contract
- Joint Mapping Tests
- Ball Command Term
- QC Env Fixtures
- QC Env Observation Rules
- Ball Target Tests
- Graphify Command Index
- G1 Tracking Config
- Tracking Termination Terms
- Tracking Reward Terms
- Tracking Runner Config
- Converted Dataset Tests
- Running Stats Tests
- Debug Marker Helpers
- Ball Variant Config
- Spawn Sampler Tests
- Spawn Lane Predicates
- Sorter Corridor Tests
- Sorter Latch Tests
- Stack Variant Invariants
- Domino Fallen Tests
- Robot Mesh Manifest
- Obs Layout Helper
- Joint Limit Mapper
- Task ID Parsing Tests
- Ball Termination Terms
- Sorter Asset Layout
- Sorter Variant Tests
- Grace Window Label Tests
- Empty Grasp Sensor Tests
- Domino Aligned Tests
- Claude Exports Reference
- Delta Action Processing
- Ball Success Terms
- Domino Topple Kernels
- Sorter Piece Specs
- Sorter Spawn Faces Tests
- OpenCode Exports Reference
- Tracking Command Config
- QC Env Sensors
- Viewer Shutdown Check
- Spawn Pitch Tests
- Sorter Latch Kernel
- Recording Mixin Guards
- Reset Gate Tests
- Chain Topple Tests
- DK1 Collision Tuning
- Tracking Viewer Ghost
- Tracking Anchor Observations
- Shared Reset Events
- Success Termination Helper
- Place Success Kernel
- Stack Progress Tests
- Claude Query Reference
- Release Gate ADR
- Mesh Converter Script
- Serial Reader Connect
- Upright Spawn Writer
- OpenCode Query Reference
- Deferred Qpos Tests
- Fake Recorder Tests
- Domino Lane Tests
- Place Progress Tests
- DK1 Grasp Contract
- Sim Profile Settings
- Fixed Spawn ADR
- Agent Workflow Guide
- Ball Command Config
- Pad Contact Force
- Domino Command Config
- Termination Grace Tests
- QC Train Script
- Query Traversal Modes
- OpenCode Pipeline Concepts
- Claude Add Watch Reference
- Claude Hook Reference
- Claude Update Reference
- Manipulation Runner
- Velocity Runner
- Follower Action Sender
- Ghost Model Setup
- OpenCode Add Watch
- OpenCode Hooks
- OpenCode Update
- DK1 Test Scenes
- Precommit Hooks
- Claude GitHub Reference
- Claude Transcribe Reference
- Tracking Debug Vis
- Task Lister Script
- Recorder Frame Writer
- Opencode Config
- Graphify Plugin
- OpenCode GitHub Flow
- OpenCode Transcribe Flow
- Domino 2 Base Run
- Domino 2 CriticOff Run
- Domino 2 QAgg Run
- Domino 3 Base Run
- Domino 4 Base Run
- Domino 6 Base Run
- Place CriticOff 100k Run
- Place CriticOff 1M Run
- Place CriticSelection 100k Run
- Place CriticSelection 1M Run
- Place SelectionSteering 100k Run
- Place SelectionSteering 1M Run
- Place CriticSteering 100k Run
- Place CriticSteering 1M Run
- Place FixedS FixedT 1M Run
- Place FixedS FixedT 200k Run
- Place FixedS FixedT 200k Pair
- Place FixedS FixedT CriticOff
- Place FixedS FixedT CriticSel
- Place FixedS RandT 200k Run
- Place FixedS RandT 200k Pair
- Place FixedS RandT 50k Run
- Place RandS FixedT 50k Run
- Place RandS RandT 50k Run
- Wiki Export Concepts
- Claude Config Pointer
- Extraction Spec Doc
- Tracking Package Intro
- Velocity Package Intro
- Install Script
- Benchmark Package Intro
- OpenCode Spec Doc
- Neo4j Export Concepts
- Cross Repo Merge Concepts
- Claude Integration Concepts
- Query Feedback Concepts
- Graph Query Fast Path
- OpenCode Add Watch Flow
- OpenCode Exports Flow
- OpenCode GitHub Flow
- OpenCode Hooks Flow
- OpenCode Transcribe Flow
- OpenCode Interpreter Detection
- Precommit Pyright Hook
- QC FQL Training Seam
- SuccessGate Glossary Concept
- Mjlab Pin ADR
- NativeCCD Setting Note
- Place Category Glossary
- Reward Config Type
- Scene Entity Type
- Sorter Latch Glossary
- Project Metadata
- Loot List Tasks CLI
- Loot Play CLI
- Loot Record CLI
- Loot Split CLI
- Numpy Scalar Type
- Numpy Int Type
- Numpy Index Type
- Pathlib Type
- MuJoCo Model Type

## God Nodes (most connected - your core abstractions)
1. `MotionCommand` - 63 edges
2. `Recorder` - 38 edges
3. `make_harness()` - 32 edges
4. `assemble_transitions()` - 31 edges
5. `ContactReleaseGate` - 26 edges
6. `SuccessGate` - 26 edges
7. `_build()` - 25 edges
8. `StackCommand` - 25 edges
9. `LootEnv` - 24 edges
10. `make_manipulation_env_cfg()` - 24 edges

## Surprising Connections (you probably didn't know these)
- `Contact-Only Release Gate` --semantically_similar_to--> `Ball Minigolf and Throw Task`  [INFERRED] [semantically similar]
  docs/adr/018-contact-only-release-gate.md → loot_bench/tasks/ball/README.md
- `manual_replay_demo.py (interactive viewer replay of corkinbox100-fullrel through the sim)` --semantically_similar_to--> `record_replay.py (replay a LeRobot dataset through the sim, no hardware)`  [INFERRED] [semantically similar]
  tests/live_bridge/README.md → loot_bench/scripts/live_bridge/README.md
- `verify_replay.py (bitwise qpos verification that an episode replays exactly)` --semantically_similar_to--> `relabel_episode recomputes the source episode's own reward, and that self-check is what catches real drift`  [INFERRED] [semantically similar]
  loot_bench/scripts/live_bridge/README.md → tests/scripts/README.md
- `Seeds drawn once and reused for every alpha, so an alpha difference is not a seed difference` --semantically_similar_to--> `A fabricated episode pins the contract more sharply than a recording: state exactly on a threshold, known answer`  [INFERRED] [semantically similar]
  run_scripts/README.md → tests/scripts/README.md
- `LOOT-bench Serena Project Config` --conceptually_related_to--> `loot-bench Robot-Manipulation Benchmark Suite`  [AMBIGUOUS]
  .serena/project.yml → CLAUDE.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **The three faults the rebuild found behind the flat-zero curve** — docs_adr_001_rebuild_the_qc_integration_from_09e5dfa_flat_zero_success_curve, docs_adr_005_one_action_unit_is_half_a_joint_travel_normalized_joint_position_action, docs_adr_007_ee_pose_publishes_a_6d_rotation_ee_pose_6d_rotation, docs_adr_008_episodes_start_at_the_home_pose_home_pose_reset [EXTRACTED 1.00]
- **Teleop recording pipeline: record, label, inspect, replay, verify, keep** — loot_bench_scripts_live_bridge_readme_live_record, loot_bench_scripts_live_bridge_readme_episode_labeling_keys, loot_bench_scripts_live_bridge_readme_debug_panel, loot_bench_scripts_live_bridge_readme_record_replay, loot_bench_scripts_live_bridge_readme_verify_replay, loot_bench_scripts_live_bridge_readme_out_scratch_dataset_kept [EXTRACTED 1.00]
- **Sim parameters decided together from one measurement campaign** — docs_sim_settings_timestep_decimation, docs_sim_settings_solver_iterations, docs_sim_settings_cone_impratio, docs_sim_settings_nativeccd_disabled, docs_sim_settings_profile_comparison [EXTRACTED 1.00]
- **The loot-bench ↔ qc cross-repo contract (no version pin, checked not assumed)** — docs_adr_002_training_lives_in_the_upstream_qc_checkout_training_in_qc, docs_adr_004_qc_venv_reaches_loot_bench_by_pythonpath_pythonpath_reach, docs_adr_009_the_gym_adapter_is_a_seam_only_gym_adapter_seam, docs_adr_009_the_gym_adapter_is_a_seam_only_contact_telemetry_key_contract, docs_adr_011_datasets_stamp_the_action_convention_convention_record, docs_adr_010_episode_boundary_and_bootstrap_mask_are_separate_terminals_vs_masks [INFERRED 0.85]
- **Graphify full pipeline flow** — _claude_skills_graphify_skill_detect, _claude_skills_graphify_skill_ast_extraction, _claude_skills_graphify_skill_semantic_extraction, _claude_skills_graphify_skill_build_cluster, _claude_skills_graphify_skill_outputs [EXTRACTED 0.85]
- **Graphify export options set** — _claude_skills_graphify_references_exports_wiki, _claude_skills_graphify_references_exports_neo4j, _claude_skills_graphify_references_exports_mcp [EXTRACTED 0.85]
- **20 Hz end-effector delta control envelope** — docs_adr_016_control_at_20hz_through_a_5cm_end_effector_delta_action_convention, context_control_step, docs_rl_settings_one_task_reward [EXTRACTED 0.85]
- **Success gating with release and at-rest** — context_success_gate, docs_adr_018_contact_only_release_gate_release_gate, loot_bench_tasks_shape_sorter_readme_sorter_task [EXTRACTED 0.85]

## Communities (266 total, 80 thin omitted)

### Community 0 - "BC Gate Type Tests"
Cohesion: 0.06
Nodes (51): float32, floating, integer, intp, Namespace, Path, build_chunks(), episode_bounds() (+43 more)

### Community 1 - "Transitions Test Helpers"
Cohesion: 0.07
Nodes (36): assemble_transitions(), _check_rewards(), _episode_bounds(), floating, integer, intp, NDArray, Half-open ``[start, stop)`` row bounds of each episode, in row order. (+28 more)

### Community 2 - "Sim Task Build Checks"
Cohesion: 0.08
Nodes (53): main(), Path, Build, reset, and step every registered LOOT task. See tests/tasks/README.md., Build, reset, and step *task_id*. Returns ``(build_error, render_warning)``…, _run(), _check_ball_at_target(), _check_ball_minigolf_using_wall(), _check_domino_3pieces() (+45 more)

### Community 3 - "Sim Settings Comparison"
Cohesion: 0.07
Nodes (38): Exception, MjModel, acc_stats(), AccStats, apply_profile(), build_env(), _fmt_diverge(), GraspScenario (+30 more)

### Community 4 - "Stack Command Term"
Cohesion: 0.06
Nodes (33): get_command(), CommandTerm, CommandTermCfg, DebugVisualizer, Entity, ManagerBasedRlEnv, ndarray, Tensor (+25 more)

### Community 5 - "Velocity Command Tracking"
Cohesion: 0.06
Nodes (35): commands_vel(), ManagerBasedRlEnv, SceneEntityCfg, Tensor, terrain_levels_vel(), VelocityStage, foot_air_time(), foot_contact() (+27 more)

### Community 6 - "Velocity Reward Terms"
Cohesion: 0.08
Nodes (38): angular_momentum_penalty(), body_angular_velocity_penalty(), feet_air_time(), feet_clearance(), feet_slip(), feet_swing_height, DebugVisualizer, ManagerBasedRlEnv (+30 more)

### Community 7 - "QC Training ADRs"
Cohesion: 0.06
Nodes (45): Flat-zero success curve (the qc-fql failure mode), ADR-001: Rebuild the QC integration from 09e5dfa, RSL-RL PPO cfg exists only to satisfy mjlab registration, ADR-002: Training lives in the upstream qc checkout, lerobot-before-mjlab import order warning, ADR-004: qc venv reaches loot_bench by PYTHONPATH, never pip install, Keep run input/output off NFS $HOME and UV_CACHE_DIR off /dev/shm, XLA_PYTHON_CLIENT_PREALLOCATE=false (JAX and mjlab share one 11 GB card) (+37 more)

### Community 8 - "Action Convention Tests"
Cohesion: 0.05
Nodes (24): built_env(), _grasp_pose(), fixture, Tensor, Tests for the shared action convention. The pure tests need no simulator. The…, The grasp site's world-frame pose, as the solver reads it., The convention as the simulator actually sees it., QC's chunk of 5 spans 0.25 s here, which is both reference domains. At the old… (+16 more)

### Community 9 - "Peg Insertion Assets"
Cohesion: 0.08
Nodes (33): _add_walls(), frame_axes(), get_board_spec(), get_peg_spec(), get_sheet_board_spec(), get_sheet_spec(), _mat2quat(), _normalize() (+25 more)

### Community 10 - "Sorter Asset Generator"
Cohesion: 0.12
Nodes (40): best_fit(), classify_hole(), cross2(), dist_to_boundary(), emit_xml(), extrude(), face_frame(), face_loops() (+32 more)

### Community 11 - "Place Spawn Tests"
Cohesion: 0.08
Nodes (17): draw_place_poses(), enforce_place_separation(), place_half_width(), place_xy_from_draws(), device, Tensor, Draw the cube pose and target over the shared envelope. Fixed draw counts in…, (E, 3): env-local target position. (+9 more)

### Community 12 - "Live Recording Guide"
Cohesion: 0.07
Nodes (38): debug_panel.py (live per-frame sanity-check panel, --debug-obs append-only mode), Letter-key episode labeling (V success / F failure / D discard / U undo / Q quit / C view), Auto-save on success, orange-ghost pause and 2 s failure auto-save on any other termination, live_bridge (DK1 teleop -> LeRobot demonstration recording), live_record.py (record episodes at ~50 Hz from live GELLO teleop), out/demos is scratch, dataset/demos/<task>/<stamp> is kept and committed, record_replay.py (replay a LeRobot dataset through the sim, no hardware), Replay determinism decided by GPU contact impulses (bitwise match until first gripper contact) (+30 more)

### Community 13 - "Tracking Motion Command"
Cohesion: 0.11
Nodes (6): MotionCommand, CommandTerm, Tensor, Clip joint positions and write root + joint state to sim., Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``. Called after…, Reset to exact reference state at a specific frame. Like ``_resample_command``…

### Community 14 - "Color Palette Tiers"
Cohesion: 0.09
Nodes (23): Difficulty, order_color(), Shared object color palette for the task categories' asset builders., Color an object by which split tier first needs it. Objects tail-extend across…, get_block_spec(), MjSpec, MjSpec builders for the stack blocks., category_of() (+15 more)

### Community 15 - "Recorder Episode Tests"
Cohesion: 0.11
Nodes (18): Drop staged episode; returns its label or None., Drop the current episode without saving (operator rejected the demo)., Recorder, _add_dummy_frame(), patch, undo_last_save() drops the staged episode and reports its label. An undone…, Single-level undo: a second undo has nothing to drop., Only the last save is undoable: staging episode 2 committed episode 1, so undo… (+10 more)

### Community 16 - "QC Episode Bookkeeping"
Cohesion: 0.09
Nodes (16): EpisodeStats, Per-episode bookkeeping for the qc adapter. Kept apart from the environment so…, Running totals for one episode, and the info dict qc reads at the end.…, Fold one step's reward, and its two contact forces, in., The per-step info dict. Published every step rather than only at the end: qc…, Unit tests for ``loot_bench/qc/episode.py``., A category with no single scored object publishes none of the three., The case that makes the latch worth having. A success gate can unlatch: a block… (+8 more)

### Community 17 - "Graphify Rebuild Flows"
Cohesion: 0.06
Nodes (30): Cluster Only Rebuild, Incremental Update Re-extract, Graph Build Cluster Analyze, Graph Health Check Gate, For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only (+22 more)

### Community 18 - "Tracking Env Config"
Cohesion: 0.08
Nodes (30): make_tracking_env_cfg(), ManagerBasedRlEnvCfg, Motion mimic task configuration. This module defines the base configuration for…, Create base tracking task configuration., InsertionVariantCfg, make_manipulation_env_cfg(), CommandTermCfg, EntityCfg (+22 more)

### Community 19 - "Peg Insertion Command"
Cohesion: 0.13
Nodes (25): get_command(), Archived peg-insertion command. Unregistered, see README. Goal bores, fixed peg…, board_pose(), peg_poses(), ManagerBasedRlEnv, Tensor, Observation terms for the peg-insertion task category (archived, unregistered)., Board pose, env-local (E, 7). Constant per task. (+17 more)

### Community 20 - "Split Demos Frame State"
Cohesion: 0.12
Nodes (28): episode_rewards(), frame_state(), proprio_width(), Width of the proprioceptive head of a recorded observation. Nothing follows the…, Narrow one episode's recorded state to its first ``take`` objects. ``n_layout``…, Rewards for frames ``0..last_frame``, each from its own state. The recording…, Relabel one source episode into the target task. Returns None when the target's…, relabel_episode() (+20 more)

### Community 21 - "Shared Observation Terms"
Cohesion: 0.13
Nodes (21): flatten_object_poses(), flatten_pose(), Observation terms shared by every task category., A single object's pose in the env-local world frame (E, 7). Position (env-…, Per-env object poses in the env-local world frame, flattened (E, N*7). Object-…, get_command(), CommandTermCfg, ManagerBasedRlEnv (+13 more)

### Community 22 - "Dataset Converter"
Cohesion: 0.11
Nodes (22): build_dataset(), convert(), _load_columns(), _load_many(), main(), Any, floating, integer (+14 more)

### Community 23 - "Shared Task Workbench"
Cohesion: 0.11
Nodes (17): Command-term helpers shared by the task categories. The success metrics of the…, Shared workbench for all tasks. Categories pass core parts as parameters. The…, Visual-only workspace table shared by every task category., Environment configuration for the domino task category., Archived peg-insertion env config. Unregistered, see README., get_cube_spec(), MjSpec, MjSpec builder for the place cube. (+9 more)

### Community 24 - "Success Gate Kernels"
Cohesion: 0.11
Nodes (17): bore_axial_lateral(), bore_axis_cos_tilt(), device, Tensor, Success gating shared by all task categories Contains the :class:`SuccessGate`…, # TODO: check with real robot, Decompose object-to-mouth offsets along/across a bore/hole axis. Shared by the…, |cos| of the angle between an object's body z-axis and a bore/hole axis. Sign-… (+9 more)

### Community 25 - "Debug Panel Output Tests"
Cohesion: 0.11
Nodes (10): Per-frame output: EEF/GOAL/OBJECT/REWARD/DONE/ANOMALIES + optional FPS line…, The min/mean/max breakdown must NOT appear every step. Only in the once-per-…, A NaN in joint_pos is checked live (cheap), but not shown until the episode-end…, capsys is not a terminal, so isatty() is False here. This is the…, The old overwrite style is the default when actually attached to a terminal:…, The FPS/step/speed/RTF line mjlab's base viewer normally prints on its own…, Nothing recorded yet -> nothing to summarize, no banner printed., The reported min/mean/max must span every frame, not just whatever the last… (+2 more)

### Community 26 - "Live Record Script"
Cohesion: 0.11
Nodes (21): ask_operator(), _build_wrist_pip(), _cast_gripper_ray(), _find_camera_id(), _get_mj_model(), _get_pip_app(), _hw_loop(), _LivePolicy (+13 more)

### Community 27 - "Leader Mapper Tests"
Cohesion: 0.14
Nodes (17): gripper_scale_offset(), leader_grasp_pose(), leader_to_delta_action(), The grasp-site pose (position, ``wxyz`` quaternion) a leader pose asks for.…, The 7-wide action that drives the sim arm toward a leader pose. Six delta…, (scale, offset) matching the task-side gripper term. Same model, same helper,…, ndarray, The one channel that is still absolute. It has to invert the gripper term's own… (+9 more)

### Community 28 - "Recording Mixin Logic"
Cohesion: 0.12
Nodes (11): Drop last staged episode; single-level undo., Extract the Episode_Termination/* sub-dict from env.step() extras., Whether the success termination fired; extras is a fresh snapshot taken during…, Names of the termination terms that fired, read from step() extras., Multi-episode recording with 2 s failure grace and reset phase., Success saves at once; other dones get a 2 s failure grace., Two-column HUD text: state, counts, and valid keys., Log stage breakdown for slow steps. (+3 more)

### Community 29 - "Place Command Term"
Cohesion: 0.13
Nodes (15): get_command(), PlaceCommand, PlaceCommandCfg, CommandTerm, CommandTermCfg, DebugVisualizer, ManagerBasedRlEnv, Place command. Samples the cube spawn and target, tests placement. (+7 more)

### Community 30 - "Graphify Subcommand Reference"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 31 - "Split Demos Adapter"
Cohesion: 0.13
Nodes (22): CategoryAdapter, _check_source_reward(), _count_disturbed(), _domino_progress_and_gate(), FrameState, _held_block_mask(), _make_domino_adapter(), _make_sorter_adapter() (+14 more)

### Community 32 - "Reward Convention Tests"
Cohesion: 0.13
Nodes (15): command_sparse_task_reward(), command_task_reward(), CommandTerm, ManagerBasedRlEnv, Tensor, Reward-term factories shared by every task category. **The convention.** A…, The task reward: ``progress - 1``, floored, and exactly ``0`` at success. Args:…, The whole reward for a category that reports progress. (+7 more)

### Community 33 - "Converter Guard Tests"
Cohesion: 0.12
Nodes (12): _build(), A bounded policy can only emit [-1, 1]. An action outside it is one the policy…, The recorder's mapping clamps and the action term clamps, so an out-of-box row…, Episode structure is read off row order, so an out-of-order file would silently…, A sort applied to the observations but not the actions would pair every state…, The sort is what makes transitions.py's contiguity rule satisfiable., Every ``place`` variant shares its widths, so without this a RandS-RandT…, Call the converter with named columns, not a splat. The splat reads better but… (+4 more)

### Community 34 - "Cartpole Example Config"
Cohesion: 0.15
Nodes (22): cartpole_balance_env_cfg(), cartpole_ppo_runner_cfg(), cartpole_smooth_reward(), cartpole_swingup_env_cfg(), _gaussian_tolerance(), _get_cartpole_cfg(), _get_spec(), _make_env_cfg() (+14 more)

### Community 35 - "Dataset Sidecar Metadata"
Cohesion: 0.13
Nodes (16): LeRobotDataset, extra_info_path(), load_extra_info(), Path, Sidecar metadata for LeRobot datasets recorded by this bridge. lerobot 0.6.0…, write_extra_info(), Flush buffered frames with label; seed restores replay layout., Write the staged episode to its dataset (point of no return). (+8 more)

### Community 36 - "QC Dataset Reader"
Cohesion: 0.11
Nodes (16): _as_scalar(), _Env, get_training_columns(), Any, NDArray, Reading a converted dataset back, and checking it against an environment.…, Only the keys ``qc``'s ``Dataset.create`` may be given. The stamp and…, What :func:`check_dataset_matches_env` needs of an environment. (+8 more)

### Community 37 - "Peg Insertion MDP"
Cohesion: 0.12
Nodes (11): InsertionCommand, InsertionCommandCfg, CommandTerm, CommandTermCfg, DebugVisualizer, ManagerBasedRlEnv, Tensor, (E,) float: task done, scored pegs at rest, released. (+3 more)

### Community 38 - "Converter Contract Tests"
Cohesion: 0.10
Nodes (16): _action(), _columns(), _FakeEnv, ndarray, Tests for the recording -> training-format converter. The episode-structure…, Just enough of the adapter to answer the stamp check., Two sessions are each numbered from episode zero., Why _load_many offsets. Two failed sessions, each numbered from zero,… (+8 more)

### Community 39 - "Lift Cube Example"
Cohesion: 0.13
Nodes (8): LiftingCommand, MultiCubeLiftingCommand, CommandTerm, DebugVisualizer, Tensor, Selects one of N cubes as the target at each reset., Geom IDs of the target cube per env. Shape: (B, K)., Position of the target cube per env. Cached per step — updated in…

### Community 40 - "Manipulator Interface Wiring"
Cohesion: 0.14
Nodes (16): ManipulatorInterface, Robot-agnostic manipulator description for benchmark tasks. Tasks use this…, get_yam_interface(), YAM manipulator interface. Thin loot-bench adapter over the mjlab asset-zoo YAM…, YAM constants for get_yam_interface from mjlab's asset zoo., ManagerBasedRlEnvCfg, Wire a robot's :class:`ManipulatorInterface` into a base task cfg. This is the…, Fill the robot placeholders left by a ``make_<cat>_env_cfg`` in place. Adds the… (+8 more)

### Community 41 - "Live Record Viewer"
Cohesion: 0.12
Nodes (12): _find_site_id(), _get_qimage_and_qpixmap(), _LiveRecordingViewer, Native viewer with recording; state machine lives in the mixin., Capture the intended visualization-flag baseline after base setup., Restore baseline vis flags (recording keys overlap viewer toggles)., Draw ghost-at-home overlay on top of task debug geoms each frame., Reseed all RNGs and reset the env so this episode's object scatter and sampled… (+4 more)

### Community 42 - "Replay Verification"
Cohesion: 0.18
Nodes (21): build_env_like_live_record(), diff_existing_replay(), Episode, FrameStats, load_episode(), main(), MjModel, ndarray (+13 more)

### Community 43 - "Ball Task Assets"
Cohesion: 0.17
Nodes (21): _add_blocker(), _add_box(), _add_incline(), _add_marker(), _add_obstacle(), _add_platform_with_hole(), _add_railing(), _add_ramp_platform() (+13 more)

### Community 44 - "Recording Mixin Ghost Tests"
Cohesion: 0.13
Nodes (8): make_harness(), V/F mid-episode end the recording manually with that label., The grace window gets its own ghost color, so an undecided episode can't be…, TestGhostColor, TestHud, TestManualEnd, TestSessionEnd, TestUndo

### Community 45 - "YAM Cube Example"
Cohesion: 0.17
Nodes (17): _cube_color(), get_cube_spec(), ManagerBasedRlEnvCfg, MjSpec, # NOTE: No noise for goal position., Generate a distinct color for cube i of n using HSV hue rotation., Multi-cube task: depth + segmentation mask for goal conditioning., yam_lift_cube_env_cfg() (+9 more)

### Community 46 - "Tracking Error Metrics"
Cohesion: 0.15
Nodes (19): compute_ee_orientation_error(), compute_ee_position_error(), compute_joint_velocity_error(), compute_mpkpe(), compute_root_relative_mpkpe(), _get_body_indices(), Tensor, Compute Mean Per-Keybody Position Error (MPKPE). MPKPE measures the average… (+11 more)

### Community 47 - "Split Plan Validation"
Cohesion: 0.27
Nodes (11): adapter_for(), plan_for(), SplitPlan, variant_from_task_id(), Synthetic Stack5-Easy episode: 10 cubes, only their z varying.…, Pads loaded to the end: no success, even with the qpos wide open. The old qpos…, _stack_episode(), _stack_state() (+3 more)

### Community 48 - "Project Context Glossary"
Cohesion: 0.13
Nodes (20): AGENTS Pointer to CLAUDE.md, loot-bench Robot-Manipulation Benchmark Suite, Convention Record Dataset Stamp, LOOT Benchmark Glossary, BC Gate Behaviour-Cloning Fit, 5cm End-Effector Delta Action Convention, Contact-Only Release Gate, Fixed Spawn Table (+12 more)

### Community 49 - "Camera Observation Example"
Cohesion: 0.19
Nodes (19): camera_depth(), camera_rgb(), camera_segmentation(), camera_target_cube_mask(), ee_to_object_distance(), ee_velocity(), object_to_goal_distance(), ManagerBasedRlEnv (+11 more)

### Community 50 - "Dataset Reader Tests"
Cohesion: 0.14
Nodes (10): DatasetReader, Find the start and end indices for the target episode. LeRobot datasets store…, The dataset's sidecar extra info (episode_seeds, seed_device, operator_id, ...…, Read frames from a saved LeRobotDataset (offline replay)., skipif, End-to-end test with a real LeRobot dataset from HuggingFace., DatasetReader wraps a LeRobotDataset to produce ReaderFrame objects., Build a mock LeRobotDataset with num_frames frames in episode 0. (+2 more)

### Community 51 - "Domino Command Term"
Cohesion: 0.15
Nodes (11): domino_row_slots(), DominoCommand, CommandTerm, DebugVisualizer, Tensor, (E, G*7): per-slot env-local target pose, in slot order., (E,) float: task done, dominoes at rest, released., Positions of the domino assigned to each slot, (E, G, 3). (+3 more)

### Community 52 - "Go1 Velocity Config"
Cohesion: 0.15
Nodes (15): ManagerBasedRlEnvCfg, Unitree Go1 velocity environment configurations., Create Unitree Go1 flat terrain velocity configuration., Create Unitree Go1 rough terrain velocity configuration., unitree_go1_flat_env_cfg(), unitree_go1_rough_env_cfg(), RslRlOnPolicyRunnerCfg, RL configuration for Unitree Go1 velocity task. (+7 more)

### Community 53 - "Play Viewer Wrapper"
Cohesion: 0.15
Nodes (11): Handle, main(), Play entry point: mjlab's player with the loot-bench tasks registered., NativeViewerWrapper, NativeMujocoViewer, Native viewer with safe teardown and uncapped FPS. close() waits for GLFW…, NativeMujocoViewer with safe teardown and throttled overlays., main() (+3 more)

### Community 54 - "Leader Pose Printer"
Cohesion: 0.20
Nodes (9): main(), Print the leader's live joint positions so you can pick a new home pose. Move…, LiveReader, Reads frames from a real DK1 robot in real-time. Connects to DK1Follower (the…, patch, LiveReader wraps DK1Follower + DK1Leader for real-time data., Can instantiate without connecting (connect_on_init=False)., Leader-only mode: no follower connect, joint_pos mirrors leader_action. (+1 more)

### Community 55 - "State Replay Script"
Cohesion: 0.15
Nodes (14): build_env(), Episode, load_episode(), main(), _NullPolicy, NativeMujocoViewer, Path, RslRlVecEnvWrapper (+6 more)

### Community 56 - "DK1 Actuator Config"
Cohesion: 0.19
Nodes (14): BuiltinPositionActuatorCfg, InitialStateCfg, _ArmJointCfg, get_arm_actuators(), get_gripper_actuator(), DK1 actuator configs (DAMIAO DM4340/DM4310). Motor specs and arm gains match…, get_bimanual_dk1_robot_cfg(), get_unimanual_dk1_robot_cfg() (+6 more)

### Community 57 - "Lift Reward Example"
Cohesion: 0.18
Nodes (15): bring_object_reward(), joint_velocity_hinge_penalty(), multi_cube_bring_object_reward(), multi_cube_staged_position_reward(), ManagerBasedRlEnv, SceneEntityCfg, Tensor, Quadratic hinge penalty on joint velocities exceeding a symmetric limit.… (+7 more)

### Community 58 - "Dataset Env Agreement"
Cohesion: 0.25
Nodes (9): check_dataset_matches_env(), Raise unless ``data`` was recorded under the environment's conventions. Every…, _env_from(), parametrize, An env agreeing with ``data`` except where told otherwise. ``overrides`` takes…, The failure this exists for: a dataset recorded under a different MDP trains…, A pre-ADR-016 file stamps ``action_scale``/``action_offset`` and no…, An env builds its stamp out of the action terms' buffers, which are torch… (+1 more)

### Community 59 - "Record Replay Script"
Cohesion: 0.16
Nodes (9): main(), ndarray, Replay a LeRobot dataset through sim and record to logs/. Viewer holds last…, Acts as the viewer's policy AND manages recording. Flow per step: 1.…, Called by the custom viewer after env.step()., NativeViewerWrapper that records obs/action/reward/done per step., Step env and pass reward/done to the recording policy., _RecordingViewer (+1 more)

### Community 60 - "Sorter Command Term"
Cohesion: 0.14
Nodes (8): (E,) float: success latched since the last reset., CommandTerm, DebugVisualizer, Tensor, (E, N*6): per hole mouth xyz and axis xyz, in piece order., (E,) float: every piece latched, released., (E, N) bool: pieces latched inside the cube., ShapeSorterCommand

### Community 61 - "Domino Observation Terms"
Cohesion: 0.22
Nodes (13): get_command(), Domino command. Spawns dominoes from the fixed table, checks slot placement,…, domino_poses(), phase_obs(), ManagerBasedRlEnv, Tensor, Observation terms for the domino task category., All domino poses in the env-local world frame, flattened (E, G*7). (+5 more)

### Community 62 - "Reader Data Envelope"
Cohesion: 0.16
Nodes (9): ABC, Uniform data envelope for one timestep from any source. Every Reader subclass…, Abstract base class for all data sources. Concrete subclasses: LiveReader (real…, Return the next frame, or raise StopIteration when exhausted., Release any resources (serial ports, file handles)., Reader, ReaderFrame, Only joint_pos is required. Everything else has defaults. (+1 more)

### Community 63 - "Action Convention Record"
Cohesion: 0.16
Nodes (14): _Array, action_convention(), A teleop recording -> the arrays ``qc`` trains on. ``live_record.py`` writes a…, The convention record a recording made right now is stamped with. Read from the…, The teleop half of the shared action convention. A leader-arm pose is joint…, delta_action_from_poses(), normalized_scale_offset(), pose_within_one_delta() (+6 more)

### Community 64 - "Debug Panel Widget"
Cohesion: 0.15
Nodes (7): LiveDebugPanel, Per-frame update; aggregate breakdown prints at finish_episode., Status is precomputed by the viewer; reading it is free., Print aggregate breakdown and banner; no-op if no frames., Redraw in place; append-only when piped or append_only set., Plain append for the episode summary (never overwritten)., A task/env shape the panel can't introspect still renders a (less detailed)…

### Community 65 - "Spawn Table Validation"
Cohesion: 0.19
Nodes (9): min_pairwise_separation(), Tensor, Check a category's spawn table; return a list of violations (``[]`` ok). Args:…, Minimum centre-to-centre distance (m) over all cell pairs., validate_spawn_table(), _example_tables(), parametrize, Synthetic category table specs exercising the validator's shape. Toy inputs for… (+1 more)

### Community 66 - "Debug Panel Env Tests"
Cohesion: 0.19
Nodes (8): _aggregate_buffered(), Terminal dashboard for live_record.py, on by default. Per-frame line plus…, Aggregate buffered per-frame tensors once at episode end., First 3 values as x/y/z (positions, goals, object poses)., _xyz(), Unit tests for the --debug-obs live dashboard (no viewer, no simulator)., TestAggregateBuffered, TestXyz

### Community 67 - "Recording Mixin Fake Env"
Cohesion: 0.14
Nodes (7): FakeEnv, FakePolicy, Harness, State-machine tests for RecordingViewerMixin, driven with fake recorder/env.…, Minimal env: steps, returns zeros, and counts how often it was asked., The two attributes the recording path reads off the live policy.…, Concrete host standing in for the real viewer subclass.

### Community 68 - "QC Env Contract Tests"
Cohesion: 0.12
Nodes (7): r in [-1, 0]. qc's --sparse depends on nothing else touching 0., qc clips to [-1, 1] itself, but a stray value must not raise., qc counts grips out of this key; nested-only would be a KeyError. It is the…, A key that appears only on some steps is harder to trust., The negative control for :class:`TestObjectContact`. Stack scores three blocks,…, Guards against handing back a stale or copied-once buffer., TestStepContract

### Community 69 - "Mapper Grasp Tests"
Cohesion: 0.17
Nodes (10): leader_is_aligned(), ndarray, The grasp-site pose every episode starts from, in the base frame.…, Whether an episode may start with the leader held at this pose. True when one…, reset_grasp_pose(), Viewer-agnostic recording state machine (record, fail-grace, reset)., The gate that replaced the opening snap. ADR-016 measured 66 of 99 demos…, It is an absolute channel with no jump to gate. (+2 more)

### Community 70 - "Ball Env Config"
Cohesion: 0.17
Nodes (14): get_ball_spec(), get_wall_spec(), MjSpec, Standalone wall entity for ThrowWallVariation., make_ball_env_cfg(), ManagerBasedRlEnvCfg, Environment configuration for the ball task category., Build the robot-agnostic ball cfg (see ``tasks/common/env_cfg.py`` for the… (+6 more)

### Community 71 - "Rotation Observation Tests"
Cohesion: 0.26
Nodes (9): The first two columns of a rotation matrix, flattened (..., 6). A continuous,…, rot6d_from_matrix(), Tensor, Unit tests for ``tasks/common/observations.py``., The sign-free rotation observation., The reason this term exists. ``quat_from_matrix`` picks its branch by an argmax…, _rot_y(), _rot_z() (+1 more)

### Community 72 - "Domino Asset Builders"
Cohesion: 0.21
Nodes (9): get_domino_spec(), MjSpec, MjSpec builders for the domino pieces., Task registration for all domino variants on all supported robots., task_id_parts(), domino_half_extents(), domino_mass(), DominoVariantCfg (+1 more)

### Community 73 - "Sweep Run Script"
Cohesion: 0.26
Nodes (12): cell_of, clamp(), die(), is_value(), note_flag(), reap_one(), repo_needs_ticket(), sanitize() (+4 more)

### Community 74 - "Lift Command Config"
Cohesion: 0.22
Nodes (9): LiftingCommandCfg, MultiCubeLiftingCommandCfg, ObjectPoseRangeCfg, CommandTermCfg, ManagerBasedRlEnv, Configuration for target position sampling in dynamic mode., Configuration for object pose sampling when resampling commands., TargetPositionRangeCfg (+1 more)

### Community 75 - "QC Gym Adapter Env"
Cohesion: 0.24
Nodes (7): LootEnv, Any, ndarray, What an action means in this env, as the stamped record. Published so a dataset…, The actor group, as a fresh float32 row. The copy is deliberate. mjlab hands…, [N] the strongest finger-pad contact force this sensor sees. ``None`` when the…, One mjlab task, presented as a single-env gymnasium environment. Observations…

### Community 76 - "Spawn Reachability Check"
Cohesion: 0.19
Nodes (12): get_unimanual_dk1_interface(), Task registration for all ball variants on all supported robots., _task_id_parts(), main(), ManagerBasedRlEnv, slice, Tensor, IK reachability check over every spawn cell. See tests/tasks/README.md. For… (+4 more)

### Community 77 - "Split Demos IO"
Cohesion: 0.24
Nodes (13): _column(), _features(), load_source_episodes(), main(), _print_summary(), DataFrame, ndarray, Relabel a recorded demo into the easier same-family tasks, without a sim. Every… (+5 more)

### Community 78 - "Ball Observation Terms"
Cohesion: 0.30
Nodes (12): get_command(), ball_pose(), ball_velocity(), field_pose(), ManagerBasedRlEnv, Tensor, Observation terms for the ball task category., Ball position in the env-local world frame (E, 3). No quaternion: a sphere's… (+4 more)

### Community 79 - "Release Gate Tests"
Cohesion: 0.24
Nodes (8): ContactReleaseGate, Per-step "pads not exerting grip force" check for a command's success gate.…, _fake_contact_sensor(), ContactSensor, Duck-typed ContactSensor stand-in: the gate only reads data.force., The gate ships reading the env's release threshold, not a copy., One loaded pad vetoes release, however lightly the other rests., TestContactReleaseGate

### Community 80 - "Fixed Spawn Envelope"
Cohesion: 0.22
Nodes (6): Shared fixed-spawn mechanism every category spawns from. Stack, domino and…, Whether one jittered ``(x, y)`` pose is inside the shared envelope., Whether a cell is inside the envelope at all four jitter corners., spawn_cell_in_envelope(), spawn_pose_in_envelope(), TestEnvelope

### Community 81 - "Delta Action Terms"
Cohesion: 0.18
Nodes (10): DifferentialIKActionCfg, JointPositionAction, JointPositionActionCfg, EndEffectorDeltaActionCfg, NormalizedJointPositionAction, NormalizedJointPositionActionCfg, ManagerBasedRlEnv, Configuration for :class:`EndEffectorDeltaAction`. ``actuator_names``,… (+2 more)

### Community 82 - "G1 Velocity Config"
Cohesion: 0.22
Nodes (10): ManagerBasedRlEnvCfg, Unitree G1 velocity environment configurations., Create Unitree G1 flat terrain velocity configuration., Create Unitree G1 rough terrain velocity configuration., unitree_g1_flat_env_cfg(), unitree_g1_rough_env_cfg(), RslRlOnPolicyRunnerCfg, RL configuration for Unitree G1 velocity task. (+2 more)

### Community 83 - "Debug Panel Env Shapes"
Cohesion: 0.21
Nodes (6): _fake_env(), Stands in for the real env.env.observation_manager shape used by…, --debug-obs (append_only=True) must keep every frame in the scrollback, even…, domino's task_obs_terms puts domino_poses (positions) first, then…, Any env shape this doesn't understand disables the named breakdown rather than…, TestObsLayout

### Community 84 - "DK1 Collision Contract"
Cohesion: 0.29
Nodes (11): _compensate_arm_gravity(), get_bimanual_spec(), get_unimanual_spec(), MjSpec, Cancel gravity on everything the arm actuators carry. **Why the arm needs…, _compiled_gripper_only_model(), _geom_names(), MjModel (+3 more)

### Community 85 - "Joint Mapping Tests"
Cohesion: 0.24
Nodes (7): lerobot_to_mjlab_ctrl(), mjlab_gripper_meters(), Convert real-robot joint_pos (7,) → MJLab data.ctrl (7,). Joints 0-5 (radians)…, Map real-robot gripper [0,1] → prismatic meters [0.0, ≈-0.0411]. Inverts the…, The teleop half of the shared action convention.…, TestAbsoluteCtrl, TestGripperConversion

### Community 86 - "Ball Command Term"
Cohesion: 0.24
Nodes (5): BallCommand, CommandTerm, Tensor, (E,) float: ball at target, at rest, and released., (E, 3): env-local target position (was a ball-relative delta).

### Community 87 - "QC Env Fixtures"
Cohesion: 0.18
Nodes (8): env(), place_env(), fixture, slice, Contract tests for the qc gym adapter, run against a real task. These build a…, Map each actor-observation term to its slice of the flat vector. Derived rather…, _term_slices(), TestSpaces

### Community 88 - "QC Env Observation Rules"
Cohesion: 0.20
Nodes (7): _peak_to_peak(), Largest per-column range. Never the standard deviation. The observations are…, The promise qc's evaluation loop rests on. With auto-reset on, the observation…, Replay verification needs a bit-identical start., Split observations into (robot columns, scene columns)., _split(), TestEpisodeBoundaries

### Community 90 - "Graphify Command Index"
Cohesion: 0.20
Nodes (11): Graphify Slash Command Trigger, Ingest URL into Corpus, Watch Folder Auto Rebuild, Extraction Subagent Prompt Spec, GitHub Clone Repos, Post Commit Auto Rebuild Hook, Whisper Video Transcription, AST Structural Extraction (+3 more)

### Community 91 - "G1 Tracking Config"
Cohesion: 0.22
Nodes (8): ManagerBasedRlEnvCfg, Unitree G1 flat tracking environment configurations., Create Unitree G1 flat terrain tracking configuration., unitree_g1_flat_tracking_env_cfg(), RslRlOnPolicyRunnerCfg, RL configuration for Unitree G1 tracking task., Create RL runner configuration for Unitree G1 tracking task., unitree_g1_tracking_ppo_runner_cfg()

### Community 92 - "Tracking Termination Terms"
Cohesion: 0.35
Nodes (8): bad_anchor_ori(), bad_anchor_pos(), bad_anchor_pos_z_only(), bad_motion_body_pos(), bad_motion_body_pos_z_only(), ManagerBasedRlEnv, SceneEntityCfg, Tensor

### Community 93 - "Tracking Reward Terms"
Cohesion: 0.47
Nodes (10): _get_body_indexes(), motion_global_anchor_orientation_error_exp(), motion_global_anchor_position_error_exp(), motion_global_body_angular_velocity_error_exp(), motion_global_body_linear_velocity_error_exp(), motion_relative_body_orientation_error_exp(), motion_relative_body_position_error_exp(), ManagerBasedRlEnv (+2 more)

### Community 94 - "Tracking Runner Config"
Cohesion: 0.24
Nodes (5): MotionTrackingOnPolicyRunner, _OnnxMotionModel, MjlabOnPolicyRunner, ONNX-exportable model that wraps the policy and bundles motion reference data., VecEnv

### Community 95 - "Converted Dataset Tests"
Cohesion: 0.24
Nodes (7): load_dataset(), Path, Load a converted ``.npz``, checked against ``env`` when one is given. The way a…, Including the two string fields. ``np.savez`` stores a ``str`` as a 0-d unicode…, The path a training run actually takes: the stamp arrives from the file, not…, The string field, through the npz round trip., TestLoadDataset

### Community 96 - "Running Stats Tests"
Cohesion: 0.25
Nodes (4): Running min/max/mean over an episode, O(1) memory., _RunningStats, The whole point: mean/min/max are over every value seen across all update()…, TestRunningStats

### Community 97 - "Debug Marker Helpers"
Cohesion: 0.20
Nodes (9): DebugVisualizer, add_box_wireframe(), marker_color(), DebugVisualizer, ndarray, Shared status coloring for the command terms' debug-vis goal markers. Every…, Draw an oriented box's 12 edges as thin cylinders. Unlike a second, larger…, Marker RGBA for one goal slot. Args: base_rgba: The target object's own color… (+1 more)

### Community 98 - "Ball Variant Config"
Cohesion: 0.18
Nodes (5): BallVariantCfg, Variant definitions for the ball task category. Two task families, each with…, Local y of the minigolf pit centre (offset for every non-direct layout, so the…, Height the ball starts/rolls on at the spawn side of the field., Height of the surface the target sits on.

### Community 99 - "Spawn Sampler Tests"
Cohesion: 0.36
Nodes (4): Draw jittered poses with object ``i`` on cell ``i``. One ``rand(E, 3)`` draw…, sample_fixed_spawns(), Tensor, TestSampleFixedSpawns

### Community 100 - "Spawn Lane Predicates"
Cohesion: 0.27
Nodes (8): domino_spawn_is_clear(), Whether a jittered ``(x, y)`` spawn pose is clear of the row lane., Whether a jittered ``(x, y)`` spawn pose is clear of the tower site., stack_spawn_is_clear(), _category_tables(), Unit tests for ``tasks/common/fixed_spawns.py``. One shared envelope, one pitch…, Real per-category spawn tables, same spec shape as :func:`_example_tables`.…, TestCategoryTables

### Community 101 - "Sorter Corridor Tests"
Cohesion: 0.22
Nodes (4): Whether a jittered ``(x, y)`` spawn pose is clear of the approach corridor., sorter_spawn_is_clear(), _sorter_radius(), TestSorterTable

### Community 102 - "Sorter Latch Tests"
Cohesion: 0.42
Nodes (4): SorterLatch, _latch(), _step(), TestSorterLatch

### Community 103 - "Stack Variant Invariants"
Cohesion: 0.25
Nodes (7): parametrize, Invariants of the stack variant catalog (spares, prefix rule, spawn table)., test_block_count_is_two_n(), test_blocks_tail_extend_across_target_height(), test_easier_reset_is_prefix_of_harder(), test_more_blocks_than_needed(), test_only_the_target_block_count_reaches_the_target()

### Community 104 - "Domino Fallen Tests"
Cohesion: 0.36
Nodes (3): _quat_axis_angle(), (4,) wxyz quaternion for a rotation of *angle* about *axis* (unit)., TestDominoFallenAlongRow

### Community 105 - "Robot Mesh Manifest"
Cohesion: 0.20
Nodes (9): base_link, finger_left, finger_right, link1-2, link2-3, link3-4, link4-5, link5-6 (+1 more)

### Community 106 - "Obs Layout Helper"
Cohesion: 0.20
Nodes (6): _ObsLayout, ndarray, Name to offset map for one flat obs vector (matches saved layout)., Best-effort layout; unknown envs disable the breakdown., Fold one frame's array in; True if it held a non-finite value., _to_numpy()

### Community 107 - "Joint Limit Mapper"
Cohesion: 0.22
Nodes (8): _dk1_scratch(), joint_pos_limits(), MjModel, Compiled DK1, built once, for joint limits and grasp-site FK. Built on first…, Hard (lower, upper) ctrl limits read from the compiled model., MjData, The regression this guards: a copy of dk1.xml's ranges going stale., TestJointLimits

### Community 108 - "Task ID Parsing Tests"
Cohesion: 0.33
Nodes (5): easier_target_task_ids(), parse_task_id(), Split ``LOOT-<Category>-<Task>[-<Variation>]-<Robot>`` into its parts., All registered same-family, splittable tasks with fewer objects., TestTaskIds

### Community 109 - "Ball Termination Terms"
Cohesion: 0.29
Nodes (9): ball_hits_wall(), ball_off_platform(), ball_out_of_bounds(), ManagerBasedRlEnv, Tensor, Termination terms for the ball task category. ``ball_off_platform`` /…, Ball strayed far from the env origin (e.g. knocked off the table)., Ball dropped below the platform's base level outside the pit. ``spawn_z`` is… (+1 more)

### Community 110 - "Sorter Asset Layout"
Cohesion: 0.27
Nodes (8): load_sorter_layout(), MjSpec builders and layout for the shape-sorter task category. The cube and its…, One face hole of the sorter cube, in the cube body frame., One STL piece; centred at its cross-section centroid, axis +z., SorterHole, SorterLayout, SorterPiece, Shape-sorter variants. A variant fixes the shape count and one spawn mode: -…

### Community 111 - "Sorter Variant Tests"
Cohesion: 0.22
Nodes (3): ShapeSorterVariantCfg, The command term reads the variants module's tolerances., TestSorterVariants

### Community 112 - "Grace Window Label Tests"
Cohesion: 0.20
Nodes (5): F during the grace window: the episode is staged as failure now, without…, V overrides a termination-labeled failure to success (e.g. the task succeeded…, Undecided after the grace period: _execute_step auto-saves failure., Within the grace period nothing is saved. The window stays open., TestProvisionalFailureActions

### Community 113 - "Empty Grasp Sensor Tests"
Cohesion: 0.22
Nodes (8): _gripper_index(), _hold_action(), ndarray, The cube-filtered sensor: the only key that answers "touched the cube".…, Neither sensor may fire on an empty grasp. The arm holds its reset pose, which…, The action that parks the arm where it is and holds the gripper open. A zero…, Which action column drives the gripper: the last one, after the 6-D delta.…, TestObjectContact

### Community 114 - "Domino Aligned Tests"
Cohesion: 0.38
Nodes (4): parametrize, Tensor, _quat_mul(), TestDominoUprightAligned

### Community 115 - "Claude Exports Reference"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 116 - "Delta Action Processing"
Cohesion: 0.28
Nodes (6): DifferentialIKAction, EndEffectorDeltaAction, Tensor, Differential IK on the grasp site, with the action box made real. See the…, The IK frame's ``(position, wxyz quaternion)``, in the world frame. Exactly the…, Clamp to [-1, 1], then apply scale and offset. The clamp binds every caller,…

### Community 117 - "Ball Success Terms"
Cohesion: 0.25
Nodes (6): Ball command. Samples spawns, tracks the ball, tests at-target success., Reward terms for the ball task category. Ball has no progress signal: a ball is…, ball_in_target_zone(), Tensor, Pure success criteria for the ball category. Plain tensor math, no mjlab or…, xy-radius target test for the ball. Hole targets additionally require the ball…

### Community 118 - "Domino Topple Kernels"
Cohesion: 0.28
Nodes (8): chain_topple_success(), domino_fallen_along_row(), domino_upright_aligned(), Tensor, Pure success criteria for the domino category. Plain tensor math, no mjlab or…, All dominoes chain-fallen the same way, in row order. Args: fallen: (E, G) bool…, Upright + row-alignment check for dominoes. A domino counts as standing when…, Per-domino "toppled along the row" test (chain semantics). A domino counts as…

### Community 119 - "Sorter Piece Specs"
Cohesion: 0.25
Nodes (9): get_sorter_piece_spec(), get_sorter_spec(), MjSpec, Shape, A free-body STL sorter piece, tinted with its socket colour., The free-body STL shape-sorter cube (visual mesh + convex parts). A…, make_shape_sorter_env_cfg(), ManagerBasedRlEnvCfg (+1 more)

### Community 120 - "Sorter Spawn Faces Tests"
Cohesion: 0.31
Nodes (5): allowed_faces(), Shape, Faces whose hidden shape is not required. For no_rotation spawn., Unit tests for the success criteria in the per-category ``mdp/success.py``…, TestNoRotationFaces

### Community 121 - "OpenCode Exports Reference"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 122 - "Tracking Command Config"
Cohesion: 0.29
Nodes (5): MotionCommandCfg, MotionLoader, CommandTermCfg, ManagerBasedRlEnv, VizCfg

### Community 123 - "QC Env Sensors"
Cohesion: 0.25
Nodes (5): ContactSensor, ManagerBasedRlEnv, The mjlab env underneath, for tools that need the simulator. Read-only: the…, The named contact sensor, or ``None`` if the task declares none., Build the task. Args: task_id: a registered mjlab task id, e.g. ``LOOT-…

### Community 124 - "Viewer Shutdown Check"
Cohesion: 0.32
Nodes (6): Wait for GLFW teardown; True if done within TEARDOWN_TIMEOUT., wait_for_viewer_teardown(), main(), Check that closing the native viewer doesn't segfault at interpreter exit. See…, Open a tiny passive viewer, close it, and return straight into interpreter…, run_child()

### Community 125 - "Spawn Pitch Tests"
Cohesion: 0.39
Nodes (3): Minimum centre-to-centre pitch (m) for two neighbouring spawn cells. Set by the…, spawn_pitch(), TestSpawnPitch

### Community 126 - "Sorter Latch Kernel"
Cohesion: 0.32
Nodes (5): device, Tensor, Latched mouth-plane crossing for the shape sorter. A piece scores the moment it…, Advance one step; returns the (E, S) bool latch state., SorterLatch

### Community 127 - "Recording Mixin Guards"
Cohesion: 0.25
Nodes (4): Accepted keys must match what the HUD advertises per state., A termination before any frame was recorded has nothing to decide. It must go…, Q must not bypass an open grace window. The un-decided episode would be…, TestStateGuards

### Community 128 - "Reset Gate Tests"
Cohesion: 0.25
Nodes (4): What ends the reset phase (ADR-016). The gate reads the *leader*, and it is one…, Before the hardware has reported once there is nothing to compare, and the gate…, Held open, the arm must keep being driven — a frozen sim is what the operator…, TestResetPhaseAlignmentGate

### Community 130 - "DK1 Collision Tuning"
Cohesion: 0.29
Nodes (6): CollisionCfg, _collision_with_pad_tuning(), DK1 collision configs: full, gripper-only, gripper-only with self-collision., CollisionCfg with shared contact tuning (grippy pads, plain rest)., # TODO: fix self-collision of dark part near link2-3/link3-4., # TODO: fix oversized collision box of link2-3.

### Community 131 - "Tracking Viewer Ghost"
Cohesion: 0.29
Nodes (4): Any, ViserServer, Create motion scrubber controls in the Viser viewer., Enable or disable the motion scrubber GUI controls.

### Community 132 - "Tracking Anchor Observations"
Cohesion: 0.57
Nodes (6): motion_anchor_ori_b(), motion_anchor_pos_b(), ManagerBasedRlEnv, Tensor, robot_body_ori_b(), robot_body_pos_b()

### Community 133 - "Shared Reset Events"
Cohesion: 0.29
Nodes (6): ManagerBasedRlEnv, SceneEntityCfg, Tensor, Reset events shared by all task categories. mjlab's…, Reset the named joints to ``pose`` plus a uniform sample. Joints absent from…, reset_joints_to_pose()

### Community 134 - "Success Termination Helper"
Cohesion: 0.29
Nodes (6): command_success_termination(), CommandTerm, ManagerBasedRlEnv, Tensor, Termination-term factories shared by every task category., Task's success condition (task-done, at-rest, and released) all met.

### Community 135 - "Place Success Kernel"
Cohesion: 0.33
Nodes (6): place_progress(), place_reached(), Tensor, Pure success criteria for the place category. Plain tensor math, no mjlab or…, Cube centre within ``pos_threshold`` of the target (3D, position-only). Args:…, Continuous distance-based shaping, in [0, 1]. 1.0 exactly at the target,…

### Community 137 - "Claude Query Reference"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 138 - "Release Gate ADR"
Cohesion: 0.33
Nodes (5): Consequences, Contact-only release gate, Why deleted rather than kept as a second opinion, Why the joint gate was a proxy, Why the pad force joins the observation

### Community 139 - "Mesh Converter Script"
Cohesion: 0.53
Nodes (5): convert_visual(), linear_to_srgb(), main(), Path, One-time conversion of the official TRLC-DK1 URDF meshes to MuJoCo-loadable…

### Community 140 - "Serial Reader Connect"
Cohesion: 0.33
Nodes (4): _connect_with_retry(), Open serial connections; retry handshake on dropped packets., Build device and connect, retrying transient ConnectionError., _T

### Community 141 - "Upright Spawn Writer"
Cohesion: 0.33
Nodes (5): Entity, Tensor, Upright-spawn writer shared by every task category. :func:`write_upright_spawn`…, Write each entity upright at its ``(xy, spawn_yaw)`` pose. Sits each entity on…, write_upright_spawn()

### Community 142 - "OpenCode Query Reference"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 147 - "DK1 Grasp Contract"
Cohesion: 0.40
Nodes (5): _build_grasp_model(), MjModel, DK1 grasp contact contract on the compiled MuJoCo model. The sim gripper is a…, Unimanual DK1 plus a free box, physics matching the play env., test_grasp_holds_piece_without_crushing()

### Community 148 - "Sim Profile Settings"
Cohesion: 0.40
Nodes (5): Control Step 20 Hz, cone elliptic, impratio 10, Sim profile benchmark campaign, iterations 100 / ls_iterations 50, timestep 0.005 / decimation 10

### Community 149 - "Fixed Spawn ADR"
Cohesion: 0.40
Nodes (4): Fixed spawn points from the arm's measured envelope, Measured envelope, Pitch from the gripper, Zoning rule

### Community 150 - "Agent Workflow Guide"
Cohesion: 0.40
Nodes (5): Flag ADR conflicts explicitly, Use the glossary's vocabulary, Read CONTEXT.md and docs/adr before exploring, Local markdown issue tracker (.scratch/), Wayfinding map, claims and blocking

### Community 151 - "Ball Command Config"
Cohesion: 0.50
Nodes (3): BallCommandCfg, CommandTermCfg, ManagerBasedRlEnv

### Community 152 - "Pad Contact Force"
Cohesion: 0.60
Nodes (4): pad_contact_force(), ContactSensor, Tensor, (E,) strongest finger-pad contact force magnitude [N], per env. The one…

### Community 153 - "Domino Command Config"
Cohesion: 0.50
Nodes (3): DominoCommandCfg, CommandTermCfg, ManagerBasedRlEnv

### Community 154 - "Termination Grace Tests"
Cohesion: 0.40
Nodes (3): Success termination → episode staged as success, reset phase entered., Non-success termination saves nothing yet. The grace window opens so the…, TestTerminationHandling

### Community 155 - "QC Train Script"
Cohesion: 0.60
Nodes (4): die(), train_qc.sh script, usage(), XLA_PYTHON_CLIENT_PREALLOCATE

### Community 156 - "Query Traversal Modes"
Cohesion: 0.67
Nodes (4): MCP Server Graph Access, BFS DFS Traversal Modes, Constrained Query Expansion, Query Path Explain Flow

### Community 157 - "OpenCode Pipeline Concepts"
Cohesion: 0.50
Nodes (4): Extraction Spec OpenCode, Query Flow OpenCode, Update Flow OpenCode, Graphify Pipeline OpenCode Variant

### Community 158 - "Claude Add Watch Reference"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 159 - "Claude Hook Reference"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 160 - "Claude Update Reference"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 163 - "Follower Action Sender"
Cohesion: 0.50
Nodes (3): Any, ndarray, Send (7,) action to follower; joints rad, gripper in [0, 1].

### Community 164 - "Ghost Model Setup"
Cohesion: 0.50
Nodes (3): build_ghost_model_and_qpos(), Set up recording state; call from subclass __init__., Build ghost model: robot visual geoms only, colored, at the start pose.

### Community 165 - "OpenCode Add Watch"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 166 - "OpenCode Hooks"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 167 - "OpenCode Update"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 168 - "DK1 Test Scenes"
Cohesion: 0.67
Nodes (4): basic_scene.py (build the robot through get_unimanual_dk1_robot_cfg, the same path training uses), basic_scene.xml (raw MJCF load of dk1.xml, also checks <include> and meshdir resolution), test_dk1_collision.py (gripper-only mode collides exactly the wrist/finger/pad geoms; pads keep tuned contact parameters), TRLC-DK1 checks (two eyeball scenes plus the pytest collision contract)

### Community 169 - "Precommit Hooks"
Cohesion: 0.67
Nodes (3): pre-commit hook: ruff check --fix, pre-commit hook: ruff format, install.sh / install.ps1

## Ambiguous Edges - Review These
- `MCP Server Graph Access` → `BFS DFS Traversal Modes`  [AMBIGUOUS]
  .claude/skills/graphify/references/exports.md · relation: conceptually_related_to
- `LOOT-bench Serena Project Config` → `loot-bench Robot-Manipulation Benchmark Suite`  [AMBIGUOUS]
  .serena/project.yml · relation: conceptually_related_to

## Knowledge Gaps
- **214 isolated node(s):** `VizCfg`, `Ranges`, `VizCfg`, `VizCfg`, `Step 6b - Wiki (only if --wiki flag)` (+209 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1240 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **80 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `MCP Server Graph Access` and `BFS DFS Traversal Modes`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `LOOT-bench Serena Project Config` and `loot-bench Robot-Manipulation Benchmark Suite`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `make_tracking_env_cfg()` connect `Tracking Env Config` to `Delta Action Terms`, `Tracking Command Config`, `G1 Tracking Config`, `Go1 Velocity Config`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Why does `MotionCommandCfg` connect `Tracking Command Config` to `Tracking Env Config`, `Tracking Termination Terms`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `make_manipulation_env_cfg()` connect `Tracking Env Config` to `Ball Env Config`, `Color Palette Tiers`, `Go1 Velocity Config`, `Shared Task Workbench`, `Sorter Piece Specs`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `MotionCommand` (e.g. with `motion_anchor_ori_b()` and `motion_anchor_pos_b()`) actually correct?**
  _`MotionCommand` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `Recorder` (e.g. with `_LiveRecordingViewer` and `_ReplayAndRecord`) actually correct?**
  _`Recorder` has 4 INFERRED edges - model-reasoned connections that need verification._