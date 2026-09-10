# Graph Report - LOOT-bench  (2026-09-10)

## Corpus Check
- 322 files · ~181,287 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 3061 nodes · 5400 edges · 237 communities (161 shown, 51 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 244 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `df2d26f3`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- check_success_reward.py
- bc_gate.py
- ADR-016: control at 20 Hz through a 5 cm end-effector delta
- compare_sim_settings.py
- ball/assets.py
- UniformVelocityCommand
- ManagerBasedRlEnv
- assemble_transitions
- live_record.py
- Recorder
- make_sorter_assets.py
- adapter_for
- MotionCommand
- convert.py
- EpisodeStats
- ball/mdp/commands.py
- peg_insertion/mdp/commands.py
- peg_insertion/assets.py
- test_fixed_spawns.py
- NativeViewerWrapper
- mapper.py
- LiveReader
- ._panel
- stack/variants.py
- RecordingViewerMixin
- shape_sorter/mdp/commands.py
- split_demos.py
- _stack_progress_and_gate
- marker_color
- LeRobotDataset
- common/commands.py
- shaped_task_reward
- write_upright_spawn
- What You Must Do When Invoked
- _build
- verify_replay.py
- delta_action_from_poses
- domino_env_cfg.py
- _quat_axis_angle
- table_entity_cfg
- What You Must Do When Invoked
- cartpole_env_cfg.py
- MultiCubeLiftingCommand
- LootEnv
- domino/mdp/commands.py
- make_harness
- run_evaluate
- dk1_constants.py
- manipulation/mdp/observations.py
- InsertionCommand
- TestCompiledTask
- ReaderFrame
- load_extra_info
- DatasetReader
- state_replay.py
- place/mdp/commands.py
- stack_progress
- test_convert.py
- manipulation/mdp/rewards.py
- check_dataset_matches_env
- DominoCommand
- stack/mdp/commands.py
- Session (one teleop recording run)
- trlc_dk1/__init__.py
- LiveDebugPanel
- live_record.py (record episodes at ~50 Hz from live GELLO teleop)
- QC seam tests (contract tests for loot_bench/qc/ plus the two standalone scripts)
- _ReplayAndRecord
- Shape-sorter task category (insert every piece into its matching hole in a free cube)
- ContactReleaseGate
- Domino task category (build a row of upright dominoes)
- make_manipulation_env_cfg
- SuccessGate
- test_debug_panel.py
- ball_env_cfg.py
- test_recording_mixin.py
- TestStepContract
- ObservationTermCfg
- sweep.sh
- _split
- Tensor
- LiftingCommandCfg
- TerminationTermCfg
- test_stack_variants.py
- TestSpaces
- ball/mdp/observations.py
- Stack task category (stack blocks to reach a target height)
- Convention record (dataset stamp)
- Task reward (one term, [-1, 0])
- unitree_g1_rough_env_cfg
- StackCommand
- save_env_render
- Place task category (carry one cube to a sampled target and let go)
- _fake_env
- Category (task family)
- make_tracking_env_cfg
- tracking/mdp/rewards.py
- leader_grasp_pose
- Workbench (shared env skeleton)
- joint_pos_limits
- SuccessGate
- tracking/mdp/terminations.py
- MotionTrackingOnPolicyRunner
- _RunningStats
- check_success_reward.py (teleport objects to their goals and check the command term's success metric fires)
- graphify reference: extra exports and benchmark
- wait_for_viewer_teardown
- Contact-only release gate
- ._at
- dataset.py
- manifest.json
- _ObsLayout
- easier_target_task_ids
- TestProvisionalFailureActions
- test_env.py
- normalized_scale_offset
- actions.py
- _connect_with_retry
- MotionCommandCfg
- BallCommand
- TestPlaceProgress
- sweep.sh (generic alpha x seed sweep, owns the sweep and nothing about the experiment)
- TestStateGuards
- TestResetPhaseAlignmentGate
- spawn_pose_in_envelope
- .create_gui
- tracking/mdp/observations.py
- TestChainToppleSuccess
- reset_joints_to_pose
- load_dataset
- .__init__
- tasks/common shared helpers
- convert_glb_meshes_to_obj.py
- graphify reference: extra exports and benchmark
- TestDeferredQposQvel
- FakeRecorder
- .send_action
- common/success.py
- Read CONTEXT.md and docs/adr before exploring
- sample_fixed_spawns
- graphify reference: query, path, explain
- graphify reference: query, path, explain
- TestTerminationHandling
- train_qc.sh
- pre-commit hook: ruff check --fix
- ManipulationOnPolicyRunner
- VelocityOnPolicyRunner
- ._init_recording
- MjSpec
- RewardTermCfg
- TestSorterTable
- ._debug_vis_impl
- list_tasks.py
- test_split_demos.py
- critic-off_100k-100k.sh
- critic-off_1m-1m.sh
- critic-selection_100k-100k.sh
- critic-selection_1m-1m.sh
- critic-selection-steering_100k-100k.sh
- critic-selection-steering_1m-1m.sh
- critic-steering_100k-100k.sh
- critic-steering_1m-1m.sh
- fixed-s_fixed-t_1m-1m.sh
- fixed-s_fixed-t_200k-0.sh
- fixed-s_fixed-t_200k-200k.sh
- fixed-s_fixed-t_critic-off_600k-0.sh
- fixed-s_fixed-t_critic-selection_600k-0.sh
- fixed-s_rand-t_200k-0.sh
- fixed-s_rand-t_200k-200k.sh
- fixed-s_rand-t_50k-600k.sh
- rand-s_fixed-t_50k-600k.sh
- rand-s_rand-t_50k-600k.sh
- DataFrame
- tracking/__init__.py
- velocity/__init__.py
- install.sh script
- loot_bench/__init__.py
- loot-bench
- SceneEntityCfg
- device
- domino/mdp/success.py
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- test_success.py
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- graphify.js
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- CLAUDE.md
- .claude/skills/graphify/references/extraction-spec.md
- .opencode/skills/graphify/references/extraction-spec.md
- _category_tables
- check_spawn_reachability.py
- .__init__
- domino_row_slots
- TestDominoFallenAlongRow
- config/yam/__init__.py
- ball_in_target_zone
- inserted_count_reward
- Fixed spawn points from the arm's measured envelope
- BallCommandCfg
- TestEpisodeBoundaries
- illegal_contact
- insertion_complete
- Entity
- device
- MjSpec

## God Nodes (most connected - your core abstractions)
1. `MotionCommand` - 63 edges
2. `Recorder` - 38 edges
3. `make_harness()` - 32 edges
4. `assemble_transitions()` - 31 edges
5. `make_manipulation_env_cfg()` - 26 edges
6. `_build()` - 25 edges
7. `LootEnv` - 24 edges
8. `StackCommand` - 23 edges
9. `RecordingViewerMixin` - 23 edges
10. `LiveDebugPanel` - 23 edges

## Surprising Connections (you probably didn't know these)
- `BC gate (tests/qc/bc_gate.py)` --semantically_similar_to--> `Splitter self-check on recomputed reward`  [INFERRED] [semantically similar]
  CONTEXT.md → CLAUDE.md
- `verify_replay.py (bitwise qpos verification that an episode replays exactly)` --semantically_similar_to--> `relabel_episode recomputes the source episode's own reward, and that self-check is what catches real drift`  [INFERRED] [semantically similar]
  loot_bench/scripts/live_bridge/README.md → tests/scripts/README.md
- `Use the glossary's vocabulary` --references--> `Category (task family)`  [INFERRED]
  docs/agents/domain.md → CONTEXT.md
- `manual_replay_demo.py (interactive viewer replay of corkinbox100-fullrel through the sim)` --semantically_similar_to--> `record_replay.py (replay a LeRobot dataset through the sim, no hardware)`  [INFERRED] [semantically similar]
  tests/live_bridge/README.md → loot_bench/scripts/live_bridge/README.md
- `Replay determinism decided by GPU contact impulses (bitwise match until first gripper contact)` --semantically_similar_to--> `conftest.py imports lerobot before mjlab; the reverse native-library order segfaults during collection`  [INFERRED] [semantically similar]
  loot_bench/scripts/live_bridge/README.md → tests/qc/README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **The 20 Hz / ±5 cm end-effector delta action convention** — docs_adr_016_control_at_20hz_through_a_5cm_end_effector_delta_delta_control, docs_adr_016_control_at_20hz_through_a_5cm_end_effector_delta_action_box, docs_adr_016_control_at_20hz_through_a_5cm_end_effector_delta_actuator_split, docs_adr_016_control_at_20hz_through_a_5cm_end_effector_delta_ik_max_dq_torque_cap, docs_adr_016_control_at_20hz_through_a_5cm_end_effector_delta_gravity_compensation, docs_adr_016_control_at_20hz_through_a_5cm_end_effector_delta_episode_budget_15s, docs_adr_011_datasets_stamp_the_action_convention_stamp_keys [EXTRACTED 1.00]
- **The three faults the rebuild found behind the flat-zero curve** — docs_adr_001_rebuild_the_qc_integration_from_09e5dfa_flat_zero_success_curve, docs_adr_005_one_action_unit_is_half_a_joint_travel_normalized_joint_position_action, docs_adr_007_ee_pose_publishes_a_6d_rotation_ee_pose_6d_rotation, docs_adr_008_episodes_start_at_the_home_pose_home_pose_reset [EXTRACTED 1.00]
- **Teleop recording pipeline: record, label, inspect, replay, verify, keep** — loot_bench_scripts_live_bridge_readme_live_record, loot_bench_scripts_live_bridge_readme_episode_labeling_keys, loot_bench_scripts_live_bridge_readme_debug_panel, loot_bench_scripts_live_bridge_readme_record_replay, loot_bench_scripts_live_bridge_readme_verify_replay, loot_bench_scripts_live_bridge_readme_out_scratch_dataset_kept [EXTRACTED 1.00]
- **One task reward in [-1, 0], exactly 0 only at success (ADR-006), floored at -MIN_SHORTFALL** — loot_bench_tasks_ball_readme_ball_rewards, loot_bench_tasks_domino_readme_domino_rewards, loot_bench_tasks_place_readme_place_rewards, loot_bench_tasks_shape_sorter_readme_sorter_rewards, loot_bench_tasks_stack_readme_stack_rewards, loot_bench_tasks_peg_insertion_readme_peg_sparse_rewards [EXTRACTED 1.00]
- **Sim parameters decided together from one measurement campaign** — docs_sim_settings_timestep_decimation, docs_sim_settings_solver_iterations, docs_sim_settings_cone_impratio, docs_sim_settings_nativeccd_disabled, docs_sim_settings_profile_comparison [EXTRACTED 1.00]
- **Success gating pipeline: criterion to latched episode metric** — context_success_criterion, context_successgate, context_released, context_episode_success [EXTRACTED 1.00]
- **Action convention contract spanning recorder, dataset and trainer** — context_action_box, context_end_effector_delta, context_control_step, context_action_convention, context_convention_record [INFERRED 0.85]
- **Demonstration-to-training-run flow: record, split, gate, smoke, sweep** — loot_bench_scripts_live_bridge_readme_live_record, tests_scripts_readme_test_split_demos, tests_qc_readme_bc_gate, run_scripts_readme_sweep_sh, tests_qc_readme_make_smoke_dataset [INFERRED 0.85]
- **The loot-bench ↔ qc cross-repo contract (no version pin, checked not assumed)** — docs_adr_002_training_lives_in_the_upstream_qc_checkout_training_in_qc, docs_adr_004_qc_venv_reaches_loot_bench_by_pythonpath_pythonpath_reach, docs_adr_009_the_gym_adapter_is_a_seam_only_gym_adapter_seam, docs_adr_009_the_gym_adapter_is_a_seam_only_contact_telemetry_key_contract, docs_adr_011_datasets_stamp_the_action_convention_convention_record, docs_adr_010_episode_boundary_and_bootstrap_mask_are_separate_terminals_vs_masks [INFERRED 0.85]

## Communities (237 total, 51 thin omitted)

### Community 0 - "check_success_reward.py"
Cohesion: 0.12
Nodes (41): _check_ball_at_target(), _check_ball_minigolf_using_wall(), _check_domino_3pieces(), _check_domino_knockover(), _check_minigolf_using_wall_blocks_direct_shots(), _check_place(), _check_shape_sorter_full(), _check_stack3_easy() (+33 more)

### Community 1 - "bc_gate.py"
Cohesion: 0.06
Nodes (51): float32, Namespace, build_chunks(), episode_bounds(), first_frames(), fit(), FitCfg, FitData (+43 more)

### Community 2 - "ADR-016: control at 20 Hz through a 5 cm end-effector delta"
Cohesion: 0.05
Nodes (58): Flat-zero success curve (the qc-fql failure mode), ADR-001: Rebuild the QC integration from 09e5dfa, RSL-RL PPO cfg exists only to satisfy mjlab registration, ADR-002: Training lives in the upstream qc checkout, ADR-003: Pin mjlab to >=1.5.1,<1.6, lerobot-before-mjlab import order warning, ADR-004: qc venv reaches loot_bench by PYTHONPATH, never pip install, Keep run input/output off NFS $HOME and UV_CACHE_DIR off /dev/shm (+50 more)

### Community 3 - "compare_sim_settings.py"
Cohesion: 0.07
Nodes (38): Exception, acc_stats(), AccStats, apply_profile(), build_env(), _fmt_diverge(), GraspScenario, main() (+30 more)

### Community 4 - "ball/assets.py"
Cohesion: 0.17
Nodes (21): _add_blocker(), _add_box(), _add_incline(), _add_marker(), _add_obstacle(), _add_platform_with_hole(), _add_railing(), _add_ramp_platform() (+13 more)

### Community 5 - "UniformVelocityCommand"
Cohesion: 0.06
Nodes (35): commands_vel(), ManagerBasedRlEnv, SceneEntityCfg, Tensor, terrain_levels_vel(), VelocityStage, foot_air_time(), foot_contact() (+27 more)

### Community 6 - "ManagerBasedRlEnv"
Cohesion: 0.08
Nodes (38): angular_momentum_penalty(), body_angular_velocity_penalty(), feet_air_time(), feet_clearance(), feet_slip(), feet_swing_height, DebugVisualizer, ManagerBasedRlEnv (+30 more)

### Community 7 - "assemble_transitions"
Cohesion: 0.07
Nodes (37): assemble_transitions(), _check_rewards(), _episode_bounds(), floating, integer, intp, NDArray, Recorded rows -> the transition arrays ``qc`` trains on. A recording is one row… (+29 more)

### Community 8 - "live_record.py"
Cohesion: 0.06
Nodes (33): ask_operator(), _build_wrist_pip(), _cast_gripper_ray(), _find_camera_id(), _find_site_id(), _get_mj_model(), _get_pip_app(), _get_qimage_and_qpixmap() (+25 more)

### Community 9 - "Recorder"
Cohesion: 0.11
Nodes (18): Drop staged episode; returns its label or None., Drop the current episode without saving (operator rejected the demo)., Recorder, _add_dummy_frame(), patch, undo_last_save() drops the staged episode and reports its label. An undone…, Single-level undo: a second undo has nothing to drop., Only the last save is undoable: staging episode 2 committed episode 1, so undo… (+10 more)

### Community 10 - "make_sorter_assets.py"
Cohesion: 0.12
Nodes (40): best_fit(), classify_hole(), cross2(), dist_to_boundary(), emit_xml(), extrude(), face_frame(), face_loops() (+32 more)

### Community 11 - "adapter_for"
Cohesion: 0.26
Nodes (11): adapter_for(), plan_for(), SplitPlan, variant_from_task_id(), Synthetic Stack5-Easy episode: 10 cubes, only their z varying.…, Pads loaded to the end: no success, even with the qpos wide open. The old qpos…, _stack_episode(), _stack_state() (+3 more)

### Community 12 - "MotionCommand"
Cohesion: 0.11
Nodes (6): MotionCommand, CommandTerm, Tensor, Clip joint positions and write root + joint state to sim., Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``. Called after…, Reset to exact reference state at a specific frame. Like ``_resample_command``…

### Community 13 - "convert.py"
Cohesion: 0.11
Nodes (25): action_convention(), build_dataset(), convert(), _load_columns(), _load_many(), main(), Any, floating (+17 more)

### Community 14 - "EpisodeStats"
Cohesion: 0.09
Nodes (16): EpisodeStats, Per-episode bookkeeping for the qc adapter. Kept apart from the environment so…, Running totals for one episode, and the info dict qc reads at the end.…, Fold one step's reward, and its two contact forces, in., The per-step info dict. Published every step rather than only at the end: qc…, Unit tests for ``loot_bench/qc/episode.py``., A category with no single scored object publishes none of the three., The case that makes the latch worth having. A success gate can unlatch: a block… (+8 more)

### Community 15 - "ball/mdp/commands.py"
Cohesion: 0.24
Nodes (12): get_command(), Ball command. Samples spawns, tracks the ball, tests at-target success., Reward terms for the ball task category. Ball has no progress signal: a ball is…, ball_hits_wall(), ball_off_platform(), ball_out_of_bounds(), ManagerBasedRlEnv, Tensor (+4 more)

### Community 16 - "peg_insertion/mdp/commands.py"
Cohesion: 0.21
Nodes (15): get_command(), Archived peg-insertion command. Unregistered, see README. Goal bores, fixed peg…, board_pose(), peg_poses(), ManagerBasedRlEnv, Tensor, Observation terms for the peg-insertion task category (archived, unregistered)., Board pose, env-local (E, 7). Constant per task. (+7 more)

### Community 17 - "peg_insertion/assets.py"
Cohesion: 0.07
Nodes (39): InsertionVariantCfg, _add_walls(), frame_axes(), get_board_spec(), get_peg_spec(), get_sheet_board_spec(), get_sheet_spec(), _mat2quat() (+31 more)

### Community 18 - "test_fixed_spawns.py"
Cohesion: 0.13
Nodes (14): min_pairwise_separation(), Tensor, Shared fixed-spawn mechanism every category spawns from. Stack, domino and…, Check a category's spawn table; return a list of violations (``[]`` ok). Args:…, Minimum centre-to-centre pitch (m) for two neighbouring spawn cells. Set by the…, Minimum centre-to-centre distance (m) over all cell pairs., spawn_pitch(), validate_spawn_table() (+6 more)

### Community 19 - "NativeViewerWrapper"
Cohesion: 0.15
Nodes (11): Handle, main(), Play entry point: mjlab's player with the loot-bench tasks registered., NativeViewerWrapper, NativeMujocoViewer, Native viewer with safe teardown and uncapped FPS. close() waits for GLFW…, NativeMujocoViewer with safe teardown and throttled overlays., main() (+3 more)

### Community 20 - "mapper.py"
Cohesion: 0.11
Nodes (18): leader_is_aligned(), lerobot_to_mjlab_ctrl(), mjlab_gripper_meters(), ndarray, The teleop half of the shared action convention. A leader-arm pose is joint…, Convert real-robot joint_pos (7,) → MJLab data.ctrl (7,). Joints 0-5 (radians)…, The grasp-site pose every episode starts from, in the base frame.…, Whether an episode may start with the leader held at this pose. True when one… (+10 more)

### Community 21 - "LiveReader"
Cohesion: 0.20
Nodes (9): main(), Print the leader's live joint positions so you can pick a new home pose. Move…, LiveReader, Reads frames from a real DK1 robot in real-time. Connects to DK1Follower (the…, patch, LiveReader wraps DK1Follower + DK1Leader for real-time data., Can instantiate without connecting (connect_on_init=False)., Leader-only mode: no follower connect, joint_pos mirrors leader_action. (+1 more)

### Community 22 - "._panel"
Cohesion: 0.11
Nodes (10): Per-frame output: EEF/GOAL/OBJECT/REWARD/DONE/ANOMALIES + optional FPS line…, The min/mean/max breakdown must NOT appear every step. Only in the once-per-…, A NaN in joint_pos is checked live (cheap), but not shown until the episode-end…, capsys is not a terminal, so isatty() is False here. This is the…, The old overwrite style is the default when actually attached to a terminal:…, The FPS/step/speed/RTF line mjlab's base viewer normally prints on its own…, Nothing recorded yet -> nothing to summarize, no banner printed., The reported min/mean/max must span every frame, not just whatever the last… (+2 more)

### Community 23 - "stack/variants.py"
Cohesion: 0.11
Nodes (18): Difficulty, order_color(), Shared object color palette for the task categories' asset builders., Color an object by which split tier first needs it. Objects tail-extend across…, get_block_spec(), MjSpec, MjSpec builders for the stack blocks., make_stack_env_cfg() (+10 more)

### Community 24 - "RecordingViewerMixin"
Cohesion: 0.12
Nodes (11): Drop last staged episode; single-level undo., Extract the Episode_Termination/* sub-dict from env.step() extras., Whether the success termination fired; extras is a fresh snapshot taken during…, Names of the termination terms that fired, read from step() extras., Multi-episode recording with 2 s failure grace and reset phase., Success saves at once; other dones get a 2 s failure grace., Two-column HUD text: state, counts, and valid keys., Log stage breakdown for slow steps. (+3 more)

### Community 25 - "shape_sorter/mdp/commands.py"
Cohesion: 0.05
Nodes (40): load_sorter_layout(), MjSpec builders and layout for the shape-sorter task category. The cube and its…, One face hole of the sorter cube, in the cube body frame., One STL piece; centred at its cross-section centroid, axis +z., SorterHole, SorterLayout, SorterPiece, get_command() (+32 more)

### Community 26 - "split_demos.py"
Cohesion: 0.14
Nodes (23): _column(), _count_disturbed(), _features(), lagged_rewards(), load_source_episodes(), main(), _print_summary(), proprio_width() (+15 more)

### Community 27 - "_stack_progress_and_gate"
Cohesion: 0.13
Nodes (23): CategoryAdapter, _check_source_reward(), _domino_progress_and_gate(), frame_state(), FrameState, _held_block_mask(), _make_domino_adapter(), _make_sorter_adapter() (+15 more)

### Community 28 - "marker_color"
Cohesion: 0.11
Nodes (16): DebugVisualizer, add_box_wireframe(), marker_color(), DebugVisualizer, ndarray, Shared status coloring for the command terms' debug-vis goal markers. Every…, Draw an oriented box's 12 edges as thin cylinders. Unlike a second, larger…, Marker RGBA for one goal slot. Args: base_rgba: The target object's own color… (+8 more)

### Community 29 - "LeRobotDataset"
Cohesion: 0.13
Nodes (9): LeRobotDataset, ndarray, Tensor, Flush buffered frames with label; seed restores replay layout., Write the staged episode to its dataset (point of no return)., Un-mocked round-trip against the installed lerobot. Verifies the staging+label…, Un-mocked: save two episodes, undo the second. Only the first lands on disk., test_recorder_real_roundtrip() (+1 more)

### Community 30 - "common/commands.py"
Cohesion: 0.10
Nodes (24): pad_contact_force(), ContactSensor, Tensor, Command-term helpers shared by the task categories. The success metrics of the…, # TODO: check with real robot, (E,) strongest finger-pad contact force magnitude [N], per env. The one…, Shared workbench for all tasks. Categories pass core parts as parameters. The…, ee_pose_w() (+16 more)

### Community 31 - "shaped_task_reward"
Cohesion: 0.13
Nodes (15): command_sparse_task_reward(), command_task_reward(), CommandTerm, ManagerBasedRlEnv, Tensor, Reward-term factories shared by every task category. **The convention.** A…, The task reward: ``progress - 1``, floored, and exactly ``0`` at success. Args:…, The whole reward for a category that reports progress. (+7 more)

### Community 32 - "write_upright_spawn"
Cohesion: 0.29
Nodes (5): Entity, Tensor, Upright-spawn writer shared by every task category. :func:`write_upright_spawn`…, Write each entity upright at its ``(xy, spawn_yaw)`` pose. Sits each entity on…, write_upright_spawn()

### Community 33 - "What You Must Do When Invoked"
Cohesion: 0.07
Nodes (26): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+18 more)

### Community 34 - "_build"
Cohesion: 0.12
Nodes (12): _build(), A bounded policy can only emit [-1, 1]. An action outside it is one the policy…, The recorder's mapping clamps and the action term clamps, so an out-of-box row…, Episode structure is read off row order, so an out-of-order file would silently…, A sort applied to the observations but not the actions would pair every state…, The sort is what makes transitions.py's contiguity rule satisfiable., Every ``place`` variant shares its widths, so without this a RandS-RandT…, Call the converter with named columns, not a splat. The splat reads better but… (+4 more)

### Community 35 - "verify_replay.py"
Cohesion: 0.17
Nodes (22): build_env_like_live_record(), diff_existing_replay(), Episode, FrameStats, load_episode(), main(), MjModel, ndarray (+14 more)

### Community 36 - "delta_action_from_poses"
Cohesion: 0.31
Nodes (5): delta_action_from_poses(), The clamped 6-D delta action that asks for ``target`` from ``current``. The…, The mapping both the recorder and the action term are built on., A bounded policy can only emit [-1, 1]. The recorder has to obey the same bound…, TestDeltaFromPoses

### Community 37 - "domino_env_cfg.py"
Cohesion: 0.20
Nodes (11): get_domino_spec(), MjSpec, MjSpec builders for the domino pieces., make_domino_env_cfg(), ManagerBasedRlEnvCfg, Environment configuration for the domino task category., Build the robot-agnostic domino cfg (see ``tasks/common/env_cfg.py`` for the…, domino_half_extents() (+3 more)

### Community 38 - "_quat_axis_angle"
Cohesion: 0.36
Nodes (6): parametrize, Tensor, _quat_axis_angle(), _quat_mul(), (4,) wxyz quaternion for a rotation of *angle* about *axis* (unit)., TestDominoUprightAligned

### Community 39 - "table_entity_cfg"
Cohesion: 0.10
Nodes (20): get_table_spec(), EntityCfg, MjSpec, Visual-only workspace table shared by every task category., Fixed-base table entity. Pin via workbench fixed_props., Fixed-base, visual-only workspace table shared by every task category., table_entity_cfg(), get_sorter_piece_spec() (+12 more)

### Community 40 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 41 - "cartpole_env_cfg.py"
Cohesion: 0.15
Nodes (22): cartpole_balance_env_cfg(), cartpole_ppo_runner_cfg(), cartpole_smooth_reward(), cartpole_swingup_env_cfg(), _gaussian_tolerance(), _get_cartpole_cfg(), _get_spec(), _make_env_cfg() (+14 more)

### Community 42 - "MultiCubeLiftingCommand"
Cohesion: 0.13
Nodes (8): LiftingCommand, MultiCubeLiftingCommand, CommandTerm, DebugVisualizer, Tensor, Selects one of N cubes as the target at each reset., Geom IDs of the target cube per env. Shape: (B, K)., Position of the target cube per env. Cached per step — updated in…

### Community 43 - "LootEnv"
Cohesion: 0.24
Nodes (7): LootEnv, Any, ndarray, What an action means in this env, as the stamped record. Published so a dataset…, The actor group, as a fresh float32 row. The copy is deliberate. mjlab hands…, [N] the strongest finger-pad contact force this sensor sees. ``None`` when the…, One mjlab task, presented as a single-env gymnasium environment. Observations…

### Community 44 - "domino/mdp/commands.py"
Cohesion: 0.15
Nodes (18): flatten_object_poses(), Per-env object poses in the env-local world frame, flattened (E, N*7). Object-…, DominoCommandCfg, get_command(), CommandTermCfg, ManagerBasedRlEnv, Domino command. Spawns dominoes from the fixed table, checks slot placement,…, domino_poses() (+10 more)

### Community 45 - "make_harness"
Cohesion: 0.13
Nodes (8): make_harness(), V/F mid-episode end the recording manually with that label., The grace window gets its own ghost color, so an undecided episode can't be…, TestGhostColor, TestHud, TestManualEnd, TestSessionEnd, TestUndo

### Community 46 - "run_evaluate"
Cohesion: 0.15
Nodes (19): compute_ee_orientation_error(), compute_ee_position_error(), compute_joint_velocity_error(), compute_mpkpe(), compute_root_relative_mpkpe(), _get_body_indices(), Tensor, Compute Mean Per-Keybody Position Error (MPKPE). MPKPE measures the average… (+11 more)

### Community 47 - "dk1_constants.py"
Cohesion: 0.08
Nodes (36): BuiltinPositionActuatorCfg, CollisionCfg, InitialStateCfg, _ArmJointCfg, get_arm_actuators(), get_gripper_actuator(), DK1 actuator configs (DAMIAO DM4340/DM4310). Motor specs and arm gains match…, _collision_with_pad_tuning() (+28 more)

### Community 48 - "manipulation/mdp/observations.py"
Cohesion: 0.19
Nodes (19): camera_depth(), camera_rgb(), camera_segmentation(), camera_target_cube_mask(), ee_to_object_distance(), ee_velocity(), object_to_goal_distance(), ManagerBasedRlEnv (+11 more)

### Community 49 - "InsertionCommand"
Cohesion: 0.17
Nodes (8): InsertionCommand, CommandTerm, Tensor, (E,) float: task done, scored pegs at rest, released., (E,) float: success latched since the last reset., (S, 3) outward bore axes in the board frame., Position of the peg assigned to each socket, (E, S, 3)., Task progress in [0, 1], (E,): fraction of sockets filled.

### Community 50 - "TestCompiledTask"
Cohesion: 0.10
Nodes (15): built_env(), _grasp_pose(), fixture, Tensor, Tests for the shared action convention. The pure tests need no simulator. The…, The grasp site's world-frame pose, as the solver reads it., The convention as the simulator actually sees it., QC's chunk of 5 spans 0.25 s here, which is both reference domains. At the old… (+7 more)

### Community 51 - "ReaderFrame"
Cohesion: 0.16
Nodes (9): ABC, Uniform data envelope for one timestep from any source. Every Reader subclass…, Abstract base class for all data sources. Concrete subclasses: LiveReader (real…, Return the next frame, or raise StopIteration when exhausted., Release any resources (serial ports, file handles)., Reader, ReaderFrame, Only joint_pos is required. Everything else has defaults. (+1 more)

### Community 52 - "load_extra_info"
Cohesion: 0.27
Nodes (9): extra_info_path(), load_extra_info(), Path, Sidecar metadata for LeRobot datasets recorded by this bridge. lerobot 0.6.0…, write_extra_info(), operator_id passed to close() must appear in the sidecar extra-info file after…, close() with no operator_id must not crash (backwards compat)., test_close_stores_operator_id() (+1 more)

### Community 53 - "DatasetReader"
Cohesion: 0.14
Nodes (10): DatasetReader, Find the start and end indices for the target episode. LeRobot datasets store…, The dataset's sidecar extra info (episode_seeds, seed_device, operator_id, ...…, Read frames from a saved LeRobotDataset (offline replay)., skipif, End-to-end test with a real LeRobot dataset from HuggingFace., DatasetReader wraps a LeRobotDataset to produce ReaderFrame objects., Build a mock LeRobotDataset with num_frames frames in episode 0. (+2 more)

### Community 54 - "state_replay.py"
Cohesion: 0.15
Nodes (14): build_env(), Episode, load_episode(), main(), _NullPolicy, NativeMujocoViewer, Path, RslRlVecEnvWrapper (+6 more)

### Community 55 - "place/mdp/commands.py"
Cohesion: 0.05
Nodes (31): draw_place_poses(), enforce_place_separation(), get_command(), place_half_width(), place_xy_from_draws(), PlaceCommand, PlaceCommandCfg, CommandTerm (+23 more)

### Community 56 - "stack_progress"
Cohesion: 0.15
Nodes (13): Entity, (E, N) bool marking the single block currently held in the gripper. Held = the…, block_top_heights(), _heights(), Tensor, Pure success criteria for the stack category. Plain tensor math, no mjlab or…, Top-face heights above the table, (E, N)., Top-face z per block. Args: pos_z: (E, N) block-centre z (world). half_h: (N,)… (+5 more)

### Community 57 - "test_convert.py"
Cohesion: 0.10
Nodes (16): _action(), _columns(), _FakeEnv, ndarray, Tests for the recording -> training-format converter. The episode-structure…, Just enough of the adapter to answer the stamp check., Two sessions are each numbered from episode zero., Why _load_many offsets. Two failed sessions, each numbered from zero,… (+8 more)

### Community 58 - "manipulation/mdp/rewards.py"
Cohesion: 0.25
Nodes (12): bring_object_reward(), joint_velocity_hinge_penalty(), multi_cube_bring_object_reward(), multi_cube_staged_position_reward(), ManagerBasedRlEnv, SceneEntityCfg, Tensor, Quadratic hinge penalty on joint velocities exceeding a symmetric limit.… (+4 more)

### Community 59 - "check_dataset_matches_env"
Cohesion: 0.25
Nodes (9): check_dataset_matches_env(), Raise unless ``data`` was recorded under the environment's conventions. Every…, _env_from(), parametrize, An env agreeing with ``data`` except where told otherwise. ``overrides`` takes…, The failure this exists for: a dataset recorded under a different MDP trains…, A pre-ADR-016 file stamps ``action_scale``/``action_offset`` and no…, An env builds its stamp out of the action terms' buffers, which are torch… (+1 more)

### Community 60 - "DominoCommand"
Cohesion: 0.18
Nodes (9): DominoCommand, CommandTerm, Tensor, (E, G*7): per-slot env-local target pose, in slot order., (E,) float: task done, dominoes at rest, released., (E,) float: success latched since the last reset., Positions of the domino assigned to each slot, (E, G, 3)., Per-piece values (E, N, D) -> per-slot values via piece_for_slot. (+1 more)

### Community 61 - "stack/mdp/commands.py"
Cohesion: 0.11
Nodes (18): command_success_termination(), CommandTerm, ManagerBasedRlEnv, Tensor, Termination-term factories shared by every task category., Task's success condition (task-done, at-rest, and released) all met., get_command(), CommandTermCfg (+10 more)

### Community 62 - "Session (one teleop recording run)"
Cohesion: 0.15
Nodes (17): delta_action_from_poses (single build site), TRLC-DK1 robot asset, Arm gravity compensation, mjlab pinned >=1.5.1,<1.6, out/ is scratch, dataset/ is kept, Action box ([-1, 1] per channel), BC gate (tests/qc/bc_gate.py), Closed experiment (+9 more)

### Community 63 - "trlc_dk1/__init__.py"
Cohesion: 0.09
Nodes (24): ManipulatorInterface, Robot-agnostic manipulator description for benchmark tasks. Tasks use this…, get_unimanual_dk1_interface(), get_yam_interface(), YAM manipulator interface. Thin loot-bench adapter over the mjlab asset-zoo YAM…, YAM constants for get_yam_interface from mjlab's asset zoo., ManagerBasedRlEnvCfg, Wire a robot's :class:`ManipulatorInterface` into a base task cfg. This is the… (+16 more)

### Community 64 - "LiveDebugPanel"
Cohesion: 0.15
Nodes (7): LiveDebugPanel, Per-frame update; aggregate breakdown prints at finish_episode., Status is precomputed by the viewer; reading it is free., Print aggregate breakdown and banner; no-op if no frames., Redraw in place; append-only when piped or append_only set., Plain append for the episode summary (never overwritten)., A task/env shape the panel can't introspect still renders a (less detailed)…

### Community 65 - "live_record.py (record episodes at ~50 Hz from live GELLO teleop)"
Cohesion: 0.18
Nodes (17): debug_panel.py (live per-frame sanity-check panel, --debug-obs append-only mode), Letter-key episode labeling (V success / F failure / D discard / U undo / Q quit / C view), Auto-save on success, orange-ghost pause and 2 s failure auto-save on any other termination, live_bridge (DK1 teleop -> LeRobot demonstration recording), live_record.py (record episodes at ~50 Hz from live GELLO teleop), out/demos is scratch, dataset/demos/<task>/<stamp> is kept and committed, record_replay.py (replay a LeRobot dataset through the sim, no hardware), USE_FOLLOWER_STATE_FOR_SIM (drive the sim from follower joint states instead of leader actions) (+9 more)

### Community 66 - "QC seam tests (contract tests for loot_bench/qc/ plus the two standalone scripts)"
Cohesion: 0.15
Nodes (17): Replay determinism decided by GPU contact impulses (bitwise match until first gripper contact), grip_contact (gate) and object_contact (telemetry only, filtered to the cube), Place observations, 42 wide (joint pos/vel, EE pose with 6D rotation, cube pose, target, last action), START_POSE_NOISE = 0.0 lives on the task so recorder and training run read one number, bc_gate.py (fit recorded actions with a plain regressor, exit status is the verdict; ADR-012), conftest.py imports lerobot before mjlab; the reverse native-library order segfaults during collection, pad_contact_force / object_contact_force: a cross-repo contract with no version pin (check eval/pad_contact_runs), QC seam tests (contract tests for loot_bench/qc/ plus the two standalone scripts) (+9 more)

### Community 67 - "_ReplayAndRecord"
Cohesion: 0.16
Nodes (9): main(), ndarray, Replay a LeRobot dataset through sim and record to logs/. Viewer holds last…, Acts as the viewer's policy AND manages recording. Flow per step: 1.…, Called by the custom viewer after env.step()., NativeViewerWrapper that records obs/action/reward/done per step., Step env and pass reward/done to the recording policy., _RecordingViewer (+1 more)

### Community 68 - "Shape-sorter task category (insert every piece into its matching hole in a free cube)"
Cohesion: 0.18
Nodes (17): Ball reward: the sparse member of the one-task-reward convention (-1 every step until scored), Insertion success: insert_depth below the mouth, laterally centred, axis-aligned, at rest, gripper open, Peg-insertion task category (archived, unregistered, kept importable), Peg insertion (one peg into one bore, vertical or horizontal orientation), Peg observations (board pose for parity with shape_sorter's cube pose, peg poses, per-socket inserted flags), Sparse peg rewards paired with a teleop demonstration pipeline instead of dense shaping, SHAPE_GEOMS (one cross-section description shared by peg and bore, clearance 1.5 mm), socket_layout (one source for the bore wall geometry and the command's goal poses) (+9 more)

### Community 69 - "ContactReleaseGate"
Cohesion: 0.24
Nodes (8): ContactReleaseGate, Per-step "pads not exerting grip force" check for a command's success gate.…, _fake_contact_sensor(), ContactSensor, Duck-typed ContactSensor stand-in: the gate only reads data.force., The gate ships reading the env's release threshold, not a copy., One loaded pad vetoes release, however lightly the other rests., TestContactReleaseGate

### Community 70 - "Domino task category (build a row of upright dominoes)"
Cohesion: 0.17
Nodes (17): Domino observations (all domino poses object-major, per-slot placed indicator, per-slot target pose, phase), Domino reward shaped by placement fraction (half placement + half toppled in the topple phase), Domino task category (build a row of upright dominoes), Fixed row pose every reset so a teleoperator always builds in the same spot, KnockOver phase latch and chain-topple criterion (same lean, monotonic first-fall times along the row axis), Per-domino identity colors and the two-box row-slot ghost marker, RandomRow variant (row start pose and yaw redrawn every reset, one shared draw sized for the hardest count), Cube-spawn variants: FixedOrientation, NoRotation, Full, RandomPositions (+9 more)

### Community 71 - "make_manipulation_env_cfg"
Cohesion: 0.11
Nodes (21): make_manipulation_env_cfg(), CommandTermCfg, EntityCfg, ManagerBasedRlEnvCfg, Assemble the shared workbench around a category core. Args: entities: all scene…, get_cube_spec(), MjSpec, MjSpec builder for the place cube. (+13 more)

### Community 72 - "SuccessGate"
Cohesion: 0.15
Nodes (16): Procedural geometry by default, One source for geometry and success metric, Bootstrap mask (masks column), Boundary (terminals column), Crossing plane, Episode success (latched), Insertion channel, SorterLatch (+8 more)

### Community 73 - "test_debug_panel.py"
Cohesion: 0.19
Nodes (8): _aggregate_buffered(), Terminal dashboard for live_record.py, on by default. Per-frame line plus…, Aggregate buffered per-frame tensors once at episode end., First 3 values as x/y/z (positions, goals, object poses)., _xyz(), Unit tests for the --debug-obs live dashboard (no viewer, no simulator)., TestAggregateBuffered, TestXyz

### Community 74 - "ball_env_cfg.py"
Cohesion: 0.12
Nodes (15): get_ball_spec(), get_wall_spec(), MjSpec, Standalone wall entity for ThrowWallVariation., make_ball_env_cfg(), ManagerBasedRlEnvCfg, Environment configuration for the ball task category., Build the robot-agnostic ball cfg (see ``tasks/common/env_cfg.py`` for the… (+7 more)

### Community 75 - "test_recording_mixin.py"
Cohesion: 0.14
Nodes (7): FakeEnv, FakePolicy, Harness, State-machine tests for RecordingViewerMixin, driven with fake recorder/env.…, Minimal env: steps, returns zeros, and counts how often it was asked., The two attributes the recording path reads off the live policy.…, Concrete host standing in for the real viewer subclass.

### Community 76 - "TestStepContract"
Cohesion: 0.12
Nodes (7): r in [-1, 0]. qc's --sparse depends on nothing else touching 0., qc clips to [-1, 1] itself, but a stray value must not raise., qc counts grips out of this key; nested-only would be a KeyError. It is the…, A key that appears only on some steps is harder to trust., The negative control for :class:`TestObjectContact`. Stack scores three blocks,…, Guards against handing back a stale or copied-once buffer., TestStepContract

### Community 77 - "ObservationTermCfg"
Cohesion: 0.20
Nodes (14): _cube_color(), get_cube_spec(), ManagerBasedRlEnvCfg, MjSpec, # NOTE: No noise for goal position., Generate a distinct color for cube i of n using HSV hue rotation., Multi-cube task: depth + segmentation mask for goal conditioning., yam_lift_cube_env_cfg() (+6 more)

### Community 78 - "sweep.sh"
Cohesion: 0.26
Nodes (12): cell_of, clamp(), die(), is_value(), note_flag(), reap_one(), repo_needs_ticket(), sanitize() (+4 more)

### Community 79 - "_split"
Cohesion: 0.22
Nodes (9): slice, _peak_to_peak(), ndarray, Largest per-column range. Never the standard deviation. The observations are…, Replay verification needs a bit-identical start., Map each actor-observation term to its slice of the flat vector. Derived rather…, Split observations into (robot columns, scene columns)., _split() (+1 more)

### Community 81 - "LiftingCommandCfg"
Cohesion: 0.22
Nodes (9): LiftingCommandCfg, MultiCubeLiftingCommandCfg, ObjectPoseRangeCfg, CommandTermCfg, ManagerBasedRlEnv, Configuration for target position sampling in dynamic mode., Configuration for object pose sampling when resampling commands., TargetPositionRangeCfg (+1 more)

### Community 82 - "TerminationTermCfg"
Cohesion: 0.15
Nodes (15): ManagerBasedRlEnvCfg, Unitree Go1 velocity environment configurations., Create Unitree Go1 flat terrain velocity configuration., Create Unitree Go1 rough terrain velocity configuration., unitree_go1_flat_env_cfg(), unitree_go1_rough_env_cfg(), RslRlOnPolicyRunnerCfg, RL configuration for Unitree Go1 velocity task. (+7 more)

### Community 83 - "test_stack_variants.py"
Cohesion: 0.25
Nodes (7): parametrize, Invariants of the stack variant catalog (spares, prefix rule, spawn table)., test_block_count_is_two_n(), test_blocks_tail_extend_across_target_height(), test_easier_reset_is_prefix_of_harder(), test_more_blocks_than_needed(), test_only_the_target_block_count_reaches_the_target()

### Community 85 - "ball/mdp/observations.py"
Cohesion: 0.23
Nodes (13): ball_pose(), ball_velocity(), field_pose(), ManagerBasedRlEnv, Tensor, Observation terms for the ball task category., Ball position in the env-local world frame (E, 3). No quaternion: a sphere's…, Field pose in the env-local world frame (E, 7). (+5 more)

### Community 86 - "Stack task category (stack blocks to reach a target height)"
Cohesion: 0.15
Nodes (14): Difficulty is the block's height:width ratio at BLOCK_DENSITY 1562.5 kg/m^3 (slenderest block is lightest), SHOW_GOAL_STACK (ghost target stack drawn beside the workspace in loot-play only), Each variant carries 2n blocks so height, not block count, is the binding constraint, Stack observations (joint pos/vel, EE pose, all block poses object-major), Stack success: some block's top face within HEIGHT_TOL of the target height, all blocks slow, gripper open, Stack task category (stack blocks to reach a target height), make_smoke_dataset.py (synthetic .npz with real widths and action stamp; no number off it means anything), compare_sim_settings.py (sim-setting profile speed/accuracy benchmark; stale, predates the ADR-016 delta action) (+6 more)

### Community 87 - "Convention record (dataset stamp)"
Cohesion: 0.17
Nodes (13): pre-commit hook: pyright, Serena project config (LOOT-bench), loot_bench package, num_qs must stay at 2 or more, run_scripts/ experiment launchers, train_qc.sh (cross-repo launcher), Action convention (scale and offset), Adapter (LootEnv gym env) (+5 more)

### Community 88 - "Task reward (one term, [-1, 0])"
Cohesion: 0.19
Nodes (13): Splitter self-check on recomputed reward, Reward lag (one control step), Shortfall floor (MIN_SHORTFALL 1e-3), Task reward (one term, [-1, 0]), Ball sparse reward (-1 until it scores), Stack progress (tallest resting block), nativeccd disabled (analytic box-box collider), Ball category (+5 more)

### Community 89 - "unitree_g1_rough_env_cfg"
Cohesion: 0.22
Nodes (10): ManagerBasedRlEnvCfg, Unitree G1 velocity environment configurations., Create Unitree G1 flat terrain velocity configuration., Create Unitree G1 rough terrain velocity configuration., unitree_g1_flat_env_cfg(), unitree_g1_rough_env_cfg(), RslRlOnPolicyRunnerCfg, RL configuration for Unitree G1 velocity task. (+2 more)

### Community 90 - "StackCommand"
Cohesion: 0.24
Nodes (6): CommandTerm, Tensor, (E, 1) target height — constant per task (``command_obs=False``)., (E,) float: target height reached, blocks at rest, released., (E,) float: success latched since the last reset., StackCommand

### Community 91 - "save_env_render"
Cohesion: 0.20
Nodes (12): main(), Path, Build, reset, and step every registered LOOT task. See tests/tasks/README.md., Build, reset, and step *task_id*. Returns ``(build_error, render_warning)``…, _run(), domain_of(), ManagerBasedRlEnv, Path (+4 more)

### Community 92 - "Place task category (carry one cube to a sampled target and let go)"
Cohesion: 0.22
Nodes (13): Domino success: every domino upright, row-aligned and within pos_threshold of its slot, at rest, gripper open, _SLOT_KEEP_OUT (spawn cells near a row slot are blocked), ContactReleaseGate (pad force at or below MIN_GRIP_FORCE before the cube counts as released), Spawn and target regions disjoint in x so the cube can never spawn at the target, PlaceCommand (scatter the cube through ScatterGrid, then draw a target point), Place success: cube centre within pos_threshold in 3D, moving slowly, released (position only, no orientation), Place reward shaped by the spawn->target travel fraction normalized by MAX_TRAVEL_DIST, Place task category (carry one cube to a sampled target and let go) (+5 more)

### Community 93 - "_fake_env"
Cohesion: 0.21
Nodes (6): _fake_env(), Stands in for the real env.env.observation_manager shape used by…, --debug-obs (append_only=True) must keep every frame in the scrollback, even…, domino's task_obs_terms puts domino_poses (positions) first, then…, Any env shape this doesn't understand disables the named breakdown rather than…, TestObsLayout

### Community 94 - "Category (task family)"
Cohesion: 0.23
Nodes (12): Automatic registration via package discovery, Task category pattern, Category (task family), Keep-out (blocked_near), Recovery demonstration, Row slot (domino), Variant, Domino progress (build / topple phases) (+4 more)

### Community 95 - "make_tracking_env_cfg"
Cohesion: 0.14
Nodes (12): ManagerBasedRlEnvCfg, Unitree G1 flat tracking environment configurations., Create Unitree G1 flat terrain tracking configuration., unitree_g1_flat_tracking_env_cfg(), RslRlOnPolicyRunnerCfg, RL configuration for Unitree G1 tracking task., Create RL runner configuration for Unitree G1 tracking task., unitree_g1_tracking_ppo_runner_cfg() (+4 more)

### Community 96 - "tracking/mdp/rewards.py"
Cohesion: 0.41
Nodes (11): _get_body_indexes(), motion_global_anchor_orientation_error_exp(), motion_global_anchor_position_error_exp(), motion_global_body_angular_velocity_error_exp(), motion_global_body_linear_velocity_error_exp(), motion_relative_body_orientation_error_exp(), motion_relative_body_position_error_exp(), ManagerBasedRlEnv (+3 more)

### Community 97 - "leader_grasp_pose"
Cohesion: 0.13
Nodes (18): gripper_scale_offset(), leader_grasp_pose(), leader_to_delta_action(), The grasp-site pose (position, ``wxyz`` quaternion) a leader pose asks for.…, The 7-wide action that drives the sim arm toward a leader pose. Six delta…, (scale, offset) matching the task-side gripper term. Same model, same helper,…, ndarray, The one channel that is still absolute. It has to invert the gripper term's own… (+10 more)

### Community 98 - "Workbench (shared env skeleton)"
Cohesion: 0.18
Nodes (11): Explicit actuator split (arm / gripper / ik_max_dq), ee_pose publishes a 6D rotation, ManipulatorInterface, Observations carry no noise, Fixed prop, Released (robot has let go), Wiring (robot into task cfg), Workbench (shared env skeleton) (+3 more)

### Community 99 - "joint_pos_limits"
Cohesion: 0.22
Nodes (8): _dk1_scratch(), joint_pos_limits(), MjModel, Compiled DK1, built once, for joint limits and grasp-site FK. Built on first…, Hard (lower, upper) ctrl limits read from the compiled model., MjData, The regression this guards: a copy of dk1.xml's ranges going stale., TestJointLimits

### Community 100 - "SuccessGate"
Cohesion: 0.22
Nodes (6): device, Tensor, At-rest + released gating with a per-episode success latch. ``update`` combines…, Advance one step; returns the (E,) float success signal. Args: task_done: (E,)…, SuccessGate, TestSuccessGate

### Community 101 - "tracking/mdp/terminations.py"
Cohesion: 0.35
Nodes (8): bad_anchor_ori(), bad_anchor_pos(), bad_anchor_pos_z_only(), bad_motion_body_pos(), bad_motion_body_pos_z_only(), ManagerBasedRlEnv, SceneEntityCfg, Tensor

### Community 102 - "MotionTrackingOnPolicyRunner"
Cohesion: 0.24
Nodes (5): MotionTrackingOnPolicyRunner, _OnnxMotionModel, MjlabOnPolicyRunner, ONNX-exportable model that wraps the policy and bundles motion reference data., VecEnv

### Community 103 - "_RunningStats"
Cohesion: 0.25
Nodes (4): Running min/max/mean over an episode, O(1) memory., _RunningStats, The whole point: mean/min/max are over every value seen across all update()…, TestRunningStats

### Community 104 - "check_success_reward.py (teleport objects to their goals and check the command term's success metric fires)"
Cohesion: 0.25
Nodes (11): Strictly additive minigolf layouts with a shared off-centre pit, Ball success: in target zone, at rest, released from the gripper, Ball observations (ball position without orientation, field pose, wall pose, ball velocity, raw target), Ball task category (bring a ball to its goal by hitting or throwing), Minigolf family (raised platform with a pit out of the arm's reach, perimeter railing), Throw family (flat field with a wall between spawn area and target), WallVariation (the wall is a separate entity re-posed every reset), build_sim_registered_tasks.py (build/reset/step every registered task, render one scene per task) (+3 more)

### Community 105 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 106 - "wait_for_viewer_teardown"
Cohesion: 0.32
Nodes (6): Wait for GLFW teardown; True if done within TEARDOWN_TIMEOUT., wait_for_viewer_teardown(), main(), Check that closing the native viewer doesn't segfault at interpreter exit. See…, Open a tiny passive viewer, close it, and return straight into interpreter…, run_child()

### Community 107 - "Contact-only release gate"
Cohesion: 0.33
Nodes (5): Consequences, Contact-only release gate, Why deleted rather than kept as a second opinion, Why the joint gate was a proxy, Why the pad force joins the observation

### Community 109 - "dataset.py"
Cohesion: 0.12
Nodes (15): _as_scalar(), _Env, get_training_columns(), Any, NDArray, Reading a converted dataset back, and checking it against an environment.…, Only the keys ``qc``'s ``Dataset.create`` may be given. The stamp and…, What :func:`check_dataset_matches_env` needs of an environment. (+7 more)

### Community 110 - "manifest.json"
Cohesion: 0.20
Nodes (9): base_link, finger_left, finger_right, link1-2, link2-3, link3-4, link4-5, link5-6 (+1 more)

### Community 111 - "_ObsLayout"
Cohesion: 0.20
Nodes (6): _ObsLayout, ndarray, Name to offset map for one flat obs vector (matches saved layout)., Best-effort layout; unknown envs disable the breakdown., Fold one frame's array in; True if it held a non-finite value., _to_numpy()

### Community 112 - "easier_target_task_ids"
Cohesion: 0.33
Nodes (5): easier_target_task_ids(), parse_task_id(), Split ``LOOT-<Category>-<Task>[-<Variation>]-<Robot>`` into its parts., All registered same-family, splittable tasks with fewer objects., TestTaskIds

### Community 113 - "TestProvisionalFailureActions"
Cohesion: 0.20
Nodes (5): F during the grace window: the episode is staged as failure now, without…, V overrides a termination-labeled failure to success (e.g. the task succeeded…, Undecided after the grace period: _execute_step auto-saves failure., Within the grace period nothing is saved. The window stays open., TestProvisionalFailureActions

### Community 114 - "test_env.py"
Cohesion: 0.19
Nodes (11): env(), _gripper_index(), _hold_action(), place_env(), fixture, Contract tests for the qc gym adapter, run against a real task. These build a…, The cube-filtered sensor: the only key that answers "touched the cube".…, Neither sensor may fire on an empty grasp. The arm holds its reset pose, which… (+3 more)

### Community 115 - "normalized_scale_offset"
Cohesion: 0.28
Nodes (6): _Array, normalized_scale_offset(), Map [-1, 1] onto [lower, upper]. Shared with the teleop mapper. Takes torch on…, The gripper's half of the convention., The whole point of the offset. A joint whose range does not straddle zero still…, TestNormalizedScaleOffset

### Community 116 - "actions.py"
Cohesion: 0.09
Nodes (23): DifferentialIKAction, DifferentialIKActionCfg, JointPositionAction, JointPositionActionCfg, EndEffectorDeltaAction, EndEffectorDeltaActionCfg, NormalizedJointPositionAction, NormalizedJointPositionActionCfg (+15 more)

### Community 117 - "_connect_with_retry"
Cohesion: 0.33
Nodes (4): _connect_with_retry(), Open serial connections; retry handshake on dropped packets., Build device and connect, retrying transient ConnectionError., _T

### Community 118 - "MotionCommandCfg"
Cohesion: 0.29
Nodes (5): MotionCommandCfg, MotionLoader, CommandTermCfg, ManagerBasedRlEnv, VizCfg

### Community 119 - "BallCommand"
Cohesion: 0.22
Nodes (6): BallCommand, CommandTerm, Tensor, (E,) float: ball at target, at rest, and released., (E,) float: success latched since the last reset., (E, 3): env-local target position (was a ball-relative delta).

### Community 120 - "TestPlaceProgress"
Cohesion: 0.21
Nodes (7): place_progress(), place_reached(), Tensor, Pure success criteria for the place category. Plain tensor math, no mjlab or…, Cube centre within ``pos_threshold`` of the target (3D, position-only). Args:…, Continuous distance-based shaping, in [0, 1]. 1.0 exactly at the target,…, TestPlaceProgress

### Community 121 - "sweep.sh (generic alpha x seed sweep, owns the sweep and nothing about the experiment)"
Cohesion: 0.32
Nodes (8): Alpha-major run pool with no barrier between alphas (--max-parallel, a failed run does not stop the others), critic-variants/ (eight single-seed pre-sweep runs, the record of a closed experiment, ADR-014), The '--' split: typed flags are tagged and named, the wrapper's fixed configuration is not, One sweep is one wandb group; run named alpha-<a>_seed-<s>; tags written in exactly one place each, No timestamp in the name, so re-running a wrapper joins the previous group (how you add seeds), run_scripts (ready-made QC-FQL launches, Linux/cluster only), sweep.sh (generic alpha x seed sweep, owns the sweep and nothing about the experiment), Sweep wrapper (pins task, dataset and one arm's configuration; exports LOOT_RUN_SCRIPT)

### Community 122 - "TestStateGuards"
Cohesion: 0.25
Nodes (4): Accepted keys must match what the HUD advertises per state., A termination before any frame was recorded has nothing to decide. It must go…, Q must not bypass an open grace window. The un-decided episode would be…, TestStateGuards

### Community 123 - "TestResetPhaseAlignmentGate"
Cohesion: 0.25
Nodes (4): What ends the reset phase (ADR-016). The gate reads the *leader*, and it is one…, Before the hardware has reported once there is nothing to compare, and the gate…, Held open, the arm must keep being driven — a frozen sim is what the operator…, TestResetPhaseAlignmentGate

### Community 124 - "spawn_pose_in_envelope"
Cohesion: 0.26
Nodes (5): Whether one jittered ``(x, y)`` pose is inside the shared envelope., Whether a cell is inside the envelope at all four jitter corners., spawn_cell_in_envelope(), spawn_pose_in_envelope(), TestEnvelope

### Community 125 - ".create_gui"
Cohesion: 0.29
Nodes (4): Any, ViserServer, Create motion scrubber controls in the Viser viewer., Enable or disable the motion scrubber GUI controls.

### Community 126 - "tracking/mdp/observations.py"
Cohesion: 0.57
Nodes (6): motion_anchor_ori_b(), motion_anchor_pos_b(), ManagerBasedRlEnv, Tensor, robot_body_ori_b(), robot_body_pos_b()

### Community 128 - "reset_joints_to_pose"
Cohesion: 0.29
Nodes (6): ManagerBasedRlEnv, SceneEntityCfg, Tensor, Reset events shared by all task categories. mjlab's…, Reset the named joints to ``pose`` plus a uniform sample. Joints absent from…, reset_joints_to_pose()

### Community 129 - "load_dataset"
Cohesion: 0.24
Nodes (7): load_dataset(), Path, Load a converted ``.npz``, checked against ``env`` when one is given. The way a…, Including the two string fields. ``np.savez`` stores a ``str`` as a 0-d unicode…, The path a training run actually takes: the stamp arrives from the file, not…, The string field, through the npz round trip., TestLoadDataset

### Community 130 - ".__init__"
Cohesion: 0.50
Nodes (3): InsertionCommandCfg, CommandTermCfg, ManagerBasedRlEnv

### Community 131 - "tasks/common shared helpers"
Cohesion: 0.40
Nodes (6): tasks/common shared helpers, Distractor object, Home pose (START_JOINT_POS), ScatterGrid (spawn sampler), Tail-extension and seed draw ordering, loot-split-demos

### Community 132 - "convert_glb_meshes_to_obj.py"
Cohesion: 0.53
Nodes (5): convert_visual(), linear_to_srgb(), main(), Path, One-time conversion of the official TRLC-DK1 URDF meshes to MuJoCo-loadable…

### Community 133 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 136 - ".send_action"
Cohesion: 0.50
Nodes (3): Any, ndarray, Send (7,) action to follower; joints rad, gripper in [0, 1].

### Community 137 - "common/success.py"
Cohesion: 0.19
Nodes (11): bore_axial_lateral(), bore_axis_cos_tilt(), Success gating shared by all task categories Contains the :class:`SuccessGate`…, # TODO: check with real robot, Decompose object-to-mouth offsets along/across a bore/hole axis. Shared by the…, |cos| of the angle between an object's body z-axis and a bore/hole axis. Sign-…, peg_inserted(), Tensor (+3 more)

### Community 138 - "Read CONTEXT.md and docs/adr before exploring"
Cohesion: 0.40
Nodes (5): Flag ADR conflicts explicitly, Use the glossary's vocabulary, Read CONTEXT.md and docs/adr before exploring, Local markdown issue tracker (.scratch/), Wayfinding map, claims and blocking

### Community 139 - "sample_fixed_spawns"
Cohesion: 0.36
Nodes (4): Draw jittered poses with object ``i`` on cell ``i``. One ``rand(E, 3)`` draw…, sample_fixed_spawns(), Tensor, TestSampleFixedSpawns

### Community 140 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 141 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 142 - "TestTerminationHandling"
Cohesion: 0.40
Nodes (3): Success termination → episode staged as success, reset phase entered., Non-success termination saves nothing yet. The grace window opens so the…, TestTerminationHandling

### Community 143 - "train_qc.sh"
Cohesion: 0.60
Nodes (4): die(), train_qc.sh script, usage(), XLA_PYTHON_CLIENT_PREALLOCATE

### Community 144 - "pre-commit hook: ruff check --fix"
Cohesion: 0.50
Nodes (4): pre-commit hook: ruff check --fix, pre-commit hook: ruff format, Platform-pinned torch build, install.sh / install.ps1

### Community 147 - "._init_recording"
Cohesion: 0.50
Nodes (3): build_ghost_model_and_qpos(), Set up recording state; call from subclass __init__., Build ghost model: robot visual geoms only, colored, at the start pose.

### Community 150 - "TestSorterTable"
Cohesion: 0.22
Nodes (4): Whether a jittered ``(x, y)`` spawn pose is clear of the approach corridor., sorter_spawn_is_clear(), _sorter_radius(), TestSorterTable

### Community 153 - "test_split_demos.py"
Cohesion: 0.16
Nodes (18): DataFrame, _domino_episode(), _episode_frame(), _head(), ndarray, parametrize, Tensor, Tests for the demo splitter (offline relabel to easier same-family tasks).… (+10 more)

### Community 204 - "domino/mdp/success.py"
Cohesion: 0.24
Nodes (8): chain_topple_success(), domino_fallen_along_row(), domino_upright_aligned(), Tensor, Pure success criteria for the domino category. Plain tensor math, no mjlab or…, All dominoes chain-fallen the same way, in row order. Args: fallen: (E, G) bool…, Upright + row-alignment check for dominoes. A domino counts as standing when…, Per-domino "toppled along the row" test (chain semantics). A domino counts as…

### Community 205 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 206 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 207 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 208 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 209 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 210 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 211 - "test_success.py"
Cohesion: 0.40
Nodes (5): SorterLatch, _latch(), Unit tests for the success criteria in the per-category ``mdp/success.py``…, _step(), TestSorterLatch

### Community 221 - "_category_tables"
Cohesion: 0.28
Nodes (7): domino_spawn_is_clear(), Whether a jittered ``(x, y)`` spawn pose is clear of the row lane., Whether a jittered ``(x, y)`` spawn pose is clear of the tower site., stack_spawn_is_clear(), _category_tables(), Real per-category spawn tables, same spec shape as :func:`_example_tables`.…, TestCategoryTables

### Community 222 - "check_spawn_reachability.py"
Cohesion: 0.31
Nodes (8): main(), ManagerBasedRlEnv, Tensor, IK reachability check over every spawn cell. See tests/tasks/README.md. For…, ``(label, x, y)`` over every cell at all four jitter corners., Drive the grasp site to ``(tx, ty, tz)``; return best residual, steps., _solve(), _targets()

### Community 223 - ".__init__"
Cohesion: 0.25
Nodes (5): ContactSensor, ManagerBasedRlEnv, The mjlab env underneath, for tools that need the simulator. Read-only: the…, The named contact sensor, or ``None`` if the task declares none., Build the task. Args: task_id: a registered mjlab task id, e.g. ``LOOT-…

### Community 224 - "domino_row_slots"
Cohesion: 0.36
Nodes (4): domino_row_slots(), Row slot xy positions (local frame) for a pinned-centre row. Slots are anchored…, Min row-slot x over every permitted row y, tilt, branch and count, and max…, TestDominoRowLane

### Community 226 - "config/yam/__init__.py"
Cohesion: 0.67
Nodes (4): RslRlOnPolicyRunnerCfg, yam_lift_cube_ppo_runner_cfg(), yam_lift_cube_vision_ppo_runner_cfg(), yam_multi_cube_seg_ppo_runner_cfg()

### Community 227 - "ball_in_target_zone"
Cohesion: 0.33
Nodes (4): ball_in_target_zone(), Tensor, Pure success criteria for the ball category. Plain tensor math, no mjlab or…, xy-radius target test for the ball. Hole targets additionally require the ball…

### Community 228 - "inserted_count_reward"
Cohesion: 0.40
Nodes (6): inserted_count_reward(), ManagerBasedRlEnv, Tensor, Fraction of sockets currently filled., Sparse bonus while the task is complete, pieces at rest and released., success_reward()

### Community 229 - "Fixed spawn points from the arm's measured envelope"
Cohesion: 0.40
Nodes (4): Fixed spawn points from the arm's measured envelope, Measured envelope, Pitch from the gripper, Zoning rule

### Community 230 - "BallCommandCfg"
Cohesion: 0.50
Nodes (3): BallCommandCfg, CommandTermCfg, ManagerBasedRlEnv

### Community 232 - "illegal_contact"
Cohesion: 0.50
Nodes (3): illegal_contact(), ManagerBasedRlEnv, Tensor

### Community 233 - "insertion_complete"
Cohesion: 0.50
Nodes (4): insertion_complete(), ManagerBasedRlEnv, Tensor, All pegs inserted, at rest, and released.

## Knowledge Gaps
- **178 isolated node(s):** `Measured envelope`, `Pitch from the gripper`, `Zoning rule`, `VizCfg`, `_ArmJointCfg` (+173 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1205 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **51 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `make_manipulation_env_cfg()` connect `make_manipulation_env_cfg` to `domino_env_cfg.py`, `table_entity_cfg`, `ball_env_cfg.py`, `ObservationTermCfg`, `peg_insertion/assets.py`, `TerminationTermCfg`, `actions.py`, `stack/variants.py`, `common/commands.py`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Why does `make_tracking_env_cfg()` connect `make_tracking_env_cfg` to `TerminationTermCfg`, `actions.py`, `ObservationTermCfg`, `MotionCommandCfg`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Why does `MotionCommandCfg` connect `MotionCommandCfg` to `tracking/mdp/terminations.py`, `make_tracking_env_cfg`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `MotionCommand` (e.g. with `motion_anchor_ori_b()` and `motion_anchor_pos_b()`) actually correct?**
  _`MotionCommand` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `Recorder` (e.g. with `_LiveRecordingViewer` and `_ReplayAndRecord`) actually correct?**
  _`Recorder` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Measured envelope`, `Pitch from the gripper`, `Zoning rule` to the rest of the system?**
  _178 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `check_success_reward.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11962833914053426 - nodes in this community are weakly interconnected._