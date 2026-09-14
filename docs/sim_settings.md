# Sim settings

## Parameters

- `timestep` — physics integration step [s]. Smaller is more accurate and
  more stable in contact, at proportionally more compute. MuJoCo default
  0.002.
- `decimation` — physics steps per policy action; control dt =
  timestep × decimation. Defines the MDP; datasets don't transfer across
  control rates.
- `iterations` / `ls_iterations` — caps on Newton solver and line-search
  iterations per physics step. The solver early-exits when converged, so
  these are ceilings. MuJoCo defaults 100/50; mjlab trims to 10/20 because
  with thousands of lockstep GPU envs every allowed iteration costs kernel
  launches. At num_envs=1 the trim doesn't matter (measured below).
- `cone="elliptic"`, `impratio=10` — stiffer, less slippery friction than
  the defaults (pyramidal, 1); the standard grasping setup.
- `disableflags=("nativeccd",)` — use the analytic box-box collider instead
  of GJK/EPA, which jitters for boxes resting on boxes.

## Compared profiles and results

| profile | dt/dec | it/ls | ctrl | note |
|---|---|---|---|---|
| train | 0.005/4 | 10/20 | 50 Hz | old training settings |
| play | 0.01/1 | 20/10 | 100 Hz | old play/demo settings |
| train_acc | 0.005/4 | 100/50 | 50 Hz | train dt, default solver caps |
| mujoco | 0.002/10 | 100/50 | 50 Hz | MuJoCo defaults, 50 Hz control |
| reference | 0.001/20 | 100/100 | 50 Hz | accuracy baseline only |

Speed (GPU PC): train 183 steps/s (3.7× real time), play 241 (2.4×),
train_acc 183, mujoco 124 (2.5×). Viewer at 1× holds the target rate for
every profile (train: 50 sps at 648 fps). Laptop: no profile holds real
time — train 0.4×, play 0.5× — so laptop teleop always ran in slow motion
and the laptop is unsuitable for demo collection regardless of settings.

Accuracy (qpos deviation from `reference` along scripted rollouts; grasp =
cube pinned into the closing gripper, then lifted and shaken 5.5 s):

- grasp: train stays ≤1.5 mm from reference, cube held (slip 2.5 mm).
  play is ~6× worse (10 mm max, diverges >1 mm at 0.5 s). mujoco ~4× better
  than train. train_acc is identical to train. Same numbers on both
  machines.
- sweep (free-space): same ordering, everything sub-mm over 6 s.
- stack (6-cube tower at rest, the pre-henge block): all profiles incl.
  reference settle ~3 mm, nothing collapses. The per-profile numbers vary
  between machines (chaotic settling), so only the gross outcome counts.
- henge (3-level trilithon of 0.065 x 0.020 x 0.020 m blocks teleported in
  and settled 2 s under the decided settings, 8 envs, GPU; measured
  2026-09-11 by `tests/tasks/check_success_reward.py`, which prints these):

  | lintel | centre error, max | `\|cos(body z, up)\|`, max |
  | --- | --- | --- |
  | 1 | 0.92 mm | 0.0032 |
  | 2 | 1.71 mm | 0.0066 |
  | 3 | 2.06 mm | 0.0070 |

  Against `HEIGHT_TOL` = 3 mm and `TILT_COS_MAX` = 0.10. The centre error
  compounds upward (each level's uprights sink into the lintel below) and is
  the number that is close to its gate, at 2.1 of 3 mm on lintel 3. The
  tilt is the gate that would decide whether the task is recordable
  (0.10 allows only 4.5 mm of end-height difference across an upright pair,
  and a tilted lintel 1 tilts everything above it), and it settles more
  than an order of magnitude inside it. Neither constant was changed on
  this measurement; see ADR-020 for why they cannot move independently.

Takeaways: the 10/20 caps cost nothing and change nothing at num_envs=1
(train_acc = train in both speed and trajectories — the solver converges
below the caps). Accuracy is set by the timestep; 0.01 is measurably the
worst, 0.005 is sufficient (grip holds, tower stands, sub-mm penetration),
0.002 is more accurate but fixes no observable defect for 32% throughput.
For RL the env is its own ground truth — consistency between demos and
training matters, fidelity to a 1 kHz reference doesn't.

From this test, the following settings are decided:

| setting | value | taken from | why |
|---|---|---|---|
| timestep / decimation | 0.005 / 10 (20 Hz control) | ADR-016 | `timestep` from the grasp test below (≤1.5 mm from reference, grip holds). `decimation` was 4 (50 Hz) until ADR-016 moved the control rate to 20 Hz to match both QC-validated reference domains; the physics timestep is deliberately untouched, so nothing in this document's accuracy results changes. |
| iterations / ls_iterations | 100 / 50 | MuJoCo defaults | Free at num_envs=1 (identical speed and trajectories to 10/20 in every scenario); headroom for contact states where the mjlab-example caps would silently under-converge. Small limits only needed for multi-env on GPU. |
| cone / impratio | elliptic / 10 | mjlab manipulation example | Stiffer, less slippery friction than the MuJoCo defaults (pyramidal, 1) which are for locomotion; the grasp test confirms the grip contract: firm hold, no drop, no crush. |
| nativeccd | disabled | loot-bench choice | In MuJoCo Warp (what mjlab simulates with) this flag has exactly one effect: box–box pairs are rerouted from the generic GJK/EPA convex pipeline (≤4 contacts sharing one depth and normal frame) to the analytic box–box collider (up to 8 contacts with individual depths and normals). That difference is why cube stacking and box-pad grasping jittered with it enabled; disabling it globally is harmless because the flag is inert for every non-box-box pair. |
