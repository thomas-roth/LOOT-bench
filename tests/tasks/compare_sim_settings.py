# pyright: reportAttributeAccessIssue=false
"""Compare sim-setting profiles for speed and accuracy.

See tests/tasks/README.md.

**Stale: needs a pass before its numbers are trusted again.** Two things below
describe a repo that no longer exists, and neither is a one-line fix.
(1) The profile table is pinned to older settings — `train` reads 0.005/4/10/20
where the registered tasks now run 0.005/10/100/50, and `play` no longer
differs from `train` in anything but `episode_length_s`. This staleness
predates ADR-016. (2) `scripted_action` and the grasp scenario's lift-and-shake
were written for absolute joint-position actions; ADR-016 made an action a
*relative* end-effector delta, so a sinusoid that used to be a bounded joint
sweep now integrates into drift. (`GRIPPER_CLOSE_ACTION = -2.0` names a scale
that is gone, but the gripper term clamps to `[-1, 1]`, so it still means
"fully closed" — the constant works, only its docstring is wrong.) Fixing (2)
means re-deriving amplitudes against
`DELTA_POS_SCALE`/`DELTA_ORI_SCALE` so the sweep stays in the workspace. The
script still runs; its accuracy comparison is still internally consistent
(every profile is driven by the same command); what it no longer measures is
the settings the benchmark actually uses.

Profiles (timestep / decimation / solver iterations / ls_iterations):

  train      0.005 / 4  /  10 / 20   current training settings (mjlab
                                     manipulation example; what QC-FQL
                                     online training runs on)
  play       0.010 / 1  /  20 / 10   current play/demo-collection settings
                                     (100 Hz control MDP, not 50 Hz!)
  train_acc  0.005 / 4  / 100 / 50   train timestep, MuJoCo-default solver
                                     caps. Isolates what 10/20 costs
  mujoco     0.002 / 10 / 100 / 50   MuJoCo defaults (OGBench-style), with
                                     decimation keeping 50 Hz control
  reference  0.001 / 20 / 100 / 100  accuracy baseline only (never a
                                     deployment candidate)

Everything else (cone, impratio, nativeccd, nconmax, ...) is taken unchanged
from the task's registered config, so the comparison isolates the four knobs
above.

Measurements per profile:

1. Accuracy: reset with a fixed seed, run a deterministically scripted
   scenario (a pure function of sim time, so every control rate samples the
   same continuous command), record qpos every 0.02 s of sim time, and
   compare against the ``reference`` profile at the same sim times. Three
   scenarios, from contact-free to contact-critical:

     sweep   free-space arm sweep over the table. No sustained contacts;
             isolates integrator/actuator differences.
     stack   every block teleported into one aligned tower (with 2 mm
             alternating level offsets so it is not knife-edge symmetric),
             arm held at home. Resting-contact stability: does the tower
             creep, jitter, or collapse? ``drift`` is the worst block's
             displacement from its teleported pose at the end (absolute,
             per profile).
     grasp   a cube teleported into the open gripper and re-pinned to the
             grasp site for the first 0.5 s of sim time while the fingers
             close (they reach the piece in <0.3 s, see
             tests/trlc_dk1/test_dk1_grasp.py; the last pin lands at exactly
             0.5 s for every profile, so all release from an identical
             pose), then carried through a lift-and-shake arm trajectory
             with the gripper held closed. Grip-contact accuracy under
             inertial load. Deviation stats start at release. ``slip`` is
             the max cube distance from the grasp site after release; a
             dropped cube shows up as ~0.2 m instead of a few mm.

   ``stack`` and ``grasp`` reach into ``HengeCommand`` for the block
   entities, so they need a LOOT-Stack-* task; on other tasks they are
   skipped with a note. Contact-rich trajectories diverge chaotically, so
   the absolute deviation grows with sim time for *every* profile. Read
   objDev comparatively (smaller = closer to reference), and lean on div@
   (first sim time the object DOFs drift >1 mm) plus the absolute
   ``drift``/``slip`` columns.

2. Speed: steps/s of a bare ``env.step()`` loop after a warmup (Warp kernel
   compile + CUDA graph capture happen on the first steps), reported with the
   real-time factor (steps/s x step_dt) and physics steps/s (x decimation).

3. ``--viewer`` (needs a display, skipped by default): the sweep scenario
   applied on top of the play config, run through the native viewer twice.
   At 1x speed (does demo collection keep up with real time on this
   machine?) and at max speed multiplier (render-loop throughput ceiling).

Usage:
    uv run python tests/tasks/compare_sim_settings.py
    uv run python tests/tasks/compare_sim_settings.py --cpu --sim-seconds 4
    uv run python tests/tasks/compare_sim_settings.py --viewer
    uv run python tests/tasks/compare_sim_settings.py --scenarios grasp
    uv run python tests/tasks/compare_sim_settings.py \\
        --task LOOT-Sorter-Cube-Circle-Dk1 --profiles train play
"""

import argparse
import dataclasses
import math
import sys
import time

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import loot_bench  # noqa: F401  (registers tasks)
from loot_bench.tasks.stack.mdp.commands import HengeCommand
from loot_bench.tasks.stack.variants import BLOCK_HALF_EXTENTS

DEFAULT_TASK = "LOOT-Stack-2Henges-Dk1"
SAMPLE_DT = 0.02  # sim-time spacing of accuracy samples; every profile's
# step_dt must divide this so all profiles are compared at identical times.
OBJECT_DRIFT_THRESHOLD = 1e-3  # 1 mm / 1 mrad: "first divergence" marker

GRIP_PIN_T = 0.5
"""[s] sim time the grasped cube stays pinned to the grasp site while the
fingers close; under gravity it would fall out of the still-open gripper."""
GRIPPER_CLOSE_ACTION = -2.0
"""Raw action for a fully closed gripper: the gripper action scale is half
the rack stroke, so open -> closed spans roughly [0, -2]."""
TOWER_XY = (0.30, 0.15)
"""Tower position on the table, clear of the home-pose gripper at
(0.30, 0.00, 0.20)."""
TOWER_LEVEL_OFFSET = 0.002
"""[m] alternating x offset per tower level (imperfect, realistic tower)."""


@dataclasses.dataclass(frozen=True)
class Profile:
    name: str
    timestep: float
    decimation: int
    iterations: int
    ls_iterations: int
    note: str

    @property
    def step_dt(self) -> float:
        return self.timestep * self.decimation


PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (
        Profile("train", 0.005, 4, 10, 20, "current training settings"),
        Profile("play", 0.01, 1, 20, 10, "current play/demo settings"),
        Profile("train_acc", 0.005, 4, 100, 50, "train dt, default solver caps"),
        Profile("mujoco", 0.002, 10, 100, 50, "MuJoCo defaults, 50 Hz control"),
        Profile("reference", 0.001, 20, 100, 100, "accuracy baseline"),
    )
}


def apply_profile(cfg, profile: Profile) -> None:
    cfg.sim.mujoco.timestep = profile.timestep
    cfg.decimation = profile.decimation
    cfg.sim.mujoco.iterations = profile.iterations
    cfg.sim.mujoco.ls_iterations = profile.ls_iterations


def scripted_action(t: float, dim: int, device: str) -> torch.Tensor:
    """Deterministic raw action as a pure function of sim time.

    A slow multi-joint sinusoidal sweep (phase-shifted per joint, so the arm
    moves through the workspace and over the table objects) plus a square-wave
    open/close on the last dim (the gripper on the DK1). Being a function of
    *time* rather than step index, every control rate samples the same
    continuous command. Required for cross-profile comparability.
    """
    j = torch.arange(dim, dtype=torch.float32, device=device)
    a = 0.35 * torch.sin(2.0 * math.pi * 0.15 * t + j * (math.pi / 3.0))
    a[-1] = 1.0 if (t % 4.0) < 2.0 else -1.0
    return a


def build_env(
    task: str, profile: Profile, device: str, *, deterministic: bool
) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(task, play=False)
    apply_profile(cfg, profile)
    cfg.scene.num_envs = 1
    if deterministic:
        # A mid-run auto-reset (success fires, or time_out) would replace the
        # trajectory with a fresh scatter and destroy the comparison. Same
        # trick as live_bridge/verify_replay.py.
        cfg.terminations.pop("success", None)
        cfg.terminations["time_out"].time_out = False
        cfg.episode_length_s = int(1e9)
    return ManagerBasedRlEnv(cfg=cfg, device=device)


def robot_qpos_mask(m: mujoco.MjModel) -> np.ndarray:
    """True for qpos entries belonging to the robot entity ("robot/..." joints)."""
    mask = np.zeros(m.nq, dtype=bool)
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        adr = int(m.jnt_qposadr[j])
        jtype = int(m.jnt_type[j])
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            n = 7
        elif jtype == mujoco.mjtJoint.mjJNT_BALL:
            n = 4
        else:
            n = 1
        if name.startswith("robot/"):
            mask[adr : adr + n] = True
    return mask


# ---------------------------------------------------------------------------
# Accuracy scenarios
# ---------------------------------------------------------------------------


class ScenarioUnavailable(Exception):
    """Raised by Scenario.prepare when the task cannot host the scenario."""


class Scenario:
    """One seeded accuracy rollout: scene setup + scripted policy + metrics."""

    name: str = ""
    title: str = ""
    extra_keys: tuple[str, ...] = ()
    stats_start_s: float = 0.0
    """Sim time before which samples are excluded from the deviation stats
    (setup phases like the grasp pin are not solver-accuracy signal)."""

    def prepare(self, env: ManagerBasedRlEnv, device: str) -> None:
        """Teleport the scene into shape, right after the seeded reset."""

    def action(self, t: float, dim: int, device: str) -> torch.Tensor:
        raise NotImplementedError

    def per_step(
        self, env: ManagerBasedRlEnv, device: str, t: float, step_dt: float
    ) -> None:
        """Hook before each env.step (*t* = sim time before the step)."""

    def on_sample(self, env: ManagerBasedRlEnv, device: str, t: float) -> None:
        """Hook at each accuracy sample (*t* = sim time after the step)."""

    def extras(self, env: ManagerBasedRlEnv, device: str) -> dict[str, float]:
        """Absolute per-profile metrics, reported next to the deviations."""
        return {}


def _stack_command(env: ManagerBasedRlEnv, scenario: str) -> HengeCommand:
    cmd = env.command_manager.get_term("task")
    if not isinstance(cmd, HengeCommand):
        raise ScenarioUnavailable(
            f"scenario '{scenario}' needs a LOOT-Stack-* task (command term is "
            f"{type(cmd).__name__})"
        )
    return cmd


class SweepScenario(Scenario):
    name = "sweep"
    title = "free-space arm sweep"

    def action(self, t: float, dim: int, device: str) -> torch.Tensor:
        return scripted_action(t, dim, device)


class StackScenario(Scenario):
    name = "stack"
    title = "teleported block tower at rest"
    extra_keys = ("drift",)

    def prepare(self, env: ManagerBasedRlEnv, device: str) -> None:
        cmd = _stack_command(env, self.name)
        origins = env.scene.env_origins
        num_envs = cmd.num_envs
        quat = torch.zeros(num_envs, 4, device=device)
        quat[:, 0] = 1.0
        poses: list[torch.Tensor] = []
        cum = 0.0
        for i, block in enumerate(cmd.blocks):
            h = BLOCK_HALF_EXTENTS[2]
            off = torch.tensor(
                [TOWER_XY[0] + (-1.0) ** i * TOWER_LEVEL_OFFSET, TOWER_XY[1]],
                device=device,
            )
            z = origins[:, 2] + cum + h
            pos = torch.cat([origins[:, :2] + off, z.unsqueeze(-1)], dim=-1)
            block.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
            block.write_root_link_velocity_to_sim(
                torch.zeros(num_envs, 6, device=device)
            )
            poses.append(pos)
            cum += 2 * h
        self._blocks = cmd.blocks
        self._pos0 = torch.stack(poses, dim=1)  # (E, N, 3)

    def action(self, t: float, dim: int, device: str) -> torch.Tensor:
        # Hold the arm at home (gripper open): only the tower's own contacts
        # act, so the rollout isolates resting-contact stability.
        return torch.zeros(dim, device=device)

    def extras(self, env: ManagerBasedRlEnv, device: str) -> dict[str, float]:
        pos = torch.stack([b.data.root_link_pos_w for b in self._blocks], dim=1)
        return {"drift": float((pos - self._pos0).norm(dim=-1).max())}


class GraspScenario(Scenario):
    name = "grasp"
    title = "cube gripped, lifted and shaken"
    extra_keys = ("slip",)
    stats_start_s = GRIP_PIN_T

    def prepare(self, env: ManagerBasedRlEnv, device: str) -> None:
        cmd = _stack_command(env, self.name)
        self._block = cmd.blocks[0]
        robot = env.scene["robot"]
        if "grasp_site" not in robot.site_names:
            raise ScenarioUnavailable(
                f"scenario '{self.name}': robot has no 'grasp_site' site"
            )
        self._robot = robot
        self._sid = robot.site_names.index("grasp_site")
        self._slip_max = 0.0
        self._pin(env, device)

    def _pin(self, env: ManagerBasedRlEnv, device: str) -> None:
        # Aligning the cube with the grasp-site frame makes its faces meet
        # the finger pads flat (the site frame is an axis permutation of the
        # ee frame. Equivalent for a cube).
        pos = self._robot.data.site_pos_w[:, self._sid]
        quat = self._robot.data.site_quat_w[:, self._sid]
        self._block.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
        self._block.write_root_link_velocity_to_sim(
            torch.zeros(pos.shape[0], 6, device=device)
        )

    def per_step(
        self, env: ManagerBasedRlEnv, device: str, t: float, step_dt: float
    ) -> None:
        # The half-step slack makes the *last* pin land at exactly
        # t == GRIP_PIN_T for every profile (0.5 is a multiple of every
        # step_dt), so all profiles release the cube from an identical pose.
        # Without it, each control rate would release from its own last pin
        # time and the deviation stats would start with a pin-cadence
        # artifact instead of solver signal.
        if t < GRIP_PIN_T + 0.5 * step_dt:
            self._pin(env, device)

    def action(self, t: float, dim: int, device: str) -> torch.Tensor:
        a = torch.zeros(dim, device=device)
        a[-1] = GRIPPER_CLOSE_ACTION
        u = t - GRIP_PIN_T
        if u > 0.0:
            # Lift-and-shake: a slow whole-arm swing plus a faster tremor,
            # ramped in over 1 s so the grip is not shock-loaded at release.
            ramp = min(u, 1.0)
            j = torch.arange(dim - 1, dtype=torch.float32, device=device)
            a[:-1] = ramp * (
                0.9 * torch.sin(2.0 * math.pi * 0.25 * u + j * (math.pi / 3.0))
                + 0.25 * torch.sin(2.0 * math.pi * 1.5 * u + j * (math.pi / 2.0))
            )
        return a

    def on_sample(self, env: ManagerBasedRlEnv, device: str, t: float) -> None:
        if t <= GRIP_PIN_T:
            return
        site = self._robot.data.site_pos_w[:, self._sid]
        block = self._block.data.root_link_pos_w
        self._slip_max = max(self._slip_max, float((block - site).norm(dim=-1).max()))

    def extras(self, env: ManagerBasedRlEnv, device: str) -> dict[str, float]:
        return {"slip": self._slip_max}


SCENARIOS: dict[str, type[Scenario]] = {
    s.name: s for s in (SweepScenario, StackScenario, GraspScenario)
}


def run_accuracy(
    task: str,
    profile: Profile,
    device: str,
    scenario_classes: list[type[Scenario]],
    sim_seconds: float,
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, dict[str, float]]], dict[str, str], np.ndarray]:
    """Run every scenario in one env (seeded reset each) under *profile*.

    Returns per-scenario (qpos trajectory sampled every SAMPLE_DT, extra
    metrics), the skip reasons of unavailable scenarios, and the robot qpos
    mask.
    """
    steps_per_sample = round(SAMPLE_DT / profile.step_dt)
    if abs(steps_per_sample * profile.step_dt - SAMPLE_DT) > 1e-9:
        raise SystemExit(
            f"[ERROR] profile '{profile.name}': step_dt {profile.step_dt:g} "
            f"does not divide the sample interval {SAMPLE_DT:g}"
        )
    env = build_env(task, profile, device, deterministic=True)
    dim = env.action_manager.total_action_dim
    mask = robot_qpos_mask(env.sim.mj_model)
    n_samples = round(sim_seconds / SAMPLE_DT)

    out: dict[str, tuple[np.ndarray, dict[str, float]]] = {}
    skipped: dict[str, str] = {}
    with torch.no_grad():
        for cls in scenario_classes:
            scenario = cls()
            env.reset(seed=seed)
            try:
                scenario.prepare(env, device)
            except ScenarioUnavailable as e:
                skipped[scenario.name] = str(e)
                continue
            samples: list[torch.Tensor] = []
            t = 0.0
            for k in range(n_samples * steps_per_sample):
                scenario.per_step(env, device, t, profile.step_dt)
                env.step(scenario.action(t, dim, device).unsqueeze(0))
                t += profile.step_dt
                if (k + 1) % steps_per_sample == 0:
                    samples.append(env.sim.data.qpos[0].clone())
                    scenario.on_sample(env, device, t)
            traj = torch.stack(samples).cpu().numpy()
            out[scenario.name] = (traj, scenario.extras(env, device))
    env.close()
    return out, skipped, mask


def run_perf(
    task: str,
    profile: Profile,
    device: str,
    steps: int,
    warmup: int,
    num_envs: int,
) -> float:
    """Return env steps/s of a bare step loop (scripted action, post-warmup)."""
    cfg = load_env_cfg(task, play=False)
    apply_profile(cfg, profile)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    dim = env.action_manager.total_action_dim
    env.reset(seed=0)

    t = 0.0
    with torch.no_grad():
        for _ in range(warmup):
            env.step(scripted_action(t, dim, device).expand(num_envs, dim))
            t += profile.step_dt
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(scripted_action(t, dim, device).expand(num_envs, dim))
            t += profile.step_dt
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    env.close()
    return steps / elapsed


def run_viewer(
    task: str, profile: Profile, device: str, steps: int, max_speed: bool
) -> tuple[float, float]:
    """Return (median fps, median steps/s) of a metered native-viewer run.

    Uses the play config (what loot-play / live_record build) with the
    profile's physics applied on top, the same scripted sweep as the headless
    passes, and the repo's NativeViewerWrapper (uncapped frame rate).
    """
    from mjlab.rl import RslRlVecEnvWrapper

    from loot_bench.scripts.utils.viewer_wrapper import NativeViewerWrapper

    cfg = load_env_cfg(task, play=True)
    apply_profile(cfg, profile)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    wrapped = RslRlVecEnvWrapper(env)
    dim = env.action_manager.total_action_dim

    fps_samples: list[float] = []
    sps_samples: list[float] = []

    class MeteredViewer(NativeViewerWrapper):
        def _update_stats(self):
            before = self._stats_last_time
            super()._update_stats()
            if self._stats_last_time != before:  # a 0.5 s window just closed
                fps_samples.append(self._fps)
                sps_samples.append(self._sps)

    clock = {"t": 0.0}

    def policy(obs):
        del obs
        action = scripted_action(clock["t"], dim, device).unsqueeze(0)
        clock["t"] += profile.step_dt
        return action

    viewer = MeteredViewer(wrapped, policy)
    if max_speed:
        viewer._speed_index = len(viewer.SPEED_MULTIPLIERS) - 1
        viewer._time_multiplier = viewer.SPEED_MULTIPLIERS[viewer._speed_index]
    viewer.run(num_steps=steps)
    env.close()

    steady_fps = sorted(fps_samples[2:])  # drop warmup windows
    steady_sps = sorted(sps_samples[2:])
    if not steady_fps:
        return float("nan"), float("nan")
    return steady_fps[len(steady_fps) // 2], steady_sps[len(steady_sps) // 2]


# ---------------------------------------------------------------------------
# Results and reporting
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AccStats:
    dev_mean: float
    dev_max: float
    robot_dev_max: float
    diverge_s: float
    extras: dict[str, float]


def acc_stats(
    traj: np.ndarray,
    ref: np.ndarray,
    robot_mask: np.ndarray,
    extras: dict[str, float],
    start_s: float = 0.0,
) -> AccStats:
    n0 = round(start_s / SAMPLE_DT)  # skip the scenario's setup phase
    n = min(len(traj), len(ref))
    dev = np.abs(traj[n0:n] - ref[n0:n])  # (n - n0, nq)
    obj = dev[:, ~robot_mask].max(axis=1)  # per-sample worst object DOF
    rob = dev[:, robot_mask].max(axis=1)
    over = np.nonzero(obj > OBJECT_DRIFT_THRESHOLD)[0]
    diverge = start_s + (over[0] + 1) * SAMPLE_DT if len(over) else math.inf
    return AccStats(
        float(obj.mean()), float(obj.max()), float(rob.max()), float(diverge), extras
    )


@dataclasses.dataclass
class Result:
    profile: Profile
    steps_per_s: float = float("nan")
    acc: dict[str, AccStats] = dataclasses.field(default_factory=dict)
    viewer_fps_1x: float = float("nan")
    viewer_sps_1x: float = float("nan")
    viewer_fps_max: float = float("nan")
    viewer_sps_max: float = float("nan")


def _fmt_diverge(v: float) -> str:
    return "  inf " if math.isinf(v) else f"{v:5.2f}s"


def print_perf_table(results: list[Result], viewer: bool) -> None:
    head = (
        f"{'profile':<10} {'dt/dec':>9} {'it/ls':>7} {'ctrl':>5} "
        f"{'steps/s':>8} {'xRT':>7} {'phys/s':>8}"
    )
    if viewer:
        head += f" {'fps@1x':>7} {'sps@1x':>7} {'fps@max':>8} {'sps@max':>8}"
    print("\n" + head)
    print("-" * len(head))
    for r in results:
        p = r.profile
        row = (
            f"{p.name:<10} {p.timestep:g}/{p.decimation:<2} "
            f"{p.iterations:>3}/{p.ls_iterations:<3} {1.0 / p.step_dt:>4.0f}Hz "
            f"{r.steps_per_s:>8.1f} {r.steps_per_s * p.step_dt:>6.1f}x "
            f"{r.steps_per_s * p.decimation:>8.0f}"
        )
        if viewer:
            row += (
                f" {r.viewer_fps_1x:>7.1f} {r.viewer_sps_1x:>7.1f}"
                f" {r.viewer_fps_max:>8.1f} {r.viewer_sps_max:>8.1f}"
            )
        print(row)
    print(
        "\n  xRT      = single-env real-time factor (steps/s x step_dt)"
        "\n  phys/s   = physics steps/s (steps/s x decimation)"
        "\n  sps@1x   = viewer steps/s at 1x speed; demo collection keeps up "
        "with real time iff sps@1x x step_dt ~= 1"
    )


def print_accuracy_tables(
    results: list[Result],
    scenario_classes: list[type[Scenario]],
    ref_extras: dict[str, dict[str, float]],
) -> None:
    for cls in scenario_classes:
        head = f"{'profile':<10} {'objDev mean/max':>16} {'div@':>6} {'robDev max':>10}"
        for key in cls.extra_keys:
            head += f" {key:>9}"
        print(f"\naccuracy vs reference — {cls.name}: {cls.title}")
        print(head)
        print("-" * len(head))
        for r in results:
            stats = r.acc.get(cls.name)
            if stats is None:
                continue
            row = (
                f"{r.profile.name:<10} "
                f"{stats.dev_mean:>7.1e}/{stats.dev_max:<8.1e} "
                f"{_fmt_diverge(stats.diverge_s)} {stats.robot_dev_max:>10.1e}"
            )
            for key in cls.extra_keys:
                row += f" {stats.extras[key]:>9.1e}"
            print(row)
        row = f"{'reference':<10} {'—':>7}/{'—':<8} {'—':>6} {'—':>10}"
        for key in cls.extra_keys:
            row += f" {ref_extras[cls.name][key]:>9.1e}"
        print(row)
    if scenario_classes:
        print(
            "\n  objDev = |qpos - reference| over non-robot DOFs, sampled "
            f"every {SAMPLE_DT:g} s sim time (comparative, not absolute)"
            f"\n  div@   = first sim time with object deviation > "
            f"{OBJECT_DRIFT_THRESHOLD:g}"
            "\n  drift  = worst block displacement from its teleported tower "
            "pose at the end (absolute)"
            "\n  slip   = max cube distance from the grasp site after grip "
            "release; a dropped cube reads ~2e-01 (absolute)"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(PROFILES),
        default=["train", "play", "train_acc", "mujoco"],
        help="profiles to compare (reference is always run as accuracy baseline)",
    )
    ap.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENARIOS),
        default=["sweep", "stack", "grasp"],
        help="accuracy scenarios (stack/grasp need a LOOT-Stack-* task)",
    )
    ap.add_argument(
        "--sim-seconds",
        type=float,
        default=6.0,
        help="sim time per accuracy rollout",
    )
    ap.add_argument("--seed", type=int, default=42, help="accuracy-rollout seed")
    ap.add_argument("--perf-steps", type=int, default=300)
    ap.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="untimed steps before the perf loop (Warp compile + graph capture)",
    )
    ap.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="perf loop only; accuracy and viewer always run a single env",
    )
    ap.add_argument(
        "--viewer",
        action="store_true",
        help="also measure native-viewer play mode (needs a display)",
    )
    ap.add_argument("--viewer-steps", type=int, default=300)
    ap.add_argument("--no-accuracy", action="store_true")
    ap.add_argument("--no-perf", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force CPU execution")
    args = ap.parse_args()

    device = "cuda:0" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"[INFO] task={args.task} device={device}")

    profiles = [PROFILES[name] for name in dict.fromkeys(args.profiles)]
    results = [Result(p) for p in profiles]

    scenario_classes: list[type[Scenario]] = []
    ref_extras: dict[str, dict[str, float]] = {}
    if not args.no_accuracy:
        scenario_classes = [SCENARIOS[n] for n in dict.fromkeys(args.scenarios)]
        print(
            f"[INFO] accuracy: scenarios {[c.name for c in scenario_classes]}, "
            f"{args.sim_seconds:g} s sim time each, seed {args.seed}, "
            f"reference = {PROFILES['reference']}"
        )
        print("[INFO] accuracy rollouts: reference")
        ref_out, ref_skipped, robot_mask = run_accuracy(
            args.task,
            PROFILES["reference"],
            device,
            scenario_classes,
            args.sim_seconds,
            args.seed,
        )
        for reason in ref_skipped.values():
            print(f"[WARN] {reason}; skipping")
        scenario_classes = [c for c in scenario_classes if c.name in ref_out]
        ref_extras = {name: extras for name, (_, extras) in ref_out.items()}
        for res in results:
            if res.profile.name == "reference":
                continue
            print(f"[INFO] accuracy rollouts: {res.profile.name}")
            out, _, _ = run_accuracy(
                args.task,
                res.profile,
                device,
                scenario_classes,
                args.sim_seconds,
                args.seed,
            )
            for cls in scenario_classes:
                traj, extras = out[cls.name]
                res.acc[cls.name] = acc_stats(
                    traj, ref_out[cls.name][0], robot_mask, extras, cls.stats_start_s
                )

    if not args.no_perf:
        for res in results:
            print(f"[INFO] perf loop: {res.profile.name} (num_envs={args.num_envs})")
            res.steps_per_s = run_perf(
                args.task,
                res.profile,
                device,
                args.perf_steps,
                args.warmup_steps,
                args.num_envs,
            )

    if args.viewer:
        for res in results:
            print(f"[INFO] viewer @1x: {res.profile.name}")
            res.viewer_fps_1x, res.viewer_sps_1x = run_viewer(
                args.task, res.profile, device, args.viewer_steps, max_speed=False
            )
            print(f"[INFO] viewer @max speed: {res.profile.name}")
            res.viewer_fps_max, res.viewer_sps_max = run_viewer(
                args.task, res.profile, device, args.viewer_steps, max_speed=True
            )

    if not args.no_perf or args.viewer:
        print_perf_table(results, viewer=args.viewer)
    print_accuracy_tables(results, scenario_classes, ref_extras)
    return 0


if __name__ == "__main__":
    sys.exit(main())
