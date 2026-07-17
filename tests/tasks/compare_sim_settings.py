# pyright: reportAttributeAccessIssue=false
"""Compare sim-setting profiles for speed and accuracy.

See tests/tasks/README.md.

Profiles (timestep / decimation / solver iterations / ls_iterations):

  train      0.005 / 4  /  10 / 20   current training settings (mjlab
                                     manipulation example; what QC-FQL
                                     online training runs on)
  play       0.010 / 1  /  20 / 10   current play/demo-collection settings
                                     (100 Hz control MDP, not 50 Hz!)
  train_acc  0.005 / 4  / 100 / 50   train timestep, MuJoCo-default solver
                                     caps — isolates what 10/20 costs
  mujoco     0.002 / 10 / 100 / 50   MuJoCo defaults (OGBench-style), with
                                     decimation keeping 50 Hz control
  reference  0.001 / 20 / 100 / 100  accuracy baseline only (never a
                                     deployment candidate)

Everything else (cone, impratio, nativeccd, nconmax, ...) is taken unchanged
from the task's registered config, so the comparison isolates the four knobs
above.

Two headless measurements per profile, plus an optional viewer pass:

1. Accuracy: reset with a fixed seed, drive the arm with a deterministic
   scripted sweep (a pure function of sim time, so every control rate samples
   the same continuous command), record qpos every 0.02 s of sim time, and
   compare against the ``reference`` profile at the same sim times.
   Contact-rich trajectories diverge chaotically, so the absolute deviation
   grows with sim time for *every* profile — read the numbers comparatively
   (smaller = closer to reference), and look at the first time the object
   DOFs drift by more than 1 mm.

2. Speed: steps/s of a bare ``env.step()`` loop after a warmup (Warp kernel
   compile + CUDA graph capture happen on the first steps), reported with the
   real-time factor (steps/s x step_dt) and physics steps/s (x decimation).

3. ``--viewer`` (needs a display, skipped by default): the same profile
   applied on top of the play config, run through the native viewer twice —
   at 1x speed (does demo collection keep up with real time on this
   machine?) and at max speed multiplier (render-loop throughput ceiling).

Usage:
    uv run python tests/tasks/compare_sim_settings.py
    uv run python tests/tasks/compare_sim_settings.py --cpu --sim-seconds 4
    uv run python tests/tasks/compare_sim_settings.py --viewer
    uv run python tests/tasks/compare_sim_settings.py \\
        --task G3A-Sorter-Cube-Circle-Dk1_uni --profiles train play
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

import g3a_bench  # noqa: F401  (registers tasks)

DEFAULT_TASK = "G3A-Stack-Low-Dk1_uni"
SAMPLE_DT = 0.02  # sim-time spacing of accuracy samples; every profile's
# step_dt must divide this so all profiles are compared at identical times.
OBJECT_DRIFT_THRESHOLD = 1e-3  # 1 mm / 1 mrad: "first divergence" marker


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
    continuous command — required for cross-profile comparability.
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
        # trajectory with a fresh scatter and destroy the comparison — same
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


def run_accuracy(
    task: str, profile: Profile, device: str, sim_seconds: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (qpos trajectory sampled every SAMPLE_DT, robot qpos mask)."""
    steps_per_sample = round(SAMPLE_DT / profile.step_dt)
    if abs(steps_per_sample * profile.step_dt - SAMPLE_DT) > 1e-9:
        raise SystemExit(
            f"[ERROR] profile '{profile.name}': step_dt {profile.step_dt:g} "
            f"does not divide the sample interval {SAMPLE_DT:g}"
        )
    env = build_env(task, profile, device, deterministic=True)
    dim = env.action_manager.total_action_dim
    env.reset(seed=seed)

    n_samples = round(sim_seconds / SAMPLE_DT)
    samples: list[torch.Tensor] = []
    t = 0.0
    with torch.no_grad():
        for k in range(n_samples * steps_per_sample):
            action = scripted_action(t, dim, device).unsqueeze(0)
            env.step(action)
            t += profile.step_dt
            if (k + 1) % steps_per_sample == 0:
                samples.append(env.sim.data.qpos[0].clone())
    traj = torch.stack(samples).cpu().numpy()
    mask = robot_qpos_mask(env.sim.mj_model)
    env.close()
    return traj, mask


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

    Uses the play config (what g3a-play / live_record build) with the
    profile's physics applied on top, the same scripted sweep as the headless
    passes, and the repo's NativeViewerWrapper (uncapped frame rate).
    """
    from mjlab.rl import RslRlVecEnvWrapper

    from g3a_bench.scripts.utils.viewer_wrapper import NativeViewerWrapper

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


@dataclasses.dataclass
class Result:
    profile: Profile
    steps_per_s: float = float("nan")
    obj_dev_mean: float = float("nan")
    obj_dev_max: float = float("nan")
    robot_dev_mean: float = float("nan")
    robot_dev_max: float = float("nan")
    obj_diverge_s: float = float("nan")
    viewer_fps_1x: float = float("nan")
    viewer_sps_1x: float = float("nan")
    viewer_fps_max: float = float("nan")
    viewer_sps_max: float = float("nan")


def accuracy_stats(
    res: Result, traj: np.ndarray, ref: np.ndarray, robot_mask: np.ndarray
) -> None:
    n = min(len(traj), len(ref))
    dev = np.abs(traj[:n] - ref[:n])  # (n, nq)
    obj = dev[:, ~robot_mask].max(axis=1)  # per-sample worst object DOF
    rob = dev[:, robot_mask].max(axis=1)
    res.obj_dev_mean = float(obj.mean())
    res.obj_dev_max = float(obj.max())
    res.robot_dev_mean = float(rob.mean())
    res.robot_dev_max = float(rob.max())
    over = np.nonzero(obj > OBJECT_DRIFT_THRESHOLD)[0]
    res.obj_diverge_s = float((over[0] + 1) * SAMPLE_DT) if len(over) else math.inf


def print_table(results: list[Result], viewer: bool) -> None:
    head = (
        f"{'profile':<10} {'dt/dec':>9} {'it/ls':>7} {'ctrl':>5} "
        f"{'steps/s':>8} {'xRT':>7} {'phys/s':>8} "
        f"{'objDev mean/max':>16} {'div@':>6} {'robDev max':>10}"
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
            f"{r.steps_per_s * p.decimation:>8.0f} "
            f"{r.obj_dev_mean:>7.1e}/{r.obj_dev_max:<8.1e} "
            f"{r.obj_diverge_s:>5.2f}s {r.robot_dev_max:>10.1e}"
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
        "\n  objDev   = |qpos - reference| over non-robot DOFs, sampled every "
        f"{SAMPLE_DT:g} s sim time (comparative, not absolute — see docstring)"
        f"\n  div@     = first sim time with object deviation > "
        f"{OBJECT_DRIFT_THRESHOLD:g}"
        "\n  sps@1x   = viewer steps/s at 1x speed; demo collection keeps up "
        "with real time iff sps@1x x step_dt ~= 1"
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

    if not args.no_accuracy:
        print(
            f"[INFO] accuracy: {args.sim_seconds:g} s sim time, seed {args.seed}, "
            f"reference = {PROFILES['reference']}"
        )
        ref_traj, robot_mask = run_accuracy(
            args.task, PROFILES["reference"], device, args.sim_seconds, args.seed
        )
        for res in results:
            if res.profile.name == "reference":
                continue
            print(f"[INFO] accuracy rollout: {res.profile.name}")
            traj, _ = run_accuracy(
                args.task, res.profile, device, args.sim_seconds, args.seed
            )
            accuracy_stats(res, traj, ref_traj, robot_mask)

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

    print_table(results, viewer=args.viewer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
