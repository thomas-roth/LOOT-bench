# pyright: reportAttributeAccessIssue=false
"""Relabel a Place ``success``-mode QC dataset for the ``progress`` variant.

``G3A-Place-Easy-Dk1_uni`` emits ``success - 1.0``; its ``-Dense`` twin emits
``progress + success - 2.0``.  ``qc.validate_reward_convention`` therefore
rejects the sparse ``.npz`` against the dense task, and the dense variant has
no data of its own.  Re-recording 100 teleop demos would also change the
trajectories, so this script recomputes the reward **in place from the
existing recording** and writes a new file.

Why no simulator is needed
--------------------------
Both dense terms are pure functions of state the ``.npz`` already stores:

    success[i]  = rewards[i] + source_reward_max          (exactly 0 or 1)
    dist[i]     = || observations[i][cube_pos] - target ||
    progress[i] = max(place_progress(dist[i]), success[i])
    rewards'[i] = progress[i] + success[i] - target_reward_max

``target`` is the per-episode-constant command slice of the same observation.
``place_progress`` and the success clamp are the task's own kernels, imported
rather than reimplemented, so the relabel cannot drift from the env.

The *pre-step* observation is the right one.  ``ManagerBasedRlEnv.step()``
computes rewards **before** it refreshes the command manager, so a frame's
reward is scored on the command metrics left standing from the previous step
-- exactly the state ``observations[i]`` records.  Using
``next_observations[i]`` instead is off by one control step and costs up to
2.6e-2 of reward on fast transport frames.

``--verify-episodes`` re-simulates whole episodes through the target task
(seeded reset + the recorded actions, the ``verify_replay.py`` recipe) and
compares the live per-frame reward with the relabelled one.  Measured
agreement over 14 episodes is 3.6e-7 .. 8.0e-5.  That residue is the *replay's*
error, not the relabel's: the recorded ``observation.qpos`` and the recorded
observations agree exactly with each other, and it is the open-loop
re-simulation that walks off by up to 35 um of cube position.  The recording
is the more faithful source, which is why the relabel reads it instead of
re-simulating.

Usage:
    # relabel + verify 5 episodes against the live dense env
    uv run python g3a_bench/scripts/live_bridge/relabel_place_dense.py \\
        --source dataset/demos/G3A-Place-Easy-Dk1_uni/G3A-Place-Easy-Dk1_uni_qc_physical.npz \\
        --target-task G3A-Place-Easy-Dense-Dk1_uni \\
        --output dataset/demos/G3A-Place-Easy-Dense-Dk1_uni/G3A-Place-Easy-Dense-Dk1_uni_qc_physical.npz \\
        --lerobot-dir dataset/demos/G3A-Place-Easy-Dk1_uni/2026-08-07_20-45-13 \\
        --verify-episodes 0 1 2 3 4
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

import g3a_bench  # noqa: F401,E402  -- registers the tasks
from g3a_bench.tasks.common.rewards import REWARD_OFFSET_TERM, reward_max  # noqa: E402
from g3a_bench.tasks.place.mdp.commands import MAX_TRAVEL_DIST  # noqa: E402
from g3a_bench.tasks.place.mdp.success import place_progress  # noqa: E402

CUBE_POSE_TERM = "cube_pose"
COMMAND_TERM = "command"
VERIFY_TOL = 2e-4
"""Per-frame |live - relabelled| a re-simulation may show before it counts as
a failure.  This is the *replay's* budget, not the relabel's: the relabel is
exact on the recording by construction, while an open-loop re-simulation walks
away from it.  Worst case over 14 Place episodes was 8.0e-5 of reward, i.e.
35 um of cube position (``dev * MAX_TRAVEL_DIST``)."""


def _obs_slices(env: ManagerBasedRlEnv) -> dict[str, slice]:
    """Actor-observation slice per term name, read off the built env.

    Never hardcoded: a future observation-term change must break this loudly
    rather than silently relabel against the wrong three columns.
    """
    mgr = env.observation_manager
    slices: dict[str, slice] = {}
    off = 0
    names = mgr.active_terms["actor"]
    dims = mgr.group_obs_term_dim["actor"]
    for name, dim in zip(names, dims, strict=True):
        width = int(np.prod(dim))
        slices[name] = slice(off, off + width)
        off += width
    return slices


def _check_target_task(cfg, target_task: str, source_max: float) -> float:
    """Assert the target task is the dense twin of the source, return its max."""
    if REWARD_OFFSET_TERM not in cfg.rewards:
        raise ValueError(
            f"{target_task} was not built by make_manipulation_env_cfg (no "
            f"{REWARD_OFFSET_TERM!r} term), so its reward convention is unknown."
        )
    target_max = reward_max(cfg.rewards)
    terms = {k for k in cfg.rewards if k != REWARD_OFFSET_TERM}
    if terms != {"success", "progress"}:
        raise ValueError(
            f"{target_task} has reward terms {sorted(terms)}; this script only "
            f"knows how to reconstruct 'success' + 'progress'."
        )
    if abs(target_max - (source_max + 1.0)) > 1e-6:
        raise ValueError(
            f"{target_task} has reward_max={target_max} but the source dataset "
            f"has reward_max={source_max}; the two differ by more than the one "
            f"'progress' term this script adds."
        )
    return target_max


def relabel_rewards(
    src: dict[str, np.ndarray],
    cube_pos: slice,
    target: slice,
    source_max: float,
    target_max: float,
) -> np.ndarray:
    """Per-frame ``progress + success - target_max`` for every episode."""
    obs = src["observations"]
    episode_idx = src["episode_index"]
    success = src["rewards"] + source_max
    if not np.all(np.isin(success, (0.0, 1.0))):
        raise ValueError(
            "Source rewards are not exactly {-reward_max, 0}, so the success "
            "indicator cannot be read back out of them. This script only "
            "relabels a sparse (success-only) recording."
        )

    dist = np.empty(len(obs), dtype=np.float32)
    for ep in np.unique(episode_idx):
        rows = episode_idx == ep
        goal = obs[rows][:, target]
        spread = float(np.max(np.abs(goal - goal[0])))
        if spread > 0.0:
            raise ValueError(
                f"Episode {ep}'s command observation is not constant (spread "
                f"{spread:.3e}); the target was resampled mid-episode and this "
                f"script's per-episode goal is wrong."
            )
        dist[rows] = np.linalg.norm(obs[rows][:, cube_pos] - goal[0], axis=-1)

    progress = (
        place_progress(torch.from_numpy(dist), MAX_TRAVEL_DIST)
        .numpy()
        .astype(np.float32)
    )
    # Same clamp PlaceCommand._update_metrics applies, and for the same
    # reason: the success frame must pay *exactly* 0 or upstream QC's
    # --sparse flag flattens it into an ordinary -1 step.
    progress = np.maximum(progress, success)

    rewards = (progress + success - target_max).astype(np.float32)
    if rewards.max() > 1e-6 or rewards.min() < -target_max - 1e-6:
        raise ValueError(
            f"Relabelled rewards span [{rewards.min():.6f}, {rewards.max():.6f}], "
            f"outside the [{-target_max}, 0] the dense task emits."
        )
    exact_zero = rewards == 0.0
    if not np.array_equal(exact_zero, success == 1.0):
        raise ValueError(
            "Relabelled reward is 0 on a different frame set than the source's "
            "success frames; the progress clamp did not reproduce the exact zero."
        )
    return rewards


def verify(
    env,
    lerobot_dir: str,
    episodes: list[int],
    src: dict[str, np.ndarray],
    dense: np.ndarray,
    device: str,
) -> bool:
    """Re-simulate whole episodes and compare the live reward to the relabel."""
    from g3a_bench.scripts.live_bridge.mapper import lerobot_to_mjlab_raw_action
    from g3a_bench.scripts.live_bridge.verify_replay import load_episode

    ok = True
    for e in episodes:
        rows = src["episode_index"] == e
        want = dense[rows]
        ep = load_episode(lerobot_dir, e)
        if len(ep.actions) != len(want):
            raise ValueError(
                f"Episode {e}: recording has {len(ep.actions)} frames but the "
                f"dataset has {len(want)} — mismatched --lerobot-dir?"
            )
        env.reset(seed=ep.seed)
        live = np.empty(len(want), dtype=np.float32)
        t0 = time.perf_counter()
        with torch.no_grad():
            for i, action in enumerate(ep.actions):
                raw = lerobot_to_mjlab_raw_action(action, gravity_compensate=True)
                _, rew, _, _, _ = env.step(
                    torch.from_numpy(raw).unsqueeze(0).to(device)
                )
                live[i] = float(rew[0])
        secs = time.perf_counter() - t0
        dev = np.abs(live - want)
        worst = int(np.argmax(dev))
        same_zero = np.array_equal(live == 0.0, want == 0.0)
        verdict = "OK" if dev.max() <= VERIFY_TOL and same_zero else "FAIL"
        print(
            f"  ep {e:3d}: {len(want):4d} frames in {secs:5.1f}s "
            f"({len(want) / secs:3.0f} steps/s) | max |Δ| {dev.max():.3e} "
            f"(= {dev.max() * MAX_TRAVEL_DIST * 1e6:.0f} um of replay drift) "
            f"@ f{worst} | success frame {'aligned' if same_zero else 'MOVED'} "
            f"| {verdict}"
        )
        ok = ok and verdict == "OK"
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True, help="Sparse Place .npz to relabel.")
    p.add_argument(
        "--target-task", required=True, help="Dense task id the output is for."
    )
    p.add_argument("--output", required=True, help="Output .npz path.")
    p.add_argument(
        "--lerobot-dir",
        default=None,
        help="Original live_record session, required by --verify-episodes.",
    )
    p.add_argument(
        "--verify-episodes",
        type=int,
        nargs="*",
        default=[],
        help="Re-simulate these episodes through the target task and compare.",
    )
    args = p.parse_args()

    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"[ERROR] {out_path} exists; refusing to overwrite.")
    if args.verify_episodes and not args.lerobot_dir:
        raise SystemExit("[ERROR] --verify-episodes needs --lerobot-dir.")

    with np.load(args.source, allow_pickle=True) as raw:
        src = {k: raw[k] for k in raw.files}
    if "reward_max" not in src:
        raise SystemExit(
            f"[ERROR] {args.source} has no reward_max metadata, so it predates "
            f"the negative reward convention. Rebuild it with lerobot_to_qc.py."
        )
    source_max = float(np.asarray(src["reward_max"]).reshape(-1)[0])
    print(f"[INFO] source: {args.source}")
    print(f"       task={str(src['task_id'])!r} reward_max={source_max} ")
    print(
        f"       {len(src['rewards'])} frames, "
        f"{len(np.unique(src['episode_index']))} episodes"
    )

    cfg = load_env_cfg(args.target_task, play=True)
    target_max = _check_target_task(cfg, args.target_task, source_max)
    # verify_replay.py's recipe: a done mid-replay auto-resets the env inside
    # step() and destroys the comparison. Termination terms only produce
    # `done`, never forces, so the physics being scored are unchanged.
    cfg.terminations["time_out"].time_out = False
    cfg.terminations.pop("ee_ground_collision", None)
    cfg.terminations.pop("success", None)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    slices = _obs_slices(env)
    missing = {CUBE_POSE_TERM, COMMAND_TERM} - set(slices)
    if missing:
        raise SystemExit(
            f"[ERROR] {args.target_task}'s actor observation has no "
            f"{sorted(missing)} term(s); cannot locate the cube pose / target."
        )
    cube_pose = slices[CUBE_POSE_TERM]
    cube_pos = slice(cube_pose.start, cube_pose.start + 3)  # pos, drop the quat
    target = slices[COMMAND_TERM]
    if (
        src["observations"].shape[1]
        != env.observation_manager.group_obs_dim["actor"][0]
    ):
        raise SystemExit(
            f"[ERROR] dataset observations are "
            f"{src['observations'].shape[1]}-wide but {args.target_task} emits "
            f"{env.observation_manager.group_obs_dim['actor'][0]}."
        )
    print(
        f"[INFO] target: {args.target_task} reward_max={target_max} | "
        f"cube_pos obs[{cube_pos.start}:{cube_pos.stop}] "
        f"target obs[{target.start}:{target.stop}]"
    )

    dense = relabel_rewards(src, cube_pos, target, source_max, target_max)
    print(
        f"[INFO] relabelled rewards: min {dense.min():.6f} max {dense.max():.6f} "
        f"mean {dense.mean():.6f} | {int((dense == 0.0).sum())} exact-zero frames"
    )

    ok = True
    if args.verify_episodes:
        print(
            f"[INFO] verifying {len(args.verify_episodes)} episode(s) "
            f"against the live env (tol {VERIFY_TOL:g}):"
        )
        ok = verify(env, args.lerobot_dir, args.verify_episodes, src, dense, device)
    env.close()
    if not ok:
        raise SystemExit("[ERROR] verification failed; nothing written.")

    out = dict(src)
    out["rewards"] = dense
    out["reward_max"] = np.array(target_max, dtype=np.float32)
    out["task_id"] = np.array(args.target_task)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(
        f"[INFO] Saved {out_path} (reward_max={target_max}, "
        f"task_id={args.target_task!r}); every other key copied unchanged."
    )


if __name__ == "__main__":
    main()
