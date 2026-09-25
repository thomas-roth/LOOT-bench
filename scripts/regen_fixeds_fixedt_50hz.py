"""Regenerate a working QC .npz from a legacy joint-absolute-50Hz recording.

The legacy FixedS-FixedT demo rows (Aug 2026, ``G3A-Place-FixedS-FixedT-Dk1_uni``)
were flown under the old MDP: 42-wide obs (with a ``last_action`` term, without
``pad_contact_force``), the old release gate, 10 s episodes. The recorded
``action`` column holds the leader pose (radians + gripper [0, 1]), NOT the raw
action ``env.step()`` took, and the old ``next.reward`` column describes the old
reward/gate. Neither can be copied through.

This script replays every recorded leader row through the CURRENT env at the
current decimation (50 Hz via the local ``CONTROL_DECIMATION = 4`` edit):

* leader pose -> raw action via the CURRENT ``mapper.lerobot_to_mjlab_raw_action``
  (current scale/offset, record-time gravity compensation on, clamp to the box),
* env seeded per episode from the recording's sidecar seeds, command resampled
  from the same seed (FixedS-FixedT is deterministic anyway),
* per frame: current 36-wide actor obs, current task reward, current
  success/time_out termination.

Output is stamped with the CURRENT convention (``joint-absolute-20hz`` name kept
as-is for now; ``control_step_s`` read off the live env = 0.02, current task id),
so ``check_dataset_matches_env`` passes against the env the data was generated
under. Transitions follow ``assemble_transitions`` semantics (success = absorbing,
timeout rows dropped).

Usage:
    uv run python scripts/regen_fixeds_fixedt_50hz.py \
        --src dataset/demos/G3A-Place-FixedS-FixedT-Dk1_uni/2026-08-28_10-57-41 \
        --task LOOT-Place-FixedS-FixedT-Dk1 \
        --output dataset/qc/place_fixed-s_fixed-t_100-demos_regen-50hz.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from loot_bench.qc.dataset import CONVENTION_NAME  # noqa: E402
from loot_bench.qc.transitions import assemble_transitions  # noqa: E402
from loot_bench.scripts.live_bridge.mapper import (  # noqa: E402
    action_scale_offset,
    lerobot_to_mjlab_raw_action,
)


def _load_parquet(path: Path) -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, help="Legacy LeRobot dataset root.")
    parser.add_argument("--task", required=True, help="Current registered task id.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--device", default="cuda", help="torch device for the sim.")
    parser.add_argument(
        "--max-episodes", type=int, default=-1, help="Replay at most N episodes."
    )
    args = parser.parse_args()

    from mjlab.tasks.registry import load_env_cfg
    from mjlab.envs import ManagerBasedRlEnv

    src = Path(args.src)
    df = _load_parquet(src / "data" / "chunk-000" / "file-000.parquet")

    sidecar = src / "meta" / "g3a_extra_info.json"
    legacy_sidecar = {}
    if sidecar.exists():
        legacy_sidecar = json.loads(sidecar.read_text())
    seeds = {str(k): int(v) for k, v in legacy_sidecar.get("episode_seeds", {}).items()}
    seed_device = legacy_sidecar.get("seed_device")
    print(f"[INFO] legacy sidecar seeds: {len(seeds)} (device {seed_device})")
    if not seeds:
        raise SystemExit("no episode seeds in legacy sidecar; cannot replay exactly")

    cur_scale, cur_offset = action_scale_offset()
    print(f"[INFO] current action scale:  {np.asarray(cur_scale).round(4)}")
    print(f"[INFO] current action offset: {np.asarray(cur_offset).round(4)}")
    grav = bool(legacy_sidecar.get("gravity_compensate", True))
    print(f"[INFO] gravity_compensate={grav} (from legacy sidecar)")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"[INFO] device: {device}")
    cfg = load_env_cfg(args.task, play=False)
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    cfg.observations["actor"].enable_corruption = False
    # FixedS-FixedT always resolves the same spawn/target/seed mapping; the
    # legacy rows were recorded under 10 s episodes. Replay under the same
    # budget so a slow-but-successful demo still terminates on success
    # instead of truncating mid-place (only the rate changed, not the task).
    cfg.episode_length_s = 10.0
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    from loot_bench.tasks.common.actions import JOINT_POS_ACTION_NAME

    term = env.action_manager.get_term(JOINT_POS_ACTION_NAME)
    env_scale = term.scale.detach().cpu().numpy().reshape(-1)
    env_offset = term.offset.detach().cpu().numpy().reshape(-1)
    step_dt = float(env.cfg.decimation * env.cfg.sim.mujoco.timestep)
    obs_dim = int(np.prod(env.observation_manager.group_obs_dim["actor"]))
    print(f"[INFO] env step_dt={step_dt:g} s, obs_dim={obs_dim}")
    if not (
        np.allclose(env_scale, cur_scale, atol=1e-5)
        and np.allclose(env_offset, cur_offset, atol=1e-5)
    ):
        raise SystemExit(
            "mapper scale/offset disagree with the built env term; aborting"
        )

    ep_ids = sorted(int(e) for e in df["episode_index"].unique())
    if args.max_episodes > 0:
        ep_ids = ep_ids[: args.max_episodes]
    print(f"[INFO] replaying {len(ep_ids)} episodes")

    # The cube's recorded spawn (x=0.24) predates the current fixed-spawn
    # table (x=0.25, df2d26f): same task, 1 cm drift. Replay each episode
    # from the RECORDED cube pose so the arm meets the cube where the
    # operator put it; the command target is identical either way.

    all_obs: list[np.ndarray] = []
    all_act: list[np.ndarray] = []
    all_rew: list[float] = []
    all_ep: list[int] = []
    all_frame: list[int] = []
    kept_eps = 0
    dropped_eps: list[int] = []
    with torch.no_grad():
        for new_i, old_i in enumerate(ep_ids):
            sub = df[df["episode_index"] == old_i].sort_values("frame_index")
            leader = np.stack(sub["action"].to_numpy()).astype(np.float32)
            n = len(sub)
            seed = seeds.get(str(old_i))
            if seed is None:
                print(f"[WARN] ep {old_i}: no seed, skipping")
                dropped_eps.append(old_i)
                continue
            env.reset(seed=seed)
            # Respawn-correction: teleport the cube to its recorded
            # frame-0 pose (root position + quat). Arm reset pose is
            # untouched, so the opening actions still match.
            rec_qpos_all = np.stack(
                df[df["episode_index"] == old_i]
                .sort_values("frame_index")["observation.qpos"]
                .to_numpy()
            ).astype(np.float32)
            qpos_now = env.sim.data.qpos[0].cpu().numpy().copy()
            qpos_now[:7] = rec_qpos_all[0][:7]
            env.sim.data.qpos[0] = torch.from_numpy(qpos_now).to(device)
            env.sim.forward()
            obs = env.get_observations()["actor"][0].detach().cpu().numpy()
            done_seen = False
            for t in range(n):
                raw = lerobot_to_mjlab_raw_action(
                    leader[t],
                    gravity_compensate=grav,
                    scale=cur_scale,
                    offset=cur_offset,
                ).astype(np.float32)
                raw = np.clip(raw, -1.0, 1.0)
                obs_t = env.get_observations()["actor"][0].detach().cpu().numpy()
                assert obs_t.shape == (obs_dim,), (
                    obs_t.shape,
                    obs_dim,
                )
                o, r, terminated, truncated, _ = env.step(
                    torch.as_tensor(raw, dtype=torch.float32, device=device).reshape(
                        1, -1
                    )
                )
                done = bool(terminated[0].item()) or bool(truncated[0].item())
                all_obs.append(np.asarray(obs_t, dtype=np.float32))
                all_act.append(raw)
                all_rew.append(float(r[0].item()))
                all_ep.append(new_i)
                all_frame.append(t)
                obs = o
                if done:
                    done_seen = True
                    if t + 1 < n:
                        print(
                            f"[WARN] ep {old_i}: env done at frame {t}, "
                            f"recording has {n} rows; truncating episode here"
                        )
                    break
            if not done_seen:
                print(
                    f"[WARN] ep {old_i}: env never signalled done "
                    f"(replayed {n} rows); episode kept as truncated"
                )
            kept_eps += 1
            if (new_i + 1) % 10 == 0:
                print(f"[INFO] replayed {new_i + 1}/{len(ep_ids)} episodes")

    obs_arr = np.stack(all_obs).astype(np.float32)
    act_arr = np.stack(all_act).astype(np.float32)
    rew_arr = np.asarray(all_rew, dtype=np.float32)
    ep_arr = np.asarray(all_ep, dtype=np.int64)
    fr_arr = np.asarray(all_frame, dtype=np.int64)
    print(f"[INFO] kept {kept_eps} episodes, dropped {len(dropped_eps)}")
    print(
        f"[INFO] rows: {len(obs_arr)}, obs {obs_arr.shape}, "
        f"act min/max {act_arr.min():.3f}/{act_arr.max():.3f}, "
        f"rew min/max {rew_arr.min():.3f}/{rew_arr.max():.3f}, "
        f"success rows {(rew_arr == 0.0).sum()}"
    )

    data = {
        **assemble_transitions(
            episode_index=ep_arr,
            observations=obs_arr,
            actions=act_arr,
            rewards=rew_arr,
        ),
        "convention": np.array(CONVENTION_NAME),
        "action_scale": np.asarray(env_scale, dtype=np.float32),
        "action_offset": np.asarray(env_offset, dtype=np.float32),
        "control_step_s": np.float32(step_dt),
        "start_pose_noise": np.float32(0.0),
        "task_id": np.array(args.task),
    }
    out = Path(args.output)
    np.savez(out, **data)  # type: ignore[arg-type]
    print(f"[INFO] wrote {out}")
    print(f"       transitions: {len(data['rewards'])}")
    print(f"       obs {data['observations'].shape} act {data['actions'].shape}")

    # Self-check against the live env convention.
    from loot_bench.qc.dataset import check_dataset_matches_env, load_dataset

    loaded = load_dataset(out, env=None)

    class _Env:
        observation_space = type("S", (), {"shape": (obs_dim,)})()
        action_space = type("S", (), {"shape": (7,)})()

        def action_convention(self):  # noqa: ANN204
            return {
                "convention": CONVENTION_NAME,
                "action_scale": np.asarray(env_scale, dtype=np.float32),
                "action_offset": np.asarray(env_offset, dtype=np.float32),
                "control_step_s": step_dt,
                "start_pose_noise": 0.0,
                "task_id": args.task,
            }

    check_dataset_matches_env(loaded, _Env())
    print("[INFO] check_dataset_matches_env: PASS")
    env.close()


if __name__ == "__main__":
    main()
