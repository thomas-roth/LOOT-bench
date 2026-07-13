"""Play a trained QC-FQL agent with the MuJoCo viewer.

Rebuilds the agent from the checkpoint's config sidecar (falls back to
CLI/default config if absent), loads the weights, and runs it in the mjlab
NativeMujocoViewer for visual inspection.

Usage:
    uv run g3a-play-qcfql \
        --checkpoint out/agent.pt \
        --dataset-path out/demos/success/combined_45demos.npz \
        --task G3a-Cube-Stack3-Dk1_uni

Installed as g3a-play-qcfql (see pyproject.toml).
"""

import argparse

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import NativeMujocoViewer

import g3a_bench  # noqa: F401  (registers tasks)
from g3a_bench.scripts.live_bridge.mapper import (
    PHYSICAL_RAW_ACTION_MAX,
    PHYSICAL_RAW_ACTION_MIN,
)
from g3a_bench.scripts.qc_fql.agent.config import QCFQLConfig
from g3a_bench.scripts.qc_fql.agent.qcfql_agent import QCFQLAgent
from g3a_bench.scripts.qc_fql.scripts.train_qcfql import (
    load_checkpoint_config,
    load_dataset,
)


class _QCFQLViewerPolicy:
    """Policy wrapper for NativeMujocoViewer that runs a QC-FQL agent.

    Maintains an action chunk queue: the agent emits H actions at once
    (config.horizon_length), and the viewer consumes one per env step.
    """

    def __init__(self, agent, action_dim: int, device: str):
        self.agent = agent
        self.action_dim = action_dim
        self.device = device
        self.action_queue: list[np.ndarray] = []

    def __call__(self, obs) -> torch.Tensor:
        """Called by viewer each step with obs → returns raw action tensor."""
        if len(self.action_queue) == 0:
            # obs is a TensorDict (not a plain dict) with keys "actor" and "critic".
            if hasattr(obs, "keys") and "actor" in obs:
                obs_np = obs["actor"][0].detach().cpu().numpy()
            else:
                obs_np = obs[0].detach().cpu().numpy()
            obs_t = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                action_chunk = self.agent.sample_actions(obs_t).squeeze(0).cpu().numpy()
            self.action_queue.extend(action_chunk.reshape(-1, self.action_dim))

        action = np.clip(self.action_queue.pop(0), -1.0, 1.0)
        # Denormalize [-1, 1] → raw action space (same as MjlabSingleEnv)
        action_raw = (action + 1.0) / 2.0 * (
            PHYSICAL_RAW_ACTION_MAX - PHYSICAL_RAW_ACTION_MIN
        ) + PHYSICAL_RAW_ACTION_MIN
        return torch.from_numpy(action_raw).float().to(self.device).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="g3a-play-qcfql",
        description="Visualize a trained QC-FQL agent in the MuJoCo viewer.",
    )
    parser.add_argument(
        "--checkpoint", required=True, help="path to agent.pt saved by train_qcfql"
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help=".npz from lerobot_to_qc.py (needed to infer obs/action dims)",
    )
    parser.add_argument(
        "--task",
        default="G3a-Cube-Stack3-Dk1_uni",
        help="mjlab task id",
    )
    parser.add_argument("--device", default="cuda", help="torch device")
    # Fallback config for checkpoints without a sidecar (older runs). New
    # checkpoints carry <checkpoint>.json and ignore these.
    parser.add_argument(
        "--horizon-length", type=int, default=5, help="fallback H if no config sidecar"
    )
    parser.add_argument(
        "--actor-type",
        default="distill-ddpg",
        choices=["distill-ddpg", "best-of-n"],
        help="fallback actor type if no config sidecar",
    )
    args = parser.parse_args()

    # Rebuild agent with same config and load weights
    dataset = load_dataset(args.dataset_path)
    ex_obs = dataset.data["observations"][0]
    ex_act = dataset.data["actions"][0]

    config = load_checkpoint_config(args.checkpoint)
    if config is None:
        print(
            f"[WARN] No config sidecar for {args.checkpoint}; using CLI/default config"
        )
        config = QCFQLConfig(
            horizon_length=args.horizon_length, actor_type=args.actor_type
        )
    else:
        print(f"Loaded agent config from {args.checkpoint}.json")
    agent = QCFQLAgent.create(config, ex_obs, ex_act)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    agent = agent.to(args.device).eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    action_dim = dataset.data["actions"].shape[-1]

    # Create env with viewer config (play=True enables visual rendering)
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env = RslRlVecEnvWrapper(env)

    policy = _QCFQLViewerPolicy(agent, action_dim, args.device)

    print("[INFO] Starting MuJoCo viewer — close the window to exit")
    viewer = NativeMujocoViewer(env=env, policy=policy)
    viewer.run()
    env.close()


if __name__ == "__main__":
    main()
