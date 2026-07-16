"""Replay LeRobot dataset in simulation using Native MuJoCo viewer.

Not a pytest test — an interactive manual-verification script.

Usage:
    uv run python tests/live_bridge/manual_replay_demo.py

Opens an interactive MuJoCo window showing the cube task environment
replaying leader actions from a recorded dataset.
"""

# Load lerobot/HF-datasets native libs before mjlab/mujoco-warp to avoid a
# native import-order segfault (HF `datasets` vs mujoco-warp). Must precede mjlab.
import lerobot.datasets.lerobot_dataset  # noqa: F401,E402,I001

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
import g3a_bench  # noqa: F401 — registers benchmark tasks in mjlab registry
from g3a_bench.scripts.live_bridge.mapper import lerobot_to_mjlab_raw_action
from g3a_bench.scripts.live_bridge.reader import DatasetReader
from g3a_bench.scripts.utils.viewer_wrapper import NativeViewerWrapper


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load a task config in play-mode (infinite episode, no noise).
    # ------------------------------------------------------------------
    cfg = load_env_cfg("G3a-Stacking-Stack3-Dk1_uni", play=True)
    cfg.terminations["time_out"].time_out = False

    # ------------------------------------------------------------------
    # 2. Create the environment.
    # ------------------------------------------------------------------
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env = RslRlVecEnvWrapper(env)
    print("[INFO] Environment created")

    # ------------------------------------------------------------------
    # 3. Open recorded dataset.
    # ------------------------------------------------------------------
    reader = DatasetReader("FabianKerj/corkinbox100-fullrel")
    print("[INFO] Dataset reader opened")

    # ------------------------------------------------------------------
    # 4. Replay policy: read one frame from dataset per env step.
    #    Loops back to the start when the episode ends.
    # ------------------------------------------------------------------
    class ReplayPolicy:
        def __call__(self, obs):
            try:
                frame = reader.read()
            except StopIteration:
                print("[INFO] Episode complete — looping back to start")
                reader._load_episode()
                frame = reader.read()

            if frame.leader_action is None:
                raise RuntimeError("DatasetReader frame is missing an action")
            raw_action = lerobot_to_mjlab_raw_action(frame.leader_action)
            return torch.from_numpy(raw_action).unsqueeze(0).to(device)

    # ------------------------------------------------------------------
    # 5. Launch the native MuJoCo viewer window.
    # ------------------------------------------------------------------
    print("[INFO] Starting Native MuJoCo viewer")
    print("[INFO] Press Ctrl+C in terminal or close the window to stop")
    NativeViewerWrapper(env=env, policy=ReplayPolicy()).run()

    # ------------------------------------------------------------------
    # 6. Cleanup.
    # ------------------------------------------------------------------
    env.close()
    reader.close()
    print("[INFO] Done")


if __name__ == "__main__":
    main()
