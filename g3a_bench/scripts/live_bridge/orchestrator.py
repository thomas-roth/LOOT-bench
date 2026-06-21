import numpy as np

from g3a_bench.scripts.live_bridge.mapper import lerobot_to_mjlab_raw_action
from g3a_bench.scripts.live_bridge.recorder import Recorder
from g3a_bench.scripts.live_bridge.simulator import Simulator


def record_episode(
    sim: Simulator,
    recorder: Recorder,
    actions: list[np.ndarray],
    task: str = "",
) -> None:
    obs = sim.reset()
    for action in actions:
        ctrl = lerobot_to_mjlab_raw_action(action)
        obs_next, reward, done, _extras = sim.step(ctrl)
        # obs is state BEFORE action, action is what was applied,
        # reward/done are the RESULT of the action (LeRobot "next." convention)
        recorder.add_frame(obs, action, reward=reward, done=done, task=task)
        obs = obs_next
        if done:
            break
    recorder.end_episode()
