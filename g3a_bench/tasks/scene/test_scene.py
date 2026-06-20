# Important: this imports your custom task registration
from mjlab.scripts.play import PlayConfig, run_play

import g3a_bench  # noqa: F401

run_play(
    task_id="Mjlab-Scene-Yam",
    cfg=PlayConfig(
        agent="random",
        device="cpu",
        viewer="viser",
    ),
)
