# Important: this imports your custom task registration
import config.yam  # noqa: F401
from mjlab.scripts.play import PlayConfig, run_play

run_play(
    task_id="Mjlab-Scene-Yam",
    cfg=PlayConfig(
        agent="random",
        device="cpu",
        viewer="viser",
    ),
)
