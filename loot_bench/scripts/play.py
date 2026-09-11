"""Play entry point: mjlab's player with the loot-bench tasks registered."""

import mjlab.scripts.play as mjlab_play

import loot_bench  # noqa: F401  (registers benchmark tasks)
from loot_bench.scripts.utils.viewer_wrapper import NativeViewerWrapper
from loot_bench.tasks.stack.mdp import commands as henge_commands


def main() -> None:
    mjlab_play.NativeMujocoViewer = NativeViewerWrapper
    # Henge build guide; a teleop recording must not show it.
    henge_commands.SHOW_GOAL_HENGE = True
    mjlab_play.main()


if __name__ == "__main__":
    main()
