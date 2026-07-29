"""Play entry point: mjlab's player with the g3a-bench tasks registered."""

import mjlab.scripts.play as mjlab_play

import g3a_bench  # noqa: F401  (registers benchmark tasks)
from g3a_bench.scripts.utils.viewer_wrapper import NativeViewerWrapper
from g3a_bench.tasks.stack.mdp import commands as stack_commands


def main() -> None:
    mjlab_play.NativeMujocoViewer = NativeViewerWrapper
    # Build guide for the stack tasks, interactive play only (teleop
    # recordings keep the plain target-height plane).
    stack_commands.SHOW_GOAL_STACK = True
    mjlab_play.main()


if __name__ == "__main__":
    main()
