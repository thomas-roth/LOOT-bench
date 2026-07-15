"""Play entry point: mjlab's player with the g3a-bench tasks registered."""

import mjlab.scripts.play as mjlab_play

import g3a_bench  # noqa: F401  (registers benchmark tasks)
from g3a_bench.scripts.utils.viewer_wrapper import NativeViewerWrapper

if __name__ == "__main__":
    mjlab_play.NativeMujocoViewer = NativeViewerWrapper
    mjlab_play.main()
