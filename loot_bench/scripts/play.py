"""Play entry point: mjlab's player with the loot-bench tasks registered."""

import mjlab.scripts.play as mjlab_play

import loot_bench  # noqa: F401  (registers benchmark tasks)
from loot_bench.scripts.utils.viewer_wrapper import NativeViewerWrapper


def main() -> None:
    mjlab_play.NativeMujocoViewer = NativeViewerWrapper
    mjlab_play.main()


if __name__ == "__main__":
    main()
