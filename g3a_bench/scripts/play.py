"""Play entry point: mjlab's player with the g3a-bench tasks registered."""

import mjlab.scripts.play as _mjlab_play

import g3a_bench  # noqa: F401  (registers benchmark tasks)
from g3a_bench.scripts.viewer_wrapper import NativeViewerWrapper


def main():
    """Wrap mjlab's play entry point with a throttled-overlay native viewer.

    mjlab's NativeMujocoViewer hardcodes ``frame_rate=60`` with no CLI
    override, and its per-frame status overlay additionally locks the loop
    to the monitor refresh rate. We always swap in the throttled-overlay
    viewer; the frame-rate defaults to ``NativeViewerWrapper.VIEWER_FPS``
    and can be overridden via the ``G3A_VIEWER_FPS`` environment variable.
    """
    _mjlab_play.NativeMujocoViewer = NativeViewerWrapper
    _mjlab_play.main()


if __name__ == "__main__":
    main()
