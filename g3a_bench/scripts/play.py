"""Play entry point: mjlab's player with the g3a-bench tasks registered."""

import functools
import os

import mjlab.scripts.play as _mjlab_play

import g3a_bench  # noqa: F401  (registers benchmark tasks)


def main():
    """Wrap mjlab's play entry point, honoring G3A_VIEWER_FPS.

    mjlab's NativeMujocoViewer hardcodes a 60 fps render cap
    (``NativeMujocoViewer(env, policy).run()``, no CLI override). Setting
    G3A_VIEWER_FPS patches the default before delegating to mjlab's main().
    """
    fps = os.environ.get("G3A_VIEWER_FPS")
    if fps is not None:
        _mjlab_play.NativeMujocoViewer = functools.partial(
            _mjlab_play.NativeMujocoViewer, frame_rate=float(fps)
        )
    _mjlab_play.main()


if __name__ == "__main__":
    main()
