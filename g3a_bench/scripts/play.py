"""Play entry point: mjlab's player with the g3a-bench tasks registered.

Not PPO residue: this delegates entirely to mjlab's own player, which is
algorithm-agnostic. ``--agent zero``/``random`` (what this project actually
uses, see README) run a trivial in-line dummy policy -- no RSL-RL/PPO
involved at all. Only mjlab's own ``--agent trained`` mode touches RSL-RL,
and that mode is unusable regardless (nothing produces an RSL-RL checkpoint
since training is QC-FQL, not PPO).
"""

import mjlab.scripts.play as mjlab_play

import g3a_bench  # noqa: F401  (registers benchmark tasks)
from g3a_bench.scripts.utils.viewer_wrapper import NativeViewerWrapper


def main() -> None:
    mjlab_play.NativeMujocoViewer = NativeViewerWrapper
    mjlab_play.main()


if __name__ == "__main__":
    main()
