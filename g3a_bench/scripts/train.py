"""Train entry point: mjlab's trainer with the g3a-bench tasks registered."""

from mjlab.scripts.train import main

import g3a_bench  # noqa: F401  (registers benchmark tasks)

if __name__ == "__main__":
    main()
