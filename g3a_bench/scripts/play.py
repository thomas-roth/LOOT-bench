"""Play entry point: mjlab's player with the g3a-bench tasks registered."""

from mjlab.scripts.play import main

import g3a_bench  # noqa: F401  (importing registers the g3a-bench tasks)

if __name__ == "__main__":
    main()
