# Pin mjlab below 1.6

`pyproject.toml` bounds mjlab to `>=1.5.1,<1.6`. mjlab 1.6.0 refuses the DK1
collision specification, so no task builds on it.

## Consequences

- `CLAUDE.md` tells the reader to use the newest mjlab release. That guidance
  stops at 1.6.0 until the DK1 asset satisfies the new specification.
- The bound is a hard constraint, not a preference. Remove it only together
  with a DK1 asset that compiles on 1.6.
