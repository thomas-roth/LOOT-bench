# Architecture decision records (ADRs)

Each file here records one decision: what we chose, and why. The code shows
what the repo does. These files show why it does that, and not the obvious
other thing.

Read the records that touch the area you are about to change. If your work
contradicts one, say so and reopen the decision instead of overturning it in
silence.

## What earns a record

All three of these must be true:

1. The decision is hard to reverse. Changing your mind later costs real work.
2. It is surprising without the reason. A future reader will look at the code
   and ask why it was done that way.
3. There was a true trade-off. Other options existed, and we rejected them for
   stated reasons.

An explicit "no" counts. Several records here exist only to stop an expensive
experiment from being run a second time.

Do not record a decision that is easy to reverse, because you will simply
reverse it. Do not record one that surprises nobody.

## Format

A title, then one to three sentences: the context, the decision, and the
reason. Add a "Considered options" or "Consequences" section only when it
carries something a reader cannot infer.

Give the numbers that decided it, so a later reader can check the reasoning
against the same measurement.

## Numbering

`NNN-short-slug.md`, in sequence. Take the highest number in this directory
and add one. Numbers are never reused, and a file is never renumbered, because
other records cite it as ADR-NNN.

A superseded record stays. Mark it with a `status:` line at the top and name
the record that replaces it.

## Related

`docs/agents/domain.md` says how the engineering agents use these files
together with `CONTEXT.md`, the project glossary.
