"""molbuilder.parse.scripts.source_dict — the dict-shaped umbrella
extractor, WITH the schema-version gate.

One pass over a generated-script body (``.fdf`` / ``.py``) collecting
what the per-block extractors read: regions, frozen atoms, user-custom
lines, provenance, and the atom-metadata schema version — refusing an
unreadable block version through the READABLE-set gate and surfacing
diagnostics as notes.

MOVED here 2026-08-29 from ``parse/dirs/bundle.py`` when the bundle
parser retired (user ruling: calculation-to-calculation passing is
gone — a calculation that builds on a finished result CITES it, and
the transport composite's `compose_junction` does the fuse).  This
extractor was never the bundle's: `script_emit` re-exports it as the
read half of the emit/extract pair.

**NO PRODUCTION CALLER TODAY (2026-09-05).**  This said
`parse/dirs/job.py` "reads it for the live run decode" until the run
decoder was deleted on 2026-09-04; the re-export at
`script_emit.py:809` is now reached only from `tests/test_script_emit.py`.
The module still carries the ONLY copy of the schema-version gate
(below), which is why it has not been folded into `scripts/source.py`'s
typed parser -- that fold was deferred pending "its own careful pass
over the run decoder", and the run decoder is gone, so the deferral has
no remaining condition.  Flagged rather than deleted: which of the two
overlapping readers survives is a design call.

**Known overlap, noted rather than hidden**: `scripts/source.py`'s
``ScriptSourceTextParser`` walks the same blocks TYPED (ScriptResult,
no version gate); this module walks them into a dict WITH the gate.
Two readers of one format is exactly the shape this codebase folds —
the fold (job.py onto the typed parser + the gate moving into it) is
its own careful pass over the run decoder, flagged in the 2026-08-29
review, not done in it.
"""

from __future__ import annotations


# --------------------------------------------------------------------- #
#  Script-source extraction (regions + frozen + user-custom + ...)       #
# --------------------------------------------------------------------- #


#: The atom-metadata schema this build WRITES (script_emit stamps it from the
#: same constant). Compared against rather than a literal: a version written
#: down in two places is how the block came to claim v4 while carrying v7.



