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

from typing import Any, Dict, List, Optional

from .atom_metadata import _extract_atom_metadata_dict
from .provenance import _extract_provenance_dict
from .user_custom import _extract_user_custom_inner

# --------------------------------------------------------------------- #
#  Script-source extraction (regions + frozen + user-custom + ...)       #
# --------------------------------------------------------------------- #


#: The atom-metadata schema this build WRITES (script_emit stamps it from the
#: same constant). Compared against rather than a literal: a version written
#: down in two places is how the block came to claim v4 while carrying v7.
from molbuilder.sidecars.molstruct import READABLE_VERSIONS as _READABLE
from molbuilder.sidecars.molstruct import SCHEMA_VERSION as _CURRENT_SCHEMA


def _extract_script_source(text: str) -> Dict[str, Any]:
    """Single-pass extract over a generated-script body for the run
    decoder.  Returns a dict with:

      * ``regions``           dict[str, list[int]] | None
      * ``frozen_atoms``      list[int] | None
      * ``user_custom_lines`` list[str] | None
      * ``provenance``        dict[str, str] | None
      * ``schema_version``    int | None
      * ``notes``             list[str]

    ``None`` distinguishes "block absent" from "block present but
    empty" (``{}`` / ``[]``) — `model/parse.md`'s absent-vs-empty rule.  A block whose
    version is not the one this build writes is READ (a finished run must stay
    readable) and surfaced as a diagnostic note naming what may be missing --
    see the comment at the check itself.
    """
    notes: List[str] = []
    atom_md = _extract_atom_metadata_dict(text)
    regions: Optional[Dict[str, List[int]]] = None
    frozen: Optional[List[int]] = None
    schema_version: Optional[int] = None
    if atom_md is not None:
        sv = atom_md.get("schema_version")
        if isinstance(sv, int):
            schema_version = sv
            if sv not in _READABLE:
                # REFUSED, NOT READ (2026-08-01, by decision; amended
                # 2026-08-20 to the READABLE SET -- v8 added only optional
                # identity columns, so a v7 block reads whole and refusing
                # it would have made every existing finished run's labels
                # unreadable for a change that loses nothing).
                #
                # It used to READ an older block and attach a warning, on the
                # reasoning that a finished run cannot be re-exported the way a
                # sidecar can.  That reasoning is wrong for a product still
                # being built: an older block stores the same facts in different
                # places, so "read it and warn" hands back a payload that LOOKS
                # complete and quietly is not -- which is how a junction's fifty
                # frozen atoms came back as an empty list.  Supporting both
                # shapes also doubles what every reader, test and debugging
                # session has to hold in its head, for data that will be
                # regenerated anyway.
                #
                # The scripts get regenerated.  That is cheaper than a format
                # nobody can reason about.
                notes.append(
                    f"atom-metadata schema_version {sv}, but this molbuilder "
                    f"writes v{_CURRENT_SCHEMA} and reads "
                    f"{sorted(_READABLE)} only. The block was "
                    f"NOT read -- an older one keeps the same facts in "
                    f"different places (before v7 the frozen atoms sat in a "
                    f"top-level key rather than in `regions`), so reading it "
                    f"would silently drop what it cannot map. Re-generate the "
                    f"script from the structure."
                )
            else:
                # A BLOCK AT ANY OTHER VERSION IS READ, AND SAID SO ABOUT.
                #
                # This is a FINISHED RUN on disk, so refusing it outright would
                # make a user's existing results unreadable -- unlike the
                # sidecar, which is refused because it is still being worked on
                # and can be re-exported. But the note has to be accurate, and
                # this one was not: it said "molbuilder expects 4 — loading with
                # current handler", which reads as a formality.
                #
                # It is not. Before the label store was unified, the reserved
                # `frozen_atoms` list sat in a top-level key; this reader takes
                # the whole store from `regions`, so on an older block the
                # LABELS COME BACK AND THE FROZEN SET DOES NOT. That is how a
                # junction's fifty pinned electrode atoms read back as an empty
                # list. The note says which fact is at risk now, instead of
                # reporting a number.
                raw_regions = atom_md.get("regions")
                if isinstance(raw_regions, dict):
                    regions = {
                        str(k): sorted({int(i) for i in v})
                        for k, v in raw_regions.items()
                    }
                else:
                    regions = {}
                # ONE designated read: v5 keeps the reserved label in
                # `regions`, v3/v4 kept it in a top-level key, and this
                # knows which without the caller spelling the name.
                from molbuilder.sidecars import molstruct as _ms
                frozen = _ms.frozen_atoms(atom_md)
        else:
            notes.append(
                "atom-metadata block has no schema_version; ignored.")
    return {
        "regions":           regions,
        "frozen_atoms":      frozen,
        "user_custom_lines": _extract_user_custom_inner(text),
        "provenance":        _extract_provenance_dict(text),
        "schema_version":    schema_version,
        "notes":             notes,
    }
