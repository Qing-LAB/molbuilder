"""PDB (``.pdb``) FileParser — a geometry file somebody else's tool wrote.

WHY IT EXISTS.  `/api/results/dir` asks the registry *what reads this
file* and the Results picker drops anything the answer is ``None`` for,
so a format with a reader but no registration is a format the browser
cannot offer.  ``structure.js`` has claimed ``.xyz`` **and** ``.pdb``
since it was written, and nothing in ``parse/`` claimed ``.pdb`` — so
half that presenter was unreachable from the tab from 2026-09-18, when
the picker moved onto the server's verdict.  The reader was already
here: :meth:`Structure.from_pdb`, and ``StructureCodec.load`` says so
in a comment — *"the file picker accepts .xyz AND .pdb, and each needs
its own parser"*.  The gap was the registration, not the reading.

Same shape as the sweep's ``job-set.json`` two files over, and the
``.transport.json`` record before it: a reader with no row in the
registry, found by asking what the door can answer for a real
directory rather than by reading the presenter and believing it.

WHAT IT DOES NOT CLAIM.  ``cell`` is ``None``.  A PDB states its
lattice in a ``CRYST1`` record and :meth:`Structure.from_pdb` reads
ATOM/HETATM only, so there is no cell to carry — and inventing one
from the coordinates is the failure `siesta_xv` exists to avoid.
``pyscf-geom`` answers ``None`` for the same honest reason.
"""

from __future__ import annotations

from pathlib import Path

from molbuilder.parse.base import FileParser
from molbuilder.parse.errors import ParseError
from molbuilder.parse.types import StructureResult

from ._helpers import build_structure_result

#: How far to look before giving up.  Generous, because the thing being
#: skipped is the HEADER and a PDB header has no small bound: measured
#: over the 305 `.pdb` in this tree, the first coordinate record sits
#: anywhere from byte 0 to byte 286011 (`1jj2.pdb`, a ribosome).
#:
#: THIS WAS 8192 FOR AN HOUR ON 2026-09-19 AND REFUSED TEN REAL FILES --
#: ordinary RCSB entries whose `REMARK`/`SEQRES`/`HELIX`/`SHEET` preamble
#: runs past 8 KB (`1kx5` nucleosome 59940, `2kei` NMR ensemble 26406,
#: `2acj` 43821).  The reasoning was "a solvated system reaches hundreds
#: of megabytes, so do not read it whole", which is true of the FILE and
#: says nothing about the header -- and the one file it was validated
#: against, `1c75.pdb`, starts at 5751, seventy per cent of the way
#: through the window it passed.  A margin that thin is not a margin.
#:
#: The scan still stops at the first coordinate record, so a real PDB
#: costs its header and not its atoms: `1jj2.pdb` is 8.3 MB and answers
#: after 286 KB.  The cap only bounds the pathological case -- a large
#: file with a `.pdb` name and no coordinates anywhere.
_SNIFF_BYTES = 4 * 1024 * 1024


def _looks_like_pdb(path: Path) -> bool:
    """Does this file carry a coordinate record?

    THE CONTENT, NOT THE SUFFIX -- the rule `sidecars/transport.py`
    states and the reason its predecessor was deleted.  A ``.pdb`` that
    holds a refusal message, a truncated download or somebody's notes
    is not a structure, and offering it puts an error card where a
    molecule should be.

    Streamed, and it RETURNS AT THE FIRST HIT, so the cost is the
    header rather than the file.  Reading a fixed head instead is what
    refused ten real structures; see :data:`_SNIFF_BYTES`.
    """
    try:
        seen = 0
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(("ATOM  ", "HETATM")):
                    return True
                seen += len(line)
                if seen > _SNIFF_BYTES:
                    return False
    except OSError:
        return False
    return False


class PdbFileParser(FileParser):
    """Parse a ``.pdb`` into a :class:`StructureResult`."""

    name = "pdb"
    label = "PDB coordinates (.pdb)"
    hint = "a .pdb holding ATOM / HETATM records"
    output = StructureResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        p = Path(path)
        return (p.suffix.lower() == ".pdb" and p.is_file()
                and _looks_like_pdb(p))

    @classmethod
    def parse(cls, path: Path) -> StructureResult:
        p = Path(path)
        # `from_pdb` takes TEXT, not a path -- its own docstring says so
        # and points at `StructureCodec.load` for files.  The codec is
        # floor 2 and reaches back into `parse`, so this reads the text
        # and calls the L1 door directly, which is what `siesta_xv` does
        # with `molbuilder.structure` for the same reason.
        # THROUGH THE ONE DOOR, because a `.pdb` here is a PERSON'S
        # structure and not an engine's output.  `read_text` + `from_pdb`
        # skips the `.molstruct.json` beside it, so the regions, frozen
        # atoms and cell its author set are silently dropped and
        # everything downstream is faithfully correct about the wrong
        # thing -- the failure `test_one_door_reads_a_structure` exists
        # to stop, and which it caught here within the hour.
        #
        # `StructureCodec.load` also reads `utf-8-sig`, so a file saved
        # with a BOM no longer parses differently here than it does when
        # the sidebar opens it.
        #
        # Wrapped because `base.FileParser.parse` requires the canonical
        # `ParseError`; an unwrapped one reaches Flask as an HTML 500 the
        # browser cannot read.  The file can vanish between `can_parse`
        # and here, and the codec raises on unparseable columns.
        from molbuilder.workingcopy_structure import StructureCodec
        try:
            structure = StructureCodec().load(p)
        except (OSError, ValueError, TypeError) as exc:
            raise ParseError(f"{p.name}: {exc}") from exc
        return build_structure_result(
            structure=structure,
            cell=None,
            parser_name=cls.name,
            source=p,
            source_format="pdb",
        )


__all__ = ["PdbFileParser"]
