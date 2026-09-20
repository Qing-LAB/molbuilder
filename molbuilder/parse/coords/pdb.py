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
from molbuilder.parse.types import StructureResult
from molbuilder.structure import Structure

from ._helpers import build_structure_result

#: How much of the file is enough to decide.  A PDB's records are
#: line-oriented and ``ATOM``/``HETATM`` start in column 1, so the
#: answer is at the top or the file is not one -- and `can_parse` runs
#: once per file in a directory, on a format that reaches hundreds of
#: megabytes for a solvated system.  Reading it whole to say "no" is
#: what makes a picker scan feel broken.
_SNIFF_BYTES = 8192


def _looks_like_pdb(path: Path) -> bool:
    """Does the head of this file carry a coordinate record?

    THE CONTENT, NOT THE SUFFIX -- the rule `sidecars/transport.py`
    states and the reason its predecessor was deleted.  A ``.pdb`` that
    holds a refusal message, a truncated download or somebody's notes
    is not a structure, and offering it puts an error card where a
    molecule should be.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(_SNIFF_BYTES)
    except OSError:
        return False
    for line in head.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            return True
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
        structure = Structure.from_pdb(
            p.read_text(encoding="utf-8", errors="replace"))
        return build_structure_result(
            structure=structure,
            cell=None,
            parser_name=cls.name,
            source=p,
            source_format="pdb",
        )


__all__ = ["PdbFileParser"]
