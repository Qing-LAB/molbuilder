"""PDB (``.pdb``) FileParser — a geometry file somebody else's tool wrote.

WHY IT EXISTS.  `/api/results/dir` asks the registry *what reads this
file* and the Results picker drops anything the answer is ``None`` for,
so a format with a reader but no registration is a format the browser
cannot offer.

WHAT IT DOES NOT CLAIM.  A box: ``cell`` is the structure's, which a PDB
read never has.  A PDB states its
lattice in a ``CRYST1`` record and :meth:`Structure.from_pdb` reads
ATOM/HETATM only, so there is no cell to carry — and inventing one
from the coordinates is the failure `siesta_xv` exists to avoid.
"""

from __future__ import annotations

from pathlib import Path

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import StructureResult

from ._helpers import build_structure_result, read_through_the_codec

#: How far to look before giving up.  Generous, because the thing being
#: skipped is the HEADER and a PDB header has no small bound: measured
#: over the 305 `.pdb` in this tree, the first coordinate record sits
#: anywhere from byte 0 to byte 286011 (`1jj2.pdb`, a ribosome).
#:
#: The scan still stops at the first coordinate record, so a real PDB
#: costs its header and not its atoms: `1jj2.pdb` is 8.3 MB and answers
#: after 286 KB.  The cap only bounds the pathological case -- a large
#: file with a `.pdb` name and no coordinates anywhere.
_SNIFF_BYTES = 4 * 1024 * 1024


def _looks_like_pdb(path: Path) -> bool:
    """Does this file carry a coordinate record?

    THE CONTENT, NOT THE SUFFIX.  A ``.pdb`` that
    holds a refusal message, a truncated download or somebody's notes
    is not a structure, and offering it puts an error card where a
    molecule should be.

    Streamed, and it RETURNS AT THE FIRST HIT, so the cost is the
    header rather than the file; see :data:`_SNIFF_BYTES`.
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
        # THROUGH THE ONE DOOR, because a `.pdb` here is a PERSON'S
        # structure and not an engine's output: the codec reads it as every
        # door does -- alone, for a `.pdb` holds no sidecar (`model/
        # structure.md` § 2.4) -- never a second reader of its own; it reads
        # `utf-8-sig`, so a file saved with a BOM parses here as it does when
        # the sidebar opens it.
        structure = read_through_the_codec(path)
        return build_structure_result(
            structure=structure,
            cell=structure.cell,
            parser_name=cls.name,
            source=Path(path),
            source_format="pdb",
        )


__all__ = ["PdbFileParser"]
