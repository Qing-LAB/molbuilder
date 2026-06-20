"""SIESTA ``.XV`` (final-coordinates) FileParser.

Phase E of parse-module.md migration: wraps
:func:`molbuilder.parsers.siesta_struct.read_xv` AND surfaces the
cell vectors that the legacy reader internally computes but
discards.

The legacy ``read_xv`` returns a :class:`Structure` (geometry-
only); cell vectors are read from the file but dropped per a
historical "Structure is geometry-only" choice.  The Phase 1
JobMonitor decoder hit this gap and had to read the cell
separately from the .fdf's ``%block LatticeVectors`` (with all
the LatticeConstant-unit-conversion bugs that came with it).

This parser closes the gap: :class:`StructureResult` carries
``cell`` as a first-class field, populated directly from the
.XV's leading 3 rows (always Bohr per SIESTA convention; we
convert to Å here).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import StructureResult
from molbuilder.parsers.siesta_struct import (
    SiestaXVError,
    read_xv as _legacy_read_xv,
)

from ._helpers import build_structure_result


# Source: CODATA 2018 + SIESTA manual § 7.3.5.  The exact value
# molbuilder uses elsewhere; cross-checked in
# tests/parse/test_round2_fixes.py.
_ANGSTROM_PER_BOHR = 0.529177249


def _read_xv_cell(path: Path) -> Optional[np.ndarray]:
    """Read JUST the 3 cell-vector rows from a .XV file and return
    them in Å as a 3x3 ndarray.  Returns None on parse failure;
    the structure portion is the legacy reader's responsibility
    (it will raise SiestaXVError on the same file).

    Cell rows are in Bohr per SIESTA convention; we convert to Å
    here.  Velocity columns (the trailing 3 floats) are discarded.
    """
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return None
    cell_bohr = np.zeros((3, 3), dtype=float)
    for i in range(3):
        toks = lines[i].split()
        if len(toks) < 3:
            return None
        try:
            cell_bohr[i] = [float(toks[0]), float(toks[1]), float(toks[2])]
        except ValueError:
            return None
    return cell_bohr * _ANGSTROM_PER_BOHR


class SiestaXVFileParser(FileParser):
    """Parse a SIESTA ``.XV`` final-coordinates file.

    Returns :class:`StructureResult` with both ``structure``
    (positions + elements in Å) and ``cell`` (3x3 Å) populated.
    """

    name   = "siesta-xv"
    label  = "SIESTA .XV final-coordinates"
    hint   = "files ending in .XV (capital X V)"
    output = StructureResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        # Case-sensitive: SIESTA emits .XV (capital), not .xv.
        # Avoid claiming things like .xml.
        return path.name.endswith(".XV") and path.is_file()

    @classmethod
    def parse(cls, path: Path) -> StructureResult:
        # Structure portion via the legacy reader (positions,
        # elements, atom count consistency checks).
        try:
            structure = _legacy_read_xv(path)
        except SiestaXVError:
            raise
        # Cell portion via the cell-only helper (legacy discards it).
        cell = _read_xv_cell(path)
        return build_structure_result(
            structure=structure,
            cell=cell,
            parser_name=cls.name,
            source=path,
            source_format="siesta-xv",
        )
