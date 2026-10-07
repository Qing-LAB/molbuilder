"""SIESTA ``.XV`` (final-coordinates) FileParser.

:class:`StructureResult` carries ``cell`` as a first-class field, populated directly from the .XV's leading 3
rows (always Bohr per SIESTA convention; we convert to Å here).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from molbuilder.chemistry import symbol_for_z
from molbuilder.parse.base import FileParser
from molbuilder.parse.types import StructureResult
from molbuilder.structure import Structure

from ._helpers import build_structure_result


# 1 Bohr in Ångström, from the one place it is spelled.
from molbuilder.constants import BOHR_ANGSTROM as _ANGSTROM_PER_BOHR


class SiestaXVError(ValueError):
    """Raised when a ``.XV`` file can't be parsed (malformed shape,
    wrong atom count, unreadable Z mapping)."""


def _nonblank_lines(p: Path) -> List[str]:
    """The file's non-blank lines.  THE ONE READ every door below shares.

    ``utf-8-sig`` accepts an optional BOM.
    """
    return [ln for ln in p.read_text(encoding="utf-8-sig",
                                     errors="replace").splitlines()
            if ln.strip()]


def _cell_from(lines: List[str], name: str) -> np.ndarray:
    """The 3x3 cell in Å from the leading three rows.  STRICT: raises.

    The tolerant door is this one with its failures swallowed.  Writing it once and
    catching, rather than twice with different strictness, is what stops
    the two drifting.
    """
    cell_bohr = np.zeros((3, 3), dtype=float)
    for i in range(3):
        toks = lines[i].split()
        if len(toks) < 3:
            raise SiestaXVError(
                f"{name}: cell row {i+1} has {len(toks)} tokens; "
                f"expected at least 3."
            )
        try:
            cell_bohr[i] = [float(toks[0]), float(toks[1]), float(toks[2])]
        except ValueError as exc:
            raise SiestaXVError(
                f"{name}: cell row {i+1} has a non-numeric component."
            ) from exc
        # `float("nan")` and `float("inf")` PARSE, so the try above does not
        # catch them, and the `Structure` this matrix goes onto refuses a
        # non-finite cell with a bare `ValueError` its callers do not catch.
        if not np.all(np.isfinite(cell_bohr[i])):
            raise SiestaXVError(
                f"{name}: cell row {i+1} has a non-finite component "
                f"({' '.join(toks[:3])}); a lattice vector must be a finite "
                f"length."
            )
    return cell_bohr * _ANGSTROM_PER_BOHR


def _atoms_from(lines: List[str], name: str):
    """``(elements, positions_ang)``.  STRICT: raises SiestaXVError."""
    try:
        n_atoms = int(lines[3].strip().split()[0])
    except (ValueError, IndexError) as exc:
        raise SiestaXVError(
            f"{name}: line 4 must be an integer atom count; got "
            f"{lines[3]!r}"
        ) from exc
    if n_atoms <= 0:
        raise SiestaXVError(
            f"{name}: atom count must be > 0; got {n_atoms}."
        )
    atom_lines = lines[4:4 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise SiestaXVError(
            f"{name}: header declares {n_atoms} atoms but only "
            f"{len(atom_lines)} lines follow."
        )

    elements: List[str] = []
    positions_bohr = np.zeros((n_atoms, 3), dtype=float)
    for i, raw in enumerate(atom_lines):
        toks = raw.split()
        if len(toks) < 5:
            raise SiestaXVError(
                f"{name}: atom row {i+1} has {len(toks)} tokens; "
                f"expected at least 5 (ispec iza x y z [vx vy vz])."
            )
        try:
            iza = int(toks[1])
        except ValueError as exc:
            raise SiestaXVError(
                f"{name}: atom row {i+1} has non-integer Z {toks[1]!r}."
            ) from exc
        # MOLBUILDER'S OWN TABLE, not ase's.
        try:
            elements.append(symbol_for_z(iza))
        except ValueError as exc:
            raise SiestaXVError(
                f"{name}: atom row {i+1} has atomic number {iza} "
                f"outside the element table."
            ) from exc
        positions_bohr[i] = [
            float(toks[2]), float(toks[3]), float(toks[4]),
        ]
    return elements, positions_bohr * _ANGSTROM_PER_BOHR


def read_xv_with_cell(path: Union[str, Path]):
    """``(Structure, cell_ang)`` from ONE pass over the file.

    THE DOOR FOR CALLERS THAT WANT BOTH.  Strict, like `read_xv`: a malformed file raises.

    THE STRUCTURE CARRIES THE CELL, and the tuple's second element is the
    same matrix for the two callers that want it bare (the FileParser fills
    `StructureResult.cell`; `read_xv_cell` answers the matrix alone).  A
    file that states a lattice produces a structure that has one, so
    `Structure.__post_init__` applies ITS default (a stated cell means
    periodic on every axis) in the one place that owns that rule.
    """
    p = Path(path)
    lines = _nonblank_lines(p)
    if len(lines) < 4:
        raise SiestaXVError(
            f"{p.name}: file too short ({len(lines)} non-blank lines); "
            f"expected at least 3 cell rows + atom count + atoms."
        )
    cell = _cell_from(lines, p.name)
    elements, positions_ang = _atoms_from(lines, p.name)
    # A `.XV` IS SIESTA'S OWN FRAME, the cell at the origin, so the structure
    # it reads STATES that origin -- an offset of 0, set together with its
    # coordinates (`model/structure-periodicity.md` § 6.0): the next deck
    # applies nothing, and a viewer draws it as the engine had it.
    return Structure(elements=elements, positions=positions_ang,
                     cell=cell, title=p.stem,
                     engine_offset=[0.0, 0.0, 0.0]), cell


def _read_xv(path: Union[str, Path]) -> Structure:
    """Read a SIESTA ``.XV`` file and return a :class:`Structure`.

    File format (SIESTA manual, ``siesta/Src/iofa.f``)::

        ax1 ax2 ax3 vx1 vx2 vx3        (cell row 1, Bohr + Bohr/fs)
        bx1 bx2 bx3 vy1 vy2 vy3        (cell row 2)
        cx1 cx2 cx3 vz1 vz2 vz3        (cell row 3)
        N                              (number of atoms)
        ispec(1) iza(1) x(1) y(1) z(1) vx(1) vy(1) vz(1)
        ispec(2) iza(2) x(2) y(2) z(2) vx(2) vy(2) vz(2)
        ...

    ``ispec`` is the species index into the ``ChemicalSpeciesLabel``
    block; ``iza`` is the atomic number Z.  We key off Z because it
    survives across re-orderings of the species block; the species
    index requires a matching ``.fdf``.

    Positions come back in Å.  Velocities are discarded.  The returned
    Structure carries the cell and states the engine's origin (offset 0);
    :func:`_read_xv_cell` reads the cell alone for callers that need only
    it.
    """
    return read_xv_with_cell(path)[0]


def _read_xv_cell(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Read JUST the 3 cell-vector rows from a .XV file and return
    them in Å as a 3x3 ndarray.  Returns None on parse failure;
    the structure portion is :func:`_read_xv`'s responsibility (it
    will raise SiestaXVError on the same file).

    TOLERANT ON PURPOSE, and `validation/identity.py` is why: it asks for
    the cell ALONE, so a file whose cell rows are sound and whose atom
    list is corrupt must still answer a cell.  That is why this reads the
    first three rows and does not touch the atoms.

    Cell rows are in Bohr per SIESTA convention; we convert to Å here.
    Velocity columns (the trailing 3 floats) are discarded.
    """
    p = Path(path)   # accept str (public alias read_xv_cell)
    try:
        lines = _nonblank_lines(p)
        if len(lines) < 3:
            return None
        return _cell_from(lines, p.name)
    except (OSError, SiestaXVError):
        return None


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
        structure, cell = read_xv_with_cell(path)
        return build_structure_result(
            structure=structure,
            cell=cell,
            parser_name=cls.name,
            source=path,
            source_format="siesta-xv",
        )


# --------------------------------------------------------------------- #
#  Public-API re-exports                                                #
# --------------------------------------------------------------------- #
read_xv = _read_xv
read_xv_cell = _read_xv_cell


__all__ = [
    "SiestaXVError",
    "SiestaXVFileParser",
    "read_xv",
    "read_xv_cell",
    "read_xv_with_cell",
]
