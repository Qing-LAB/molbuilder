"""SIESTA ``.XV`` (final-coordinates) FileParser.

Absorbed from the legacy ``molbuilder.parsers.siesta_struct``
(``SiestaXVError`` + ``read_xv``); that package was deleted 2026-06-21
and this is the only ``.XV`` reader (provenance:
`docs/archive/old_docs/protocols/parse-module.md` § 8).

The legacy ``read_xv`` returned a :class:`Structure` (geometry-only);
cell vectors were read from the file but dropped per a historical
"Structure is geometry-only" choice.  The Phase 1 JobMonitor decoder
hit this gap and had to read the cell separately from the .fdf's
``%block LatticeVectors`` (with all the LatticeConstant-unit-
conversion bugs that came with it).

This parser closes the gap: :class:`StructureResult` carries ``cell``
as a first-class field, populated directly from the .XV's leading 3
rows (always Bohr per SIESTA convention; we convert to Å here).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import StructureResult
from molbuilder.structure import Structure

from ._helpers import build_structure_result


# 1 Bohr in Ångström, from the one place it is spelled.  This module and
# `transport.preflight` both read `.XV` files and carried DIFFERENT
# values (0.5291772108 here, 0.529177 there), so the same file gave
# coordinates 4e-7 apart depending on which reader was asked.
from molbuilder.constants import BOHR_ANGSTROM as _ANGSTROM_PER_BOHR


class SiestaXVError(ValueError):
    """Raised when a ``.XV`` file can't be parsed (malformed shape,
    wrong atom count, unreadable Z mapping)."""


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

    Positions come back in Å.  Velocities are discarded.  Cell vectors
    are read but NOT surfaced on the returned Structure (geometry-only
    dataclass); :func:`_read_xv_cell` exposes the cell for callers
    that need it.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        raise SiestaXVError(
            f"{p.name}: file too short ({len(lines)} non-blank lines); "
            f"expected at least 3 cell rows + atom count + atoms."
        )

    # Cell: 3 rows × 6 floats; the trailing 3 are cell-velocity (vc/dt).
    cell_bohr = np.zeros((3, 3), dtype=float)
    for i in range(3):
        toks = lines[i].split()
        if len(toks) < 3:
            raise SiestaXVError(
                f"{p.name}: cell row {i+1} has {len(toks)} tokens; "
                f"expected at least 3."
            )
        cell_bohr[i] = [float(toks[0]), float(toks[1]), float(toks[2])]
    # Cell is read for the helper :func:`_read_xv_cell`; the returned
    # Structure stays geometry-only.

    try:
        n_atoms = int(lines[3].strip().split()[0])
    except (ValueError, IndexError) as exc:
        raise SiestaXVError(
            f"{p.name}: line 4 must be an integer atom count; got "
            f"{lines[3]!r}"
        ) from exc
    if n_atoms <= 0:
        raise SiestaXVError(
            f"{p.name}: atom count must be > 0; got {n_atoms}."
        )

    atom_lines = lines[4:4 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise SiestaXVError(
            f"{p.name}: header declares {n_atoms} atoms but only "
            f"{len(atom_lines)} lines follow."
        )

    elements: List[str] = []
    positions_bohr = np.zeros((n_atoms, 3), dtype=float)
    # Lazy import: ase isn't a build-time dep for every consumer.
    try:
        from ase.data import chemical_symbols as _SYMBOLS
    except ImportError as exc:                                  # pragma: no cover
        raise SiestaXVError(
            "ase is required to map .XV atomic numbers to element "
            "symbols.  Install ase or read .XV via molbuilder's own "
            "tools."
        ) from exc

    for i, raw in enumerate(atom_lines):
        toks = raw.split()
        if len(toks) < 5:
            raise SiestaXVError(
                f"{p.name}: atom row {i+1} has {len(toks)} tokens; "
                f"expected at least 5 (ispec iza x y z [vx vy vz])."
            )
        try:
            iza = int(toks[1])
        except ValueError as exc:
            raise SiestaXVError(
                f"{p.name}: atom row {i+1} has non-integer Z {toks[1]!r}."
            ) from exc
        if iza <= 0 or iza >= len(_SYMBOLS):
            raise SiestaXVError(
                f"{p.name}: atom row {i+1} has atomic number {iza} "
                f"outside the element table (1..{len(_SYMBOLS)-1})."
            )
        elements.append(_SYMBOLS[iza])
        positions_bohr[i] = [
            float(toks[2]), float(toks[3]), float(toks[4]),
        ]

    return Structure(
        elements=elements,
        positions=positions_bohr * _ANGSTROM_PER_BOHR,
        title=p.stem,
    )


def _read_xv_cell(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Read JUST the 3 cell-vector rows from a .XV file and return
    them in Å as a 3x3 ndarray.  Returns None on parse failure;
    the structure portion is :func:`_read_xv`'s responsibility (it
    will raise SiestaXVError on the same file).

    Cell rows are in Bohr per SIESTA convention; we convert to Å here.
    Velocity columns (the trailing 3 floats) are discarded.
    """
    path = Path(path)   # accept str (public alias read_xv_cell)
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
        structure = _read_xv(path)
        cell = _read_xv_cell(path)
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
#
# H4a (test-cleanup): consumers reach for the publicly-named helpers
# from the coords module by file-type association.  The richer .fdf
# parsers live in ``parse/dirs/_assembler_helpers`` (the assembler
# composes them); re-export here so tests + future callers have one
# import path per file type.
read_xv = _read_xv
read_xv_cell = _read_xv_cell


def xv_to_xyz(xv_path: Union[str, Path],
              xyz_path: Optional[Union[str, Path]] = None) -> str:
    """Translate a SIESTA ``.XV`` into extended-XYZ text (written to
    ``xyz_path`` when given), **preserving the periodic cell**.

    A SIESTA ``.XV`` carries the lattice; a plain ``.xyz`` would drop it
    and a downstream generator would invent a vacuum cell -- wrong for a
    periodic junction.  So the cell is emitted on the comment line as the
    ASE extended-XYZ ``Lattice="..."`` header (row-major, Å), which
    ``molbuilder.siesta.convert`` (via ASE) round-trips back into the FDF
    cell.  Coordinates come from :func:`read_xv` (Å), the cell from
    :func:`read_xv_cell` (Å).

    Returns the extended-XYZ text.  This is the convenient ``.XV`` data-
    extraction entry (also exposed as the ``molbuilder xv2xyz`` CLI).
    """
    xv_path = Path(xv_path)
    struct = _read_xv(xv_path)
    cell = _read_xv_cell(xv_path)
    if cell is not None:
        flat = " ".join(f"{v:.8f}"
                        for v in np.asarray(cell, dtype=float).reshape(-1))
        comment = f'Lattice="{flat}" Properties=species:S:1:pos:R:3'
    else:
        comment = struct.title or xv_path.stem
    return struct.to_xyz(str(xyz_path) if xyz_path is not None else None,
                         comment=comment)


from molbuilder.parse.dirs._assembler_helpers import (   # noqa: E402
    SiestaFdfStructureError,
    check_fdf_handedness,
    check_xv_handedness,
    extract_system_label,
    read_fdf_initial_coords,
)


__all__ = [
    "SiestaXVError",
    "SiestaXVFileParser",
    "SiestaFdfStructureError",
    "read_xv",
    "read_fdf_initial_coords",
    "extract_system_label",
    "check_xv_handedness",
    "check_fdf_handedness",
]
