"""Private helpers for the bundle + job DirParsers.

H1 of parse-module.md migration: absorbed from
``molbuilder.parsers.{siesta_struct,pyscf_struct}`` (the read-side
helpers the legacy ``script_bundle.assemble_from_run_dir`` consumed
beyond just ``read_xv`` / ``read_optimized_xyz``).  The legacy
modules stay in place until H4; consumers that still use them keep
working.

What lives here
---------------

SIESTA side:
  * :exc:`SiestaFdfStructureError`
  * :func:`read_fdf_initial_coords` — .fdf -> Structure (initial)
  * :func:`extract_system_label`     — read SystemLabel directive
  * :func:`check_xv_handedness`      — left-handed-cell diagnostic
  * :func:`check_fdf_handedness`     — same, for .fdf LatticeVectors

PySCF side:
  * :exc:`PyscfStructureError`
  * :func:`read_py_initial_coords`   — .py -> Structure (initial)
  * :func:`extract_pyscf_job`        — JOB literal from a .py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from molbuilder.structure import Structure


# 1 Bohr in Ångström.  Same value the absorbed .XV reader uses
# (``parse/coords/siesta_xv.py:_ANGSTROM_PER_BOHR``); pinned here to
# avoid a cross-module constant import.
_ANGSTROM_PER_BOHR = 0.5291772108


# --------------------------------------------------------------------- #
#  SIESTA .fdf -> Structure                                              #
# --------------------------------------------------------------------- #


class SiestaFdfStructureError(ValueError):
    """Raised when a ``.fdf`` lacks the blocks needed to reconstruct
    a :class:`Structure`."""


def _block_re(name: str) -> re.Pattern[str]:
    """Build a ``%block <name> ... %endblock <name>`` regex.  Tolerant
    of any whitespace and any case of the ``%block`` / ``%endblock``
    keywords."""
    return re.compile(
        rf"^\s*%block\s+{re.escape(name)}\s*$"
        rf"([\s\S]*?)"
        rf"^\s*%endblock\s+{re.escape(name)}\s*$",
        re.IGNORECASE | re.MULTILINE,
    )


_BLOCK_SPECIES = _block_re("ChemicalSpeciesLabel")
_BLOCK_COORDS  = _block_re("AtomicCoordinatesAndAtomicSpecies")
_BLOCK_CELL    = _block_re("LatticeVectors")
# LatticeConstant: per the SIESTA manual the unit is OPTIONAL and
# defaults to Bohr when omitted.  Make the unit group optional so
# bare ``LatticeConstant 5.0`` parses; the per-unit branch below
# applies the Bohr default when group(2) is None.
_LATTICE_CONSTANT_RE = re.compile(
    r"^\s*LatticeConstant\s+([0-9.eE+\-]+)(?:\s+(\w+))?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_COORD_FMT_RE = re.compile(
    r"^\s*AtomicCoordinatesFormat\s+(\w+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# SystemLabel directive: SIESTA writes ``<SystemLabel>.XV`` (and other
# output files), which is NOT the same as the .fdf basename when the
# generator emits stage-suffixed filenames (``h2-stage2.fdf``) over a
# single SystemLabel (``h2``).
_SYSTEM_LABEL_RE = re.compile(
    r"^\s*SystemLabel\s+(\S+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_species_block(text: str) -> List[str]:
    """Return a list mapping 1-based species index to element symbol.

    The block carries rows of ``<species_idx> <Z> <label>``; we key on
    Z (authoritative across re-orderings of the species block) rather
    than label (which may be an arbitrary string like ``Au_bulk``).
    """
    try:
        from ase.data import chemical_symbols as _SYMBOLS
    except ImportError as exc:                                  # pragma: no cover
        raise SiestaFdfStructureError(
            "ase is required to map ChemicalSpeciesLabel Z values to "
            "element symbols."
        ) from exc
    m = _BLOCK_SPECIES.search(text)
    if m is None:
        raise SiestaFdfStructureError(
            "no ChemicalSpeciesLabel block in .fdf; can't map species "
            "indices to element symbols."
        )
    species: List[Optional[str]] = []
    for raw in m.group(1).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) < 2:
            continue
        try:
            idx = int(toks[0])
            z   = int(toks[1])
        except ValueError:
            continue
        if z <= 0 or z >= len(_SYMBOLS):
            raise SiestaFdfStructureError(
                f"ChemicalSpeciesLabel row {raw!r}: Z={z} is outside "
                f"the element table."
            )
        while len(species) < idx:
            species.append(None)
        species[idx - 1] = _SYMBOLS[z]
    if not species:
        raise SiestaFdfStructureError(
            "ChemicalSpeciesLabel block is empty."
        )
    if any(s is None for s in species):
        missing = [i+1 for i, s in enumerate(species) if s is None]
        raise SiestaFdfStructureError(
            f"ChemicalSpeciesLabel has gaps at species indices {missing}; "
            f"can't assign symbols."
        )
    return species  # type: ignore[return-value]


def extract_system_label(text: str) -> Optional[str]:
    """Return the ``SystemLabel`` directive value from a ``.fdf`` body,
    or ``None`` when no recognisable line is present."""
    m = _SYSTEM_LABEL_RE.search(text)
    return m.group(1) if m else None


def read_fdf_initial_coords(text_or_path: Union[str, Path]) -> Structure:
    """Read the AtomicCoordinatesAndAtomicSpecies block of a ``.fdf``
    and return a :class:`Structure`.

    Used as a last-resort fallback when ``.XV`` is missing.  Reflects
    the structure as it would be at the START of the run -- NOT
    converged geometry.

    Handles units per ``AtomicCoordinatesFormat``:
      * ``Ang`` / ``NotScaledCartesianAng``           — Å
      * ``Bohr`` / ``NotScaledCartesianBohr``         — Å = Bohr × 0.5291772108
      * ``ScaledCartesian``                           — Å = value × LatticeConstant
      * ``Fractional`` / ``ScaledByLatticeVectors``   — Å = fractional · cell_vectors

    Reads cell vectors from ``LatticeVectors`` (+ optional
    ``LatticeConstant`` scale) for the Fractional / ScaledCartesian
    projection but does not surface them on the returned Structure
    (the dataclass is geometry-only today).
    """
    # Type-based dispatch: Path -> read file; str -> treat as fdf
    # body text directly.  Older "if str-and-file-exists then read"
    # heuristic blew up on long body strings (os.stat ENAMETOOLONG).
    if isinstance(text_or_path, Path):
        text = text_or_path.read_text(encoding="utf-8", errors="replace")
    else:
        text = str(text_or_path)

    species = _parse_species_block(text)
    coords_m = _BLOCK_COORDS.search(text)
    if coords_m is None:
        raise SiestaFdfStructureError(
            "no AtomicCoordinatesAndAtomicSpecies block in .fdf."
        )

    # Cell + lattice-constant.  Default LatticeConstant unit is Bohr.
    lattice_ang: Optional[np.ndarray] = None
    cell_block = _BLOCK_CELL.search(text)
    if cell_block:
        rows: List[List[float]] = []
        for raw in cell_block.group(1).splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) < 3:
                continue
            rows.append([float(toks[0]), float(toks[1]), float(toks[2])])
        if len(rows) == 3:
            lattice_ang = np.asarray(rows, dtype=float)
    lc_m = _LATTICE_CONSTANT_RE.search(text)
    if lc_m:
        lc_val  = float(lc_m.group(1))
        # Unit is optional in the .fdf grammar; per the SIESTA manual
        # the default is Bohr when omitted.
        lc_unit = (lc_m.group(2) or "bohr").lower()
        if lc_unit.startswith("ang"):
            lc_ang = lc_val
        elif lc_unit.startswith("bohr"):
            lc_ang = lc_val * _ANGSTROM_PER_BOHR
        else:
            raise SiestaFdfStructureError(
                f"unsupported LatticeConstant unit {lc_unit!r}; "
                f"expected Ang or Bohr."
            )
        if lattice_ang is not None:
            lattice_ang = lattice_ang * lc_ang

    # Coordinate format.  Default per SIESTA manual is "Bohr".
    fmt_m = _COORD_FMT_RE.search(text)
    fmt = fmt_m.group(1).lower() if fmt_m else "bohr"

    elements: List[str] = []
    raw_xyz: List[List[float]] = []
    for raw in coords_m.group(1).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) < 4:
            continue
        try:
            x = float(toks[0])
            y = float(toks[1])
            z = float(toks[2])
            ispec = int(toks[3])
        except ValueError:
            continue
        if ispec < 1 or ispec > len(species):
            raise SiestaFdfStructureError(
                f"coordinate row references species index {ispec}, "
                f"but ChemicalSpeciesLabel has {len(species)} entries."
            )
        elements.append(species[ispec - 1])
        raw_xyz.append([x, y, z])

    if not elements:
        raise SiestaFdfStructureError(
            "AtomicCoordinatesAndAtomicSpecies block has no atoms."
        )
    coords = np.asarray(raw_xyz, dtype=float)

    if fmt in ("ang", "notscaledcartesianang"):
        positions_ang = coords
    elif fmt in ("bohr", "notscaledcartesianbohr"):
        positions_ang = coords * _ANGSTROM_PER_BOHR
    elif fmt == "scaledcartesian":
        if lc_m is None:
            raise SiestaFdfStructureError(
                "AtomicCoordinatesFormat ScaledCartesian needs a "
                "LatticeConstant directive."
            )
        positions_ang = coords * lc_ang   # lc_ang from above
    elif fmt in ("fractional", "scaledbylatticevectors"):
        if lattice_ang is None:
            raise SiestaFdfStructureError(
                "AtomicCoordinatesFormat Fractional needs a "
                "LatticeVectors block."
            )
        positions_ang = coords @ lattice_ang
    else:
        raise SiestaFdfStructureError(
            f"unsupported AtomicCoordinatesFormat {fmt!r}; supported: "
            f"Ang, Bohr, ScaledCartesian, Fractional."
        )

    return Structure(
        elements=elements,
        positions=positions_ang,
    )


# --------------------------------------------------------------------- #
#  Chirality / handedness diagnostics                                    #
# --------------------------------------------------------------------- #


def _left_handed_warning(source_label: str, det_value: float) -> str:
    """Loud multi-line warning string for ``RunBundle.notes`` /
    ``BundleResult.notes``.  Phrased loudly because a silent chirality
    flip on a chiral molecule is genuinely wrong physics."""
    return (
        f"WARNING: LEFT-HANDED CELL DETECTED in {source_label} "
        f"(det = {det_value:.4g}).  "
        f"This silently MIRRORS the structure (chirality flip).  "
        f"For chiral molecules this is wrong physics.  "
        f"Verify the LatticeVectors block; if the inversion is "
        f"intentional, ignore."
    )


def check_xv_handedness(path: Union[str, Path]) -> Optional[str]:
    """Return a left-handed-cell warning when ``path`` is a `.XV` whose
    first three rows form a negative-determinant cell.  ``None`` when
    the cell is right-handed, the file is unreadable, or the .XV is
    malformed (the caller's own ``read_xv`` will surface those)."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return None
    rows: List[List[float]] = []
    for i in range(3):
        toks = lines[i].split()
        if len(toks) < 3:
            return None
        try:
            rows.append([float(toks[0]), float(toks[1]), float(toks[2])])
        except ValueError:
            return None
    det = float(np.linalg.det(np.asarray(rows, dtype=float)))
    if det < 0:
        return _left_handed_warning(p.name, det)
    return None


def check_fdf_handedness(text: str) -> Optional[str]:
    """Return a left-handed-cell warning when ``text``'s
    ``%block LatticeVectors`` has negative determinant.  ``None`` when
    no cell block is present or the cell is right-handed.

    The ``LatticeConstant`` scale is a positive scalar that can't flip
    the determinant sign, so the sign of ``det(lattice * scale)``
    equals the sign of ``det(lattice)``.  We skip the multiplication.
    """
    cell_block = _BLOCK_CELL.search(text)
    if cell_block is None:
        return None
    rows: List[List[float]] = []
    for raw in cell_block.group(1).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) < 3:
            continue
        try:
            rows.append([float(toks[0]), float(toks[1]), float(toks[2])])
        except ValueError:
            continue
    if len(rows) != 3:
        return None
    lattice = np.asarray(rows, dtype=float)
    det = float(np.linalg.det(lattice))
    if det < 0:
        return _left_handed_warning("LatticeVectors block (.fdf)", det)
    return None


# --------------------------------------------------------------------- #
#  PySCF .py -> Structure                                                #
# --------------------------------------------------------------------- #


class PyscfStructureError(ValueError):
    """Raised when a ``.py`` lacks the gto.M atom block, or the block
    can't be parsed back to a Structure."""


# Match the canonical line the generator emits: ``JOB = "<label>"``.
# Accepts single or double quotes.
_JOB_RE = re.compile(
    r"""^\s*JOB\s*=\s*(['"])(?P<label>[^'"]+)\1\s*$""",
    re.MULTILINE,
)


def extract_pyscf_job(text: str) -> Optional[str]:
    """Return the ``JOB`` literal from a molbuilder-generated PySCF
    script, or ``None`` when no recognisable ``JOB = "..."`` line is
    present.  Used to derive ``<JOB>_optimized.xyz`` as the preferred
    final-coords source."""
    m = _JOB_RE.search(text)
    return m.group("label") if m else None


# Match the molbuilder atom-block emit format (input.py §§ 381-384):
#     mol = gto.M(
#         atom = '''
#         <El>  x  y  z
#         ...
#         ''',
#         ...
#     )
_ATOM_BLOCK_RE = re.compile(
    r"atom\s*=\s*'''(?P<block>.*?)'''",
    re.DOTALL,
)


def read_py_initial_coords(text_or_path: Union[str, Path]) -> Structure:
    """Parse the ``atom = '''…'''`` block of a molbuilder-generated
    PySCF script and return a :class:`Structure`.

    Used as a last-resort fallback when ``<JOB>_optimized.xyz`` is
    missing.  Coords come back in Å (the generator emits Å per
    ``unit = 'Ang'`` in the gto.M call).

    Only the molbuilder-generated emit format is recognised
    (whitespace-delimited rows of ``element x y z``).  Hand-written
    PySCF scripts using list-of-tuple or other formats will raise
    :class:`PyscfStructureError` and the bundle layer surfaces it as
    an actionable error.
    """
    if isinstance(text_or_path, Path):
        text = text_or_path.read_text(encoding="utf-8", errors="replace")
    else:
        text = str(text_or_path)

    m = _ATOM_BLOCK_RE.search(text)
    if m is None:
        raise PyscfStructureError(
            "no triple-quoted ``atom = '''…'''`` block in the .py.  "
            "Only molbuilder-generated PySCF scripts are supported "
            "for bundle assembly; hand-written scripts using list-of-"
            "tuple atom specs need to be re-rendered through "
            "molbuilder first."
        )
    elements: List[str] = []
    positions: List[List[float]] = []
    for raw in m.group("block").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) < 4:
            continue
        try:
            x = float(toks[1])
            y = float(toks[2])
            z = float(toks[3])
        except ValueError:
            continue
        # Canonicalise the element symbol the same way Structure.from_xyz
        # does: "FE" or "fe" become "Fe" so downstream consumers
        # (validation, ase) all see the canonical form.
        elements.append(toks[0].capitalize())
        positions.append([x, y, z])

    if not elements:
        raise PyscfStructureError(
            "the atom block in the .py is empty after parsing; the "
            "rows may not be whitespace-delimited (`El x y z`) as "
            "the molbuilder generator emits them."
        )
    return Structure(
        elements=elements,
        positions=np.asarray(positions, dtype=float),
    )


__all__ = [
    # SIESTA
    "SiestaFdfStructureError",
    "extract_system_label",
    "read_fdf_initial_coords",
    "check_xv_handedness",
    "check_fdf_handedness",
    # PySCF
    "PyscfStructureError",
    "extract_pyscf_job",
    "read_py_initial_coords",
]
