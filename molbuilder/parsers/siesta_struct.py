"""Structure-side SIESTA readers: ``.XV`` (final converged coords)
and ``.fdf`` initial coordinates.

Used by :mod:`molbuilder.script_bundle` to recover the run-time
structure for handoff into the next workflow stage.  Separate from
:mod:`molbuilder.parsers.siesta` (which parses ``.out`` for the
trajectory inspector) because the consumers and the file shapes are
different:

* ``.XV`` is a compact, fixed-format dump of final cell + per-atom
  coords + velocities.  SIESTA writes it on every successful step.
* ``.fdf`` initial coords come from the ``AtomicCoordinatesAndAtomicSpecies``
  block + the ``ChemicalSpeciesLabel`` block + the
  ``AtomicCoordinatesFormat`` directive.  Read for fallback when
  ``.XV`` is missing (run died before first XV write).

Both return a :class:`molbuilder.structure.Structure`.  Positions are
always in Å regardless of the source's native unit (Bohr → Å done
here).  Cell vectors are computed internally for Fractional /
ScaledCartesian re-projection, but ``Structure`` is geometry-only
today (no ``lattice`` field) — they're not surfaced on the returned
object.

Bundle-contract reference: ``docs/protocols/bundle-contract.md § 4.1``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from molbuilder.structure import Structure


# 1 Bohr in Ångström.  Same value used elsewhere in the project
# (molbuilder/parsers/pyscf.py uses 0.5291772108).  Pinned here to
# avoid a cross-module import for one constant.
_ANGSTROM_PER_BOHR = 0.5291772108


# --------------------------------------------------------------------- #
#  .XV reader                                                           #
# --------------------------------------------------------------------- #


class SiestaXVError(ValueError):
    """Raised when a ``.XV`` file can't be parsed (malformed shape,
    wrong atom count, unreadable Z mapping)."""


def read_xv(path: Union[str, Path]) -> Structure:
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

    Positions come back in Å.  Velocities are discarded (bundle
    handoff is geometry-only; downstream re-render starts a fresh
    MD if needed).  Cell vectors are read but NOT surfaced on the
    returned Structure -- the dataclass is geometry-only today.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        raise SiestaXVError(
            f"{p.name}: file too short ({len(lines)} non-blank lines); "
            f"expected at least 3 cell rows + atom count + atoms."
        )

    # Cell: 3 rows × 6 floats; the trailing 3 are cell-velocity
    # (vc/dt) and we drop them.
    cell_bohr = np.zeros((3, 3), dtype=float)
    for i in range(3):
        toks = lines[i].split()
        if len(toks) < 3:
            raise SiestaXVError(
                f"{p.name}: cell row {i+1} has {len(toks)} tokens; "
                f"expected at least 3."
            )
        cell_bohr[i] = [float(toks[0]), float(toks[1]), float(toks[2])]
    cell_ang = cell_bohr * _ANGSTROM_PER_BOHR

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
    # Lazy import: ase isn't a build-time dep for every consumer of
    # script_bundle, but reading .XV needs Z → symbol mapping.  The
    # standard XYZ + PDB readers in structure.py already import
    # ase.data.atomic_numbers on demand; mirror the pattern.
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

    # Structure today doesn't carry cell vectors; we still convert
    # them here because Fractional / ScaledCartesian fdf-coords need
    # the cell to project to Å, but the returned Structure is
    # geometry-only.  ``cell_ang`` lands in caller's hands if they
    # need it (today no consumer in the bundle path uses it).
    del cell_ang  # silence lint for unused-after-conversion
    return Structure(
        elements=elements,
        positions=positions_bohr * _ANGSTROM_PER_BOHR,
        title=p.stem,
    )


# --------------------------------------------------------------------- #
#  .fdf initial-coords reader                                           #
# --------------------------------------------------------------------- #


class SiestaFdfStructureError(ValueError):
    """Raised when a ``.fdf`` lacks the blocks needed to reconstruct
    a :class:`Structure`."""


# Block matcher: tolerates any whitespace and any case of the
# ``%block`` / ``%endblock`` keywords.
def _block_re(name: str) -> re.Pattern[str]:
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
# single SystemLabel (``h2``).  Used by script_bundle to find .XV.
_SYSTEM_LABEL_RE = re.compile(
    r"^\s*SystemLabel\s+(\S+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_species_block(text: str) -> List[str]:
    """Return a list mapping 1-based species index to element symbol.

    The block carries rows of ``<species_idx> <Z> <label>``; we want
    the element symbol.  Z is authoritative because the label can be
    arbitrary (``Au_bulk``, ``H1``, etc.) and won't always be a clean
    element symbol.
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
        # 1-based -> ensure list is large enough.
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
    or ``None`` when no recognisable line is present.

    SIESTA writes ``<SystemLabel>.XV`` and other output files keyed
    on this label, NOT on the ``.fdf`` basename.  The bundle layer
    uses this to find the .XV file when the .fdf basename
    differs from the SystemLabel (stage-suffixed runs, hand-renamed
    files, etc.).
    """
    m = _SYSTEM_LABEL_RE.search(text)
    return m.group(1) if m else None


def read_fdf_initial_coords(text_or_path: Union[str, Path]) -> Structure:
    """Read the AtomicCoordinatesAndAtomicSpecies block of a ``.fdf``
    and return a :class:`Structure`.

    Used as a last-resort fallback when ``.XV`` is missing.  Reflects
    the structure as it would be at the START of the run -- NOT
    converged geometry.  Caller is responsible for recording this
    in ``RunBundle.final_coords_from = "fdf-initial"`` so audit
    consumers can tell.

    Handles units per ``AtomicCoordinatesFormat``:
      * ``Ang`` / ``NotScaledCartesianAng``           — Å (no conversion)
      * ``Bohr`` / ``NotScaledCartesianBohr``         — Å = Bohr × 0.5291772108
      * ``ScaledCartesian``                           — Å = value × LatticeConstant
      * ``Fractional`` / ``ScaledByLatticeVectors``   — Å = fractional · cell_vectors

    Reads cell vectors from ``LatticeVectors`` (+ optional
    ``LatticeConstant`` scale) but does NOT surface them on the
    returned Structure -- the dataclass is geometry-only today.
    The cell is used internally to project Fractional /
    ScaledCartesian coords back to Å.
    """
    # Type-based dispatch: Path -> read file; str -> treat as fdf
    # body text directly.  Older "if str-and-file-exists then read"
    # heuristic blew up on long body strings (os.stat raises ENAMETOOLONG).
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

    # Cell + lattice-constant.  Default LatticeConstant unit is Bohr
    # per the SIESTA manual.
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

    # Same caveat as read_xv: Structure today is geometry-only.
    # ``lattice_ang`` informed the Fractional/ScaledCartesian
    # projection above; we don't surface it on Structure.
    return Structure(
        elements=elements,
        positions=positions_ang,
    )


# --------------------------------------------------------------------- #
#  Chirality / handedness diagnostic                                    #
# --------------------------------------------------------------------- #
#
# SIESTA accepts a ``%block LatticeVectors`` with negative determinant
# (a left-handed cell).  When that is paired with ``AtomicCoordinatesFormat
# Fractional``, the fractional → Å projection silently MIRRORS the
# structure -- chirality flipped.  For chiral molecules this is the
# wrong physics.  These functions are diagnostic-only: they return a
# loud multi-line warning string when a left-handed cell is detected,
# else None.  The bundle layer appends the warning to ``RunBundle.notes``
# so the user sees it in the result panel.


def _left_handed_warning(source_label: str, det_value: float) -> str:
    """Return a multi-line warning string suitable for RunBundle.notes.

    Phrased loudly because a silent chirality flip on a chiral
    molecule is genuinely wrong physics; the user MUST notice.
    """
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
    first three rows form a negative-determinant cell.

    Cheap to call: reads only the cell rows of the file, doesn't
    parse atoms.  Returns ``None`` when the cell is right-handed,
    the file is unreadable, or the .XV is malformed (the caller's
    own read_xv will surface those classes properly).
    """
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
    ``%block LatticeVectors`` (scaled by ``LatticeConstant`` per
    spec) has negative determinant.

    Returns ``None`` when no cell block is present (Fractional
    coords would then fail in read_fdf_initial_coords anyway) or
    the cell is right-handed.  Tracks the same parse rules as
    read_fdf_initial_coords: LatticeConstant defaults to Bohr when
    the unit is omitted.
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
    # Apply LatticeConstant scale per the spec: positive scalar can't
    # flip the determinant sign, so the sign of det(lattice * scale)
    # equals the sign of det(lattice).  Skip the multiplication.
    det = float(np.linalg.det(lattice))
    if det < 0:
        return _left_handed_warning("LatticeVectors block (.fdf)", det)
    return None


__all__ = [
    "SiestaXVError",
    "SiestaFdfStructureError",
    "read_xv",
    "read_fdf_initial_coords",
    "extract_system_label",
    "check_xv_handedness",
    "check_fdf_handedness",
]
