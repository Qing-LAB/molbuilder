"""An XYZ with no sidecar that no other reader is for -- SIESTA's
``<label>.xyz`` and ``.ANI``, another tool's file -- read alone, one frame or
many (`model/parse.md` § 4).

**Each XYZ file is read by the reader that is for it, by its name**:
geomeTRIC's ``<stem>_geom_optim.xyz`` by ``pyscf``, with its run's files
beside it; a PySCF deck's ``<label>_initial.xyz`` and ``<label>_optimized.xyz``
by ``pyscf-geom``; a ``.xyz`` with its sidecar beside it by
``structure-pair`` (`parse/coords/pair.py`), the pair being the file; every
other XYZ by this one.  What makes a file an XYZ is
checked here once (:func:`is_xyz`), which XYZ no reader claims by its name
once (:func:`unclaimed_by_name`), and its frames are read here once
(:func:`read_frames`); the geomeTRIC reader reads with the check and the
frames, the pair's reader asks the same unclaimed-by-name question.

Frame format::

    {N}
    {comment -- free text}
    {El}  {x}  {y}  {z}
    ...

The comment line is free text (`model/structure.md` § 2.3), so this reader
takes nothing from it: each frame states its atoms and positions (Å), no
energy, and the file says nothing of how a run ended.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from molbuilder.frame import Frame, Trajectory
from molbuilder.parse.base import FileParser
from molbuilder.parse.errors import ParseError
from molbuilder.parse.types import TrajectoryResult
from molbuilder.runfiles import (PYSCF_OPTIMIZED, ROLE_GEOM_TRAJ, ROLE_INITIAL,
                                 stem_before)
from molbuilder.sidecars.molstruct import paired_sidecar
from molbuilder.structure import Structure

from ._helpers import wrap_trajectory


#: THE XYZ FILES ANOTHER READER IS FOR, by the role their name ends in
#: (`model/parse.md` § 4): geomeTRIC's trajectory (``pyscf``) and a PySCF
#: deck's two geometries (``pyscf-geom``).
NAMED_FOR_ANOTHER = (ROLE_GEOM_TRAJ, ROLE_INITIAL, PYSCF_OPTIMIZED)


def unclaimed_by_name(path) -> bool:
    """Is ``path`` an XYZ (:func:`is_xyz`) that no reader claims by its name
    (:data:`NAMED_FOR_ANOTHER`)?  Such a file is this reader's, or the
    pair's (`parse/coords/pair.py`) when its sidecar is beside it -- the
    one question both ask, so the two split every such file between them."""
    p = Path(path)
    return (not any(stem_before(p.name, role) is not None
                    for role in NAMED_FOR_ANOTHER)
            and is_xyz(str(p)))

# Bounded sanity-check on line-0 atom count when detecting an XYZ.
_MAX_PLAUSIBLE_ATOMS = 1_000_000


def is_xyz(path: str) -> bool:
    """Structural XYZ check, not banner-matching.  Verifies the
    canonical XYZ invariant: positive integer N on line 0, a free-text
    comment on line 1, and N atom lines of ``element x y z`` form.
    Samples at most the first 3 atom lines."""
    try:
        with open(path, "r", errors="replace") as fh:
            line0 = fh.readline().strip()
            if not line0.isdigit():
                return False
            n_atoms = int(line0)
            if n_atoms <= 0 or n_atoms > _MAX_PLAUSIBLE_ATOMS:
                return False
            fh.readline()  # comment line; any content accepted
            for _ in range(min(n_atoms, 3)):
                parts = fh.readline().split()
                if len(parts) < 4:
                    return False
                # COLUMN 0 IS AN ELEMENT: SIESTA's `.FA` is `N` then
                # `index fx fy fz` -- same count, same three floats, an
                # integer where the symbol goes.
                #
                # "NOT A NUMBER" rather than a periodic-table lookup, and
                # that is deliberate: a dummy atom (`X`), a ghost (`Bq`) or
                # an isotope label is a legal XYZ column 0 and belongs to
                # whoever writes it, while an integer index never is.
                # Measured over all 114 real `.xyz` in the tree: 114 carry a
                # symbol, 0 carry a number.
                try:
                    float(parts[0])
                except ValueError:
                    pass          # a symbol -- which is what an XYZ has
                else:
                    return False  # a number -- this is somebody's data file
                try:
                    float(parts[1])
                    float(parts[2])
                    float(parts[3])
                except ValueError:
                    return False
            return True
    except OSError:
        return False


def read_frames(path: str) -> List[Tuple[str, List[str], np.ndarray]]:
    """Each whole frame of the XYZ at ``path``, in order: ``(comment,
    symbols, positions)``, the positions an ``(N, 3)`` array in Å.  A last
    frame torn mid-write -- the file of a run still being written -- is
    dropped, as is anything after a line that is no atom count."""
    frames: List[Tuple[str, List[str], np.ndarray]] = []
    with open(path, "r", errors="replace") as fh:
        while True:
            header = fh.readline()
            if not header:
                break                     # clean EOF
            header = header.strip()
            if not header.isdigit():
                # Probably a torn write at EOF; bail out cleanly.
                break
            n_atoms = int(header)
            comment = fh.readline()
            if not comment:
                break                     # torn frame
            symbols: List[str] = []
            rows: List[List[float]] = []
            for _ in range(n_atoms):
                parts = fh.readline().split()
                if len(parts) < 4:
                    break
                try:
                    rows.append([float(parts[1]), float(parts[2]),
                                 float(parts[3])])
                except ValueError:
                    break
                symbols.append(parts[0])
            if len(rows) != n_atoms:
                # Last frame is mid-write; drop it.
                break
            frames.append((comment.strip(), symbols,
                           np.array(rows, dtype=float).reshape(n_atoms, 3)))
    return frames


class XyzFileParser(FileParser):
    """An XYZ with no sidecar that no other reader is for, read alone: one
    frame per block, its atoms and positions, as a
    :class:`TrajectoryResult`."""

    name   = "xyz"
    label  = "XYZ geometry, one frame or many"
    hint   = ("an XYZ file with no sidecar -- SIESTA's <label>.xyz or .ANI, "
              "another tool's: an atom count, a comment line, then a symbol "
              "and three coordinates (Å) per atom; read alone, its comment "
              "line free text")
    output = TrajectoryResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        # A PAIR IS ITS READER'S (`parse/coords/pair.py`): read here it would
        # lose what its sidecar states.
        return unclaimed_by_name(path) and paired_sidecar(path) is None

    @classmethod
    def parse(cls, path: Path) -> TrajectoryResult:
        # A PARSER RAISES `ParseError`: the file can vanish between
        # `can_parse` and here.
        try:
            frames = read_frames(str(path))
        except OSError as exc:
            raise ParseError(f"{Path(path).name}: {exc}") from exc
        traj = Trajectory(
            source_format=cls.name,
            frames=[Frame(structure=Structure(elements=symbols,
                                              positions=positions),
                          step_index=i)
                    for i, (_comment, symbols, positions)
                    in enumerate(frames)],
            lattice=None,
            # A FILE NO RUN NAMES says nothing of how a run ended.
            run_state="unknown")
        return wrap_trajectory(traj, cls.name, path)


__all__ = ["XyzFileParser", "is_xyz", "read_frames", "NAMED_FOR_ANOTHER",
           "unclaimed_by_name"]
