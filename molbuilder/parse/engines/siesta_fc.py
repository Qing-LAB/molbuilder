"""SIESTA's force-constant file, read into the block a vibration needs.

``MD.TypeOfRun FC`` nudges each atom of the ``FC.First``..``FC.Last`` range
by ``FC.Displacement`` along x, y and z, both ways, and writes
``<SystemLabel>.FC``: one header line, then -- for each displaced atom in
range order, each direction, each side (minus, then plus) -- one row per
atom of the structure holding the force-constant contribution
``-dF_b/dR_a`` in **eV/Å²**.  Measured on a two-atom run (2026-09-23,
SIESTA 5.4.2): the z-block's two sides average to 41.713, and two single
points displaced by hand give −ΔF/2Δ = 41.713 eV/Å² for the same
element.

``<SystemLabel>.FCC`` is the same file with the HELD atoms' force rows
zeroed (SIESTA's "constrained" variant).  The free block is identical, so
this reader takes ``.FC`` and the caller slices the free atoms -- the
partial Hessian of `science/normal-modes.md` § 3, the free-free block of
the forces computed with every atom present.

The file does not say WHICH atoms were displaced -- only how many.  The
caller knows: it is the range the deck wrote, the free atoms of the sorted
copy (`model/overview.md` § 2.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# TWO WAYS: the SIESTA vibration's finish reads the file beside the job
# (`runwrap.VIBRATION_COMPANIONS`), where the package is not installed.
try:                                        # inside molbuilder
    from ...constants import BOHR_ANGSTROM, HARTREE_EV
    from ..errors import ParseError
except ImportError:                         # beside a job, in mb_vibration.pyz
    from constants import BOHR_ANGSTROM, HARTREE_EV
    from errors import ParseError

#: eV/Å² -> Hartree/Bohr², from the two constants it is made of.
EV_PER_ANG2_TO_HARTREE_PER_BOHR2 = (BOHR_ANGSTROM ** 2) / HARTREE_EV


@dataclass(frozen=True)
class ForceConstantFile:
    """What a ``.FC`` file says, with its units left as written."""
    path: str
    n_atoms: int
    #: the nudge, in Å, as the header states it
    displacement_ang: float
    #: how many atoms were nudged -- the length of the FC range
    n_displaced: int
    #: ``k[p, i, side, b, j]``: displaced atom ``p`` (by position in the
    #: range), direction ``i``, side (0: minus, 1: plus), atom ``b``,
    #: force component ``j`` -- eV/Å²
    k: np.ndarray


def read_fc(path) -> ForceConstantFile:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].lower().startswith("force constants matrix"):
        raise ParseError(
            f"{p.name}: not a SIESTA force-constant file (the first line "
            f"should begin 'Force constants matrix')")
    head = lines[0].split(":", 1)
    if len(head) != 2:
        raise ParseError(f"{p.name}: the header carries no 'n_atoms, "
                         f"displacement' pair after the colon")
    fields = head[1].split()
    try:
        n_atoms = int(fields[0])
        displacement_ang = float(fields[1].replace("D", "E"))
    except (IndexError, ValueError) as exc:
        raise ParseError(f"{p.name}: cannot read n_atoms and the "
                         f"displacement from the header: {exc}") from exc
    try:
        rows = np.array([[float(x.replace("D", "E")) for x in ln.split()]
                         for ln in lines[1:]], dtype=float)
    except ValueError as exc:
        raise ParseError(f"{p.name}: a force row is not three numbers: "
                         f"{exc}") from exc
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ParseError(f"{p.name}: force rows must carry three "
                         f"components; got shape {rows.shape}")
    per_displaced = 6 * n_atoms
    if rows.shape[0] == 0 or rows.shape[0] % per_displaced:
        raise ParseError(
            f"{p.name}: {rows.shape[0]} force rows is not a multiple of "
            f"6 x {n_atoms} atoms -- the file is truncated or is not an FC "
            f"file for this structure")
    n_displaced = rows.shape[0] // per_displaced
    k = rows.reshape(n_displaced, 3, 2, n_atoms, 3)
    return ForceConstantFile(path=str(p), n_atoms=n_atoms,
                             displacement_ang=displacement_ang,
                             n_displaced=n_displaced, k=k)


def _two_sided_mean(fc: ForceConstantFile,
                    displaced: Sequence[int]) -> np.ndarray:
    """The central-difference table in the file's own eV/Å², shape
    ``(n_atoms, n_atoms, 3, 3)``, NOT yet symmetrised -- the one place the
    two sides of each nudge are averaged, so the Hessian and its asymmetry
    diagnostic read the same numbers."""
    displaced = [int(i) for i in displaced]
    if len(displaced) != fc.n_displaced:
        raise ParseError(
            f"{Path(fc.path).name}: the file holds {fc.n_displaced} "
            f"displaced atom(s) but the caller names {len(displaced)} -- "
            f"the FC range and the free set disagree")
    if any(not 0 <= a < fc.n_atoms for a in displaced):
        raise ParseError(f"{Path(fc.path).name}: a displaced-atom index is "
                         f"outside the {fc.n_atoms} atoms of the file")
    n = fc.n_atoms
    H = np.zeros((n, n, 3, 3))
    for p, a in enumerate(displaced):
        # k[p, i, side, b, j]: mean over the two sides -> H[a, b, i, j]
        H[a, :, :, :] = np.transpose(0.5 * (fc.k[p, :, 0] + fc.k[p, :, 1]),
                                     (1, 0, 2))
    return H


def hessian_from_fc(fc: ForceConstantFile,
                    displaced: Sequence[int]) -> np.ndarray:
    """The second-derivative table in Hartree/Bohr², shape
    ``(n_atoms, n_atoms, 3, 3)``, filled where the file says something.

    ``displaced`` names, in range order, which atoms the FC range covered
    (0-based, in the order of the structure the deck was written from).
    The two sides of each nudge are averaged -- a central difference --
    and the block over the displaced atoms is symmetrised.  Rows of atoms
    the file did not nudge stay zero and are never read: the one path
    (`spectra.normal_modes.vibrational_modes`) slices the free block.
    """
    H = _two_sided_mean(fc, displaced)
    ix = np.ix_([int(i) for i in displaced], [int(i) for i in displaced])
    block = H[ix]
    H[ix] = 0.5 * (block + np.transpose(block, (1, 0, 3, 2)))
    return H * EV_PER_ANG2_TO_HARTREE_PER_BOHR2


def fc_block_asymmetry(fc: ForceConstantFile,
                       displaced: Sequence[int]) -> float:
    """``max |H_ij - H_ji|`` over the displaced atoms' block BEFORE it is
    symmetrised, in the file's own eV/Å² -- the numerical diagnostic
    `science/normal-modes.md` § 4b.6 C names, recorded with the result
    (`engines/vibration.md` § 5.5).  A step too small leaves noise here, a
    step too large leaves anharmonicity.  On a block whose off-diagonals
    vanish by symmetry (H₂ on its axis) it says nothing; an atom at a
    low-symmetry site shows it from one free atom on.
    """
    H = _two_sided_mean(fc, displaced)
    ix = np.ix_([int(i) for i in displaced], [int(i) for i in displaced])
    block = H[ix]
    if block.size == 0:
        return 0.0
    return float(np.max(np.abs(block - np.transpose(block, (1, 0, 3, 2)))))


def reference_frame_of(reading, *, name: str = "the output"):
    """The force-constant run's FC step 0 -- the UNDISPLACED geometry with
    the forces SIESTA evaluated there -- out of SIESTA's reading pass over
    the run's output (`siesta_reader.SiestaReader.finish`): its steps carry
    every step's coordinates, as ``[label, x, y, z]`` rows in Å, and forces
    in eV/Å, so nothing here re-reads the file.  The same records the
    parser's frames are built from, read beside the job where the parser is
    not.

    SIESTA evaluates the reference geometry before the first nudge, so it is
    the first step carrying forces.  The finish takes both halves from it:
    the coordinates are the geometry the force constants belong to -- the
    relaxed one after a `relax` stage, the input one when the structure was
    stated relaxed -- and the forces are what stationarity is judged by (R5
    on this route; `engines/vibration.md` § 5.2a, § 5.5).  Returns the step
    record.
    """
    for step in reading.get("steps") or ():
        if step.get("forces") and step.get("coords"):
            return step
    raise ParseError(f"{name}: no force block for the reference geometry "
                     f"(FC step 0) -- the force-constant run has not written "
                     f"its first step")


__all__ = ["ForceConstantFile", "read_fc", "hessian_from_fc",
           "fc_block_asymmetry", "reference_frame_of",
           "EV_PER_ANG2_TO_HARTREE_PER_BOHR2"]
