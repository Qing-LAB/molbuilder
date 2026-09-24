"""SIESTA's force-constant file, read into the block a vibration needs.

``MD.TypeOfRun FC`` nudges each atom of the ``FC.First``..``FC.Last`` range
by ``FC.Displacement`` along x, y and z, both ways, and writes
``<SystemLabel>.FC``: one header line, then -- for each displaced atom in
range order, each direction, each side (minus, then plus) -- one row per
atom of the structure holding the force-constant contribution
``-dF_b/dR_a`` in **eV/Å²**.  Measured on a two-atom run
(``tests/fixtures/siesta_fc``): the z-block's two sides average to
41.713, and two single points displaced by hand give −ΔF/2Δ = 41.713
eV/Å² for the same element.

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

from ...constants import BOHR_ANGSTROM, HARTREE_EV
from ..errors import ParseError

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
    ix = np.ix_(displaced, displaced)
    block = H[ix]
    H[ix] = 0.5 * (block + np.transpose(block, (1, 0, 3, 2)))
    return H * EV_PER_ANG2_TO_HARTREE_PER_BOHR2


__all__ = ["ForceConstantFile", "read_fc", "hessian_from_fc",
           "EV_PER_ANG2_TO_HARTREE_PER_BOHR2"]
