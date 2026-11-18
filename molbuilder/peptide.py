"""Peptide builder.

Wraps the well-tested :mod:`PeptideBuilder` library to assemble a fully
extended chain from sequence, then converts the Bio.PDB output into our
:class:`Structure` dataclass.

Modified residues (phosphoSer, methylLys, ...) are handled in two
phases:
    1. Build the peptide using the parent standard residue (Ser, Lys).
    2. Add the modification's heavy atoms to that residue, each placed in
       the residue's local frame by PeptideBuilder's own placement
       (``calculateCoordinates``: a bond, an angle and a dihedral from three
       atoms already there), as :data:`residues.MODIFIED_RESIDUES` states.

This keeps the heavy lifting (bond geometry, Ramachandran-valid
backbone) inside PeptideBuilder -- the tip's atoms placed by its function
too -- and only the chemistry of the tip in our own table.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import chemistry
from .residues import (
    MODIFIED_RESIDUES,
    AA_THREE_TO_ONE,
    parse_peptide_sequence,
)
from .structure import Structure


#: PEPTIDEBUILDER'S OWN GEOMETRY, corrected where it is wrong -- by residue
#: letter, the kit's ``Geo`` attribute and the value it is built with.
#: Lysine's chain end: the kit places CD-CE 1.46 A, CE-NZ 1.33 A at
#: CD-CE-NZ 124.8 degrees -- a C=N double bond's length and angle, so the
#: hydrogens are placed for an imine and every lysine is built two
#: hydrogens short (measured 2026-11-18); AmberTools' own LYS residue
#: (its force field's ``amino12.lib``, measured 2026-11-18) holds CD-CE
#: 1.525 A, CE-NZ 1.470 A at 109.5 degrees, the sp3 amine the residue is.
_GEOMETRY_CORRECTED = {
    "K": {"CD_CE_length": 1.525, "CE_NZ_length": 1.470,
          "CD_CE_NZ_angle": 109.5},
}


def _geometry(letter: str):
    """The kit's geometry of the residue ``letter`` -- its own door
    (``Geometry.geometry``) -- with :data:`_GEOMETRY_CORRECTED`'s values."""
    from PeptideBuilder.Geometry import geometry
    geo = geometry(letter)
    for name, value in _GEOMETRY_CORRECTED.get(letter, {}).items():
        setattr(geo, name, value)
    return geo


def build_peptide(
    sequence: str,
    *,
    title: Optional[str] = None,
    add_hydrogens: bool = True,
) -> Structure:
    """Build a fully-extended polypeptide from a sequence string.

    Parameters
    ----------
    sequence
        Sequence in **1-letter codes** (``"ARNDC"``, case-insensitive;
        whitespace ignored), with **``[XXX]`` brackets** for a modified or
        non-standard 3/4-letter residue, e.g. ``"AR[SEP]C"`` for
        Ala-Arg-phosphoSer-Cys.  Dashes and parentheses are rejected --
        the bracket grammar (``residues._parse``) keeps parsing
        unambiguous.  See :mod:`molbuilder.residues` for the modified codes.
    title
        Optional title written into the XYZ comment / PDB TITLE line.
    add_hydrogens
        If True (default), add explicit hydrogens via OpenBabel or RDKit
        (whichever is installed first).  Required for any quantum-
        chemistry calculation.  If neither is installed and this is True,
        :class:`~molbuilder.chemistry.BackendUnavailable` is raised,
        naming both.

    Returns
    -------
    Structure
        All-atom structure with PDB-style metadata (atom names, residue
        ids/names) filled in.
    """
    # PeptideBuilder declares Bio.PDB as a dependency and imports it at
    # load, so the one import is the probe for both.
    PeptideBuilder = chemistry.require("peptidebuilder",
                                       needed_for="Building a peptide")

    codes = parse_peptide_sequence(sequence)

    # Track which residues are modified, but build with the standard parent
    # so PeptideBuilder is happy.
    parent_codes: List[str] = []
    patches: Dict[int, str] = {}    # 1-based residue id -> modified code
    for i, code in enumerate(codes, start=1):
        if code in MODIFIED_RESIDUES:
            parent_codes.append(MODIFIED_RESIDUES[code]["parent"])
            patches[i] = code
        else:
            parent_codes.append(code)

    # PeptideBuilder API: 1-letter codes
    one_letter = "".join(AA_THREE_TO_ONE[c] for c in parent_codes)

    # Built extended (the kit's default phi/psi), through the kit's own door
    # from each residue's geometry -- `make_extended_structure` is this door
    # over the kit's defaults -- the wrong ones corrected (`_geometry`).
    bio_struct = PeptideBuilder.make_structure_from_geos(
        [_geometry(letter) for letter in one_letter])

    # THE MODIFICATIONS, on the kit's own residues before conversion: each
    # added atom placed in its residue's local frame (`_patch_residue`).
    model = next(iter(bio_struct))          # the kit writes one model
    chain = next(model.get_chains())
    for rid, code in patches.items():
        _patch_residue(chain[rid], code)

    # THE KIT'S CHAIN AS A STRUCTURE, by the one conversion of a built
    # molecule (`chemistry.molecules_to_structure`): each atom its PDB
    # record -- a modified residue by its own code -- one molecule.  The
    # element field read stripped (BioPython can return " C") and in the
    # periodic table's case.
    atoms = [(res, atom) for chain in model for res in chain
             for atom in res]
    struct = chemistry.molecules_to_structure(
        [atom.element.strip().capitalize() for _res, atom in atoms],
        [[float(c) for c in atom.coord] for _res, atom in atoms],
        [list(range(len(atoms)))],
        pdb=[(atom.get_name(), res.get_id()[1], codes[res.get_id()[1] - 1],
              res.get_parent().id or "A") for res, atom in atoms],
        title=title or f"peptide {sequence}")

    if add_hydrogens:
        struct = chemistry.add_hydrogens(struct)

    # ITS C-TERMINUS AS THE KIT WROTE IT -- an aldehyde, no OXT; the check
    # after the build says so (`validation.builders.check_built`).
    return struct


# ---------------------------------------------------------------------- #
#  Modified-residue side-chain patching                                  #
# ---------------------------------------------------------------------- #


def _patch_residue(residue, modified_code: str) -> None:
    """Add ``modified_code``'s heavy atoms to ``residue``, the kit's
    Bio.PDB residue of its parent: each placed in the residue's local frame
    by PeptideBuilder's own placement -- bonded to the third atom it names,
    at the bond length, the angle and the dihedral its row states
    (:data:`residues.MODIFIED_RESIDUES`; `engines/builders.md` § 2)."""
    from Bio.PDB.Atom import Atom
    from PeptideBuilder.PeptideBuilder import calculateCoordinates
    for name, element, frame, bond, angle, dihedral in \
            MODIFIED_RESIDUES[modified_code]["add_atoms"]:
        a, b, c = (residue[n] for n in frame)
        coord = calculateCoordinates(a, b, c, bond, angle, dihedral)
        residue.add(Atom(name, np.asarray(coord, dtype=float), 0.0, 1.0,
                         " ", f" {name:<3}", 0, element))
