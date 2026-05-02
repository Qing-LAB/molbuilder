"""Peptide builder.

Wraps the well-tested :mod:`PeptideBuilder` library to assemble a fully
extended chain from sequence, then converts the Bio.PDB output into our
:class:`Structure` dataclass.

Modified residues (phosphoSer, methylLys, ...) are handled in two
phases:
    1. Build the peptide using the parent standard residue (Ser, Lys).
    2. Patch the side chain in-place: remove specified atoms and add the
       new ones at offsets defined in :mod:`residues.MODIFIED_RESIDUES`.

This keeps the heavy lifting (bond geometry, Ramachandran-valid
backbone) inside PeptideBuilder and only does the chemistry-of-the-tip
in our own code.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .residues import (
    MODIFIED_RESIDUES,
    AA_THREE_TO_ONE,
    parse_peptide_sequence,
)
from .structure import Structure


def _import_peptidebuilder():
    """Lazy import so users who only build DNA don't need PeptideBuilder."""
    try:
        import PeptideBuilder
        # PeptideBuilder declares Bio.PDB as a dep and imports it eagerly
        # at module load, so importing PeptideBuilder is sufficient as a
        # combined probe -- no need for a redundant Bio.PDB import here.
        return PeptideBuilder
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Building peptides needs PeptideBuilder + biopython.  Install "
            "with:  pip install PeptideBuilder biopython"
        ) from exc


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
        Free-form sequence in 1-letter ("ARNDC"), 3-letter
        ("Ala-Arg-Asn-Asp-Cys"), or mixed notation; modified residues
        in parentheses, e.g. ``"AR(SEP)C"`` for Ala-Arg-phosphoSer-Cys.
        See :mod:`molbuilder.residues` for the supported modified codes.
    title
        Optional title written into the XYZ comment / PDB TITLE line.
    add_hydrogens
        If True (default), add explicit hydrogens via OpenBabel or RDKit
        (whichever is installed first).  Required for any quantum-
        chemistry calculation.  If neither is installed and this is True,
        a warning is printed and the heavy-atom-only structure is
        returned -- you will have to protonate it yourself.

    Returns
    -------
    Structure
        All-atom structure with PDB-style metadata (atom names, residue
        ids/names) filled in.
    """
    PeptideBuilder = _import_peptidebuilder()

    codes = parse_peptide_sequence(sequence)
    if not codes:
        raise ValueError("Empty sequence.")

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

    # Build with default phi/psi (=trans / extended)
    bio_struct = PeptideBuilder.make_extended_structure(one_letter)

    # Convert Bio.PDB structure -> our Structure
    elements:      List[str] = []
    positions:     List[Tuple[float, float, float]] = []
    atom_names:    List[str] = []
    residue_ids:   List[int] = []
    residue_names: List[str] = []
    chain_ids:     List[str] = []

    for model in bio_struct:
        for chain in model:
            for res in chain:
                rid = res.get_id()[1]
                rname = parent_codes[rid - 1]
                for atom in res:
                    # BioPython sometimes returns the element field with
                    # leading whitespace (e.g. " C" for backbone carbon).
                    # Strip first, otherwise downstream species detection
                    # in the FDF / pyscf modules sees " C" as a separate
                    # element from "C" and emits a malformed input.
                    el_raw = atom.element.strip()
                    elements.append(el_raw.title() if len(el_raw) > 1
                                    else el_raw.upper())
                    positions.append(tuple(float(c) for c in atom.coord))
                    atom_names.append(atom.get_name())
                    residue_ids.append(rid)
                    residue_names.append(rname)
                    chain_ids.append(chain.id or "A")
        break  # first model only

    struct = Structure(
        elements=elements,
        positions=np.asarray(positions, dtype=float),
        atom_names=atom_names,
        residue_ids=residue_ids,
        residue_names=residue_names,
        chain_ids=chain_ids,
        title=title or f"peptide {sequence}",
    )

    # Apply modifications in residue-id order so atom indices stay stable
    # within each call to _patch_residue (we rebuild the arrays each time).
    for rid in sorted(patches):
        struct = _patch_residue(struct, rid, patches[rid])

    if add_hydrogens:
        struct = _add_hydrogens(struct)

    return struct


# ---------------------------------------------------------------------- #
#  Hydrogen addition                                                     #
# ---------------------------------------------------------------------- #


def _add_hydrogens(struct: Structure) -> Structure:
    """Protonate using whichever of (OpenBabel, RDKit) is installed.

    Falls back to the heavy-atom-only structure with a warning if
    neither is available.
    """
    # ---- try OpenBabel first (fastest, doesn't reorder atoms) --------
    try:
        from openbabel import openbabel as ob
    except ImportError:
        ob = None

    if ob is not None:
        return _protonate_openbabel(struct, ob)

    # ---- fall back to RDKit ------------------------------------------
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        Chem = None  # type: ignore

    if Chem is not None:
        return _protonate_rdkit(struct, Chem, AllChem)

    import warnings
    warnings.warn(
        "Cannot add hydrogens: neither OpenBabel (`pip install openbabel`) "
        "nor RDKit (`conda install -c conda-forge rdkit`) is installed.  "
        "Returning a HEAVY-ATOM-ONLY structure -- protonate before running "
        "any quantum-chemistry calculation.",
        RuntimeWarning, stacklevel=3,
    )
    return struct


def _protonate_openbabel(struct: Structure, ob) -> Structure:
    obconv = ob.OBConversion()
    obconv.SetInAndOutFormats("pdb", "pdb")
    mol = ob.OBMol()
    obconv.ReadString(mol, struct.to_pdb())
    mol.AddHydrogens()
    out = obconv.WriteString(mol)
    return _drop_overlapping_hydrogens(
        _structure_from_pdb_string(out, title=struct.title)
    )


def _protonate_rdkit(struct: Structure, Chem, AllChem) -> Structure:
    mol = Chem.MolFromPDBBlock(struct.to_pdb(), removeHs=False, sanitize=False)
    if mol is None:
        # RDKit can choke on partial / unusual PDBs -- return as-is.
        import warnings
        warnings.warn(
            "RDKit failed to parse the heavy-atom PDB; returning "
            "heavy-atom-only structure.  Try installing OpenBabel.",
            RuntimeWarning, stacklevel=3,
        )
        return struct
    mol = Chem.AddHs(mol, addCoords=True)
    pdb_out = Chem.MolToPDBBlock(mol)
    return _drop_overlapping_hydrogens(
        _structure_from_pdb_string(pdb_out, title=struct.title)
    )


def _drop_overlapping_hydrogens(struct: Structure) -> Structure:
    """Remove H atoms that overlap (< 0.05 Å) with any other atom.

    Both protonation paths (OpenBabel's AddHydrogens and RDKit's
    AddHs(addCoords=True)) occasionally fail to compute positions for
    ambiguous-valence Hs (most often the N-terminal -NH3+ extras and
    the second backbone-amine H of a free N-terminus); the H ends up
    at the same coordinates as its anchor heavy atom or another
    template atom.  These Hs are guaranteed broken (no real H sits
    < 0.05 Å from another atom) and a downstream validator flags them
    as zero-distance pairs that abort the run.  Drop them at the
    source so the caller gets a clean structure.

    Heavy atoms are never removed.
    """
    pos      = struct.positions
    elements = struct.elements
    n        = len(pos)
    keep     = np.ones(n, dtype=bool)
    for i in range(n):
        if elements[i] != "H":
            continue
        for j in range(n):
            if i == j:
                continue
            if float(np.linalg.norm(pos[i] - pos[j])) < 0.05:
                keep[i] = False
                break
    if keep.all():
        return struct
    return Structure(
        elements      = [e for k, e in zip(keep, elements)             if k],
        positions     = pos[keep],
        atom_names    = [a for k, a in zip(keep, struct.atom_names)    if k],
        residue_ids   = [r for k, r in zip(keep, struct.residue_ids)   if k],
        residue_names = [n for k, n in zip(keep, struct.residue_names) if k],
        chain_ids     = [c for k, c in zip(keep, struct.chain_ids)     if k],
        title         = struct.title,
    )


def _structure_from_pdb_string(pdb: str, *, title: str) -> Structure:
    elements:      List[str] = []
    positions:     List[Tuple[float, float, float]] = []
    atom_names:    List[str] = []
    residue_ids:   List[int] = []
    residue_names: List[str] = []
    chain_ids:     List[str] = []
    for line in pdb.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        # Fixed-width PDB columns
        atom_name = line[12:16].strip()
        res_name  = line[17:20].strip()
        chain_id  = line[21:22].strip() or "A"
        try:
            res_id = int(line[22:26])
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        element = line[76:78].strip()
        if not element:
            # Some writers omit the element column; fall back to atom name
            element = "".join(c for c in atom_name if c.isalpha())[:1].upper()
        elements.append(element)
        positions.append((x, y, z))
        atom_names.append(atom_name)
        residue_ids.append(res_id)
        residue_names.append(res_name)
        chain_ids.append(chain_id)
    return Structure(
        elements=elements,
        positions=np.asarray(positions, dtype=float),
        atom_names=atom_names,
        residue_ids=residue_ids,
        residue_names=residue_names,
        chain_ids=chain_ids,
        title=title,
    )


# ---------------------------------------------------------------------- #
#  Modified-residue side-chain patching                                  #
# ---------------------------------------------------------------------- #


def _patch_residue(struct: Structure, rid: int, modified_code: str) -> Structure:
    """Apply the SEP/TPO/PTR/MLY/M3L/ALY-style patch to a single residue."""
    spec = MODIFIED_RESIDUES[modified_code]
    remove = set(spec.get("remove_atoms", []))
    additions = spec.get("add_atoms", [])

    # Index lookups within this residue
    atom_index_by_name: Dict[str, int] = {}
    keep_mask = np.ones(struct.n_atoms, dtype=bool)
    for i in range(struct.n_atoms):
        if struct.residue_ids[i] != rid:
            continue
        if struct.atom_names[i] in remove:
            keep_mask[i] = False
        else:
            atom_index_by_name[struct.atom_names[i]] = i

    elements      = [e for i, e in enumerate(struct.elements)      if keep_mask[i]]
    positions     = struct.positions[keep_mask]
    atom_names    = [a for i, a in enumerate(struct.atom_names)    if keep_mask[i]]
    residue_ids   = [r for i, r in enumerate(struct.residue_ids)   if keep_mask[i]]
    residue_names = [n for i, n in enumerate(struct.residue_names) if keep_mask[i]]
    chain_ids     = [c for i, c in enumerate(struct.chain_ids)     if keep_mask[i]]

    # Re-fetch indices in the trimmed array so anchor lookups resolve
    atom_index_by_name = {n: i for i, n in enumerate(atom_names)
                          if residue_ids[i] == rid}

    extra_pos: List[np.ndarray] = []
    for name, element, anchor_name, dx, dy, dz in additions:
        if anchor_name not in atom_index_by_name:
            raise ValueError(
                f"Patch for {modified_code} at residue {rid} expects an "
                f"anchor atom {anchor_name!r}, which isn't present in the "
                f"parent residue."
            )
        anchor_pos = positions[atom_index_by_name[anchor_name]]
        new_pos = anchor_pos + np.array([dx, dy, dz], dtype=float)
        extra_pos.append(new_pos)
        elements.append(element)
        atom_names.append(name)
        residue_ids.append(rid)
        residue_names.append(modified_code)   # mark patched residue
        chain_ids.append(chain_ids[atom_index_by_name[anchor_name]])

    # Update the parent residue's resname to the modified code so PDB
    # writes consistently.
    for i in range(len(residue_ids)):
        if residue_ids[i] == rid and residue_names[i] != modified_code:
            residue_names[i] = modified_code

    if extra_pos:
        positions = np.vstack([positions, np.array(extra_pos)])

    return Structure(
        elements=elements,
        positions=positions,
        atom_names=atom_names,
        residue_ids=residue_ids,
        residue_names=residue_names,
        chain_ids=chain_ids,
        title=struct.title,
    )
