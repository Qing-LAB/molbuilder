"""Shared helpers across nucleic-acid backends."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...engine_atom_index import pdb_numbering
from ...structure import Structure


def rdkit_mol_to_structure(mol, *, title: Optional[str] = None) -> Structure:
    """Convert an RDKit Mol with a 3D conformer + PDB residue info into
    our :class:`Structure` dataclass.

    Atoms missing PDB info (e.g. hydrogens added without coords) get
    sensible defaults so the result is still a valid Structure.
    """
    elements = []
    positions = []
    atom_names = []
    residue_ids = []
    residue_names = []
    chain_ids = []

    if mol.GetNumConformers() == 0:
        raise ValueError("RDKit Mol has no 3D conformer; cannot extract positions")

    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms()):
        elements.append(atom.GetSymbol())
        p = conf.GetAtomPosition(i)
        positions.append([p.x, p.y, p.z])
        info = atom.GetPDBResidueInfo()
        if info is not None:
            atom_names.append(info.GetName().strip() or f"{atom.GetSymbol()}{pdb_numbering(i)}")
            residue_ids.append(info.GetResidueNumber() or 1)
            residue_names.append(info.GetResidueName().strip() or "MOL")
            chain_ids.append(info.GetChainId().strip() or "A")
        else:
            atom_names.append(f"{atom.GetSymbol()}{pdb_numbering(i)}")
            residue_ids.append(1)
            residue_names.append("MOL")
            chain_ids.append("A")

    return Structure(
        elements=elements,
        positions=np.asarray(positions, dtype=float),
        atom_names=atom_names,
        residue_ids=residue_ids,
        residue_names=residue_names,
        chain_ids=chain_ids,
        title=title or "",
    )


def verify_backbone_connectivity(struct: Structure, kind: str,
                                 max_O3_P: float = 1.80,
                                 ) -> Optional[str]:
    """Return None if the inter-residue P-O3' distances within each chain are
    all under `max_O3_P` Angstrom (implies the backbone is bonded), else a
    string describing the worst violation.  A duplex's strands number their
    residues on, so a residue's neighbour by number may be the other strand's,
    which no backbone joins.

    Used as a self-check at the end of every backend run.  If it fails
    the backend should raise rather than ship a broken structure.
    """
    P_pos:  dict = {}
    O3_pos: dict = {}
    for i in range(struct.n_atoms):
        where = (struct.chain_ids[i], struct.residue_ids[i])
        n = struct.atom_names[i]
        if n == "P":
            P_pos[where] = struct.positions[i]
        elif n == "O3'":
            O3_pos[where] = struct.positions[i]
    worst = None
    worst_d = 0.0
    for chain, r in sorted(P_pos):
        if (chain, r - 1) in O3_pos:
            d = float(np.linalg.norm(P_pos[chain, r] - O3_pos[chain, r - 1]))
            if d > max_O3_P and d > worst_d:
                worst = (chain, r - 1, r, d)
                worst_d = d
    if worst is None:
        return None
    chain, a, b, d = worst
    return (f"chain {chain} residue {a} O3' -> residue {b} P distance = "
            f"{d:.2f} A (should be <= {max_O3_P}); backbone is broken")


def select_chain(struct: Structure, chain: str) -> Structure:
    """Return a new Structure with only atoms whose chain_id matches."""
    keep = [i for i in range(struct.n_atoms) if struct.chain_ids[i] == chain]
    if not keep:
        raise ValueError(f"chain {chain!r} not present in structure "
                         f"(have {sorted(set(struct.chain_ids))})")
    return Structure(
        elements      = [struct.elements[i]      for i in keep],
        positions     = struct.positions[keep],
        atom_names    = [struct.atom_names[i]    for i in keep],
        residue_ids   = [struct.residue_ids[i]   for i in keep],
        residue_names = [struct.residue_names[i] for i in keep],
        chain_ids     = [struct.chain_ids[i]     for i in keep],
        title         = struct.title,
    )
