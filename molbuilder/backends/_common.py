"""Shared helpers across nucleic-acid backends."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..structure import Structure


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
            atom_names.append(info.GetName().strip() or f"{atom.GetSymbol()}{i + 1}")
            residue_ids.append(info.GetResidueNumber() or 1)
            residue_names.append(info.GetResidueName().strip() or "MOL")
            chain_ids.append(info.GetChainId().strip() or "A")
        else:
            atom_names.append(f"{atom.GetSymbol()}{i + 1}")
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


def parse_pdb_to_structure(pdb_text: str, *, title: Optional[str] = None) -> Structure:
    """Parse standard PDB ATOM records into a Structure."""
    elements = []
    positions = []
    atom_names = []
    residue_ids = []
    residue_names = []
    chain_ids = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        res_name  = line[17:20].strip()
        chain_id  = (line[21:22].strip() or "A")
        try:
            res_id = int(line[22:26])
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        element = line[76:78].strip()
        if not element:
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
        title=title or "",
    )


def verify_backbone_connectivity(struct: Structure, kind: str,
                                 max_O3_P: float = 1.80,
                                 ) -> Optional[str]:
    """Return None if the inter-residue P-O3' distances are all under
    `max_O3_P` Angstrom (implies the backbone is bonded), else a
    string describing the worst violation.

    Used as a self-check at the end of every backend run.  If it fails
    the backend should raise rather than ship a broken structure.
    """
    P_pos:  dict = {}
    O3_pos: dict = {}
    for i in range(struct.n_atoms):
        rid = struct.residue_ids[i]
        n = struct.atom_names[i]
        if n == "P":
            P_pos[rid] = struct.positions[i]
        elif n == "O3'":
            O3_pos[rid] = struct.positions[i]
    worst = None
    worst_d = 0.0
    for r in sorted(P_pos):
        if r - 1 in O3_pos:
            d = float(np.linalg.norm(P_pos[r] - O3_pos[r - 1]))
            if d > max_O3_P and d > worst_d:
                worst = (r - 1, r, d)
                worst_d = d
    if worst is None:
        return None
    a, b, d = worst
    return (f"residue {a} O3' -> residue {b} P distance = {d:.2f} A "
            f"(should be <= {max_O3_P}); backbone is broken")


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
