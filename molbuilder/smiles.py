"""SMILES -> 3D Structure via RDKit.

Generates a single conformer with ETKDG (Riniker-Landrum) and cleans up
with the MMFF94s force field.  For very small molecules (< ~50 atoms)
this is essentially instant; for larger drug-like molecules it can
take a few seconds.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .structure import Structure


def _import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "build_from_smiles needs RDKit; install with "
            "`conda install -c conda-forge rdkit` or `pip install rdkit`"
        ) from exc
    return Chem, AllChem


def build_from_smiles(
    smiles: str,
    *,
    title: Optional[str] = None,
    optimize: bool = True,
    seed: int = 0xF00D,
) -> Structure:
    """Build a 3-D molecule from a SMILES string.

    Parameters
    ----------
    smiles
        SMILES string, e.g. ``"c1ccccc1"`` (benzene),
        ``"Sc1ccc(S)cc1"`` (1,4-benzenedithiol).
    title
        Optional title for the resulting structure.
    optimize
        If True (default), run MMFF94s force-field optimisation on the
        embedded conformer to clean up obviously bad geometries.
    seed
        ETKDG random seed.  Fixed for reproducibility.

    Returns
    -------
    Structure
        All-atom structure with explicit hydrogens.

    Raises
    ------
    ValueError if the SMILES is invalid or RDKit can't embed a 3-D
    conformer (rare for typical organic molecules; common for very
    strained structures).
    """
    Chem, AllChem = _import_rdkit()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) == -1:
        # Fallback: try the simpler 2D->3D approach with random coords
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) == -1:
            raise ValueError(
                f"RDKit could not embed a 3-D conformer for SMILES {smiles!r}"
            )

    if optimize:
        # MMFF94s preferred; fall back to UFF if MMFF can't parameterise.
        # RDKit return convention for MMFFOptimizeMolecule:
        #     0  = converged
        #     1  = max-iter exceeded (still produced valid coords)
        #    -1  = MMFF could not parameterise the molecule
        # We treat 0 and 1 both as "MMFF was useful"; only fall back to
        # UFF when MMFF is unable to handle the chemistry (-1) or raises.
        mmff_ok = False
        try:
            rc = AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s",
                                              maxIters=400)
            mmff_ok = (rc != -1)
        except Exception:
            mmff_ok = False
        if not mmff_ok:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=400)
            except Exception:
                pass  # optimisation is best-effort

    elements = []
    positions = []
    atom_names = []
    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms()):
        elements.append(atom.GetSymbol())
        p = conf.GetAtomPosition(i)
        positions.append([p.x, p.y, p.z])
        atom_names.append(f"{atom.GetSymbol()}{i + 1}")

    return Structure(
        elements=elements,
        positions=np.asarray(positions, dtype=float),
        atom_names=atom_names,
        title=title or f"smiles {smiles}",
    )
