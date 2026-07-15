"""SMILES -> 3D Structure.

Primary path: **RDKit** — ETKDGv3 (Riniker-Landrum) conformer + MMFF94s cleanup.
This gives the best geometry + reliable stereochemistry for typical organic
molecules, and its output is what a DFT optimisation wants as a starting point.

Fallback path: **OpenBabel** — for the inputs RDKit rejects (metal-organics /
non-kekulizable aromatics like heme, where MolFromSmiles returns None) and the
cages RDKit can't embed (fullerenes like C60, where ETKDG fails).  OpenBabel is
more lenient but its geometry is rougher, so it is a fallback, not the default.

The result records WHICH backend produced the geometry (``return_backend=True``)
so the caller can tell the user when they are on the lower-fidelity path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .structure import Structure

# Provenance strings surfaced to the user (build.py -> the result banner).
BACKEND_RDKIT = "RDKit ETKDGv3"
BACKEND_OPENBABEL = "OpenBabel make3D (RDKit fallback — lower-fidelity geometry)"


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


def _import_openbabel():
    try:
        from openbabel import pybel
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the OpenBabel SMILES fallback needs the openbabel Python "
            "bindings; install with `conda install -c conda-forge openbabel`"
        ) from exc
    return pybel


def build_from_smiles(
    smiles: str,
    *,
    title: Optional[str] = None,
    optimize: bool = True,
    seed: int = 0xF00D,
    return_backend: bool = False,
):
    """Build a 3-D molecule from a SMILES string.

    RDKit first (best geometry + stereo); OpenBabel fallback for inputs RDKit
    can't parse (metal-organics, e.g. heme) or embed (cages, e.g. C60).

    Parameters
    ----------
    smiles
        SMILES string, e.g. ``"c1ccccc1"`` (benzene).
    title
        Optional title for the resulting structure.
    optimize
        If True (default), run a force-field cleanup on the embedded conformer.
    seed
        ETKDG random seed.  Fixed for reproducibility.
    return_backend
        If True, return ``(Structure, backend_str)`` where ``backend_str`` names
        the engine that produced the geometry (``BACKEND_RDKIT`` /
        ``BACKEND_OPENBABEL``).  Default False returns just the ``Structure``.

    Returns
    -------
    Structure, or (Structure, str) when ``return_backend=True``.

    Raises
    ------
    ValueError if BOTH RDKit and OpenBabel fail to parse / embed the SMILES.
    """
    struct, backend = _build(smiles, title=title, optimize=optimize, seed=seed)
    return (struct, backend) if return_backend else struct


def _build(smiles, *, title, optimize, seed) -> Tuple[Structure, str]:
    Chem, AllChem = _import_rdkit()

    mol = Chem.MolFromSmiles(smiles)
    rdkit_stage = None            # which RDKit step failed: "parse" | "embed"
    if mol is not None:
        molH = Chem.AddHs(mol)
        if _rdkit_embed(AllChem, molH, seed):
            if optimize:
                _rdkit_optimize(AllChem, molH)
            return _struct_from_rdkit(molH, smiles, title), BACKEND_RDKIT
        rdkit_stage = "embed"
    else:
        rdkit_stage = "parse"

    # RDKit couldn't handle it (parse OR embed) -> OpenBabel fallback.
    try:
        pybel = _import_openbabel()
    except ImportError:
        if rdkit_stage == "parse":
            raise ValueError(
                f"Invalid SMILES {smiles!r}: RDKit rejected it and the "
                f"OpenBabel fallback is not installed.")
        raise ValueError(
            f"RDKit could not embed a 3-D conformer for SMILES {smiles!r} "
            f"and the OpenBabel fallback is not installed.")
    return _openbabel_build(pybel, smiles, title, optimize, rdkit_stage)


def _rdkit_embed(AllChem, mol, seed) -> bool:
    """ETKDGv3, then a random-coords retry.  True iff a conformer was embedded."""
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != -1:
        return True
    params.useRandomCoords = True
    return AllChem.EmbedMolecule(mol, params) != -1


def _rdkit_optimize(AllChem, mol) -> None:
    # MMFF94s preferred; fall back to UFF if MMFF can't parameterise (-1) or
    # raises.  Best-effort — a failed cleanup keeps the raw embedded coords.
    mmff_ok = False
    try:
        rc = AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=400)
        mmff_ok = (rc != -1)
    except Exception:
        mmff_ok = False
    if not mmff_ok:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=400)
        except Exception:
            pass


def _struct_from_rdkit(mol, smiles, title) -> Structure:
    elements, positions, atom_names = [], [], []
    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        p = conf.GetAtomPosition(i)
        elements.append(sym)
        positions.append([p.x, p.y, p.z])
        atom_names.append(f"{sym}{i + 1}")
    return Structure(
        elements=elements,
        positions=np.asarray(positions, dtype=float),
        atom_names=atom_names,
        title=title or f"smiles {smiles}",
    )


def _openbabel_build(pybel, smiles, title, optimize, rdkit_stage) -> Tuple[Structure, str]:
    try:
        mol = pybel.readstring("smi", smiles)
    except Exception as exc:
        raise ValueError(
            f"Invalid SMILES {smiles!r}: RDKit rejected it and OpenBabel could "
            f"not parse it either ({exc}).")
    mol.addh()
    # make3D builds a 3-D conformer (rule-based) + a short FF cleanup; localopt
    # tightens it.  Works on cages/metal-organics ETKDG can't handle.
    try:
        mol.make3D(forcefield="mmff94", steps=100)
        if optimize:
            mol.localopt(forcefield="mmff94", steps=250)
    except Exception:
        # As a last resort, make3D with the default UFF-ish path.
        mol.make3D()

    elements, positions, atom_names = [], [], []
    for i, atom in enumerate(mol.atoms):
        sym = pybel.ob.GetSymbol(atom.atomicnum)
        elements.append(sym)
        positions.append(list(atom.coords))
        atom_names.append(f"{sym}{i + 1}")
    if not elements:
        raise ValueError(
            f"OpenBabel produced no atoms for SMILES {smiles!r}.")
    return (
        Structure(
            elements=elements,
            positions=np.asarray(positions, dtype=float),
            atom_names=atom_names,
            title=title or f"smiles {smiles}",
        ),
        BACKEND_OPENBABEL,
    )
