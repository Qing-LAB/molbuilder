"""SMILES -> 3D Structure.

Primary path: **RDKit** — ETKDGv3 (Riniker-Landrum) conformer + MMFF94s cleanup.
This gives the best geometry + reliable stereochemistry for typical organic
molecules, and its output is what a DFT optimisation wants as a starting point.

Fallback path: **OpenBabel** — for the chemistry RDKit refuses though the
text reads (a valence or an aromatic ring it cannot assign, where MolFromSmiles
returns None) and the cages RDKit can't embed (fullerenes like C60, where ETKDG
fails).  OpenBabel is more lenient but its geometry is rougher, so it is a
fallback, not the default.  A text that does not read as SMILES is refused,
never handed to it: it would build ``C(`` as methane.

The result records WHICH backend produced the geometry (``return_backend=True``)
so the caller can tell the user when they are on the lower-fidelity path.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .chemistry import (ETKDG_SEED, BuildRefused, embed_conformer,
                        molecules_to_structure, rdkit_to_structure, require)
from .structure import Structure

# Provenance strings surfaced to the user (build.py -> the result banner).
BACKEND_RDKIT = "RDKit ETKDGv3"
BACKEND_OPENBABEL = "OpenBabel make3D (RDKit fallback — lower-fidelity geometry)"

#: What RDKit is needed for here, as a missing-tool answer says it.
_NEEDED_FOR = "Building from a SMILES"


def build_from_smiles(
    smiles: str,
    *,
    title: Optional[str] = None,
    optimize: bool = True,
    seed: int = ETKDG_SEED,
    return_backend: bool = False,
):
    """Build a 3-D molecule from a SMILES string.

    RDKit first (best geometry + stereo); OpenBabel fallback for chemistry
    RDKit refuses (a valence, an aromatic ring it cannot assign) or cannot
    embed (cages, e.g. C60).

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
    chemistry.BuildRefused for a SMILES that holds no atom or does not read,
    or that BOTH RDKit and OpenBabel fail to build; chemistry.BackendUnavailable
    if RDKit is not installed, or
    OpenBabel when RDKit could not and the fallback is needed.
    """
    struct, backend = _build(smiles, title=title, optimize=optimize, seed=seed)
    return (struct, backend) if return_backend else struct


def _build(smiles, *, title, optimize, seed) -> Tuple[Structure, str]:
    Chem = require("rdkit", needed_for=_NEEDED_FOR)
    AllChem = require("rdkit", needed_for=_NEEDED_FOR,
                      module="rdkit.Chem.AllChem")

    mol = Chem.MolFromSmiles(smiles) if smiles.strip() else None
    if not smiles.strip() or (mol is not None and mol.GetNumAtoms() == 0):
        raise BuildRefused(f"The SMILES {smiles!r} holds no atom: nothing to "
                           f"build.")
    # Text after a space that RDKit reads as the SMILES's NAME, dropping it
    # (`C C(=O)O` read as methane), and a text RDKit cannot read even
    # without its chemistry checks: each refused, never built as something
    # else -- nor handed to OpenBabel, which reads leniently (`C(` as
    # methane; `engines/builders.md` § 6).  An extended-SMILES block after
    # the space (`C |^1:0|`, a radical) is part of the SMILES, and built.
    read = Chem.SmilesParserParams()
    read.parseName, read.sanitize = True, False
    named = Chem.MolFromSmiles(smiles.strip(), read)
    if named is not None and named.HasProp("_Name") and named.GetProp("_Name"):
        raise BuildRefused(
            f"The SMILES {smiles!r} holds text after a space "
            f"({named.GetProp('_Name')!r}), which a SMILES reader takes as "
            f"its name: it would not be built.")
    if mol is None and named is None:
        raise BuildRefused(
            f"The SMILES {smiles!r} does not read: its text breaks the SMILES "
            f"grammar (an unbalanced bracket or parenthesis, a ring left "
            f"open, or a symbol that is no element).")
    if mol is not None:
        molH = Chem.AddHs(mol)
        if embed_conformer(molH, seed):
            if optimize:
                _rdkit_optimize(AllChem, molH)
            return (rdkit_to_structure(molH,
                                       title=title or f"smiles {smiles}"),
                    BACKEND_RDKIT)
    # RDKit could not -- refused its chemistry, or could not place it in
    # 3-D: OpenBabel reads and embeds what RDKit cannot (C60, an aromatic
    # ring RDKit cannot assign), its absence a tool this install lacks, said
    # through the one door (`chemistry.require`; `engines/builders.md` § 1).
    did = ("rejected it" if mol is None
           else "could not place it in 3-D")
    pybel = require("openbabel", module="openbabel.pybel",
                    needed_for=f"The SMILES {smiles!r} -- RDKit {did} --")
    return _openbabel_build(pybel, smiles, title, optimize, did)


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


def _openbabel_build(pybel, smiles, title, optimize, did) -> Tuple[Structure, str]:
    """The SMILES built by OpenBabel, where RDKit ``did`` not."""
    try:
        mol = pybel.readstring("smi", smiles)
    except Exception as exc:
        raise BuildRefused(
            f"The SMILES {smiles!r} cannot be built: RDKit {did}, and "
            f"OpenBabel could not read it ({exc}).")
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

    # ONE RESIDUE FOR EACH MOLECULE the SMILES names, by the one conversion
    # (`chemistry.molecules_to_structure`): OpenBabel's own split of its
    # bonds into molecules (``ContigFragList``, atoms from 1).
    atoms = list(mol.atoms)
    if not atoms:
        raise BuildRefused(
            f"OpenBabel produced no atoms for SMILES {smiles!r}.")
    split = pybel.ob.vectorvInt()
    mol.OBMol.ContigFragList(split)
    return (
        molecules_to_structure(
            [pybel.ob.GetSymbol(atom.atomicnum) for atom in atoms],
            [atom.coords for atom in atoms],
            [[i - 1 for i in molecule] for molecule in split],
            title=title or f"smiles {smiles}"),
        BACKEND_OPENBABEL,
    )
