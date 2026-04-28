"""RDKit-based DNA / RNA builder.

Uses ``rdkit.Chem.MolFromSequence(seq, flavor=...)`` to construct a
chemically correct polymer (proper backbone, all atoms incl. H), then
embeds a 3-D conformer with ETKDG and cleans up with UFF.

Important caveat: RDKit's embedder is conformation-aware but not
*helix*-aware.  For oligos longer than ~6 residues the conformer is
typically a folded clump rather than a B-form helix.  Fine if you'll
run a thorough DFT optimisation anyway; not what you want for
production B-DNA / A-RNA structures.  Use the ``amber`` backend
for an extended chain with proper Amber force-field topology, or
build with 3DNA externally for a true B-form / A-form helix.
"""

from __future__ import annotations

import warnings
from typing import Optional

from ..structure import Structure
from ._common import rdkit_mol_to_structure


# rdkit.Chem.MolFromSequence flavor codes (probed empirically:
# residues with O2' are RNA, residues without are DNA -- the flavors
# 2-5 produce O2', so they're RNA, and flavors 6-7 are DNA):
#   2 = RNA, 5'-OH
#   3 = RNA, 5'-phosphate
#   4 = RNA, 3'-phosphate
#   5 = RNA, both phosphates
#   6 = DNA, 5'-OH
#   7 = DNA, 5'-phosphate
_FLAVOR = {
    ("dna", "OH"): 6,  ("dna", "5P"): 7,
    ("dna", "3P"): 6,  ("dna", "PP"): 7,    # rdkit DNA only has 6 / 7
    ("rna", "OH"): 2,  ("rna", "5P"): 3,
    ("rna", "3P"): 4,  ("rna", "PP"): 5,
}


def is_available() -> bool:
    # importlib.util.find_spec checks the module's installability without
    # actually executing its top-level code -- faster than `import rdkit`
    # on every call and pyflakes-clean.
    import importlib.util
    return importlib.util.find_spec("rdkit") is not None


def build(kind: str, sequence: str, form: str, terminal: str,
          title: Optional[str] = None) -> Structure:
    if not is_available():
        from . import BackendUnavailable
        raise BackendUnavailable(
            "rdkit not installed; run `pip install rdkit` "
            "or `conda install -c conda-forge rdkit`"
        )

    if form != "B":
        warnings.warn(
            f"rdkit backend can't enforce {form}-form geometry; the "
            f"embedded conformer is whatever ETKDG produces. "
            f"Use the `amber` backend for proper backbone topology, "
            f"or 3DNA externally for canonical helical geometry.",
            RuntimeWarning, stacklevel=3,
        )

    if kind not in ("dna", "rna"):
        raise ValueError(f"rdkit backend supports kind in 'dna'|'rna'; got {kind!r}")
    flavor = _FLAVOR.get((kind, terminal))
    if flavor is None:
        raise ValueError(
            f"rdkit backend doesn't support terminal={terminal!r} for {kind}; "
            f"valid: 'OH', '5P', '3P', 'PP'"
        )

    from rdkit import Chem
    from rdkit.Chem import AllChem

    one_letter = "".join(c for c in sequence.upper() if c.isalpha())
    mol = Chem.MolFromSequence(one_letter, flavor=flavor)
    if mol is None:
        raise ValueError(
            f"RDKit could not build a {kind} polymer from sequence "
            f"{one_letter!r} (flavor={flavor})"
        )
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    if AllChem.EmbedMolecule(mol, params) == -1:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) == -1:
            raise RuntimeError(
                f"RDKit ETKDG could not embed a 3-D conformer for "
                f"{kind} {one_letter!r}"
            )

    # MMFF doesn't have parameters for nucleic acids -- use UFF.
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass  # optimisation is best-effort

    return rdkit_mol_to_structure(
        mol,
        title=title or f"{kind} {one_letter} (rdkit, embedded conformer)",
    )
