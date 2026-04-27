"""Common-name / IUPAC-name -> 3-D Structure via PubChem + RDKit.

Looks the name up in PubChem (over the network), pulls the canonical
SMILES, and then defers to :func:`molbuilder.smiles.build_from_smiles`.

Requires both ``pubchempy`` and ``rdkit``.
"""

from __future__ import annotations

from typing import Optional

from .smiles import build_from_smiles
from .structure import Structure


def _import_pubchempy():
    try:
        import pubchempy as pcp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "build_from_name needs PubChemPy; install with "
            "`pip install pubchempy`"
        ) from exc
    return pcp


def smiles_for_name(name: str) -> str:
    """Look up the canonical SMILES for a name in PubChem.

    Raises ValueError if the name doesn't resolve.
    """
    pcp = _import_pubchempy()
    compounds = pcp.get_compounds(name, "name")
    if not compounds:
        raise ValueError(f"PubChem returned no compounds for name {name!r}")
    smi = (compounds[0].canonical_smiles
           or compounds[0].isomeric_smiles)
    if not smi:
        raise ValueError(f"PubChem entry for {name!r} has no SMILES")
    return smi


def build_from_name(
    name: str,
    *,
    title: Optional[str] = None,
    **smiles_kwargs,
) -> Structure:
    """Build a 3-D molecule from a PubChem-resolvable name.

    Parameters
    ----------
    name
        Common or IUPAC name, e.g. ``"benzene"``,
        ``"1,4-benzenedithiol"``, ``"caffeine"``.
    title
        Optional title.  Defaults to the name itself.
    **smiles_kwargs
        Forwarded to :func:`build_from_smiles` (e.g. ``optimize=False``).

    Returns
    -------
    Structure
        All-atom 3-D structure.
    """
    smiles = smiles_for_name(name)
    return build_from_smiles(
        smiles,
        title=title or name,
        **smiles_kwargs,
    )
