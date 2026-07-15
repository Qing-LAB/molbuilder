"""SMILES -> 3D: RDKit-first, OpenBabel-fallback + backend provenance.

Pins the compatibility fix (task #54): RDKit handles typical organics (best
geometry + stereo); OpenBabel rescues the inputs RDKit can't parse
(metal-organics -- heme) or can't embed (cages -- C60), and the caller learns
WHICH engine produced the geometry.

The SMILES here are from LEGITIMATE sources, hardcoded so the tests need no
network:
  * hemeC   -- Wikipedia (https://en.wikipedia.org/wiki/Heme_C).
  * C60     -- PubChem canonical SMILES for buckminsterfullerene (CID 123591).
"""
from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("openbabel")

from molbuilder.smiles import (          # noqa: E402
    build_from_smiles, BACKEND_RDKIT, BACKEND_OPENBABEL,
)

# Wikipedia, Heme C.
HEME_C = (
    r"OC(=O)CC/c6c(\C)c3n7c6cc2c(/CCC(O)=O)c(/C)c1cc5n8c(cc4n([Fe]78n12)"
    r"c(c=3)c(C(S)=C)c4c)c(\C(S)=C)c5\C"
)
# PubChem canonical SMILES, buckminsterfullerene (CID 123591).
C60 = (
    "C12=C3C4=C5C6=C1C7=C8C9=C1C%10=C%11C(=C29)C3=C2C3=C4C4=C5C5=C9C6=C7C6=C7"
    "C8=C1C1=C8C%10=C%10C%11=C2C2=C3C3=C4C4=C5C5=C%11C%12=C(C6=C95)C7=C1C1=C%12"
    "C5=C%11C4=C3C3=C5C(=C81)C%10=C23"
)


def test_normal_molecule_uses_rdkit():
    """A typical organic stays on the high-fidelity RDKit path."""
    struct, backend = build_from_smiles("CCO", return_backend=True)   # ethanol
    assert backend == BACKEND_RDKIT
    assert len(struct.elements) == 9          # C2H6O = 9 atoms with explicit H


def test_metal_organic_falls_back_to_openbabel_parse():
    """hemeC: RDKit's MolFromSmiles returns None (aromatic N on [Fe]); OpenBabel
    parses it.  Item 3."""
    from rdkit import Chem
    assert Chem.MolFromSmiles(HEME_C) is None, "precondition: RDKit rejects hemeC"
    struct, backend = build_from_smiles(HEME_C, return_backend=True)
    assert backend == BACKEND_OPENBABEL
    assert any(e == "Fe" for e in struct.elements)
    assert len(struct.elements) > 40


def test_fullerene_cage_falls_back_to_openbabel_embed():
    """C60: RDKit PARSES it but ETKDG (even with random coords) can't embed the
    cage; OpenBabel make3D builds it.  Item 4 (the Name-lookup C60 case)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(C60))
    p = AllChem.ETKDGv3(); p.randomSeed = 0xF00D
    r1 = AllChem.EmbedMolecule(m, p)
    p.useRandomCoords = True
    r2 = AllChem.EmbedMolecule(m, p) if r1 == -1 else 0
    assert r1 == -1 and r2 == -1, "precondition: RDKit can't embed C60"

    struct, backend = build_from_smiles(C60, return_backend=True)
    assert backend == BACKEND_OPENBABEL
    assert len(struct.elements) == 60         # 60 carbons, no H
    assert all(e == "C" for e in struct.elements)


def test_default_return_is_just_structure():
    """return_backend defaults False -> the plain Structure (CLI contract)."""
    from molbuilder.structure import Structure
    assert isinstance(build_from_smiles("CCO"), Structure)


def test_openbabel_backend_string_names_the_fallback():
    """The provenance string must make the lower-fidelity path obvious."""
    assert "OpenBabel" in BACKEND_OPENBABEL and "fallback" in BACKEND_OPENBABEL.lower()
