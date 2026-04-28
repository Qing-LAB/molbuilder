"""Peptide builder smoke tests.

Skipped cleanly if PeptideBuilder isn't installed in the test
environment.  The protonation tests additionally need OpenBabel or
RDKit; they soft-skip with a warning if neither is available.
"""

from __future__ import annotations

import pytest

molbuilder = pytest.importorskip("molbuilder")
build_peptide = pytest.importorskip("molbuilder").build_peptide


@pytest.fixture
def heavy_only_arndc():
    """ARNDC built without protonation -- 38 heavy atoms."""
    return build_peptide("ARNDC", add_hydrogens=False)


def test_heavy_atom_count(heavy_only_arndc):
    s = heavy_only_arndc
    assert s.n_residues == 5
    assert sorted(set(s.residue_names)) == ["ALA", "ARG", "ASN", "ASP", "CYS"]
    assert s.n_atoms == 38   # 5 + 11 + 8 + 8 + 6
    assert "H" not in s.elements


def test_xyz_round_trip(heavy_only_arndc):
    s = heavy_only_arndc
    xyz = s.to_xyz()
    assert int(xyz.splitlines()[0]) == s.n_atoms


def test_pdb_atom_count(heavy_only_arndc):
    s = heavy_only_arndc
    pdb = s.to_pdb()
    assert pdb.count("ATOM") == s.n_atoms


def test_pyscf_listing(heavy_only_arndc):
    s = heavy_only_arndc
    py = s.to_pyscf()
    assert len(py) == s.n_atoms


def test_full_protonation_keeps_heavy_atom_counts(heavy_only_arndc):
    """build_peptide(...) with default add_hydrogens=True should keep
    the same heavy-atom counts but add explicit Hs."""
    s_full = build_peptide("ARNDC")
    if "H" not in s_full.elements:
        pytest.skip("no protonation backend installed (openbabel/rdkit)")
    n_h = s_full.elements.count("H")
    assert n_h >= 25
    for el in ("C", "N", "O"):
        assert s_full.elements.count(el) == heavy_only_arndc.elements.count(el)


def test_modified_residue_phosphoserine():
    s = build_peptide("AR[SEP]C", add_hydrogens=False)
    assert "SEP" in s.residue_names
    assert s.elements.count("P") == 1
