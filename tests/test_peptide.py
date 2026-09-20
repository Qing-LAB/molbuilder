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
    # 5 + 11 + 8 + 8 + 6.  The final 6 is C-terminal CYS WITHOUT its OXT:
    # the structure kit does not write one and molbuilder does not add it
    # (user ruling 2026-09-20 -- adding H is justified and cross-checked,
    # placing an oxygen is a structural guess).  A free acid would be 39.
    # `build_peptide` warns; see test_the_c_terminus_limit_is_stated.
    assert s.n_atoms == 38
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
    # 35, derivable: backbone 12 + side chains 23 (Ala 3, Arg 10, Asn 4,
    # Asp 3, Cys 3).  NOT asserted as == 35, and the reason is the point:
    # the total is invariant under every protonation error that matters --
    # a pH-7.4 assignment (Arg+, Asp-, N-term+, C-term-) also gives 35, and
    # so does the aldehyde-vs-acid C-terminus.  A per-residue assertion is
    # what would discriminate; see plan.md § 11.
    assert n_h >= 25
    for el in ("C", "N", "O"):
        assert s_full.elements.count(el) == heavy_only_arndc.elements.count(el)


def test_modified_residue_phosphoserine():
    s = build_peptide("AR[SEP]C", add_hydrogens=False)
    assert "SEP" in s.residue_names
    assert s.elements.count("P") == 1


def test_the_c_terminus_limit_is_stated(recwarn):
    """The person is told what they got -- molbuilder does not fix it.

    User ruling 2026-09-20: adding hydrogens is a deliberate exception
    (predictable geometry, two kits cross-checking each other); adding the
    C-terminal oxygen is a structural guess the tool should not make.  So
    the limit has to be visible instead.

    This is the ONLY check that can see the defect: the hydrogen count is
    35 whether the terminus is an aldehyde or a free acid, so every
    atom-count assertion in this file is blind to it by construction.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = build_peptide("ARNDC")

    said = " ".join(str(w.message) for w in caught)
    assert "ALDEHYDE" in said, f"the C-terminus limit was not stated: {said!r}"
    assert "OXT" in said, "the warning does not name the missing atom"
    # And it is DETECTED, not assumed: the day the kit writes OXT it stops.
    assert "OXT" not in s.atom_names
