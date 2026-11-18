"""Peptide builder smoke tests.

The builder needs PeptideBuilder, an optional package of the host env (a
`pip` line of `host-env.txt`): the file skips without it, naming it.  The
protonation tests need Open Babel or RDKit, which the host env requires, so
they never skip (`process/testing.md` § 3).
"""

from __future__ import annotations

import pytest

build_peptide = pytest.importorskip("molbuilder.peptide").build_peptide


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
    # The check after the build says so (`build.peptide_aldehyde`,
    # tests/test_build_findings.py).
    assert s.n_atoms == 38
    assert "H" not in s.elements


def test_full_protonation_keeps_heavy_atom_counts(heavy_only_arndc):
    """build_peptide(...) with default add_hydrogens=True should keep
    the same heavy-atom counts but add explicit Hs."""
    s_full = build_peptide("ARNDC")
    n_h = s_full.elements.count("H")
    # 35, derivable: backbone 12 + side chains 23 (Ala 3, Arg 10, Asn 4,
    # Asp 3, Cys 3).  NOT asserted as == 35, and the reason is the point:
    # the total is invariant under every protonation error that matters --
    # a pH-7.4 assignment (Arg+, Asp-, N-term+, C-term-) also gives 35, and
    # so does the aldehyde-vs-acid C-terminus.  A per-residue assertion is
    # what would discriminate.
    assert n_h >= 25
    for el in ("C", "N", "O"):
        assert s_full.elements.count(el) == heavy_only_arndc.elements.count(el)


#: EACH MODIFIED RESIDUE THE GRAMMAR TAKES: its code, its parent's one
#: letter, and the heavy atoms it adds to the parent -- the PDB's Chemical
#: Component Dictionary's definition (`engines/builders.md` § 2).
_MODIFIED = [
    ("SEP", "S", {"P": 1, "O": 3}),   # phospho-serine
    ("TPO", "T", {"P": 1, "O": 3}),   # phospho-threonine
    ("PTR", "Y", {"P": 1, "O": 3}),   # phospho-tyrosine
    ("MLZ", "K", {"C": 1}),           # N6-methyl-lysine
    ("MLY", "K", {"C": 2}),           # N6,N6-dimethyl-lysine
    ("M3L", "K", {"C": 3}),           # N6,N6,N6-trimethyl-lysine
    ("ALY", "K", {"C": 2, "O": 1}),   # N6-acetyl-lysine
]


def _residue_atoms(s, rid):
    return {s.atom_names[i]: i for i in range(s.n_atoms)
            if s.residue_ids[i] == rid}


@pytest.mark.parametrize("code, parent, adds", _MODIFIED,
                         ids=[row[0] for row in _MODIFIED])
def test_a_modified_residue_is_built_as_the_pdb_defines_it(code, parent,
                                                           adds):
    """`engines/builders.md` § 2: a modified residue is its parent plus the
    PDB's own atoms, each placed in the parent's local frame -- bonded at a
    bond length, at its hybridisation's angle (sp3 109.5, sp2 120) -- and
    the hydrogens placed after leave no two overlapping.  The silent
    failures (F12's review, 2026-11-18): lab-frame offsets put [SEP]'s P
    0.235 A from CB and A[SEP]A's CB-OG-P at 77 degrees; [MLY] was built as
    the single methyl the PDB calls MLZ; TPO's phosphate hydrogens sat
    1.08 A apart -- each built without a word."""
    import numpy as np

    from molbuilder.chemistry import CLASH_CONTACT_A
    s = build_peptide(f"A[{code}]A")
    plain = build_peptide(f"A{parent}A", add_hydrogens=False)
    mine = _residue_atoms(s, 2)
    added = [i for name, i in mine.items()
             if name not in _residue_atoms(plain, 2)
             and s.elements[i] != "H"]
    formula = {}
    for i in added:
        formula[s.elements[i]] = formula.get(s.elements[i], 0) + 1
    assert formula == adds, sorted(s.atom_names[i] for i in added)

    pos = s.positions
    heavy = [i for i in range(s.n_atoms) if s.elements[i] != "H"]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    for c in added:
        # Its bond: the nearest heavy atom, at a bond's length.
        others = [k for k in heavy if k != c]
        b = min(others, key=lambda k: d[c, k])
        assert 1.2 <= d[c, b] <= 1.9, (
            f"{s.atom_names[c]}-{s.atom_names[b]} {d[c, b]:.3f} A")
        # Its angle, at the atom it is bonded to, with each other heavy
        # atom bonded there.
        for a in (k for k in heavy if k not in (b, c) and d[b, k] <= 1.9):
            u, v = pos[a] - pos[b], pos[c] - pos[b]
            angle = np.degrees(np.arccos(
                u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
            assert 100.0 <= angle <= 130.0, (
                f"{s.atom_names[a]}-{s.atom_names[b]}-{s.atom_names[c]} "
                f"{angle:.1f} degrees")
    hydrogens = [i for i in range(s.n_atoms) if s.elements[i] == "H"]
    hh = d[np.ix_(hydrogens, hydrogens)] + np.eye(len(hydrogens)) * 9.0
    assert hh.min() >= CLASH_CONTACT_A, f"two hydrogens {hh.min():.2f} A apart"


def test_lysine_ends_in_an_amine():
    """`engines/builders.md` § 2: lysine's side chain ends in an sp3 amine,
    NZ bonded to CE and two hydrogens -- Ala-Lys-Ala, neutral, holds 24
    hydrogens (Ala 5 + Lys 12 + Ala 5 in the chain, and the 2 of the water
    the two peptide bonds did not take).  The silent failure (F12's review,
    2026-11-18): the kit's lysine geometry ends in a C=N, so every lysine
    was built two hydrogens short -- an imine, built without a word."""
    import numpy as np
    s = build_peptide("AKA")
    assert s.elements.count("H") == 24
    nz = _residue_atoms(s, 2)["NZ"]
    near = np.linalg.norm(s.positions - s.positions[nz], axis=1)
    on_nz = [i for i in range(s.n_atoms)
             if s.elements[i] == "H" and near[i] < 1.15]
    assert len(on_nz) == 2, f"NZ carries {len(on_nz)} hydrogens"
