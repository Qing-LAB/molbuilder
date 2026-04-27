"""Peptide builder smoke test.

Skipped if PeptideBuilder isn't installed in the test environment.
"""

from __future__ import annotations

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    try:
        from molbuilder import build_peptide
    except ImportError as e:
        print(f"SKIP -- PeptideBuilder/biopython not installed ({e})")
        return

    # First: heavy-atom-only build (no protonation).  ARNDC = 38 heavies.
    s_heavy = build_peptide("ARNDC", add_hydrogens=False)
    assert s_heavy.n_residues == 5
    assert sorted(set(s_heavy.residue_names)) == ["ALA", "ARG", "ASN", "ASP", "CYS"]
    assert s_heavy.n_atoms == 38, s_heavy.n_atoms          # 5 + 11 + 8 + 8 + 6
    assert "H" not in s_heavy.elements

    # Output formats round-trip on the heavy-atom version
    xyz = s_heavy.to_xyz()
    assert int(xyz.splitlines()[0]) == s_heavy.n_atoms
    pdb = s_heavy.to_pdb()
    assert pdb.count("ATOM") == s_heavy.n_atoms
    py = s_heavy.to_pyscf()
    assert len(py) == s_heavy.n_atoms

    # Now: full protonation.  Should add ~30 hydrogens for ARNDC.
    s_full = build_peptide("ARNDC")
    if "H" not in s_full.elements:
        print("  (warn: no protonation backend installed -- "
              "install openbabel or rdkit)")
    else:
        n_h = s_full.elements.count("H")
        assert n_h >= 25, n_h
        # Sanity: same heavy-atom count
        assert s_full.elements.count("C") == s_heavy.elements.count("C")
        assert s_full.elements.count("N") == s_heavy.elements.count("N")
        assert s_full.elements.count("O") == s_heavy.elements.count("O")
        print(f"  protonation OK: added {n_h} hydrogens "
              f"(total {s_full.n_atoms} atoms)")

    # Modified residue: phosphoserine via [SEP] escape
    s2 = build_peptide("AR[SEP]C", add_hydrogens=False)
    assert "SEP" in s2.residue_names, s2.residue_names
    # SEP adds a phosphate (P + 3 O)
    assert s2.elements.count("P") == 1
    assert "P"  in s2.elements

    print(f"OK -- build_peptide('ARNDC', add_hydrogens=False) = "
          f"{s_heavy.n_atoms} heavy atoms; "
          f"build_peptide('AR[SEP]C') has phosphate group attached.")


if __name__ == "__main__":
    main()
