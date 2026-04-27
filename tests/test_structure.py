"""Structure dataclass + writers (no external deps beyond numpy)."""

from __future__ import annotations

import os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from molbuilder.structure import Structure


def main() -> None:
    # Tiny H2O
    s = Structure(
        elements=["O", "H", "H"],
        positions=np.array([
            [0.000, 0.000, 0.000],
            [0.957, 0.000, 0.000],
            [-0.240, 0.927, 0.000],
        ]),
        atom_names=["O", "H1", "H2"],
        residue_ids=[1, 1, 1],
        residue_names=["HOH", "HOH", "HOH"],
        chain_ids=["A", "A", "A"],
        title="water",
    )
    assert s.n_atoms == 3
    assert s.n_residues == 1
    assert "H2O" in s.summary()

    # XYZ round-trip
    xyz = s.to_xyz()
    assert xyz.startswith("3\n")
    assert "O" in xyz and "H" in xyz
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "out.xyz")
        s.to_xyz(p)
        assert open(p).read() == xyz

    # PDB
    pdb = s.to_pdb()
    assert pdb.startswith("TITLE")
    assert "ATOM" in pdb
    assert "HOH" in pdb
    assert "END" in pdb

    # PySCF list form
    py = s.to_pyscf()
    assert py == [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.957, 0.0, 0.0)),
        ("H", (-0.240, 0.927, 0.0)),
    ]

    # PySCF string form
    py_str = s.to_pyscf(as_string=True)
    assert "O " in py_str and "H " in py_str
    assert py_str.count("\n") == 2  # 3 lines, 2 newlines

    # ASE (optional dep)
    try:
        atoms = s.to_ase()
        assert len(atoms) == 3
        assert list(atoms.get_chemical_symbols()) == ["O", "H", "H"]
    except ImportError:
        print("  (skip: ASE not installed)")

    # Translate / center
    s2 = s.centered()
    assert np.allclose(s2.positions.mean(axis=0), 0.0, atol=1e-9)

    # Concat + renumbering
    s_cat = Structure.concat([s, s])
    assert s_cat.n_atoms == 6
    # residue ids should be 1, 1, 1, 2, 2, 2
    assert s_cat.residue_ids == [1, 1, 1, 2, 2, 2]

    print("OK -- Structure + XYZ + PDB + PySCF + ASE + concat all pass.")


if __name__ == "__main__":
    main()
