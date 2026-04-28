"""Structure dataclass + writers (no external deps beyond numpy)."""

from __future__ import annotations

import numpy as np
import pytest

from molbuilder.structure import Structure


def test_basic_construction(water_structure):
    s = water_structure
    assert s.n_atoms == 3
    assert s.n_residues == 1
    assert "H2O" in s.summary()


def test_to_xyz_round_trip_to_disk(water_structure, tmp_path):
    s = water_structure
    text = s.to_xyz()
    assert text.startswith("3\n")
    assert "O" in text and "H" in text
    p = tmp_path / "out.xyz"
    s.to_xyz(str(p))
    assert p.read_text() == text


def test_to_pdb_basic_structure(water_structure):
    s = water_structure
    pdb = s.to_pdb()
    assert pdb.startswith("TITLE")
    assert "ATOM" in pdb
    assert "HOH" in pdb
    assert "END" in pdb


def test_to_pyscf_list_form(water_structure):
    s = water_structure
    py = s.to_pyscf()
    assert py == [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.957, 0.0, 0.0)),
        ("H", (-0.240, 0.927, 0.0)),
    ]


def test_to_pyscf_string_form(water_structure):
    s = water_structure
    py_str = s.to_pyscf(as_string=True)
    assert "O " in py_str and "H " in py_str
    assert py_str.count("\n") == 2  # 3 lines, 2 newlines


def test_to_ase_optional_dep(water_structure):
    """ASE is a hard dep of molbuilder.siesta; assert if installed."""
    pytest.importorskip("ase")
    atoms = water_structure.to_ase()
    assert len(atoms) == 3
    assert list(atoms.get_chemical_symbols()) == ["O", "H", "H"]


def test_centered_centroid_at_origin(water_structure):
    s2 = water_structure.centered()
    np.testing.assert_allclose(s2.positions.mean(axis=0), 0.0, atol=1e-9)


def test_concat_renumbers_residues(water_structure):
    s_cat = Structure.concat([water_structure, water_structure])
    assert s_cat.n_atoms == 6
    assert s_cat.residue_ids == [1, 1, 1, 2, 2, 2]
