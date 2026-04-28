"""Structure.from_xyz / from_pdb and the top-level molbuilder.load().

Verifies that:
  * Structure round-trips through XYZ (lossless for elements + positions)
  * Structure round-trips through PDB (lossless for atom names,
    residue ids, residue names, chain ids, plus positions)
  * the top-level molbuilder.load() detects format from extension
  * loaded structures feed render_fdf without further preparation
  * malformed inputs raise informative errors
"""

from __future__ import annotations

import numpy as np
import pytest

import molbuilder
from molbuilder.structure import Structure


# --------------------------------------------------------------------- #
#  XYZ                                                                  #
# --------------------------------------------------------------------- #


def test_from_xyz_text():
    text = (
        "3\n"
        "water-like\n"
        "O   0.000  0.000  0.000\n"
        "H   0.957  0.000  0.000\n"
        "H  -0.239  0.927  0.000\n"
    )
    s = Structure.from_xyz(text)
    assert s.n_atoms == 3
    assert s.elements == ["O", "H", "H"]
    assert s.title == "water-like"
    np.testing.assert_allclose(s.positions[1], [0.957, 0.0, 0.0])


def test_from_xyz_path(tmp_path):
    s = molbuilder.build_peptide("ARNDC")
    p = tmp_path / "pep.xyz"
    s.to_xyz(str(p))
    s2 = Structure.from_xyz(str(p))
    assert s2.n_atoms == s.n_atoms
    np.testing.assert_allclose(s2.positions, s.positions, atol=1e-4)
    assert s2.elements == list(s.elements)


def test_from_xyz_empty_raises():
    with pytest.raises(ValueError):
        Structure.from_xyz("")


def test_from_xyz_bad_header_raises():
    with pytest.raises(ValueError):
        Structure.from_xyz("not-a-number\ncomment\nH 0 0 0\n")


def test_from_xyz_short_atom_line_raises():
    with pytest.raises(ValueError):
        Structure.from_xyz("1\nshort\nH 0.0\n")


# --------------------------------------------------------------------- #
#  PDB                                                                  #
# --------------------------------------------------------------------- #


def test_from_pdb_text():
    """A PDB written by molbuilder must round-trip without losing the
    residue-level metadata that the writer puts there."""
    s = molbuilder.build_peptide("ARNDC")
    pdb = s.to_pdb()
    s2 = Structure.from_pdb(pdb)
    assert s2.n_atoms == s.n_atoms
    np.testing.assert_allclose(s2.positions, s.positions, atol=1e-3)
    assert s2.atom_names    == list(s.atom_names)
    assert s2.residue_ids   == list(s.residue_ids)
    assert s2.residue_names == list(s.residue_names)


def test_from_pdb_path(tmp_path):
    s = molbuilder.build_peptide("ARNDC")
    p = tmp_path / "pep.pdb"
    s.to_pdb(str(p))
    s2 = Structure.from_pdb(str(p))
    assert s2.n_atoms == s.n_atoms


def test_from_pdb_no_atoms_raises():
    with pytest.raises(ValueError):
        Structure.from_pdb("HEADER    something\nEND\n")


def test_from_pdb_first_model_only():
    """If multiple MODEL blocks are present, only the first is read."""
    pdb = (
        "MODEL        1\n"
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ENDMDL\n"
        "MODEL        2\n"
        "ATOM      2  C   ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ENDMDL\n"
    )
    s = Structure.from_pdb(pdb)
    assert s.n_atoms == 1
    assert s.elements == ["N"]


# --------------------------------------------------------------------- #
#  Top-level molbuilder.load                                            #
# --------------------------------------------------------------------- #


def test_load_dispatches_by_extension(tmp_path):
    s = molbuilder.build_peptide("AC")
    xyz_p = tmp_path / "x.xyz"
    pdb_p = tmp_path / "y.pdb"
    s.to_xyz(str(xyz_p))
    s.to_pdb(str(pdb_p))
    sx = molbuilder.load(str(xyz_p))
    sp = molbuilder.load(str(pdb_p))
    assert sx.n_atoms == s.n_atoms
    assert sp.n_atoms == s.n_atoms
    # PDB carries residue info; XYZ does not.
    assert sp.atom_names == list(s.atom_names)
    assert sx.atom_names == list(sx.elements)   # default-filled to elements


def test_load_unknown_extension_raises(tmp_path):
    p = tmp_path / "thing.txt"
    p.write_text("hello\n")
    with pytest.raises(ValueError):
        molbuilder.load(str(p))


# --------------------------------------------------------------------- #
#  Loaded structure -> SIESTA FDF                                       #
# --------------------------------------------------------------------- #


def test_loaded_structure_renders_fdf(tmp_path):
    """The whole point: load an existing file and feed it to render_fdf."""
    from molbuilder.siesta import SiestaConfig, convert
    s = molbuilder.build_peptide("AC")
    pdb_p = tmp_path / "ac.pdb"
    s.to_pdb(str(pdb_p))
    fdf_p = tmp_path / "ac.fdf"
    summary = convert(str(pdb_p), str(fdf_p),
                      SiestaConfig(verbose_comments=False, system_label="ac"))
    assert summary["n_atoms"] == s.n_atoms
    fdf_text = fdf_p.read_text()
    assert "%block AtomicCoordinatesAndAtomicSpecies" in fdf_text
    assert "SystemLabel       ac" in fdf_text
