"""Structure.from_xyz / from_pdb and the one reading door, StructureCodec.

Verifies that:
  * Structure round-trips through XYZ (lossless for elements + positions)
  * Structure round-trips through PDB (lossless for atom names,
    residue ids, residue names, chain ids, plus positions)
  * StructureCodec dispatches by extension AND reads the sidecar pair
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


def test_the_door_dispatches_by_extension_and_reads_the_pair(tmp_path):
    """`StructureCodec` is the one reader, and it reads BOTH files.

    This tested `molbuilder.load()` until 2026-09-07.  That function read the
    geometry and not the `.molstruct.json` beside it, so everything it handed
    back was quietly smaller than what was on disk -- which is how `jobset
    init` came to write descriptions with the author's regions and frozen
    atoms missing.  It is deleted; the codec is the door.

    The pair half is asserted here rather than only the dispatch, because
    dispatch is what the old function got right.
    """
    from molbuilder.workingcopy_structure import StructureCodec

    water = Structure.from_xyz(
        "3\nwater-like\n"
        "O   0.000  0.000  0.000\n"
        "H   0.957  0.000  0.000\n"
        "H  -0.239  0.927  0.000\n")
    xyz_p = tmp_path / "s.xyz"
    water.to_xyz(str(xyz_p))
    pdb_p = tmp_path / "s.pdb"
    pdb_p.write_text(water.to_pdb())

    sx = StructureCodec().load(xyz_p)
    sp = StructureCodec().load(pdb_p)
    assert sx.n_atoms == 3 and sp.n_atoms == 3

    # ...and a sidecar beside the geometry comes back WITH it.
    marked = water
    marked.regions = {"frozen_atoms": [0], "L-electrode": [1, 2]}
    StructureCodec().write(marked, tmp_path / "pair.xyz")
    back = StructureCodec().load(tmp_path / "pair.xyz")
    assert back.regions == {"frozen_atoms": [0], "L-electrode": [1, 2]}, (
        "the sidecar beside the geometry was not applied -- this is the "
        "defect that deleting molbuilder.load() closed")
    assert back.frozen_atoms == [0]


def test_the_door_refuses_an_extension_it_does_not_know(tmp_path):
    from molbuilder.workingcopy_structure import StructureCodec
    p = tmp_path / "s.mol2"
    p.write_text("whatever")
    with pytest.raises(ValueError, match="unsupported structure format"):
        StructureCodec().load(p)


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
