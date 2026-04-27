"""Tests for Structure.from_xyz / from_pdb and molbuilder.load.

Verifies that:
  * a Structure round-trips through XYZ (lossless for elements + positions)
  * a Structure round-trips through PDB (lossless for atom names,
    residue ids, residue names, chain ids, in addition to positions)
  * the top-level molbuilder.load() detects format from extension
  * loaded structures feed render_fdf without further preparation
  * malformed inputs raise informative errors
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import molbuilder
from molbuilder.structure import Structure


# --------------------------------------------------------------------- #
#  XYZ                                                                  #
# --------------------------------------------------------------------- #


def test_from_xyz_text() -> None:
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


def test_from_xyz_path(tmp_path_str: str) -> None:
    s = molbuilder.build_peptide("ARNDC")
    p = os.path.join(tmp_path_str, "pep.xyz")
    s.to_xyz(p)
    s2 = Structure.from_xyz(p)
    assert s2.n_atoms == s.n_atoms
    np.testing.assert_allclose(s2.positions, s.positions, atol=1e-4)
    assert s2.elements == list(s.elements)


def test_from_xyz_empty_raises() -> None:
    try:
        Structure.from_xyz("")
    except ValueError:
        return
    assert False, "expected ValueError"


def test_from_xyz_bad_header_raises() -> None:
    try:
        Structure.from_xyz("not-a-number\ncomment\nH 0 0 0\n")
    except ValueError:
        return
    assert False, "expected ValueError"


def test_from_xyz_short_atom_line_raises() -> None:
    try:
        Structure.from_xyz("1\nshort\nH 0.0\n")
    except ValueError:
        return
    assert False, "expected ValueError"


# --------------------------------------------------------------------- #
#  PDB                                                                  #
# --------------------------------------------------------------------- #


def test_from_pdb_text(tmp_path_str: str) -> None:
    """A PDB written by molbuilder must round-trip without losing the
    residue-level metadata that the writer puts there."""
    s = molbuilder.build_peptide("ARNDC")
    pdb = s.to_pdb()
    s2 = Structure.from_pdb(pdb)
    assert s2.n_atoms == s.n_atoms
    np.testing.assert_allclose(s2.positions, s.positions, atol=1e-3)
    # Residue metadata should survive the round trip
    assert s2.atom_names == list(s.atom_names)
    assert s2.residue_ids == list(s.residue_ids)
    assert s2.residue_names == list(s.residue_names)


def test_from_pdb_path(tmp_path_str: str) -> None:
    s = molbuilder.build_peptide("ARNDC")
    p = os.path.join(tmp_path_str, "pep.pdb")
    s.to_pdb(p)
    s2 = Structure.from_pdb(p)
    assert s2.n_atoms == s.n_atoms


def test_from_pdb_no_atoms_raises() -> None:
    try:
        Structure.from_pdb("HEADER    something\nEND\n")
    except ValueError:
        return
    assert False, "expected ValueError"


def test_from_pdb_first_model_only() -> None:
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


def test_load_dispatches_by_extension(tmp_path_str: str) -> None:
    s = molbuilder.build_peptide("AC")
    xyz_p = os.path.join(tmp_path_str, "x.xyz")
    pdb_p = os.path.join(tmp_path_str, "y.pdb")
    s.to_xyz(xyz_p)
    s.to_pdb(pdb_p)
    sx = molbuilder.load(xyz_p)
    sp = molbuilder.load(pdb_p)
    assert sx.n_atoms == s.n_atoms
    assert sp.n_atoms == s.n_atoms
    # PDB carries residue info; XYZ does not.
    assert sp.atom_names == list(s.atom_names)
    assert sx.atom_names == list(sx.elements)   # default-filled to elements


def test_load_unknown_extension_raises(tmp_path_str: str) -> None:
    p = os.path.join(tmp_path_str, "thing.txt")
    with open(p, "w") as fh:
        fh.write("hello\n")
    try:
        molbuilder.load(p)
    except ValueError:
        return
    assert False, "expected ValueError for unknown extension"


# --------------------------------------------------------------------- #
#  Loaded structure -> SIESTA FDF                                       #
# --------------------------------------------------------------------- #


def test_loaded_structure_renders_fdf(tmp_path_str: str) -> None:
    """The whole point: load an existing file and feed it to render_fdf."""
    from molbuilder.siesta import Config, render_fdf, convert
    s = molbuilder.build_peptide("AC")
    pdb_p = os.path.join(tmp_path_str, "ac.pdb")
    s.to_pdb(pdb_p)
    fdf_p = os.path.join(tmp_path_str, "ac.fdf")
    summary = convert(pdb_p, fdf_p, Config(verbose_comments=False,
                                           system_label="ac"))
    assert summary["n_atoms"] == s.n_atoms
    fdf_text = open(fdf_p).read()
    assert "%block AtomicCoordinatesAndAtomicSpecies" in fdf_text
    assert "SystemLabel       ac" in fdf_text


# --------------------------------------------------------------------- #
#  Test runner with a simple tempdir fixture                            #
# --------------------------------------------------------------------- #


def main() -> None:
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in sorted(globals()):
            if not name.startswith("test_"):
                continue
            fn = globals()[name]
            try:
                if "tmp_path_str" in fn.__code__.co_varnames:
                    fn(tmp)
                else:
                    fn()
                print(f"  ok   {name}")
            except AssertionError as e:
                print(f"  FAIL {name}: {e}")
                failures.append(name)
            except Exception as e:
                print(f"  ERR  {name}: {type(e).__name__}: {e}")
                failures.append(name)
    if failures:
        sys.exit(f"FAILED: {failures}")
    print("OK -- load + from_xyz + from_pdb all pass.")


if __name__ == "__main__":
    main()
