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


# --------------------------------------------------------------------- #
#  Round-trip identity through writers / readers                        #
#                                                                        #
#  Loose round-trips (write to disk, read back) only cover I/O glue;    #
#  identity round-trips (back-construct a Structure from the writer's   #
#  output, compare element + position arrays) pin the contract that    #
#  XYZ / PDB writers don't drop or reorder data on a single hop.       #
# --------------------------------------------------------------------- #


def test_xyz_round_trip_identity(water_structure):
    s = water_structure
    s2 = Structure.from_xyz(s.to_xyz())
    assert s2.elements == s.elements
    np.testing.assert_allclose(s2.positions, s.positions, atol=1e-6)


def test_pdb_round_trip_identity(water_structure):
    """PDB carries more metadata than XYZ (atom_names / residue_names /
    chain_ids); the identity round-trip should preserve it all."""
    s = water_structure
    s2 = Structure.from_pdb(s.to_pdb())
    assert s2.elements == s.elements
    np.testing.assert_allclose(s2.positions, s.positions, atol=1e-3)
    # PDB metadata that XYZ would have dropped:
    assert s2.atom_names    == s.atom_names
    assert s2.residue_names == s.residue_names
    assert s2.chain_ids     == s.chain_ids


# --------------------------------------------------------------------- #
#  Transport-oriented metadata: regions + frozen_atoms                    #
#                                                                        #
#  Both are validated in __post_init__: indices must be in range,       #
#  region membership is NOT mutually exclusive (multi-label model);     #
#  frozen_atoms is normalised to sorted-unique.  These tests pin the    #
#  contract the transport pipeline relies on -- malformed input must    #
#  fail loudly at the Structure boundary, not silently propagate into   #
#  engine scripts.                                                       #
# --------------------------------------------------------------------- #


class TestStructureRegions:
    def test_default_is_empty(self, water_structure):
        assert water_structure.regions == {}

    def test_basic_assignment_round_trips(self):
        s = Structure(
            elements=["C", "C", "C", "C"],
            positions=np.zeros((4, 3)),
            regions={"L-electrode": [0, 1], "R-electrode": [3]},
        )
        assert s.regions == {"L-electrode": [0, 1], "R-electrode": [3]}

    def test_indices_are_normalised_to_sorted_unique(self):
        """The constructor should sort + dedupe per region so engines
        don't see duplicates or reverse-order lists from sloppy input."""
        s = Structure(
            elements=["C"] * 4,
            positions=np.zeros((4, 3)),
            regions={"bridge": [3, 1, 1, 0]},
        )
        assert s.regions["bridge"] == [0, 1, 3]

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            Structure(
                elements=["C", "C"], positions=np.zeros((2, 3)),
                regions={"L-electrode": [5]},
            )

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            Structure(
                elements=["C", "C"], positions=np.zeros((2, 3)),
                regions={"L-electrode": [-1]},
            )

    def test_atom_can_belong_to_multiple_regions(self):
        """Region membership is NOT mutually exclusive.  An atom may
        carry several labels (e.g. ``"L-electrode"`` + ``"interface"``);
        engines that need a disjoint partition enforce that
        separately at engine-load time."""
        s = Structure(
            elements=["C"] * 3, positions=np.zeros((3, 3)),
            regions={"L-electrode": [0, 1], "bridge": [1, 2]},
        )
        assert s.regions["L-electrode"] == [0, 1]
        assert s.regions["bridge"] == [1, 2]

    def test_empty_label_raises(self):
        with pytest.raises(ValueError, match="region label"):
            Structure(
                elements=["C"], positions=np.zeros((1, 3)),
                regions={"": [0]},
            )


class TestStructureFrozenAtoms:
    def test_default_is_empty(self, water_structure):
        assert water_structure.frozen_atoms == []

    def test_sorted_unique_normalisation(self):
        s = Structure(
            elements=["C"] * 4, positions=np.zeros((4, 3)),
            frozen_atoms=[3, 1, 1, 0],
        )
        assert s.frozen_atoms == [0, 1, 3]

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            Structure(
                elements=["C", "C"], positions=np.zeros((2, 3)),
                frozen_atoms=[7],
            )

    def test_can_overlap_a_region_atom(self):
        """A fixed atom MAY also be tagged as part of a region (typical
        for electrode buffer-layer atoms held in place during relax).
        The two lists are independent invariants."""
        s = Structure(
            elements=["C"] * 4, positions=np.zeros((4, 3)),
            regions={"L-electrode": [0, 1]},
            frozen_atoms=[0, 1],
        )
        assert s.regions["L-electrode"] == [0, 1]
        assert s.frozen_atoms == [0, 1]
