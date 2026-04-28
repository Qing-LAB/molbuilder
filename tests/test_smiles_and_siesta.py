"""SMILES builder + siesta module smoke tests."""

from __future__ import annotations

import os

import pytest

import molbuilder
from molbuilder.siesta import SiestaConfig, render_fdf, convert


# --------------------------------------------------------------------- #
#  SMILES builder (RDKit-dependent)                                     #
# --------------------------------------------------------------------- #


def _smiles_or_skip(smiles: str, **kw):
    try:
        return molbuilder.build_from_smiles(smiles, **kw)
    except ImportError as e:
        pytest.skip(f"RDKit not installed: {e}")


def test_smiles_benzene_planarity():
    s = _smiles_or_skip("c1ccccc1", title="benzene")
    assert s.n_atoms == 12
    assert s.elements.count("C") == 6
    assert s.elements.count("H") == 6
    z_spread = s.positions[:, 2].max() - s.positions[:, 2].min()
    assert z_spread < 0.1, z_spread


def test_smiles_xyz_header_count():
    s = _smiles_or_skip("c1ccccc1")
    xyz = s.to_xyz()
    assert int(xyz.splitlines()[0]) == 12


def test_smiles_bdt_has_two_sulphurs():
    s2 = _smiles_or_skip("Sc1ccc(S)cc1", title="bdt")
    assert "S" in s2.elements
    assert s2.elements.count("S") == 2


# --------------------------------------------------------------------- #
#  siesta.render_fdf                                                    #
# --------------------------------------------------------------------- #


def test_render_fdf_dna_4mer():
    dna = molbuilder.build_dna("ATGC")
    cfg = SiestaConfig(system_name="test_dna", system_label="dna",
                       kgrid=(4, 4, 1), mesh_cutoff=350.0,
                       relax_type="none")
    fdf = render_fdf(dna, cfg)
    assert "SystemName        test_dna" in fdf
    assert "MeshCutoff 350.0 Ry" in fdf
    assert "%block kgrid_Monkhorst_Pack" in fdf
    assert "MD.TypeOfRun" not in fdf      # relax_type='none' must drop MD block
    # ChemicalSpeciesLabel for the elements DNA contains: C, H, N, O, P
    for el in ("C", "H", "N", "O", "P"):
        assert f" {el}\n" in fdf, el


def test_convert_xyz_to_fdf(tmp_path):
    """End-to-end XYZ -> FDF round-trip."""
    dna = molbuilder.build_dna("ATGC")
    xyz_path = tmp_path / "dna.xyz"
    fdf_path = tmp_path / "out" / "dna.fdf"
    dna.to_xyz(str(xyz_path))
    summary = convert(str(xyz_path), str(fdf_path),
                      SiestaConfig(system_label="dna4",
                                   kgrid=(2, 2, 1), relax_type="CG"))
    assert summary["n_atoms"] == dna.n_atoms
    assert os.path.isfile(str(fdf_path))
    text = fdf_path.read_text()
    assert "SystemLabel       dna4" in text
