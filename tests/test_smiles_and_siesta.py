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


def test_block_size_auto_pick_rule():
    """The auto-picked BlockSize must satisfy ``BlockSize <= n_atoms``
    (otherwise SIESTA's per-atom distribution pass hits propor IMAX=0
    on multi-rank MPI runs)."""
    from molbuilder.siesta import _auto_block_size
    # Known thresholds: each step at a power-of-2 boundary.
    assert _auto_block_size(2)  == 1
    assert _auto_block_size(3)  == 1
    assert _auto_block_size(4)  == 2
    assert _auto_block_size(7)  == 2
    assert _auto_block_size(8)  == 4
    assert _auto_block_size(15) == 4
    assert _auto_block_size(16) == 8
    assert _auto_block_size(50) == 8
    assert _auto_block_size(500) == 8
    # Invariant: BlockSize must never exceed n_atoms (the trigger
    # condition for `propor: ERROR: IMAX = 0`).
    for n in range(1, 64):
        assert _auto_block_size(n) <= n, n


def test_fdf_emits_explicit_blocksize_and_paralleloverk(tmp_path):
    """Generated FDF must always carry an explicit BlockSize and
    Diag.ParallelOverK -- relying on SIESTA defaults is non-portable
    (the defaults differ between 4.0 / 4.1 / MaX-1.x builds and have
    caused real `propor: IMAX = 0` failures)."""
    import numpy as np
    from molbuilder.structure import Structure
    s = Structure(
        elements=["H", "H"],
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        title="h2",
    )
    text = render_fdf(s, SiestaConfig(system_label="h2"))
    # Both lines must appear, regardless of system size.
    import re
    assert re.search(r"^BlockSize\s+\d+", text, re.MULTILINE)
    assert re.search(r"^Diag\.ParallelOverK\s+\.(true|false)\.",
                     text, re.MULTILINE)


def test_paralleloverk_auto_from_kgrid(tmp_path):
    """1x1x1 k-grid -> Diag.ParallelOverK .false. (parallelise the
    diagonaliser over orbitals).  Multi-k -> .true."""
    import numpy as np
    from molbuilder.structure import Structure
    s = Structure(
        elements=["H", "H"],
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        title="h2",
    )
    gamma = render_fdf(s, SiestaConfig(system_label="h2", kgrid=(1, 1, 1)))
    assert "Diag.ParallelOverK .false." in gamma
    multi = render_fdf(s, SiestaConfig(system_label="h2", kgrid=(4, 4, 4)))
    assert "Diag.ParallelOverK .true." in multi


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
