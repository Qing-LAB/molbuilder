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


# --------------------------------------------------------------------- #
#  Spin block (S2) — pin the keyword spelling so a SIESTA-version       #
#  regression at this layer fails loudly                                 #
# --------------------------------------------------------------------- #


def _h2_struct():
    """Single shared two-atom test structure."""
    import numpy as np
    from molbuilder.structure import Structure
    return Structure(
        elements=["H", "H"],
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        title="h2",
    )


def test_spin_polarized_emits_v5_keyword():
    """``cfg.spin_polarized=True`` MUST emit the v5 single-line
    ``Spin polarized`` form.  Without this line, SIESTA defaults to
    closed-shell DFT and any open-shell system (radical / transition
    metal / triplet) silently produces the wrong electronic state."""
    fdf = render_fdf(_h2_struct(), SiestaConfig(spin_polarized=True))
    assert "Spin polarized" in fdf
    # When spin_total is unset, neither constraint line is emitted.
    assert "Spin.Fix" not in fdf
    assert "Spin.Total" not in fdf


def test_spin_total_emits_constraint_pair():
    """``cfg.spin_total`` requires BOTH ``Spin.Fix .true.`` AND
    ``Spin.Total <v>`` -- without ``Spin.Fix`` the constraint is
    silently ignored by SIESTA, leaving multiplicity unconstrained."""
    fdf = render_fdf(
        _h2_struct(),
        SiestaConfig(spin_polarized=True, spin_total=2.0),
    )
    assert "Spin polarized" in fdf
    assert "Spin.Fix          .true." in fdf
    assert "Spin.Total        2.0" in fdf


def test_spin_total_ignored_without_polarization():
    """``spin_total`` set but ``spin_polarized=False`` -> nothing
    spin-related lands in the FDF."""
    fdf = render_fdf(
        _h2_struct(),
        SiestaConfig(spin_polarized=False, spin_total=2.0),
    )
    assert "Spin polarized" not in fdf
    assert "Spin.Fix" not in fdf
    assert "Spin.Total" not in fdf


def test_spin_total_zero_with_polarization_emits_constrained_singlet_note():
    """SP-A: ``spin_polarized=True`` AND ``spin_total=0.0`` produces a
    constrained singlet ON TOP of open-shell DFT.  This is unusual
    (the cheaper path is spin-restricted KS) and the verbose-mode FDF
    must surface a comment so a user who landed here by accident sees
    the contradiction."""
    fdf = render_fdf(
        _h2_struct(),
        SiestaConfig(spin_polarized=True, spin_total=0.0,
                     verbose_comments=True),
    )
    assert "constrained singlet" in fdf
    assert "Spin.Fix          .true." in fdf
    assert "Spin.Total        0.0"    in fdf


def test_spin_total_nonzero_does_not_emit_constrained_singlet_note():
    """SP-A negative case: a real open-shell run (spin_total>0) must
    NOT pick up the constrained-singlet note -- that note is reserved
    for the unusual zero case."""
    fdf = render_fdf(
        _h2_struct(),
        SiestaConfig(spin_polarized=True, spin_total=2.0,
                     verbose_comments=True),
    )
    assert "constrained singlet" not in fdf


def test_default_fdf_has_no_spin_block():
    """Default (closed-shell) FDF must not mention Spin at all -- the
    presence of any ``Spin`` keyword would force open-shell DFT."""
    fdf = render_fdf(_h2_struct(), SiestaConfig())
    assert "Spin polarized"      not in fdf
    assert "Spin.Fix"            not in fdf
    assert "Spin.Total"          not in fdf
    assert "SpinPolarized"       not in fdf  # v4 form not emitted either


# --------------------------------------------------------------------- #
#  Verlet/Nose dynamics (S1) — temperature/timestep are config-driven,  #
#  Nose gets MD.TargetTemperature so the thermostat target isn't 0 K  #
# --------------------------------------------------------------------- #


def test_verlet_uses_config_temperature_and_timestep():
    """Verlet dynamics emits MD.InitialTemperature and
    MD.LengthTimeStep from cfg fields (not hard-coded).  No
    MD.TargetTemperature for Verlet (NVE has no thermostat)."""
    cfg = SiestaConfig(
        relax_type="Verlet",
        md_initial_temperature=500.0,
        md_length_timestep=0.5,
    )
    fdf = render_fdf(_h2_struct(), cfg)
    assert "MD.InitialTemperature 500.0 K" in fdf
    assert "MD.LengthTimeStep 0.5 fs"      in fdf
    # Verlet is NVE: no thermostat target.
    assert "MD.TargetTemperature" not in fdf


def test_nose_emits_md_target_temperature_default_to_initial():
    """Nose-Hoover NVT MUST emit MD.TargetTemperature.  When
    md_target_temperature is None, fall back to md_initial_temperature
    so the thermostat target isn't 0 K (which would quench the run)."""
    cfg = SiestaConfig(
        relax_type="Nose",
        md_initial_temperature=400.0,
        md_target_temperature=None,
    )
    fdf = render_fdf(_h2_struct(), cfg)
    assert "MD.InitialTemperature 400.0 K" in fdf
    assert "MD.TargetTemperature  400.0 K" in fdf


def test_nose_target_temperature_explicit_override():
    """When md_target_temperature is set, it overrides initial."""
    cfg = SiestaConfig(
        relax_type="Nose",
        md_initial_temperature=400.0,
        md_target_temperature=298.15,
    )
    fdf = render_fdf(_h2_struct(), cfg)
    assert "MD.InitialTemperature 400.0 K"  in fdf
    assert "MD.TargetTemperature  298.15 K" in fdf


def test_cg_relax_does_not_emit_md_temperature_block():
    """CG (and Broyden / FIRE) relaxations don't need temperature or
    timestep -- they're not MD."""
    fdf = render_fdf(_h2_struct(), SiestaConfig(relax_type="CG"))
    assert "MD.InitialTemperature" not in fdf
    assert "MD.LengthTimeStep"     not in fdf
    assert "MD.TargetTemperature"  not in fdf

