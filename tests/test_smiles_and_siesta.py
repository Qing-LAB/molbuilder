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
    # 2026-05-27: system_name dropped from dataclass.  SystemName is
    # now driven by system_label (one job-name field).
    cfg = SiestaConfig(system_label="test_dna",
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
    """Size-only baseline (mpi_np unknown or 1).  The auto-picked
    BlockSize must satisfy ``BlockSize <= n_atoms`` (otherwise
    SIESTA's per-atom distribution pass hits propor IMAX=0 on
    multi-rank MPI runs)."""
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
    # Explicit mpi_np=1 must behave the same as None.  Single-process
    # has no rank-constraint to apply.
    assert _auto_block_size(81, None) == 8
    assert _auto_block_size(81, 1)    == 8
    # Invariant: BlockSize must never exceed n_atoms (the trigger
    # condition for `propor: ERROR: IMAX = 0`).
    for n in range(1, 64):
        assert _auto_block_size(n) <= n, n


def test_block_size_honours_mpi_rank_constraint():
    """With mpi_np set, BlockSize must satisfy ``BlockSize * mpi_np
    <= n_atoms`` so every rank gets >= 1 block.  Violating this is
    the 2026-05-28 hemeC-dithiol failure (81 atoms x 15 ranks ->
    BlockSize 8 left ranks 11-14 empty -> propor IMAX=0 abort)."""
    from molbuilder.siesta import _auto_block_size

    # The hemeC sighting itself.  Without rank-awareness the function
    # returned 8 (size-only baseline) which crashed the run.
    assert _auto_block_size(81, mpi_np=15) == 4

    # A few more cases sweeping the rank constraint.  With mpi_np
    # known there is NO artificial cap -- BlockSize scales up
    # naturally as the system + rank count grow (ScaLAPACK wants
    # bigger blocks for cache efficiency on big matrices).
    # 200 atoms / 16 ranks -> floor = 12, largest pow2 <= 12 is 8.
    assert _auto_block_size(200, mpi_np=16) == 8
    # 2000 atoms / 64 ranks -> floor = 31, largest pow2 <= 31 is 16.
    # (Before the cap-removal this was clamped to 8.)
    assert _auto_block_size(2000, mpi_np=64) == 16
    # 10000 atoms / 32 ranks -> floor = 312, largest pow2 <= 312 is
    # 256.  Big system + few ranks -> big BlockSize, as ScaLAPACK
    # wants.
    assert _auto_block_size(10000, mpi_np=32) == 256
    # 100 atoms / 16 ranks -> floor = 6, largest pow2 <= 6 is 4.
    assert _auto_block_size(100, mpi_np=16) == 4
    # 80 atoms / 32 ranks -> floor = 2, pow2 = 2.
    assert _auto_block_size(80, mpi_np=32) == 2
    # 20 atoms / 32 ranks (oversubscribed: rank > atoms) -> floor=0
    # -> cap=1 -> pow2 1.  No choice of BlockSize >= 1 can give
    # every rank a block here; the right user fix is lower mpi_np.
    assert _auto_block_size(20, mpi_np=32) == 1
    # 17 atoms / 4 ranks -> floor = 4, pow2 = 4.
    assert _auto_block_size(17, mpi_np=4) == 4
    # Universal invariant: with mpi_np set AND mpi_np <= n_atoms,
    # BlockSize * mpi_np must NEVER exceed n_atoms + (BlockSize - 1).
    # When mpi_np > n_atoms the run is OVERSUBSCRIBED -- no choice of
    # BlockSize >= 1 can give every rank a block (mathematically
    # impossible).  We floor at BlockSize=1 and let SIESTA report
    # propor IMAX=0 for the trailing ranks; the right user fix is to
    # lower mpi_np, not change BlockSize.
    for n in (5, 7, 11, 17, 19, 31, 47, 81, 199, 250):
        for r in (1, 2, 4, 7, 15, 16, 32, 64):
            bs = _auto_block_size(n, mpi_np=r)
            assert bs >= 1, f"BlockSize must be >=1, got {bs}"
            if 2 <= r <= n:
                assert bs * r <= n + (bs - 1), (
                    f"BlockSize={bs} x mpi_np={r} > n_atoms={n} "
                    f"-- last rank would be empty (propor IMAX=0)"
                )


def test_fdf_picks_safe_blocksize_for_hemec_case():
    """End-to-end: render an FDF for 81 atoms with mpi_np=15 (the
    exact 2026-05-28 hemeC-dithiol failure) and assert the emitted
    BlockSize is 4 (or smaller), not 8."""
    import re
    import numpy as np
    from molbuilder.structure import Structure
    # 81 atoms on a coarse 1.5-Å lattice so geometry preflight doesn't
    # complain about atom overlaps (BlockSize math is independent of
    # the positions; we only need n_atoms to be 81).
    side = int(np.ceil(81 ** (1 / 3)))      # 5 -> 5^3 = 125 cells, 81 used
    coords = np.array([
        [i * 1.5, j * 1.5, k * 1.5]
        for i in range(side) for j in range(side) for k in range(side)
    ])[:81]
    s = Structure(
        elements=["C"] * 81,
        positions=coords,
        title="hemeC-shaped",
    )
    cfg = SiestaConfig(mpi_np=15, relax_type="none")
    fdf = render_fdf(s, cfg)
    m = re.search(r"^BlockSize\s+(\d+)", fdf, re.MULTILINE)
    assert m, "FDF must carry an explicit BlockSize line"
    bs = int(m.group(1))
    # Strict: must be 4 (the rank-aware pick); a regression that
    # returns the old 8 would fail here.
    assert bs == 4, f"expected BlockSize=4 for 81 atoms x 15 ranks, got {bs}"
    # And the universal invariant: BlockSize x mpi_np <= n_atoms + slack.
    assert bs * 15 <= 81 + (bs - 1), (
        f"emitted BlockSize={bs} would leave trailing ranks empty"
    )


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


def test_spin_polarized_emits_v4_keyword_for_aux_compat():
    """``cfg.spin_polarized=True`` emits ``SpinPolarized .true.`` (v4
    form), NOT the v5 single-line ``Spin polarized``.  Reason: SIESTA
    5.4.2's v5 unified parser path does NOT subsequently read the
    auxiliary ``Spin.Fix`` / ``Spin.Total`` keys we depend on for
    open-shell metals (verified 2026-05-24 against the hemeC-dithiol
    failure: with ``Spin polarized``, both auxiliary keys are silently
    ignored and propor aborts; with ``SpinPolarized .true.`` both are
    honored).  The v4 form is marked deprecated in the v5 manual but
    is fully accepted in the parser."""
    fdf = render_fdf(_h2_struct(), SiestaConfig(spin_polarized=True))
    assert "SpinPolarized .true." in fdf
    # When spin_total is unset, neither constraint LINE is emitted
    # (the keywords may still appear in verbose-comments / template
    # banners; we only check the actual key-value emissions).
    assert "Spin.Fix          .true." not in fdf
    assert "Spin.Total       " not in fdf and "Spin.Total        " not in fdf


def test_spin_total_emits_constraint_pair():
    """``cfg.spin_total`` requires BOTH ``Spin.Fix .true.`` AND
    ``Spin.Total <v>`` -- without ``Spin.Fix`` the constraint is
    silently ignored by SIESTA, leaving multiplicity unconstrained.
    The leading ``SpinPolarized .true.`` is what TRIGGERS the parser
    to read the auxiliary keys at all (see preceding test)."""
    fdf = render_fdf(
        _h2_struct(),
        SiestaConfig(spin_polarized=True, spin_total=2.0),
    )
    assert "SpinPolarized .true." in fdf
    assert "Spin.Fix          .true." in fdf
    assert "Spin.Total        2.0" in fdf


def test_spin_total_ignored_without_polarization():
    """``spin_total`` set but ``spin_polarized=False`` -> nothing
    spin-related lands in the FDF."""
    fdf = render_fdf(
        _h2_struct(),
        SiestaConfig(spin_polarized=False, spin_total=2.0),
    )
    assert "SpinPolarized" not in fdf
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
    assert "SpinPolarized"       not in fdf
    assert "Spin polarized"      not in fdf
    assert "Spin.Fix"            not in fdf
    assert "Spin.Total"          not in fdf


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



# ---- Staged-relaxation suffix (job-layout v1) ---------------------------- #
#
# When ``cfg.stage`` is 1/2/3 the FDF's "Run with:" hint advertises a
# ``<basename>-stage<N>.fdf`` filename and the convert() preview-write
# uses ``<basename>-stage<N>.molwatch.log`` so multiple stages
# accumulate in one directory and the Watch tab's multi-stage merge
# picks them up automatically.


def test_fdf_stage_suffix_appears_in_run_with_block():
    fdf = render_fdf(_h2_struct(),
                     SiestaConfig(system_label="my-job", stage=2))
    assert "my-job-stage2.fdf" in fdf
    assert "my-job-stage2.out" in fdf
    assert "my-job-stage2.molwatch.log" in fdf
    assert "Stage 2" in fdf


def test_fdf_no_stage_suffix_when_stage_is_none():
    fdf = render_fdf(_h2_struct(),
                     SiestaConfig(system_label="my-job", stage=None))
    assert "my-job.fdf" in fdf
    assert "my-job.molwatch.log" in fdf
    assert "Stage 1" not in fdf and "Stage 2" not in fdf


def test_convert_writes_stage_suffixed_preview_log(tmp_path):
    """convert() drops a preview log next to the FDF; the filename
    follows the protocol basename + stage suffix, NOT the FDF stem."""
    import os
    in_p = tmp_path / "anything.xyz"
    in_p.write_text("2\nh2\nH 0 0 0\nH 0 0 0.74\n")
    fdf_p = tmp_path / "deliberately_unrelated_name.fdf"
    summary = convert(str(in_p), str(fdf_p),
                      SiestaConfig(system_label="my-job", stage=3))
    assert os.path.basename(summary["molwatch_log"]) \
        == "my-job-stage3.molwatch.log"


# --------------------------------------------------------------------- #
#  Frozen atoms -> Geometry.Constraints (three-stage contract)          #
#                                                                       #
#  Empirically validated 2026-05-25 against SIESTA 5.4.2: feeding the   #
#  emitted ``position N N N`` block makes SIESTA report                 #
#  ``siesta: Constraint (3): pos`` in its .out (one Constraint per      #
#  fixed-atom 3-tuple).  Without this block SIESTA's relaxer moves      #
#  every atom even when /modify froze them.                             #
# --------------------------------------------------------------------- #


def _struct_with_frozen(frozen):
    import numpy as np
    from molbuilder.structure import Structure
    return Structure(
        elements=["H", "O", "H", "C", "N"],
        positions=np.array([[i*1.5, 0, 0] for i in range(5)], dtype=float),
        frozen_atoms=list(frozen),
    )


def test_siesta_frozen_atoms_emit_geometry_constraints_block():
    """struct.frozen_atoms = [1, 4]  ->  ``position 2 5`` (1-based) inside
    a ``%block Geometry.Constraints`` block.  The block must appear AFTER
    %endblock AtomicCoordinatesAndAtomicSpecies (SIESTA reads sequentially)."""
    fdf = render_fdf(_struct_with_frozen([1, 4]),
                     SiestaConfig(verbose_comments=False))
    assert "%block Geometry.Constraints" in fdf
    assert "%endblock Geometry.Constraints" in fdf
    assert "position 2 5" in fdf
    coord_close = fdf.find("%endblock AtomicCoordinatesAndAtomicSpecies")
    constraint_open = fdf.find("%block Geometry.Constraints")
    assert 0 <= coord_close < constraint_open, (
        "Geometry.Constraints must come AFTER the coords block (SIESTA "
        "reads sequentially)"
    )


def test_siesta_no_frozen_atoms_emits_no_constraints_block():
    """Default Structure (no frozen atoms) -> no constraint block at all."""
    fdf = render_fdf(_struct_with_frozen([]),
                     SiestaConfig(verbose_comments=False))
    assert "%block Geometry.Constraints" not in fdf
    assert "position 2 5" not in fdf


def test_siesta_frozen_atoms_large_count_chunks_lines():
    """Many frozen indices -> emitted as multiple ``position`` lines for
    readability (single line gets unwieldy past ~20 atoms)."""
    frozen = list(range(50))   # 50 atoms.
    # Bump structure to have at least 50 atoms.
    import numpy as np
    from molbuilder.structure import Structure
    s = Structure(
        elements=["C"] * 51,
        positions=np.array([[i*1.5, 0, 0] for i in range(51)], dtype=float),
        frozen_atoms=frozen,
    )
    fdf = render_fdf(s, SiestaConfig(verbose_comments=False))
    # Count ``position`` lines inside the block.
    inside = False
    n = 0
    for line in fdf.splitlines():
        if "%block Geometry.Constraints" in line: inside = True; continue
        if "%endblock Geometry.Constraints" in line: inside = False; continue
        if inside and line.strip().startswith("position "):
            n += 1
    assert n >= 2, f"expected >= 2 ``position`` lines for 50 atoms; got {n}"


def test_pyscf_frozen_atoms_emit_constraints_file_and_optimize_kwarg():
    """struct.frozen_atoms -> writes <JOB>.constraints.txt at run time
    AND passes ``constraints=<path>`` to geometric.optimize().  Geometric
    parses the emitted ``xyz 2,5`` form (verified 2026-05-25 against
    geomeTRIC's prepare.parse_constraints in molbuilder-pySCF env)."""
    from molbuilder.pyscf import PySCFConfig, render_script
    script = render_script(
        _struct_with_frozen([1, 4]),
        PySCFConfig(verbose_comments=False, optimize=True,
                     optimizer="geometric", spin=1, method="UKS"),
    )
    # The runtime constraints-file emission.
    assert "_FROZEN_CONSTRAINTS_PATH" in script
    assert 'JOB + ".constraints.txt"' in script
    assert '"$freeze\\n"' in script
    # 1-based atom numbers in xyz line.
    assert "xyz 2,5" in script
    # The optimize() call gets the constraints= kwarg.
    assert "constraints           = _FROZEN_CONSTRAINTS_PATH" in script


def test_pyscf_no_frozen_atoms_no_constraints_emission():
    """Default Structure (no frozen atoms) -> no constraints code path."""
    from molbuilder.pyscf import PySCFConfig, render_script
    script = render_script(
        _struct_with_frozen([]),
        PySCFConfig(verbose_comments=False, optimize=True,
                     optimizer="geometric", spin=1, method="UKS"),
    )
    assert "_FROZEN_CONSTRAINTS_PATH" not in script
    assert '"$freeze' not in script
    assert "constraints           =" not in script


def test_pyscf_frozen_atoms_with_non_geometric_optimizer_emits_warning_comment():
    """If user picks berny (no constraint support in PySCF's berny API),
    DO NOT emit the constraints file -- emit a warning comment instead
    so the user sees their /modify freeze isn't honored."""
    from molbuilder.pyscf import PySCFConfig, render_script
    script = render_script(
        _struct_with_frozen([1, 4]),
        PySCFConfig(verbose_comments=False, optimize=True,
                     optimizer="berny", spin=1, method="UKS"),
    )
    assert "_FROZEN_CONSTRAINTS_PATH" not in script
    assert "WARNING" in script and "frozen_atoms" in script
    assert "geometric" in script  # the suggested fix
