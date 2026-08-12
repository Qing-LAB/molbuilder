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


def test_explicit_blocksize_override_passes_through_verbatim():
    """User-set BlockSize is honored verbatim regardless of the
    BlockSize × mpi_np vs n_atoms ratio.

    2026-05-28 correction: previously this test asserted that the
    render path AUTO-DOWNGRADED an "unsafe" BlockSize to avoid
    propor IMAX = 0.  Direct empirical sweep (see /tmp/siesta-mpi-
    probe) disproved that theory -- BlockSize = 1, 2, 4 all crash
    at the same mpi_np, so BlockSize is NOT the relevant lever for
    the propor crash.  The auto-downgrade was based on the wrong
    theory and has been removed.  This test now pins the new
    invariant: user-set BlockSize comes through unchanged, no
    WARNING comment generated.

    Tunability for propor: use the wrapper's ``-np`` flag at run
    time.  The wrapper prints a focused diagnostic naming the safe
    retry values when propor fires.
    """
    import re
    import numpy as np
    from molbuilder.structure import Structure
    side = int(np.ceil(50 ** (1 / 3)))
    coords = np.array([
        [i * 1.5, j * 1.5, k * 1.5]
        for i in range(side) for j in range(side) for k in range(side)
    ])[:50]
    s = Structure(elements=["C"] * 50, positions=coords, title="x", vacuum=(12.0, 12.0, 12.0))
    cfg = SiestaConfig(parallel_block_size=32, mpi_np=4, relax_type="none")
    fdf = render_fdf(s, cfg)
    m = re.search(r"^BlockSize\s+(\d+)", fdf, re.MULTILINE)
    assert m, "FDF must carry an explicit BlockSize line"
    bs = int(m.group(1))
    assert bs == 32, (
        f"user-set BlockSize=32 must come through verbatim; got {bs}"
    )
    # The old auto-downgrade WARNING must NOT appear.
    assert "WARNING: user-set BlockSize" not in fdf
    assert "Downgraded to BlockSize" not in fdf


def test_explicit_blocksize_override_safe_value_passes_through():
    """Counter-case: user sets a SAFE explicit BlockSize -- must be
    honored verbatim, no downgrade, no warning."""
    import re
    import numpy as np
    from molbuilder.structure import Structure
    side = int(np.ceil(50 ** (1 / 3)))
    coords = np.array([
        [i * 1.5, j * 1.5, k * 1.5]
        for i in range(side) for j in range(side) for k in range(side)
    ])[:50]
    s = Structure(elements=["C"] * 50, positions=coords, title="x", vacuum=(12.0, 12.0, 12.0))
    cfg = SiestaConfig(parallel_block_size=4, mpi_np=4, relax_type="none")
    fdf = render_fdf(s, cfg)
    m = re.search(r"^BlockSize\s+(\d+)", fdf, re.MULTILINE)
    assert int(m.group(1)) == 4, "safe user override must be honored"
    assert "WARNING: user-set BlockSize" not in fdf


def test_fdf_charged_system_emits_makov_payne_notice():
    """2026-05-28: the FDF carries an inline Makov-Payne notice
    block near the ``NetCharge`` line whenever the resolved charge
    is non-zero.  Surfaces the image-charge artefact in the artifact
    the user is most likely to read.  No correction is applied (open
    capability; see design.md decisions log + validation pass)."""
    import numpy as np
    from molbuilder.structure import Structure
    s = Structure(elements=["O", "H"],
                  positions=np.array([[0, 0, 0], [0.96, 0, 0]]),
                  title="OH-", vacuum=(12.0, 12.0, 12.0))
    cfg = SiestaConfig(net_charge=-1, relax_type="none")
    fdf = render_fdf(s, cfg)
    # NetCharge line emitted as before.
    assert "NetCharge       -1" in fdf
    # Informational block: name the artefact, the formula, the
    # typical magnitude order.
    assert "Makov-Payne" in fdf
    assert "q^2" in fdf
    assert "0.5-1.5 eV" in fdf
    # Phase-after-Phase-6 (#172): the block now points the user at
    # the companion ``makov_payne_correction.py`` script the wrapper
    # emits.  Was previously a "do the arithmetic yourself" notice.
    assert "makov_payne_correction.py" in fdf
    assert "python3 makov_payne_correction.py" in fdf
    # Cites the paper.
    assert "Makov & Payne" in fdf


def test_fdf_neutral_system_no_makov_payne_block():
    """No NetCharge -> no Makov-Payne block.  The notice is gated on
    actual charged state, not always-on noise."""
    import numpy as np
    from molbuilder.structure import Structure
    s = Structure(elements=["O", "H", "H"],
                  positions=np.array([[0, 0, 0],
                                       [0.96, 0, 0],
                                       [-0.24, 0.93, 0]]),
                  title="H2O", vacuum=(12.0, 12.0, 12.0))
    cfg = SiestaConfig(relax_type="none")
    fdf = render_fdf(s, cfg)
    assert "NetCharge" not in fdf
    assert "Makov-Payne" not in fdf


def test_cell_volume_below_one_per_atom_raises():
    """The 2026-05-28 cell-volume sanity tightening: ``vol <
    n_atoms * 1.0 A^3`` raises with an actionable message naming
    the most common cause (unit confusion producing sub-Angstrom
    lattice vectors).

    Pre-2026-05-28 the threshold was a flat ``vol < 1.0 A^3`` --
    only fired on truly catastrophic input.  The per-atom floor
    scales with molecule size and catches realistic mistakes like
    a coplanar lattice vector or a typo in a single component.
    """
    import re
    import numpy as np
    from molbuilder.structure import Structure
    # 10 carbon atoms; needs cell volume >= 10 A^3 to pass.
    s = Structure(elements=["C"] * 10,
                  positions=np.zeros((10, 3)) + np.arange(10).reshape(10, 1) * 0.5,
                  title="dense", vacuum=(12.0, 12.0, 12.0))
    # Tiny cell: 1.5 A^3 cubic = 0.5 A side.  Below 10 A^3 threshold.
    tiny_cell = np.eye(3) * 1.5 ** (1.0 / 3.0)
    cfg = SiestaConfig(relax_type="none")
    with pytest.raises(ValueError) as exc_info:
        render_fdf(s, cfg, cell=tiny_cell)
    msg = str(exc_info.value)
    # Message must name the actual volume + the per-atom requirement.
    assert re.search(r"volume \d+\.\d+ A\^3", msg), (
        f"error message must quote the actual volume: {msg!r}"
    )
    assert "10 atom" in msg, (
        f"error message must name the atom count (10): {msg!r}"
    )
    assert "10.0 A^3" in msg or "= 1 A^3 per atom" in msg, (
        f"error message must name the per-atom requirement: {msg!r}"
    )
    # Diagnostic must mention the canonical "nm vs A" + "coplanar"
    # causes so the user knows where to look.
    assert ("nm" in msg or "coplanar" in msg), (
        f"error message must name likely causes: {msg!r}"
    )


def test_cell_volume_just_above_threshold_passes():
    """Boundary: vol slightly above n_atoms A^3 must pass (no
    false-positive on legit dense cells)."""
    import numpy as np
    from molbuilder.structure import Structure
    # 10 atoms in a packed arrangement; 12 A^3 cell (1.2x threshold).
    s = Structure(elements=["C"] * 10,
                  positions=np.linspace([0, 0, 0],
                                         [10, 10, 10], 10),
                  title="dense", vacuum=(12.0, 12.0, 12.0))
    # Cell with vol = 12 A^3 = (12)^(1/3) ~= 2.29 A side.
    edge = 12.0 ** (1.0 / 3.0)
    cell = np.eye(3) * edge
    cfg = SiestaConfig(relax_type="none",
                       wrap_into_cell=False)
    # render_fdf will issue downstream warnings (atom overlap, etc.),
    # but the volume gate itself must pass.
    try:
        fdf = render_fdf(s, cfg, cell=cell)
        assert "BlockSize" in fdf, "render must produce a real FDF"
    except ValueError as e:
        if "below the minimum physical volume" in str(e):
            raise AssertionError(
                f"vol = 12 A^3 (1.2x of 10-atom threshold) should "
                f"NOT trigger the volume gate; got: {e}"
            )
        # Other errors (overlap etc.) are not this test's concern.


def test_cell_volume_per_atom_threshold_scales_with_n_atoms():
    """The new threshold is per-atom: a cell that's fine for 1 atom
    must fail for many atoms in the same cell.  This is the
    physical-fit invariant: the cell must contain the atoms."""
    import numpy as np
    from molbuilder.structure import Structure
    # 100 atoms in a 50 A^3 cell -- way below 100 A^3 threshold.
    s = Structure(elements=["H"] * 100,
                  positions=np.linspace([0, 0, 0],
                                         [3, 3, 3], 100),
                  title="crowded", vacuum=(12.0, 12.0, 12.0))
    cell = np.eye(3) * 50.0 ** (1.0 / 3.0)   # vol = 50 A^3
    cfg = SiestaConfig(relax_type="none")
    with pytest.raises(ValueError, match="below the minimum"):
        render_fdf(s, cfg, cell=cell)
    # Same cell with 1 atom: still passes (1.0 A^3 floor for n=1).
    s_single = Structure(elements=["H"],
                         positions=np.array([[1.0, 1.0, 1.0]]),
                         title="single", vacuum=(12.0, 12.0, 12.0))
    cell_single = np.eye(3) * 2.0   # vol = 8 A^3 >> 1 A^3 (n=1)
    fdf = render_fdf(s_single, cfg, cell=cell_single)
    assert "BlockSize" in fdf


def test_wrap_into_cell_boundary_handling():
    """The 2026-05-28 audit found ``_wrap_into_cell`` did the opposite
    of its docstring: it wrapped frac=0.9999999999 to -1e-10 (outside
    cell, counted as moved) while leaving frac=-1e-10 at -1e-10
    (still outside, not counted).

    Contract pinned here:
      * Every atom lands in [0, 1) fractional after the wrap.
      * Atoms within 1e-9 of the cell boundary are NOT counted as
        moved (numerical noise, not a real translation).
      * Atoms genuinely outside the cell ARE counted as moved.
    """
    import numpy as np
    from molbuilder.siesta.input import _wrap_into_cell

    cell = np.eye(3) * 10.0     # 10 A cubic cell -- fractional = cart/10.

    # Case 1: atom at frac=0.9999999999 (in cell, noise near boundary).
    # Old code: wrapped to -1e-9 (OUT of cell), counted as moved.
    # New code: stays at 9.999999999 A, NOT counted.
    p = np.array([[9.999999999, 5.0, 5.0]])
    new, n = _wrap_into_cell(p, cell)
    assert n == 0, "frac~0.9999... is on-boundary noise, not a move"
    assert np.allclose(new, p), \
        f"on-boundary atom should keep original Cartesian, got {new}"

    # Case 2: atom at frac=-1e-11 (just outside in -x).
    # Old code: wrapped = -1e-11 (still outside), not counted.
    # New code: cleanly wraps to ~10 A (inside cell), boundary so
    # not counted as a real motion.
    p = np.array([[-1e-10, 5.0, 5.0]])
    new, n = _wrap_into_cell(p, cell)
    assert n == 0, "frac~-eps is on-boundary noise, not a move"
    # Cleanly inside [0, 10): the x coord should be in [0, 10).
    frac_x = new[0, 0] / 10.0
    assert 0.0 <= frac_x < 1.0, (
        f"after wrap atom must be in cell; got frac_x={frac_x}"
    )

    # Case 3: atom at frac=1.0 + 1e-11 (just outside in +x).
    # Old code: wrapped to ~+1e-11 (inside cell), but counted as
    # moved by ~1.0 (wrong -- it was boundary noise).
    # New code: cleanly wraps to ~0 + 1e-10 A, NOT counted.
    p = np.array([[10.0 + 1e-10, 5.0, 5.0]])
    new, n = _wrap_into_cell(p, cell)
    assert n == 0, "frac~1+eps is on-boundary noise, not a move"
    frac_x = new[0, 0] / 10.0
    assert 0.0 <= frac_x < 1.0, (
        f"after wrap atom must be in cell; got frac_x={frac_x}"
    )

    # Case 4: a GENUINE wrap (frac = 1.5).  Must wrap to 0.5 and be
    # counted.
    p = np.array([[15.0, 5.0, 5.0]])
    new, n = _wrap_into_cell(p, cell)
    assert n == 1, "genuine out-of-cell atom must be counted as moved"
    assert np.isclose(new[0, 0], 5.0), \
        f"frac=1.5 must wrap to frac=0.5 (cart=5.0); got {new[0,0]}"

    # Case 5: atom genuinely inside (frac=0.5) -- no motion, no count.
    p = np.array([[5.0, 5.0, 5.0]])
    new, n = _wrap_into_cell(p, cell)
    assert n == 0
    assert np.allclose(new, p)


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
        title="hemeC-shaped", vacuum=(12.0, 12.0, 12.0))
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
        title="h2", vacuum=(12.0, 12.0, 12.0))
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
        title="h2", vacuum=(12.0, 12.0, 12.0))
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
        title="h2", vacuum=(12.0, 12.0, 12.0))


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
# ``<label>_<NN>_<name>.fdf`` filename and the convert() preview-write
# uses ``<label>_<NN>_<name>.molwatch.log`` so multiple stages
# accumulate in one directory and the Watch tab's multi-stage merge
# picks them up automatically.


def test_fdf_stage_suffix_appears_in_run_with_block():
    # ``stage`` is the stage's ARTIFACT TOKEN since 2026-08-10, not an int:
    # ``-stage2`` was a hyphen (which § 6.3 reserves for a counter) plus a bare
    # position (which R5 forbids in a filename).
    fdf = render_fdf(_h2_struct(),
                     SiestaConfig(system_label="my-job", stage="02_medium"))
    assert "my-job_02_medium.fdf" in fdf
    assert "my-job_02_medium.out" in fdf
    assert "my-job_02_medium.molwatch.log" in fdf
    assert "Stage 02_medium" in fdf


def test_fdf_no_stage_suffix_when_stage_is_none():
    fdf = render_fdf(_h2_struct(),
                     SiestaConfig(system_label="my-job", stage=None))
    assert "my-job.fdf" in fdf
    assert "my-job.molwatch.log" in fdf
    assert "Stage 0" not in fdf


def test_convert_writes_stage_suffixed_preview_log(tmp_path):
    """convert() drops a preview log next to the FDF; the filename
    follows the protocol basename + stage suffix, NOT the FDF stem."""
    import os
    in_p = tmp_path / "anything.xyz"
    in_p.write_text("2\nh2\nH 0 0 0\nH 0 0 0.74\n")
    fdf_p = tmp_path / "deliberately_unrelated_name.fdf"
    summary = convert(str(in_p), str(fdf_p),
                      SiestaConfig(system_label="my-job", stage="03_tight"),
                      vacuum=(12.0, 12.0, 12.0))
    assert os.path.basename(summary["molwatch_log"]) \
        == "my-job_03_tight.molwatch.log"


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
        frozen_atoms=list(frozen), vacuum=(12.0, 12.0, 12.0))


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
        frozen_atoms=frozen, vacuum=(12.0, 12.0, 12.0))
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


# --------------------------------------------------------------------- #
#  SIESTA 5.4.2 universal MD keyword gate (2026-06-23 silent-failure)   #
# --------------------------------------------------------------------- #
#
# The 2026-06-23 incident: the generator emitted ``MD.NumBroydenSteps``
# / ``MD.NumFIRESteps`` + ``MD.MaxDispl`` when ``cfg.relax_type`` was
# Broyden or FIRE, assuming SIESTA had per-algorithm keyword aliases.
# SIESTA 5.4.2 does NOT recognize those names + emits NO warning -- a
# Broyden / FIRE relax silently fell back to Single-point calculation,
# wasting CPU on a 444-atom Au-junction run.
#
# Empirical proof (against SIESTA 5.4.2):
#   MD.TypeOfRun Broyden + MD.NumCGsteps N + MD.MaxCGDispl X Ang
#     -> redata: Dynamics option = Broyden coord. optimization
#     -> redata: Maximum number of optimization moves = N
#
# These tests pin the universal-keyword emission so the bug class
# cannot return.  Layer: L3 render-shape (no SIESTA binary needed);
# the L4 binary-smoke test in test_siesta_keyword_smoke.py exercises
# the same path end-to-end against SIESTA 5.4.2.


import pytest as _pytest_md


@_pytest_md.mark.parametrize("relax_type", ["CG", "Broyden", "FIRE"])
def test_relaxation_emits_universal_md_step_count(relax_type):
    """``MD.NumCGsteps`` is the universal SIESTA 5.4.2 step-count
    keyword across CG, Broyden, AND FIRE despite the CG-prefixed
    name.  The generator must emit it for every relax type; the
    phantom per-algorithm aliases must never be emitted (SIESTA
    drops them silently)."""
    from molbuilder.siesta import render_fdf, SiestaConfig
    from molbuilder.structure import Structure
    import numpy as _np_md
    s = Structure(
        elements=["H", "H"],
        positions=_np_md.array([[0, 0, 0], [0.74, 0, 0]]), vacuum=(12.0, 12.0, 12.0))
    cfg = SiestaConfig(relax_type=relax_type, relax_steps=42,
                       psml_lib=None)
    text = render_fdf(s, cfg)
    # The recognized keyword IS emitted with the user value.
    assert "MD.NumCGsteps 42" in text, (
        f"relax_type={relax_type}: generator must emit "
        f"MD.NumCGsteps (universal in SIESTA 5.4.2 -- silent "
        f"single-point fallback if missing)")
    # The phantom per-algorithm aliases are NOT emitted.  If either
    # of these reappears, SIESTA 5.4.2 will silently drop it and
    # the run will degenerate to Single-point -- exactly the
    # 2026-06-23 incident.
    assert "MD.NumBroydenSteps" not in text, (
        f"relax_type={relax_type}: MD.NumBroydenSteps is a phantom "
        f"keyword in SIESTA 5.4.2 (not recognized, silently dropped). "
        f"Use MD.NumCGsteps universally.")
    assert "MD.NumFIRESteps" not in text, (
        f"relax_type={relax_type}: MD.NumFIRESteps is a phantom "
        f"keyword in SIESTA 5.4.2 (not recognized, silently dropped). "
        f"Use MD.NumCGsteps universally.")


@_pytest_md.mark.parametrize("relax_type", ["CG", "Broyden", "FIRE"])
def test_relaxation_emits_universal_md_displ_cap(relax_type):
    """``MD.MaxCGDispl`` is the universal SIESTA 5.4.2 displacement-
    cap keyword across CG, Broyden, AND FIRE.  ``MD.MaxDispl`` is a
    real SIESTA keyword too BUT it does NOT control the Broyden
    integrator's per-step displacement -- it's silently misapplied
    (recognized + parsed but never used by Broyden).  Emit the
    universal MD.MaxCGDispl form only."""
    from molbuilder.siesta import render_fdf, SiestaConfig
    from molbuilder.structure import Structure
    import numpy as _np_md
    s = Structure(
        elements=["H", "H"],
        positions=_np_md.array([[0, 0, 0], [0.74, 0, 0]]), vacuum=(12.0, 12.0, 12.0))
    cfg = SiestaConfig(relax_type=relax_type, relax_max_displ=0.07,
                       psml_lib=None)
    text = render_fdf(s, cfg)
    assert "MD.MaxCGDispl 0.07 Ang" in text, (
        f"relax_type={relax_type}: generator must emit "
        f"MD.MaxCGDispl (universal in SIESTA 5.4.2; the only key "
        f"Broyden's integrator actually honors)")
    # Phantom keyword check: MD.MaxDispl is recognized by SIESTA but
    # silently mis-applied for Broyden; never emit it.
    assert "MD.MaxDispl" not in text, (
        f"relax_type={relax_type}: MD.MaxDispl is silently "
        f"mis-applied by Broyden in SIESTA 5.4.2 (recognized + "
        f"parsed but never used).  Emit MD.MaxCGDispl only.")


def test_savehs_keyword_emitted_always():
    """``SaveHS`` is the SIESTA 5.4.2 keyword to control H+S matrix
    writing to .HSX.  ``WriteHS`` (which we emitted pre-2026-06-23)
    is a phantom keyword silently dropped by SIESTA; the default-T
    behavior of SaveHS masked the bug whenever cfg.write_hs=True.

    Emit ``SaveHS`` UNCONDITIONALLY so the user's choice (T or F)
    lands in the .HSX file as intended + the fdf-echo proves it.
    """
    from molbuilder.siesta import render_fdf, SiestaConfig
    from molbuilder.structure import Structure
    import numpy as _np_md
    s = Structure(
        elements=["H", "H"],
        positions=_np_md.array([[0, 0, 0], [0.74, 0, 0]]), vacuum=(12.0, 12.0, 12.0))
    def _has_active_writehs(text):
        """Return True if any non-comment line carries a ``WriteHS``
        directive (i.e. an actual keyword emission, not a historical-
        context comment).  SIESTA's fdf comment char is ``#``."""
        import re as _re_md
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if _re_md.match(r"^\s*WriteHS\b", line):
                return True
        return False

    # write_hs=True: SaveHS .true. lands.
    text_on = render_fdf(s, SiestaConfig(write_hs=True, psml_lib=None))
    assert "SaveHS             .true." in text_on
    assert not _has_active_writehs(text_on), (
        "WriteHS is a phantom keyword in SIESTA 5.4.2 (silently "
        "dropped, no warning).  Emit SaveHS only as an ACTIVE "
        "directive; the keyword string may still appear in "
        "historical-context comments.")
    # write_hs=False: SaveHS .false. lands -- proves the override
    # actually reaches SIESTA (pre-fix this was silently no-op).
    text_off = render_fdf(s, SiestaConfig(write_hs=False, psml_lib=None))
    assert "SaveHS             .false." in text_off
    assert not _has_active_writehs(text_off)


# --------------------------------------------------------------------- #
#  SIESTA --stage {1,2,3} per-stage overlay (2026-06-23)                #
#                                                                       #
#  Minimum-viable per-stage defaults so the user can generate           #
#  stage1.fdf / stage2.fdf / stage3.fdf in one flag instead of four     #
#  separate --relax-* overrides.  The CLI overlay is the precursor to  #
#  the staged-opt ladder, which lives in task.json (not the config).    #
# --------------------------------------------------------------------- #


class TestSiestaStageOverlay:
    """Stage overlay: --stage {1,2,3} populates relax_type +
    relax_steps + relax_force_tol + relax_max_displ to tier-aligned
    values per docs/engines/tuning.md sect. 2.3.1.
    """

    def test_stage_1_is_cg_warmup(self):
        from molbuilder.config.siesta import SiestaConfig, apply_siesta_stage
        out = apply_siesta_stage(SiestaConfig(), 1)
        assert out.relax_type == "CG"
        assert out.relax_steps == 600
        assert out.relax_force_tol == 0.05
        assert out.relax_max_displ == 0.20

    def test_stage_2_is_broyden_publishable(self):
        from molbuilder.config.siesta import SiestaConfig, apply_siesta_stage
        out = apply_siesta_stage(SiestaConfig(), 2)
        # Broyden -- not CG -- is the load-bearing choice for stage 2+
        # (per docs/engines/tuning.md sect. 2.1 "stage 2
        # Broyden refine").  Pre-2026-06-23 the SIESTA generator
        # defaulted to CG and the user had to hand-edit stage2.fdf.
        assert out.relax_type == "Broyden"
        assert out.relax_steps == 200
        # Gaussian-OPT default (0.04 eV/A == "publishable").
        assert out.relax_force_tol == 0.04
        assert out.relax_max_displ == 0.05

    def test_stage_3_is_broyden_tight_crystal_practical(self):
        from molbuilder.config.siesta import SiestaConfig, apply_siesta_stage
        out = apply_siesta_stage(SiestaConfig(), 3)
        assert out.relax_type == "Broyden"
        assert out.relax_steps == 100
        # VASP EDIFFG=-0.01 standard (crystal/surface production).
        # NOT Gaussian GAU_TIGHT (0.001 eV/A) -- that's molecule-only
        # vib/IR territory; chases SCF noise on 100+ atom metals.
        assert out.relax_force_tol == 0.01
        assert out.relax_max_displ == 0.02

    def test_stage_overlay_preserves_other_user_choices(self):
        """The overlay is intentionally narrow: only the 4 tier
        knobs are overlaid.  basis_size, mesh_cutoff, psml_lib,
        spin, k-grid etc. ride through verbatim from the input cfg.
        Catches a regression that broadens the overlay to clobber
        user-tuned compute knobs."""
        from molbuilder.config.siesta import SiestaConfig, apply_siesta_stage
        cfg = SiestaConfig(
            basis_size="DZP",
            mesh_cutoff=500,
            kgrid=(4, 4, 1),
            spin_polarized=True,
        )
        out = apply_siesta_stage(cfg, 3)
        assert out.basis_size == "DZP"
        assert out.mesh_cutoff == 500
        assert out.kgrid == (4, 4, 1)
        assert out.spin_polarized is True
        # Stage 3 values overlaid as expected.
        assert out.relax_type == "Broyden"
        assert out.relax_force_tol == 0.01

    def test_unknown_stage_raises_value_error(self):
        from molbuilder.config.siesta import SiestaConfig, apply_siesta_stage
        import pytest as _p
        with _p.raises(ValueError, match="unknown SIESTA stage"):
            apply_siesta_stage(SiestaConfig(), 99)

    def test_siesta_stage_presets_match_optim_tuning_doc(self):
        """The SIESTA tier values are duplicated across three surfaces:

          (1) ``SIESTA_STAGE_PRESETS`` in molbuilder/config/siesta.py
          (2) The numeric assertions in the per-stage tests above
          (3) The doc table in docs/engines/tuning.md
              sect. 2.3 ("Per-tier" rows for MD.MaxForceTol + the
              displacement-cap subsection)

        PySCF has a parity test that catches drift between (1) and
        the wire fixture; SIESTA had no equivalent.  This test adds
        the missing gate: assert the constant values in (1) match
        the numeric claims in the doc (3).

        The reference values come straight from
        ``docs/engines/tuning.md`` sect. 2.3.1 (system-
        type-aware tier framework):

          - publishable: 0.04 eV/A force, 0.05 A displacement
          - tight (crystal/surface): 0.01 eV/A, 0.02 A
            (VASP EDIFFG=-0.01 standard; Au(111) production
            convention).

        Stage 1 (loose preopt) uses 0.05 eV/A + 0.20 A which is the
        SIESTA-default-MaxForceTol class anchored at sect. 2.3 row 2
        (loose preopt).

        A change to these values in either surface that is NOT
        applied to the other fails this test.  Per the doc-as-sole-
        source-of-truth rule (feedback_design_doc_first memory): if
        a value mismatch surfaces here, FIRST decide which is right
        per design considerations, THEN update the failing surface.
        """
        from molbuilder.config.siesta import SIESTA_STAGE_PRESETS

        # Stage 1: CG warm-up (loose preopt tier).
        s1 = SIESTA_STAGE_PRESETS[1]
        assert s1["relax_type"]      == "CG", (
            f"stage 1 must be CG per the recipe in doc 2.1; "
            f"got {s1['relax_type']!r}")
        assert s1["relax_force_tol"] == 0.05, (
            f"stage 1 force_tol must be 0.05 eV/A (loose preopt tier "
            f"per engines/tuning.md sect. 2.3); got {s1['relax_force_tol']}")
        assert s1["relax_max_displ"] == 0.20, (
            f"stage 1 max_displ must be 0.20 A (loose preopt per "
            f"engines/tuning.md sect. 2.2); got {s1['relax_max_displ']}")

        # Stage 2: Broyden publishable (Gaussian-OPT default).
        s2 = SIESTA_STAGE_PRESETS[2]
        assert s2["relax_type"]      == "Broyden", (
            f"stage 2 must be Broyden per the recipe in doc 2.1; "
            f"got {s2['relax_type']!r}")
        assert s2["relax_force_tol"] == 0.04, (
            f"stage 2 force_tol must be 0.04 eV/A (publishable per "
            f"engines/tuning.md sect. 2.3; Gaussian-OPT default); "
            f"got {s2['relax_force_tol']}")
        assert s2["relax_max_displ"] == 0.05, (
            f"stage 2 max_displ must be 0.05 A (publishable per "
            f"engines/tuning.md sect. 2.2); got {s2['relax_max_displ']}")

        # Stage 3: Broyden tight (crystal/surface production).
        s3 = SIESTA_STAGE_PRESETS[3]
        assert s3["relax_type"]      == "Broyden", (
            f"stage 3 must be Broyden (do NOT switch back to CG); "
            f"got {s3['relax_type']!r}")
        assert s3["relax_force_tol"] == 0.01, (
            f"stage 3 force_tol must be 0.01 eV/A (crystal/surface "
            f"production per engines/tuning.md sect. 2.3.1; matches "
            f"VASP EDIFFG=-0.01).  Pre-realignment was 0.001 eV/A "
            f"which is GAU_TIGHT and chases SCF noise on 100+ atom "
            f"metals.  Got {s3['relax_force_tol']}")
        assert s3["relax_max_displ"] == 0.02, (
            f"stage 3 max_displ must be 0.02 A (crystal-tight per "
            f"engines/tuning.md sect. 2.2); got {s3['relax_max_displ']}")

    # ``test_cli_stage_flag_emits_broyden_for_stage_2_and_3`` was RETIRED
    # 2026-08-11 with `molbuilder fdf` (C2), the only channel it drove.  It
    # gated three properties, and each has a home on the live path:
    #
    #   (a) the per-stage tier values (CG vs Broyden, force tol) -- the
    #       preset tables are asserted directly by the tests above, and the
    #       rendered rung by test_prep_calculation.py::
    #       test_the_named_stages_overrides_are_what_got_rendered;
    #   (b) the stage's artifact token reaching the artifact NAMES --
    #       test_trajectory_log_stage_targets.py (repointed the same day);
    #   (c) the token + `# Stage <token> --` header inside the RENDERED deck
    #       -- test_prep_calculation.py::
    #       test_the_deck_carries_its_stages_token_and_header, added with
    #       this retirement so the property never lost a producer-side gate.
