"""SIESTA .fdf input generator.

Takes a Structure (or an XYZ/PDB file path) and emits a .fdf input file
ready to drop into a SIESTA run, with optional auto-copy of PSML
pseudopotentials from a flat library on disk.

Public API:
    SiestaConfig      -- dataclass holding every FDF parameter
    Config            -- backwards-compat alias for SiestaConfig
    render_fdf(...)   -- format an in-memory Structure as FDF text
    convert(...)      -- read XYZ/PDB, write FDF, optionally copy psml
    copy_pseudopotentials(...) -- standalone psml copy helper

The CLI lives in :mod:`molbuilder.cli` as the ``fdf`` subcommand.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from ase.data import atomic_numbers
    from ase.io import read as _ase_read
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "molbuilder.siesta needs ASE; install with `pip install ase`"
    ) from exc

from ..structure import Structure
# SiestaConfig is the L1 dataclass; this module imports it for use by
# the generator below.  External callers can import it from either
# molbuilder.config.siesta (the canonical location) or from
# molbuilder.siesta (re-exported by siesta/__init__.py).
from ..config.siesta import Config, SiestaConfig




# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _auto_block_size(n_atoms: int) -> int:
    """Pick a SIESTA ``BlockSize`` that's safe across typical 1-8 MPI
    rank counts for a structure of this many atoms.

    The constraint that triggers ``propor: ERROR: IMAX = 0`` is
    ``BlockSize > n_atoms`` -- when the BlockSize exceeds the atom
    count, SIESTA's per-atom distribution pass ends up with one
    partial block and 3+ ranks idle, and ``propor`` reports IMAX = 0.

    Returning a power of 2 that's at most ``n_atoms / 2`` guarantees
    at least 2 atom blocks exist, so even on 1 rank the count is
    well-defined.  Capped at 8 because larger values give negligible
    performance benefit on the structures molbuilder typically
    generates (peptides / DNA up to a few hundred atoms).
    """
    if n_atoms >= 16:  return 8
    if n_atoms >= 8:   return 4
    if n_atoms >= 4:   return 2
    return 1


def _detect_species(elements: Iterable[str]) -> List[str]:
    """Unique species, sorted by atomic number, preserving first-seen order
    only as a tiebreaker."""
    seen: List[str] = []
    for s in elements:
        if s not in seen:
            seen.append(s)
    return sorted(seen, key=lambda s: atomic_numbers[s])


def _wrap_into_cell(positions: np.ndarray, cell: np.ndarray
                    ) -> Tuple[np.ndarray, int]:
    """Fold atoms so their fractional coordinates sit in [0, 1).

    Treats ``cell`` as a (3, 3) matrix whose ROWS are the lattice
    vectors a, b, c -- i.e. a Cartesian position P satisfies
    ``P = (u, v, w) @ cell`` for some fractional triple (u, v, w).

    Returns ``(wrapped_positions, n_wrapped_atoms)`` so callers can
    print a useful note in the FDF.  Atoms whose fractional coordinates
    lie within a small tolerance of [0, 1] are NOT counted as wrapped
    (avoids spurious notices for atoms that happen to sit on the
    boundary).
    """
    inv = np.linalg.inv(cell)
    fractional = positions @ inv
    # Wrap into [0, 1) with %; numerical noise can produce fractional
    # very close to 1 (e.g. 0.9999999999), which we want to leave alone
    # rather than wrap to 0.
    wrapped = fractional - np.floor(fractional + 1e-9)
    # Count atoms that were genuinely outside the cell.
    moved_mask = np.any(np.abs(wrapped - fractional) > 1e-6, axis=1)
    n_moved = int(moved_mask.sum())
    new_positions = wrapped @ cell
    # For atoms that didn't move, keep the original Cartesian to
    # 1e-12 (avoid any matrix-product round-trip drift).
    new_positions[~moved_mask] = positions[~moved_mask]
    return new_positions, n_moved


def find_psml(element: str, lib: Path) -> Optional[Path]:
    """Locate a pseudopotential file for `element` in a flat lib folder."""
    for name in (f"{element}.psml", f"{element.lower()}.psml",
                 f"{element.upper()}.psml"):
        p = lib / name
        if p.is_file():
            return p
    matches = sorted(lib.glob(f"{element}*.psml"))
    if matches:
        if len(matches) > 1:
            print(f"  note: {len(matches)} variants of {element}.psml in "
                  f"{lib}; using {matches[0].name}", file=sys.stderr)
        return matches[0]
    return None


def copy_pseudopotentials(species: Sequence[str], lib: Path,
                          dest_dir: Path) -> List[str]:
    """Copy psml files for each species. Returns list of missing species."""
    missing: List[str] = []
    for s in species:
        src = find_psml(s, lib)
        if src is None:
            missing.append(s)
            print(f"  WARN: no psml file found for {s!r} in {lib}",
                  file=sys.stderr)
            continue
        dst = dest_dir / f"{s}.psml"
        if src.resolve() == dst.resolve():
            print(f"  ok:   {dst.name} (already present)", file=sys.stderr)
            continue
        shutil.copyfile(src, dst)
        print(f"  ok:   {src} -> {dst}", file=sys.stderr)
    return missing


# --------------------------------------------------------------------- #
#  FDF emitter                                                          #
# --------------------------------------------------------------------- #


def render_fdf(struct: Structure, config: Optional["SiestaConfig"] = None,
               *, cell: Optional[np.ndarray] = None) -> str:
    """Format a Structure as SIESTA .fdf text.

    If ``cell`` is None (default), an *orthorhombic* vacuum cell is
    generated, sized per-axis as ``extent[i] + 2 * cell_padding``, and
    the atom coordinates are translated so the structure sits centred
    in the box (``cell_padding`` of vacuum on every face).

    Pass an explicit ``(3, 3) cell`` (Angstrom, row vectors) to override
    -- in that case atom coordinates are passed through unchanged, since
    a user-supplied cell typically goes with a known atom frame
    (e.g. crystallographic positions).
    """
    cfg = config or SiestaConfig()
    species = (list(cfg.species_order) if cfg.species_order
               else _detect_species(struct.elements))
    species_index = {s: i + 1 for i, s in enumerate(species)}

    # ---------- net charge: explicit override or auto-detect ----------
    if cfg.net_charge is not None:
        auto_charge = int(cfg.net_charge)
        charge_source = "user-specified"
    else:
        from ..chemistry import formal_charge_from_phosphates
        auto_charge = formal_charge_from_phosphates(struct)
        charge_source = "auto (phosphate protonation)"

    # For charged systems in vacuum, the SIESTA-recommended padding is
    # >=25 A on each face to suppress image-image Coulomb interactions.
    # If the user is on the auto-vacuum path (cell=None) and hasn't
    # already bumped cell_padding past that, raise it for them.
    effective_padding = cfg.cell_padding
    auto_bumped_padding = False
    if cell is None and auto_charge != 0 and cfg.cell_padding < 25.0:
        effective_padding = 25.0
        auto_bumped_padding = True

    # Validate every element has a species index
    for el in struct.elements:
        if el not in species_index:
            raise ValueError(
                f"Atom element {el!r} not in --species-order "
                f"{list(species_index)!r}"
            )

    # Cell + atom positioning.
    #
    # Two cases:
    #
    #   (1) No cell provided -> molecule in vacuum.
    #       Build an orthorhombic box sized per-axis as
    #       (extent + 2 * cell_padding), then translate the structure
    #       so its bounding-box midpoint sits at the cell centre.
    #       Result: atom coordinates fall in [cell_padding,
    #       size - cell_padding] on every axis -- atoms are always
    #       inside the cell, no wrapping needed.
    #
    #   (2) Cell provided -> periodic system (slab, crystal, junction).
    #       Trust the cell, but check whether atoms fall inside.  If
    #       any are outside [0, 1) in fractional coordinates and
    #       `wrap_into_cell` is True (default), fold them back via
    #       fractional arithmetic.  This is what every PBC-aware
    #       structure tool does (ASE's `wrap`, VASP's POSCAR Direct,
    #       3DNA's fiber output, etc.) and avoids surprises in the
    #       SIESTA mesh and in the post-relaxation visualisation.
    positions = np.asarray(struct.positions, dtype=float)
    if cell is None:
        ext   = positions.max(axis=0) - positions.min(axis=0)
        sizes = ext + 2 * effective_padding
        cell  = np.diag(sizes)
        if cfg.center_in_vacuum:
            center_now    = (positions.max(axis=0) + positions.min(axis=0)) / 2.0
            center_target = sizes / 2.0
            positions = positions + (center_target - center_now)
        bump_note = (
            f"; padding auto-bumped from {cfg.cell_padding} -> "
            f"{effective_padding} A because NetCharge != 0"
            if auto_bumped_padding else ""
        )
        cell_note = (
            f"# (auto-generated orthorhombic vacuum cell "
            f"{sizes[0]:.2f} x {sizes[1]:.2f} x {sizes[2]:.2f} A; "
            f"cell_padding = {effective_padding} A on each face{bump_note}; "
            f"atoms centred)"
        )
    else:
        cell = np.asarray(cell, dtype=float).reshape(3, 3)
        # Sanity: positive cell volume
        vol = abs(float(np.linalg.det(cell)))
        if vol < 1.0:
            raise ValueError(
                f"Provided cell has near-zero volume ({vol:.3f} A^3); "
                f"check the lattice vectors."
            )
        if cfg.wrap_into_cell:
            positions, n_wrapped = _wrap_into_cell(positions, cell)
            cell_note = (
                "# (using user-supplied lattice;"
                + (f" {n_wrapped} atom(s) wrapped into the unit cell)"
                   if n_wrapped else " all atoms already inside the cell)")
            )
        else:
            cell_note = "# (using user-supplied lattice; wrap_into_cell=False)"

    # ---------- pre-emission validation ----------
    # By now `cell` and `positions` are final; run the validation pass
    # before any FDF text is generated so error-severity issues block
    # emission cleanly.  Warnings print to stderr but the run proceeds.
    # See molbuilder.validation and docs/design.md for the check list.
    from ..validation import validate, report
    validation_struct = Structure(
        elements      = list(struct.elements),
        positions     = positions,
        atom_names    = list(struct.atom_names),
        residue_ids   = list(struct.residue_ids),
        residue_names = list(struct.residue_names),
        chain_ids     = list(struct.chain_ids),
        title         = struct.title,
    )
    report(validate(validation_struct, cfg, cell=cell))

    out: List[str] = []

    out.append(f"SystemName        {cfg.system_name}")
    out.append(f"SystemLabel       {cfg.system_label}")
    out.append("")
    out.append(f"NumberOfAtoms     {struct.n_atoms}")
    out.append(f"NumberOfSpecies   {len(species)}")
    out.append("")

    # Lattice
    out.append("# --- Lattice ---")
    if cell_note:
        out.append(cell_note.rstrip())
    out.append("LatticeConstant 1.0 Ang")
    out.append("%block LatticeVectors")
    for v in cell:
        out.append(f"{v[0]:.12f} {v[1]:.12f} {v[2]:.12f}")
    out.append("%endblock LatticeVectors")
    out.append("")

    # Species
    out.append("# --- Species ---")
    out.append("%block ChemicalSpeciesLabel")
    for i, s in enumerate(species):
        out.append(f"{i + 1} {atomic_numbers[s]} {s}")
    out.append("%endblock ChemicalSpeciesLabel")
    out.append("")

    # Coordinates
    out.append("# --- Atomic coordinates ---")
    out.append("AtomicCoordinatesFormat Ang")
    out.append("%block AtomicCoordinatesAndAtomicSpecies")
    for el, (x, y, z) in zip(struct.elements, positions):
        out.append(f"{x:.10f} {y:.10f} {z:.10f} {species_index[el]}")
    out.append("%endblock AtomicCoordinatesAndAtomicSpecies")
    out.append("")

    v = cfg.verbose_comments

    # Basis & grid
    out.append("# --- Basis & grid ---")
    if v: out += [
        "# MeshCutoff: real-space mesh cutoff (Ry).  Range 200-500.",
        "#   Higher = more accurate forces, slower (cost cubic in cutoff).",
        "#   Increase if forces look noisy or your pseudo has a hard core",
        "#   (transition metals, oxides typically need 400+).",
    ]
    out.append(f"MeshCutoff {cfg.mesh_cutoff} Ry")
    if v: out += [
        "",
        "# PAO.BasisSize: orbital basis quality (cheap -> expensive)",
        "#   SZ    minimal -- screening only",
        "#   SZP   single-zeta + polarization",
        "#   DZ    double-zeta",
        "#   DZP   double-zeta + polarization  (recommended for production)",
        "#   TZP   triple-zeta + polarization  (accurate, ~2x slower)",
    ]
    out.append(f"PAO.BasisSize {cfg.basis_size}")
    if v: out += [
        "",
        "# PAO.EnergyShift: how diffuse the PAO orbitals are (Ry).",
        "# Range 0.001 - 0.05.  Smaller = more diffuse + accurate + slower.",
        "#   0.02 Ry      typical production value",
        "#   0.001 Ry     accuracy-critical (band gaps, weak interactions)",
        "#   0.05 Ry      fast screening only",
    ]
    out.append(f"PAO.EnergyShift {cfg.pao_energy_shift} Ry")
    out.append("")

    # XC
    out.append("# --- Exchange-correlation ---")
    if v: out += [
        "# XC.functional: GGA (recommended for most systems),",
        "#                LDA (faster, underestimates band gaps),",
        "#                VDW (dispersion-dominated systems),",
        "#                HYB (hybrid -- much more expensive).",
        "# XC.authors:    PBE (standard GGA),  BLYP, PW92 (LDA),",
        "#                DRSLL / KBM (vdW),  HSE06 (hybrid).",
    ]
    out.append(f"XC.functional {cfg.xc_functional}")
    out.append(f"XC.authors    {cfg.xc_authors}")
    out.append("")

    # SCF
    out.append("# --- SCF ---")
    if v: out += [
        "# SolutionMethod:  diagon       standard diagonalisation, O(N^3)",
        "#                  OMM          order-N, for systems > 500 atoms",
        "#                  transiesta   non-equilibrium transport",
    ]
    out.append(f"SolutionMethod    {cfg.solution_method}")

    if v: out += [
        "",
        "# DM.MixingWeight: density-matrix mixing weight (0.001 - 0.5).",
        "#   Smaller = more conservative, stable, slower.",
        "#   Larger  = aggressive, may oscillate.",
        "# Tuning hints:",
        "#   - SCF oscillating?     reduce to 0.005",
        "#   - SCF stalled?         increase or add Pulay history",
        "#   - Metals:              0.005 - 0.02",
        "#   - Insulators:          0.05 - 0.10 is often fine",
    ]
    out.append(f"DM.MixingWeight   {cfg.mixing_weight}")

    if v: out += [
        "",
        "# DM.NumberPulay: # of past SCF iterations kept for Pulay mixing.",
        "# Range 2-10.  More = better convergence + more memory.",
        "#   3      fine for most cases",
        "#   5-8    hard cases (metals, magnetic systems)",
    ]
    out.append(f"DM.NumberPulay    {cfg.pulay_history}")

    if v: out += [
        "",
        "# DM.Tolerance: density-matrix convergence threshold.",
        "# Range 1e-6 - 1e-4.  Tighter = more accurate + slower.",
        "#   1e-5    standard for relaxation",
        "#   1e-6    band structure / accurate forces",
        "#   1e-4    quick screening",
    ]
    out.append(f"DM.Tolerance      {cfg.dm_tolerance:.0e}")

    if v: out += [
        "",
        "# DM.Energy.Tolerance: redundant energy-based SCF check (eV).",
        "# Catches the rare case where DM is converged but energy keeps",
        "# drifting -- usually triggered by ill-conditioned mixing.",
    ]
    out.append(f"DM.Energy.Tolerance {cfg.dm_energy_tolerance:.0e} eV")

    if v: out += [
        "",
        "# MaxSCFIterations: SCF iteration cap.  500 is generous for the",
        "# first geometry; well-mixed systems converge in 30-100.",
    ]
    out.append(f"MaxSCFIterations  {cfg.max_scf_iter}")

    if v: out += [
        "",
        "# ElectronicTemperature: Fermi-Dirac smearing temperature.",
        "#   25 K     2 meV  -- molecular / cold properties",
        "#   300 K    25 meV -- room temperature default (ok for most)",
        "#   1000-2000 K     metals; helps SCF convergence",
        "#   < 100 K  for very accurate band-edge properties",
    ]
    out.append(f"ElectronicTemperature {cfg.electronic_temperature} K")

    if cfg.use_save_dm:
        if v: out += [
            "",
            "# DM.UseSaveDM: read .DM from previous run if present.  Free",
            "# warm-start; SIESTA silently ignores if no file exists.",
        ]
        out.append("DM.UseSaveDM      true")

    # ---- Spin polarisation ---------------------------------------
    # SIESTA's default is spin-restricted (no SpinPolarized line ->
    # closed-shell DFT).  For radicals, transition metals, or any
    # open-shell system the user MUST set spin_polarized=True or the
    # electronic structure is silently wrong.  When set, we emit
    # SpinPolarized true and (optionally) SpinTotal so SIESTA's
    # initial guess targets the correct multiplicity.
    if cfg.spin_polarized:
        if v: out += [
            "",
            "# SpinPolarized: open-shell DFT (collinear).  Required for",
            "# any system with unpaired electrons.  SIESTA's default is",
            "# closed-shell -- omitting this for a radical / transition-",
            "# metal / triplet system gives the wrong electronic state.",
        ]
        out.append("SpinPolarized     true")
        if cfg.spin_total is not None:
            if v: out += [
                "# SpinTotal: target total spin moment in mu_B (= number",
                "# of unpaired electrons).  Helps SIESTA's initial guess",
                "# converge to the right multiplicity; without it SIESTA",
                "# may settle into a wrong spin state.",
            ]
            out.append(f"SpinTotal         {cfg.spin_total}")

    # ---- NetCharge -----------------------------------------------
    # Either user-specified (cfg.net_charge != None) or auto-detected
    # from phosphate protonation state.  SIESTA defaults to neutral and
    # silently adds compensating electrons; we MUST set NetCharge for
    # any non-zero charge or the electronic structure is wrong.
    if auto_charge != 0:
        if v: out += [
            "",
            f"# NetCharge: {auto_charge:+d} ({charge_source}).",
            "# Note: SIESTA adds a uniform compensating background charge",
            "# for periodic-cell consistency.  For vacuum calcs of charged",
            "# molecules use cell_padding >= 25 A (we already auto-bump to",
            "# 25 A in the auto-vacuum case) to suppress image-image",
            "# Coulomb interactions.  To make a neutral system instead,",
            "# either build with protonate_phosphates=True or pass a",
            "# Config(net_charge=0) override.",
        ]
        out.append(f"NetCharge       {auto_charge:+d}")
    out.append("")

    # k-grid
    kx, ky, kz = cfg.kgrid
    out.append(f"# --- k-points ({kx}x{ky}x{kz}) ---")
    if v: out += [
        "# Monkhorst-Pack mesh.  Cost scales linearly with # of k-points.",
        "#   1x1x1               vacuum / molecule (only Gamma matters)",
        "#   4x4x4 to 8x8x8      periodic 3D crystals",
        "#   kx x ky x 1         2D slabs (no k along the vacuum direction)",
        "# Convergence test: rerun with 1.5x density on each axis -> total",
        "# energy should change < 1 meV/atom.",
    ]
    out.append("%block kgrid_Monkhorst_Pack")
    out.append(f"{kx} 0 0 0.0")
    out.append(f"0 {ky} 0 0.0")
    out.append(f"0 0 {kz} 0.0")
    out.append("%endblock kgrid_Monkhorst_Pack")
    out.append("")

    # ---- Parallel execution (MPI) -------------------------------
    # Always emit explicit values so the FDF behaves the same across
    # SIESTA versions and across system sizes.  Without an explicit
    # `BlockSize`, SIESTA auto-picks one as `ceil(Norb / Nrank)`,
    # which is fine for distributing the orbital matrix but is
    # *also* used elsewhere in the pipeline -- including a per-atom
    # distribution step right before mesh setup.  When the system
    # is small (e.g., a 14-atom molecule -> Norb~134, auto-picked
    # BlockSize=34 with 4 MPI ranks), the auto value is much larger
    # than the atom count, and that earlier step bails with
    #     propor: ERROR: IMAX = 0
    # An explicit smaller BlockSize keeps every distribution step
    # well-conditioned regardless of system size.
    out.append("# --- Parallel execution (MPI) ---")
    if v: out += [
        "# These settings matter only with `mpirun -np N siesta`",
        "# (single-rank runs ignore them).",
        "#",
        "# BlockSize: global block size used for ScaLAPACK orbital",
        "# distribution AND for several per-atom / per-projector",
        "# distribution passes earlier in the pipeline.  Without it",
        "# SIESTA auto-picks ceil(Norb / Nrank), which works for the",
        "# orbital matrix but can be too large for the small per-atom",
        "# passes on small molecules, giving:",
        "#     propor: ERROR: IMAX = 0",
        "# molbuilder picks a power-of-2 BlockSize from n_atoms:",
        "#   n_atoms >= 16  ->  BlockSize 8   (typical molecules)",
        "#   n_atoms >=  8  ->  BlockSize 4",
        "#   n_atoms >=  4  ->  BlockSize 2",
        "#   smaller        ->  BlockSize 1",
        "# This is conservative and rank-count-agnostic.  For >1000-",
        "# atom systems on >=16 MPI ranks, raising to 16 or 32 helps",
        "# ScaLAPACK efficiency by a few percent (override via",
        "# cfg.parallel_block_size = 16).",
        "#",
        "# Diag.ParallelOverK: parallelise the diagonaliser over",
        "# k-points (.true.) or over orbitals (.false.).  Auto-",
        "# selected here from the kgrid above: .false. for 1x1x1",
        "# (molecule / vacuum), .true. for multi-k periodic runs.",
    ]
    if cfg.parallel_block_size is None:
        block_size = _auto_block_size(struct.n_atoms)
    else:
        block_size = int(cfg.parallel_block_size)
    out.append(f"BlockSize          {block_size}")
    if cfg.parallel_over_k is None:
        over_k = (kx, ky, kz) != (1, 1, 1)
    else:
        over_k = bool(cfg.parallel_over_k)
    out.append(f"Diag.ParallelOverK {'.true.' if over_k else '.false.'}")
    out.append("")

    # Relaxation
    if cfg.relax_type and cfg.relax_type.lower() != "none":
        out.append("# --- Geometry optimisation ---")
        if v: out += [
            "# MD.TypeOfRun:  CG       Conjugate Gradients (robust default)",
            "#                Broyden  often faster on flat surfaces",
            "#                FIRE     fast for big systems",
            "#                MD       real molecular dynamics (Nose-Hoover etc.)",
        ]
        out.append(f"MD.TypeOfRun {cfg.relax_type}")
        if v: out += [
            "",
            "# MD.NumCGsteps: maximum # of relaxation steps.  Typical run",
            "# converges in 30-150; 200+ is a safety cap.",
        ]
        out.append(f"MD.NumCGsteps {cfg.relax_steps}")
        if v: out += [
            "",
            "# MD.MaxForceTol: convergence threshold on max atomic force.",
            "# Range 0.005 - 0.05 eV/Ang.",
            "#   0.04    SIESTA default (loose)",
            "#   0.02    typical production",
            "#   0.01    accurate properties",
            "#   0.005   vibrational analysis / phonons",
        ]
        out.append(f"MD.MaxForceTol {cfg.relax_force_tol} eV/Ang")
        if v: out += [
            "",
            "# MD.MaxCGDispl: maximum atom displacement per step (Ang).",
            "# Smaller = cautious + stable.  0.05 is safe for nearly-converged",
            "# structures; 0.10-0.20 is fine far from the minimum.",
        ]
        out.append(f"MD.MaxCGDispl {cfg.relax_max_displ} Ang")
        if cfg.use_save_cg or cfg.use_save_xv:
            if v: out += [
                "",
                "# MD.UseSaveCG / UseSaveXV: read .CG / .XV from previous run",
                "# if present.  Restart-friendly for long jobs.",
            ]
            if cfg.use_save_cg: out.append("MD.UseSaveCG      true")
            if cfg.use_save_xv: out.append("MD.UseSaveXV      true")
        out.append("")

    # Output
    out.append("# --- Output ---")
    if v: out += [
        "# WriteForces      forces in .FA (required for relaxation)",
        "# WriteCoorStep    coords at every MD step in main .out",
        "# WriteCoorXmol    .xyz at every step (movie viewer)",
        "# WriteMDhistory   trajectory to .ANI (xcrysden / vmd / OVITO)",
        "# WriteHS          H + S matrices, needed for TranSIESTA / DOS",
    ]
    out.append(f"WriteForces        {'true' if cfg.write_forces else 'false'}")
    out.append(f"WriteCoorStep      {'true' if cfg.write_coor_step else 'false'}")
    out.append(f"WriteCoorXmol      {'true' if cfg.write_coor_xmol else 'false'}")
    out.append(f"WriteMDhistory     {'true' if cfg.write_md_history else 'false'}")
    if cfg.write_hs:
        out.append("WriteHS            true")

    # Troubleshooting block at the end (verbose mode only).  We only
    # emit the relaxation-specific tips when an MD block is actually
    # being written -- otherwise mentioning `MD.TypeOfRun` here would
    # surprise downstream code that scans the FDF.
    if v:
        out += [
            "",
            "# ============================================================",
            "# TROUBLESHOOTING / TUNING HINTS                              ",
            "# ============================================================",
            "#",
            "# SCF doesn't converge:",
            "#   * lower DM.MixingWeight to 0.005",
            "#   * increase DM.NumberPulay to 5-8",
            "#   * raise ElectronicTemperature to 1000-2000 K (metals)",
            "#   * verify all .psml pseudopotentials are in this directory",
            "#",
            "# Forces look noisy / break symmetry:",
            "#   * raise MeshCutoff to 400-500 Ry",
            "#   * tighten DM.Tolerance to 1e-6",
            "#   * smaller PAO.EnergyShift (0.005 Ry, more diffuse basis)",
            "#",
            "# Calculation is too slow:",
            "#   * try PAO.BasisSize SZP for screening runs",
            "#   * reduce MeshCutoff to 200 Ry  (forces ~0.05 eV/Ang noisier)",
            "#   * reduce k-grid (periodic systems)",
            "#   * SolutionMethod OMM for >500 atoms",
            "#",
            "# Energy fluctuates during SCF:",
            "#   * lower DM.MixingWeight to 0.005",
            "#   * raise DM.NumberPulay to 6",
            "#   * tighten DM.Energy.Tolerance to 1e-5 eV",
            "#",
            "# 'propor: ERROR: IMAX = 0' on parallel run:",
            "#   * caused by SIESTA's auto-picked BlockSize (which",
            "#     scales with Norb / Nrank) being too large for an",
            "#     unrelated per-atom distribution step earlier in",
            "#     the pipeline.  Setting BlockSize explicitly above",
            "#     overrides the auto-choice -- if it still fails,",
            "#     drop BlockSize to 4 or 1.",
            "#   * for 1x1x1 (Gamma) k-grids, also confirm that",
            "#     Diag.ParallelOverK is .false. (we set it above",
            "#     based on the kgrid).",
            "#   * single-rank `siesta` (no mpirun) is often fastest",
            "#     for molecules under ~50 atoms anyway, since",
            "#     ScaLAPACK overhead dominates.",
        ]
        if cfg.relax_type and cfg.relax_type.lower() != "none":
            out += [
                "#",
                "# Relaxation oscillates near minimum:",
                "#   * shrink MD.MaxCGDispl to 0.02 Ang",
                "#   * loosen MD.MaxForceTol to 0.04 eV/Ang",
                "#   * switch MD.TypeOfRun to Broyden (often robust on",
                "#     flat regions) or FIRE (better for >100 atoms)",
            ]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------- #
#  File -> (Structure, cell) loader                                     #
# --------------------------------------------------------------------- #


def _struct_from_file(path: str) -> Tuple[Structure, Optional[np.ndarray]]:
    """Read an XYZ or PDB and return ``(Structure, cell_or_None)``.

    Format is detected from the file extension.  XYZ files may carry
    a periodic cell in the comment line (ASE's extended XYZ format);
    if present, it is returned alongside the structure so the caller
    can preserve it in the FDF.  PDB has no native cell concept here,
    so the cell is always ``None``.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdb":
        return Structure.from_pdb(p), None
    if ext in (".xyz", ""):
        # Use ASE for XYZ -- it understands extended-XYZ headers and
        # gives us the lattice when present, which our hand-rolled
        # parser doesn't.
        atoms = _ase_read(path)
        elements = [a.symbol for a in atoms]
        positions = atoms.get_positions()
        cell = atoms.cell.array if hasattr(atoms.cell, "array") else atoms.cell
        cell = np.asarray(cell, dtype=float)
        return (
            Structure(elements=elements, positions=positions, title=p.stem),
            cell if cell.any() else None,
        )
    raise ValueError(
        f"unsupported input extension {ext!r}; expected .xyz or .pdb"
    )


# Kept for backwards compatibility with code that imported _struct_from_xyz
_struct_from_xyz = _struct_from_file


def convert(
    input_path: str,
    fdf_path: str,
    config: Optional["SiestaConfig"] = None,
) -> dict:
    """Read an XYZ or PDB file, write an FDF, optionally copy psml files.

    Returns a summary dict with keys: ``fdf``, ``n_atoms``, ``species``,
    ``missing_psml``.
    """
    cfg = config or SiestaConfig()
    struct, cell = _struct_from_file(input_path)

    species = (list(cfg.species_order) if cfg.species_order
               else _detect_species(struct.elements))
    fdf_text = render_fdf(struct, cfg, cell=cell)

    fdf_p = Path(fdf_path)
    fdf_p.parent.mkdir(parents=True, exist_ok=True)
    fdf_p.write_text(fdf_text)

    summary = {
        "fdf": str(fdf_p),
        "n_atoms": struct.n_atoms,
        "species": species,
        "missing_psml": [],
    }

    if cfg.psml_lib and cfg.copy_psml:
        lib = Path(cfg.psml_lib).expanduser()
        if not lib.is_dir():
            print(f"  WARN: --psml-lib {lib} is not a directory; skipping psml copy",
                  file=sys.stderr)
        else:
            summary["missing_psml"] = copy_pseudopotentials(species, lib, fdf_p.parent)

    # Drop a preview <fdf-stem>.molwatch.log next to the .fdf so molwatch
    # can render the initial geometry the moment the user loads it -- no
    # waiting for SIESTA to write its first outcoor block.  The file is
    # static (one preview block, no live updates); for live updates while
    # SIESTA is running, point molwatch at the .out file instead.
    if cfg.write_molwatch_log:
        from ..molwatch_log import write_initial_preview
        mw_path = fdf_p.with_suffix(".molwatch.log")
        write_initial_preview(
            struct,
            mw_path,
            job=fdf_p.stem,
            engine="siesta",
        )
        summary["molwatch_log"] = str(mw_path)

    return summary
