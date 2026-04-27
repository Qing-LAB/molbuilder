"""SIESTA .fdf input generator.

Takes a Structure (or an XYZ file path) and emits a .fdf input file
ready to drop into a SIESTA run, with optional auto-copy of PSML
pseudopotentials from a flat library on disk.

Public API:
    Config            -- dataclass holding every FDF parameter
    render_fdf(...)   -- format an in-memory Structure as FDF text
    convert(...)      -- read XYZ, write FDF, optionally copy psml
    copy_pseudopotentials(...) -- standalone psml copy helper

The CLI lives in :mod:`molbuilder.cli` as the ``fdf`` subcommand.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
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

from .structure import Structure


# --------------------------------------------------------------------- #
#  Config                                                               #
# --------------------------------------------------------------------- #


@dataclass
class Config:
    """All FDF parameters in one place.

    Defaults follow current SIESTA best-practice for a small / medium
    organic-or-inorganic system that's about to be relaxed:

        * MeshCutoff 300 Ry, PAO.BasisSize DZP, GGA-PBE.
        * DM mixing weight 0.02 with Pulay history 3 (SIESTA tutorials
          recommend these for relaxation; the older default of 0.01 is
          stable but slow, the v5 default of 0.25 is too aggressive
          without the v5 mixing scheme).
        * DM tolerance 1e-5 plus a redundant DM.Energy.Tolerance 1e-4 eV
          guard.
        * MaxSCFIterations 500 -- typical relaxation runs need < 100
          per geometry, but a generous limit avoids stalls on the first
          step where the DM is fresh.
        * Force tol 0.02 eV/Ang and CG max-displ 0.05 Ang -- tighter
          than SIESTA's defaults (0.04 / 0.20 Bohr) but appropriate for
          structures destined for property calculations afterwards.
        * Continuation flags (UseSaveDM/CG/XV) all on -- SIESTA silently
          ignores them when no checkpoint exists, but they're free
          insurance for restartable jobs.
    """

    # System
    system_name: str = "siesta_run"
    system_label: str = "siesta"

    # Cell handling for non-periodic XYZ files
    cell_padding: float = 15.0

    # Basis
    basis_size: str = "DZP"
    pao_energy_shift: float = 0.02       # Ry, controls PAO diffuseness

    # XC
    xc_functional: str = "GGA"
    xc_authors: str = "PBE"

    # SCF
    mesh_cutoff: float = 300.0           # Ry
    mixing_weight: float = 0.02          # SIESTA tutorial default for relax
    pulay_history: int = 3
    dm_tolerance: float = 1e-5
    dm_energy_tolerance: float = 1e-4    # eV, redundant SCF guard
    max_scf_iter: int = 500
    electronic_temperature: float = 300.0  # K
    solution_method: str = "diagon"      # default; OMM for very large systems

    # k-grid
    kgrid: Tuple[int, int, int] = (1, 1, 1)

    # Relaxation; relax_type="none" disables the MD block entirely
    relax_type: str = "CG"
    relax_steps: int = 200
    relax_force_tol: float = 0.02        # eV / Ang
    relax_max_displ: float = 0.05        # Ang

    # SCF / MD continuation flags (free insurance for restartable jobs)
    use_save_dm: bool = True
    use_save_cg: bool = True
    use_save_xv: bool = True

    # Atom positioning relative to the cell:
    #   wrap_into_cell -- when an explicit cell is given (e.g. read from
    #                     a periodic XYZ), fold atoms whose fractional
    #                     coordinates fall outside [0, 1) back into the
    #                     unit cell.  Has no effect on auto-vacuum cells
    #                     because the centring step already places atoms
    #                     inside the box.
    #   center_in_vacuum -- for the auto-vacuum case, place the structure
    #                     so its bounding-box midpoint sits at the cell
    #                     centre (default).  Disable to keep raw input
    #                     coordinates (useful when several runs share a
    #                     reference frame).
    wrap_into_cell: bool = True
    center_in_vacuum: bool = True

    # When True, every section in the emitted FDF carries inline tuning
    # hints (parameter ranges, what to change when SCF / CG misbehave,
    # etc.) plus a "Troubleshooting" block at the end.  Set False for
    # a clean machine-readable FDF.
    verbose_comments: bool = True

    # Output flags
    write_forces: bool = True
    write_coor_step: bool = True
    write_coor_xmol: bool = True         # .xyz of every relaxation step
    write_md_history: bool = True        # .ANI trajectory file
    write_hs: bool = False               # H + S matrices (TranSIESTA / DOS)

    # Pseudopotentials
    psml_lib: Optional[str] = None
    copy_psml: bool = True

    # Misc -- pin the species order if you want a specific layout
    species_order: Optional[Sequence[str]] = None


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


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


def render_fdf(struct: Structure, config: Optional[Config] = None,
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
    cfg = config or Config()
    species = (list(cfg.species_order) if cfg.species_order
               else _detect_species(struct.elements))
    species_index = {s: i + 1 for i, s in enumerate(species)}

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
        sizes = ext + 2 * cfg.cell_padding
        cell  = np.diag(sizes)
        if cfg.center_in_vacuum:
            center_now    = (positions.max(axis=0) + positions.min(axis=0)) / 2.0
            center_target = sizes / 2.0
            positions = positions + (center_target - center_now)
        cell_note = (
            f"# (auto-generated orthorhombic vacuum cell "
            f"{sizes[0]:.2f} x {sizes[1]:.2f} x {sizes[2]:.2f} A; "
            f"cell_padding = {cfg.cell_padding} A on each face; "
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
                f"# (using user-supplied lattice;"
                + (f" {n_wrapped} atom(s) wrapped into the unit cell)"
                   if n_wrapped else " all atoms already inside the cell)")
            )
        else:
            cell_note = "# (using user-supplied lattice; wrap_into_cell=False)"

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

    # ---- NetCharge (auto-detected) -------------------------------
    # If the structure has unprotonated phosphate oxygens it carries
    # a non-zero formal charge; we MUST tell SIESTA, otherwise it
    # quietly assumes 0 and adds compensating electrons to neutralise,
    # which gives the wrong electronic structure.
    from .chemistry import formal_charge_from_phosphates
    auto_charge = formal_charge_from_phosphates(struct)
    if auto_charge != 0:
        if v: out += [
            "",
            f"# NetCharge: detected {auto_charge:+d} from {abs(auto_charge)} "
            "deprotonated phosphate oxygen(s).",
            "# Note: SIESTA will add a uniform compensating background charge",
            "# for periodic-cell consistency.  For vacuum calcs of charged",
            "# molecules, use a larger cell_padding (>= 25 A) to suppress",
            "# image-image Coulomb interactions.  To get a neutral system",
            "# instead, build with build_dna(..., protonate_phosphates=True)",
            "# (CLI: --protonate-phosphates) or add explicit counter-ions.",
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
#  XYZ-file convenience wrapper                                         #
# --------------------------------------------------------------------- #


def _struct_from_xyz(xyz_path: str) -> Tuple[Structure, Optional[np.ndarray]]:
    """Read an XYZ via ASE and return (Structure, cell_or_None)."""
    atoms = _ase_read(xyz_path)
    elements = [a.symbol for a in atoms]
    positions = atoms.get_positions()
    cell = atoms.cell.array if hasattr(atoms.cell, "array") else atoms.cell
    cell = np.asarray(cell, dtype=float)
    return (
        Structure(elements=elements, positions=positions, title=Path(xyz_path).stem),
        cell if cell.any() else None,
    )


def convert(
    xyz_path: str,
    fdf_path: str,
    config: Optional[Config] = None,
) -> dict:
    """Read an XYZ, write an FDF, optionally copy pseudopotentials.

    Returns a summary dict with keys: ``fdf``, ``n_atoms``, ``species``,
    ``missing_psml``.
    """
    cfg = config or Config()
    struct, cell = _struct_from_xyz(xyz_path)

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

    return summary
