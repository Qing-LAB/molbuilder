"""SiestaConfig -- every parameter the SIESTA .fdf generator emits.

L1 dataclass.  Field metadata (label / unit / range / tier / help)
drives the CLI option list, the web form schema, and the validation
pass at ``molbuilder/validation.py``; the SIESTA generator at
``molbuilder/siesta/input.py:render_fdf`` is the only consumer of the
configured values themselves.

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

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

# Job-layout v1 protocol (docs/spec/job-layout.md): the basename
# (= SystemLabel for SIESTA, job_name for PySCF) drives EVERY output
# filename, including SIESTA's restart files (.XV / .DM / .CG).  It
# must be safe to embed in a filesystem path without quoting; we keep
# it strict: letters, digits, hyphens, underscores, dots.  Reject
# slashes, whitespace, shell metacharacters, leading-dot.
#
# The same regex is re-used by PySCFConfig.job_name (see
# molbuilder/config/pyscf.py) so the two configs share one rule.
_BASENAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._\-]*$")


def _validate_basename(label: str):
    """Return a validate callable for SiestaConfig.system_label /
    PySCFConfig.job_name.  Used as ``metadata["validate"]`` -- the
    validation pass surfaces a clean error-severity ``Issue`` instead
    of letting a malformed basename reach the filesystem.
    """
    def _check(value, _cfg=None):
        # Local import to avoid an L1->L1 cycle (issues sits next door).
        from ..issues import Issue
        if not isinstance(value, str) or not _BASENAME_RE.fullmatch(value):
            return Issue(
                severity="error",
                message=(
                    f"{label}={value!r} is not a valid job basename. "
                    "Must match [A-Za-z0-9._-]+ (no slashes, no spaces, "
                    "no leading dot).  See docs/spec/job-layout.md."
                ),
                where=f"config.{label}",
            )
        return None
    return _check


@dataclass
class SiestaConfig:
    # System
    system_name: str = field(default="siesta_run", metadata={
        "label": "SystemName",
        "help": "FDF SystemName label written into the .fdf header",
    })
    system_label: str = field(default="siesta", metadata={
        "label":    "SystemLabel",
        "help":     "FDF SystemLabel; output files get this prefix.  "
                    "Must match [A-Za-z0-9._-]+ (job-layout v1).",
        "validate": _validate_basename("system_label"),
    })

    # Cell handling for non-periodic XYZ files
    cell_padding: float = field(default=15.0, metadata={
        "label": "Cell padding", "unit": "Å",
        "range": (5.0, 50.0),
        "tier":  "basic",
        "help":  "vacuum padding (Å) around the molecule on each face of the auto-cell",
    })

    # Basis
    basis_size: str = field(default="DZP", metadata={
        "label": "PAO.BasisSize",
        "help": "PAO basis size: SZ / DZ / SZP / DZP / TZP (rough -> tight)",
    })
    pao_energy_shift: float = field(default=0.01, metadata={
        "label": "PAO.EnergyShift", "unit": "Ry",
        # Upper bound tightened to 0.05 (SP4): 0.1 Ry contracts PAO
        # cutoff radii to ~3 Bohr, putting bond energies hundreds of
        # meV off -- well outside any defensible production window.
        "range": (0.001, 0.05),
        "tier":  "advanced",
        # Default tightened from 0.02 -> 0.01 Ry (gap #5).  SIESTA's
        # own internal default (0.02) is fine for screening / quick
        # scans but produces under-converged PAO tails for production
        # work; the SIESTA manual itself recommends 0.001-0.01 Ry for
        # "well-converged" calculations.  0.01 is the production-side
        # of "well-converged" -- ~2x slower than 0.02 but bond
        # energies converge to within a few meV instead of tens.
        # Loosen back to 0.02 only for screening; tighten to 0.005
        # for phonon / vibrational work.
        "help":  "smaller = more diffuse / more accurate; production work uses 0.005-0.01 Ry",
    })

    # XC
    xc_functional: str = field(default="GGA", metadata={
        "label": "XC.Functional",
        "help": "XC functional family: LDA / GGA / VDW",
    })
    xc_authors: str = field(default="PBE", metadata={
        "label": "XC.Authors",
        "help": "XC parameterisation: PBE / revPBE / BLYP / DRSLL ...",
    })

    # SCF
    mesh_cutoff: float = field(default=300.0, metadata={
        "label": "MeshCutoff", "unit": "Ry",
        "range": (50.0, 1000.0),
        "tier":  "basic",
        "help":  "real-space integration grid; 200-300 typical, 400+ for tight basis",
    })
    mixing_weight: float = field(default=0.02, metadata={
        "label": "DM.MixingWeight",
        "range": (0.001, 0.5),
        "tier":  "advanced",
        "help":  "DM mixing weight; smaller = more conservative SCF, lower if oscillating",
    })
    pulay_history: int = field(default=3, metadata={
        "label": "DM.NumberPulay",
        "range": (0, 20),
        "tier":  "advanced",
        "help":  "Pulay history depth; 3 is SIESTA-tutorial default for relaxation",
    })
    dm_tolerance: float = field(default=1e-5, metadata={
        "label": "DM.Tolerance",
        "range": (1e-8, 1e-3),
        "tier":  "advanced",
        "help":  "DM-element SCF convergence threshold",
    })
    dm_energy_tolerance: float = field(default=1e-4, metadata={
        "label": "DM.Energy.Tolerance", "unit": "eV",
        "range": (1e-8, 1e-1),
        "tier":  "advanced",
        "help":  "redundant SCF energy guard (eV)",
    })
    max_scf_iter: int = field(default=500, metadata={
        "label": "MaxSCFIterations",
        "range": (10, 5000),
        "tier":  "advanced",
        "help":  "max SCF iterations per geometry step",
    })
    electronic_temperature: float = field(default=300.0, metadata={
        "label": "ElectronicTemperature", "unit": "K",
        "range": (0.0, 5000.0),
        "tier":  "advanced",
        "help":  "electronic temperature for Fermi-Dirac smearing (K)",
    })
    solution_method: str = field(default="diagon", metadata={
        "label": "SolutionMethod",
        "choices": ("diagon", "OMM", "transiesta"),
        "help": "diagon / OMM / transiesta (transiesta requires the TranSIESTA build)",
    })

    # k-grid -- Tuple field with custom CLI parsing; not auto-generated
    # by add_dataclass_options (the bridge handles only scalar types).
    kgrid: Tuple[int, int, int] = field(default=(1, 1, 1), metadata={
        "label": "kgrid_Monkhorst_Pack",
        "tier":  "basic",
        "help":  "Monkhorst-Pack mesh (e.g. 4x4x1 in CLI or [4,4,1] in code)",
        "skip_cli": True,
    })

    # Relaxation; relax_type="none" disables the MD block entirely.
    # The actual SIESTA keyword for step count and max-displacement
    # depends on relax_type -- see siesta/input.py:render_fdf for the
    # full mapping (CG -> MD.NumCGsteps + MD.MaxCGDispl;
    # Broyden / FIRE -> MD.NumBroydenSteps / MD.NumFIRESteps + MD.MaxDispl;
    # Verlet / Nose -> MD.FinalTimeStep + MD.InitialTemperature).  The
    # labels below are therefore generic; per-engine help text lives
    # in the FDF's verbose comments.
    relax_type: str = field(default="CG", metadata={
        "label": "MD.TypeOfRun",
        "choices": ("CG", "Broyden", "FIRE", "Verlet", "Nose", "none"),
        "help": "MD/relax algorithm: CG / Broyden / FIRE / Verlet / Nose / none",
    })
    relax_steps: int = field(default=200, metadata={
        "label": "MD step count",
        "range": (1, 10000),
        "tier":  "advanced",
        "help":  "max relaxation steps (CG/Broyden/FIRE) or MD time steps (Verlet/Nose)",
    })
    relax_force_tol: float = field(default=0.02, metadata={
        "label": "MD.MaxForceTol", "unit": "eV/Å",
        "range": (0.001, 0.5),
        "tier":  "advanced",
        "help":  "force-tol stop criterion (CG/Broyden/FIRE only; ignored in Verlet/Nose)",
    })
    relax_max_displ: float = field(default=0.05, metadata={
        "label": "MD max-displ", "unit": "Å",
        "range": (0.001, 0.5),
        "tier":  "advanced",
        "help":  "displacement cap per step (MD.MaxCGDispl for CG, MD.MaxDispl otherwise)",
    })

    # ---- Verlet / Nose dynamics (only emitted when relax_type is in
    # ("Verlet", "Nose"); ignored otherwise).  Defaults are chosen to
    # match SIESTA's room-temperature biomolecular MD convention.
    # md_target_temperature defaults to None -> "use the same value as
    # md_initial_temperature" so the Nose-Hoover thermostat has a
    # sensible target without forcing the user to set both fields.
    md_initial_temperature: float = field(default=300.0, metadata={
        "label": "MD.InitialTemperature", "unit": "K",
        "range": (0.0, 5000.0),
        "tier":  "advanced",
        "help":  "initial-velocity-seed temperature for Verlet/Nose dynamics (K)",
    })
    md_target_temperature: Optional[float] = field(default=None, metadata={
        "label": "MD.TargetTemperature", "unit": "K",
        "tier":  "advanced",
        "help":  ("Nose-Hoover NVT target temperature (K).  None -> use "
                  "md_initial_temperature; ignored unless relax_type=Nose"),
    })
    md_length_timestep: float = field(default=1.0, metadata={
        "label": "MD.LengthTimeStep", "unit": "fs",
        "range": (0.1, 5.0),
        "tier":  "advanced",
        "help":  ("integration timestep for Verlet/Nose dynamics (fs).  "
                  "1.0 fs is SIESTA's default and works for systems without "
                  "H; bonded H typically needs 0.5 fs for stable energy "
                  "conservation"),
    })

    # SCF / MD continuation flags (free insurance for restartable jobs)
    use_save_dm: bool = field(default=True, metadata={
        "help": "read .DM from a prior run if present (free warm-start)",
    })
    use_save_cg: bool = field(default=True, metadata={
        "help": "read .CG from a prior CG relaxation if present",
    })
    use_save_xv: bool = field(default=True, metadata={
        "help": "read .XV (final geometry/velocities) from a prior run if present",
    })

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
    wrap_into_cell: bool = field(default=True, metadata={
        "help": "fold atoms with fractional coords outside [0,1) back into the cell",
    })
    center_in_vacuum: bool = field(default=True, metadata={
        "help": "centre the molecule in the auto-vacuum cell (auto-cell case)",
    })

    # When True, every section in the emitted FDF carries inline tuning
    # hints (parameter ranges, what to change when SCF / CG misbehave,
    # etc.) plus a "Troubleshooting" block at the end.
    verbose_comments: bool = field(default=True, metadata={
        "help": "emit inline tuning hints and a Troubleshooting block in the FDF",
    })

    # Staged-relaxation marker (job-layout v1, Cut 3+).  When 1, 2, or 3,
    # the preview ``<basename>.molwatch.log`` is written as
    # ``<basename>-stage<N>.molwatch.log`` so multiple stages of a
    # coarse->medium->tight relaxation accumulate in one directory and
    # the Watch tab's multi-stage merge picks them up automatically.
    # ``None`` (default) keeps the unsuffixed filename for single-run
    # workflows.  Constraints: SystemLabel stays identical across stages
    # (so SIESTA's .XV / .DM / .CG restart files transfer cleanly); only
    # the preview-log filename gets the suffix.
    stage: Optional[int] = field(default=None, metadata={
        "label": "Relaxation stage",
        "help":  "stage marker (1/2/3) for the preview .molwatch.log "
                 "filename; None keeps the unsuffixed name",
        "range": (1, 3),
    })

    # Output flags
    write_forces: bool = field(default=True, metadata={
        "help": "write forces to the .FA file (required for relaxation)",
    })
    write_coor_step: bool = field(default=True, metadata={
        "help": "write coordinates at every MD step in the main .out",
    })
    write_coor_xmol: bool = field(default=True, metadata={
        "help": "write .xyz of every relaxation step (movie viewer)",
    })
    write_md_history: bool = field(default=True, metadata={
        "help": "write the .ANI trajectory file (xcrysden / vmd / OVITO)",
    })
    write_hs: bool = field(default=False, metadata={
        "help": "write H + S matrices (TranSIESTA / DOS / transport)",
    })
    write_molwatch_log: bool = field(default=True, metadata={
        "help": "write <job>.molwatch.log preview (lets molwatch render before SIESTA does)",
    })

    # ---------------- Parallel execution (MPI) ----------------
    # Only matter when running `mpirun -np N siesta`; single-rank runs
    # ignore them.  Defaults below avoid the most common parallel
    # failure mode -- `propor: ERROR: IMAX = 0` -- by overriding
    # SIESTA's auto-picked BlockSize, which can be too coarse for
    # the per-atom distribution pass on small molecules.
    # Both default to None -> auto-detect.  The CLI bridge can't
    # represent a tri-state Optional[bool] (None / True / False) with
    # a flag pair, so we mark them skip_cli=True; users who need to
    # override go through the Python API.  Block-size auto picks a
    # power-of-2 from n_atoms; over_k auto turns on when the k-grid
    # has multiple k-points.
    parallel_block_size: Optional[int] = field(default=None, metadata={
        "help": "MPI block size; None=auto (power-of-2 from n_atoms)",
        "skip_cli": True,
    })
    parallel_over_k: Optional[bool] = field(default=None, metadata={
        "help": "MPI parallelise over k-points; None=auto from kgrid",
        "skip_cli": True,
    })

    # Pseudopotentials -- psml_lib uses click.Path() in the CLI so it's
    # hand-rolled there; species_order needs comma-string parsing on
    # the CLI side, also hand-rolled.
    psml_lib: Optional[str] = field(default=None, metadata={
        "help": "path to a flat directory of .psml pseudopotentials",
        "skip_cli": True,
    })
    copy_psml: bool = field(default=True, metadata={
        "help": "copy psml files into the output directory (alongside the FDF)",
    })
    species_order: Optional[Sequence[str]] = field(default=None, metadata={
        "help": "comma-separated species order (e.g. 'C,H,S,Au')",
        "skip_cli": True,
    })

    # Net charge.  When None (default), render_fdf auto-detects from the
    # phosphate protonation state via formal_charge_from_phosphates.
    net_charge: Optional[int] = field(default=None, metadata={
        "help": ("net charge override (default: auto-detect from phosphates; "
                 "set explicitly for charged side chains -- carboxylates, "
                 "amines, sulfonates -- the heuristic doesn't see)"),
    })

    # Spin polarisation.  Default off (closed-shell DFT).
    spin_polarized: bool = field(default=False, metadata={
        "help": ("open-shell DFT (collinear); required for radicals / "
                 "transition metals / triplet systems"),
    })
    spin_total: Optional[float] = field(default=None, metadata={
        "help": ("target total spin moment (mu_B); only emitted with "
                 "--spin-polarized"),
    })


# Backwards-compatible alias.  External code that imports `Config` from
# molbuilder.siesta or molbuilder.config.siesta keeps working; new code
# should prefer `SiestaConfig` so it can coexist with PySCFConfig /
# future engine configs in the same module.
Config = SiestaConfig


__all__ = ["SiestaConfig", "Config"]
