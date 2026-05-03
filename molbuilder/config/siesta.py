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

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


@dataclass
class SiestaConfig:
    # System
    system_name: str = "siesta_run"
    system_label: str = "siesta"

    # Cell handling for non-periodic XYZ files
    cell_padding: float = field(default=15.0, metadata={
        "label": "Cell padding", "unit": "Å",
        "range": (5.0, 50.0),
        "tier":  "basic",
        "help":  "vacuum padding around the molecule on each face of the auto-cell",
    })

    # Basis
    basis_size: str = "DZP"
    pao_energy_shift: float = field(default=0.02, metadata={
        "label": "PAO.EnergyShift", "unit": "Ry",
        "range": (0.001, 0.1),
        "tier":  "advanced",
        "help":  "smaller = more diffuse / more accurate; production work uses 0.005-0.01",
    })

    # XC
    xc_functional: str = "GGA"
    xc_authors: str = "PBE"

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
        "help":  "smaller = more conservative SCF; lower if oscillating",
    })
    pulay_history: int = field(default=3, metadata={
        "label": "DM.NumberPulay",
        "range": (0, 20),
        "tier":  "advanced",
    })
    dm_tolerance: float = field(default=1e-5, metadata={
        "label": "DM.Tolerance",
        "range": (1e-8, 1e-3),
        "tier":  "advanced",
    })
    dm_energy_tolerance: float = field(default=1e-4, metadata={
        "label": "DM.Energy.Tolerance", "unit": "eV",
        "range": (1e-8, 1e-1),
        "tier":  "advanced",
    })
    max_scf_iter: int = field(default=500, metadata={
        "label": "MaxSCFIterations",
        "range": (10, 5000),
        "tier":  "advanced",
    })
    electronic_temperature: float = field(default=300.0, metadata={
        "label": "ElectronicTemperature", "unit": "K",
        "range": (0.0, 5000.0),
        "tier":  "advanced",
    })
    solution_method: str = "diagon"      # default; OMM for very large systems

    # k-grid
    kgrid: Tuple[int, int, int] = (1, 1, 1)

    # Relaxation; relax_type="none" disables the MD block entirely
    relax_type: str = "CG"
    relax_steps: int = field(default=200, metadata={
        "label": "MD.Steps",
        "range": (1, 10000),
        "tier":  "advanced",
    })
    relax_force_tol: float = field(default=0.02, metadata={
        "label": "MD.MaxForceTol", "unit": "eV/Å",
        "range": (0.001, 0.5),
        "tier":  "advanced",
    })
    relax_max_displ: float = field(default=0.05, metadata={
        "label": "MD.MaxCGDispl", "unit": "Å",
        "range": (0.001, 0.5),
        "tier":  "advanced",
    })

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
    write_molwatch_log: bool = True      # write <job>.molwatch.log alongside the
                                         # .fdf with the initial geometry as a
                                         # preview block, so molwatch can render
                                         # the structure immediately -- before
                                         # SIESTA has produced any output.

    # ---------------- Parallel execution (MPI) ----------------
    # Only matter when running `mpirun -np N siesta`; single-rank runs
    # ignore them.  Defaults below avoid the most common parallel
    # failure mode -- `propor: ERROR: IMAX = 0` -- by overriding
    # SIESTA's auto-picked BlockSize, which can be too coarse for
    # the per-atom distribution pass on small molecules.
    parallel_block_size: Optional[int] = None
                                         # None -> auto: pick a power-of-2
                                         # block size based on n_atoms (see
                                         # _auto_block_size below).  Set an
                                         # explicit int to override (8 is a
                                         # safe value for most molecules at
                                         # typical 1-8 MPI rank counts; raise
                                         # to 16-32 for >1000 atoms / >=16
                                         # ranks to recover ScaLAPACK
                                         # efficiency).
    parallel_over_k: Optional[bool] = None
                                         # None -> auto: False if k-grid is
                                         # 1x1x1 (Gamma; molecule/vacuum),
                                         # True otherwise (periodic crystal
                                         # with multiple k-points).

    # Pseudopotentials
    psml_lib: Optional[str] = None
    copy_psml: bool = True

    # Misc -- pin the species order if you want a specific layout
    species_order: Optional[Sequence[str]] = None

    # Net charge.  When None (default), render_fdf auto-detects from the
    # phosphate protonation state via formal_charge_from_phosphates.  Set
    # an explicit integer to override -- needed for any system whose net
    # charge comes from groups other than phosphates (carboxylates,
    # protonated amines, sulfonates, etc., which the heuristic does NOT
    # detect).  An explicit 0 disables auto-detection (treats system as
    # neutral even if the heuristic would have flagged it).
    net_charge: Optional[int] = None

    # Spin polarisation.  Default off (closed-shell DFT).  Set True for
    # any system with unpaired electrons (radicals, transition metals,
    # certain charged biomolecules) -- without it SIESTA assumes
    # spin-restricted and silently produces a wrong electronic
    # structure for open-shell systems.
    spin_polarized: bool = False
    # Optional: total spin moment in units of Bohr magnetons (=
    # number of unpaired electrons).  None lets SIESTA decide.
    spin_total: Optional[float] = None


# Backwards-compatible alias.  External code that imports `Config` from
# molbuilder.siesta or molbuilder.config.siesta keeps working; new code
# should prefer `SiestaConfig` so it can coexist with PySCFConfig /
# future engine configs in the same module.
Config = SiestaConfig


__all__ = ["SiestaConfig", "Config"]
