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

# Job-layout v1 protocol (docs/protocols/job-layout.md): the basename
# (= SystemLabel for SIESTA, job_name for PySCF) drives EVERY output
# filename, including SIESTA's restart files (.XV / .DM / .CG).  It
# must be safe to embed in a filesystem path without quoting AND
# play nicely with ``<basename>.molwatch.log`` stem/extension
# parsing -- so we BAN dots in addition to slashes / whitespace.
# Allowed: letters, digits, hyphens, underscores.  The HTML form's
# ``pattern=`` attribute and the spec list the same rule; this is
# the single Python source of truth.
#
# The same regex is re-used by PySCFConfig.job_name (see
# molbuilder/config/pyscf.py) so the two configs share one rule.
_BASENAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


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
                    "Must match [A-Za-z0-9_-]+ (letters, digits, hyphens, "
                    "underscores).  No dots / slashes / spaces.  See "
                    "docs/protocols/job-layout.md."
                ),
                where=f"config.{label}",
            )
        return None
    return _check


@dataclass
class SiestaConfig:
    # Explicit form-section order for the schema-driven Build form.
    # Without this, sections would appear in the order their first
    # field is declared below -- which puts Spin / Parallel at the
    # end (they were grouped with the late-arriving "extras" rather
    # than near the SCF block).  This list keeps the form's visual
    # order close to SIESTA's own .fdf reading order: setup first,
    # then SCF + sampling, then the optimisation algorithm, then I/O.
    # 2026-05-27 reorder: "Parallel execution" moved from 5th to LAST
    # (just above the Generate / Save action row in the form).  Reason:
    # the section holds CODE/EXECUTION knobs (MPI ranks, OMP threads,
    # ScaLAPACK BlockSize, memory cap) -- the "how to run it" axis,
    # orthogonal to the "what to compute" axis the physics sections
    # (System / Basis / XC / SCF / Spin / k-grid / Relaxation /
    # Output) cover.  Interleaving Parallel between SCF and Spin
    # broke the mental flow.  Now physics-first, plumbing-last so
    # the user designs the calculation then sizes the machine.
    _form_section_order = (
        "System",
        "Basis & grid",
        "Exchange-correlation",
        "SCF",
        "Spin",
        "k-grid (Monkhorst-Pack)",
        "Relaxation",
        "Output & positioning",
        "Parallel execution",   # ← code/execution, sits right above Generate
    )

    # System
    # 2026-05-27 cleanup: SystemName / SystemLabel are functionally one
    # field for our generated .fdf -- the web UI exposed a single "Job
    # name" input and JS forced ``system_name = system_label`` before
    # POST.  The Python API kept the duplicate dataclass field as a
    # courtesy alias, but it was an attractive nuisance: a Python user
    # who set system_name without system_label got an .fdf where the
    # two diverged, and our own SIESTA wrappers (output names,
    # SystemLabel-prefixed scratch files) would not match the
    # SystemName header.  We drop ``system_name`` outright and emit
    # ``SystemName {cfg.system_label}`` in the FDF.  No alias kept --
    # per project "no backwards compatibility" mandate.
    system_label: str = field(default="siesta", metadata={
        "section":  "System",
        # Run-profile identity — what the run IS named.  Lives in the
        # Run profile workflow-group card alongside the system-character
        # knobs (mixing weight, electronic temperature, spin) because
        # the user sets these together at the start of a run and
        # rarely revisits.
        "workflow_group": "profile",
        "label":    "SystemLabel",
        "engine_key":  'SystemLabel',
        "id_suffix": "system-label",
        "help":     "FDF SystemLabel; output files get this prefix.  "
                    "Must match [A-Za-z0-9_-]+ (job-layout v1; no dots).",
        "pattern":  r"^[A-Za-z0-9_\-]+$",
        "validate": _validate_basename("system_label"),
    })

    # Cell handling for non-periodic XYZ files.  Not exposed in the web
    # form today (the auto-cell pad is fine for the typical workflow);
    # leave unsectioned so it stays a Python-API knob.
    cell_padding: float = field(default=15.0, metadata={
        "label": "Cell padding", "unit": "Å",
        "engine_key":  '(molbuilder: auto-cell build only)',
        "range": (5.0, 50.0),
        "tier":  "basic",
        "help":  "vacuum padding (Å) around the molecule on each face of the auto-cell",
    })

    # Basis
    basis_size: str = field(default="DZP", metadata={
        "section": "Basis & grid",
        "label": "PAO.BasisSize",
        "engine_key":  'PAO.BasisSize',
        "choices": ("SZ", "DZ", "SZP", "DZP", "TZP"),
        "help": "PAO basis size: SZ / DZ / SZP / DZP / TZP (rough -> tight)",
    })
    pao_energy_shift: float = field(default=0.01, metadata={
        "section": "Basis & grid",
        "workflow_group": "stage",
        "label": "PAO.EnergyShift", "unit": "Ry",
        "engine_key":  'PAO.EnergyShift',
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

    # Mesh cutoff lives in the "Basis & grid" section in the form
    # (next to the basis-size dropdown) even though it's strictly a
    # real-space-grid parameter; SIESTA users think of "basis + grid"
    # together when sizing their run.
    mesh_cutoff: float = field(default=300.0, metadata={
        "section": "Basis & grid",
        # Workflow-group tag (2026-06-13): "stage" means switching the
        # relaxation-stage preset MAY rewrite this field.  Three
        # tag values exist (system / stage / budget) — see docs/protocols/
        # results-tab.md § 4.6 and viewer.js STAGE_PRESETS for the
        # design rationale.  Untagged fields render bare (outside any
        # workflow-group card) and STAGE_PRESETS never touches them.
        "workflow_group": "stage",
        "label": "MeshCutoff", "unit": "Ry",
        "engine_key":  'MeshCutoff',
        # 2026-05-28 tightening: slider lower bound raised from 50
        # to 100 Ry.  50 Ry is a screening-grade value that produces
        # noticeably wrong forces / energies for any production work;
        # letting it sit at the slider floor invited silent garbage.
        # 100 Ry is still a reasonable "I'm doing a quick estimate"
        # floor; the validation pass warns at < 150 Ry separately
        # (see _check_siesta_mesh_cutoff in validation.py) so users
        # picking a low-but-not-tiny value see a soft nudge.
        "range": (100.0, 1000.0),
        "tier":  "basic",
        "help":  "real-space integration grid (Ry).  200-300 is "
                 "production-typical; 400+ for tight basis (TZP) or "
                 "vibrational work.  Below 150 Ry the forces / "
                 "energies are noticeably wrong on organic / "
                 "biomolecule systems; the validation pass warns "
                 "below that floor.",
    })

    # XC
    xc_functional: str = field(default="GGA", metadata={
        "section": "Exchange-correlation",
        "label":   "XC.Functional",
        "engine_key":  'XC.functional',
        "choices": ("LDA", "GGA", "VDW"),
        "help":    "XC functional family.  GGA (default) is the safe "
                   "production choice for organic / biomolecule work + "
                   "metals; LDA over-binds (bond lengths ~2-3% too "
                   "short, energies ~10 kcal/mol off); VDW adds a "
                   "non-local dispersion kernel and matters for "
                   "non-covalent / vdW-stacked systems (DNA bases, "
                   "MOFs).  IMPORTANT: the pseudopotential MUST match "
                   "the functional family -- a PBE pseudo on an LDA "
                   "calculation (or vice versa) silently gives wrong "
                   "bond lengths.  PseudoDojo ships separate families "
                   "for PBE / PBEsol / LDA -- pick the matching set.",
    })
    xc_authors: str = field(default="PBE", metadata={
        "section": "Exchange-correlation",
        "label":   "XC.Authors",
        "engine_key":  'XC.authors',
        # Choices match what /api/siesta/check-pseudos accepts when
        # mapping authors->family for the coverage check (see
        # build.py).  Free-text was needed historically (unusual
        # functionals); dropdown covers the 99% case and the user
        # can still set unusual values via the Python API.
        "choices": ("PBE", "PBEsol", "revPBE", "RPBE", "BLYP",
                    "CA", "PZ", "PW", "DRSLL", "LMKLL"),
        "help":    "XC parameterisation within the family.  GGA: PBE "
                   "(default, all-purpose), PBEsol (better lattice "
                   "constants for solids), revPBE / RPBE (better "
                   "thermochemistry, slightly different binding), "
                   "BLYP (rare but accepted).  VDW: DRSLL (vdW-DF1) / "
                   "LMKLL (vdW-DF2-C09).  LDA: CA (Ceperley-Alder, "
                   "default), PZ, PW.  This name MUST match what your "
                   "pseudopotential was generated for -- mismatched "
                   "XC + pseudo gives silently-wrong bond lengths.  "
                   "PseudoDojo organises downloads by this name.",
    })

    # SCF
    solution_method: str = field(default="diagon", metadata={
        "section": "SCF",
        "label": "SolutionMethod",
        "engine_key":  'SolutionMethod',
        "choices": ("diagon", "OMM", "transiesta"),
        "help": "diagon / OMM / transiesta (transiesta requires the TranSIESTA build)",
    })
    mixing_weight: float = field(default=0.02, metadata={
        "section": "SCF",
        # System characteristic — depends on what the system IS
        # (metallic / organic / open-shell), NOT on the stage.
        # Switching stages MUST NOT rewrite this.
        "workflow_group": "profile",
        "label": "DM.MixingWeight",
        "engine_key":  'DM.MixingWeight',
        "range": (0.001, 0.5),
        "tier":  "advanced",
        "help":  "DM mixing weight; smaller = more conservative SCF, lower if oscillating",
    })
    pulay_history: int = field(default=3, metadata={
        "section": "SCF",
        "label": "DM.NumberPulay",
        "engine_key":  'DM.NumberPulay',
        "range": (0, 20),
        "tier":  "advanced",
        "help":  "Pulay history depth; 3 is SIESTA-tutorial default for relaxation",
    })
    dm_tolerance: float = field(default=1e-5, metadata={
        "section": "SCF",
        "workflow_group": "stage",
        "label": "DM.Tolerance",
        "engine_key":  'DM.Tolerance',
        "range": (1e-8, 1e-3),
        "tier":  "advanced",
        "help":  "DM-element SCF convergence threshold",
    })
    dm_energy_tolerance: float = field(default=1e-4, metadata={
        "section": "SCF",
        "workflow_group": "stage",
        "label": "DM.Energy.Tolerance", "unit": "eV",
        "engine_key":  'DM.Energy.Tolerance',
        "range": (1e-8, 1e-1),
        "tier":  "advanced",
        "help":  "redundant SCF energy guard (eV)",
    })
    max_scf_iter: int = field(default=500, metadata={
        "section": "SCF",
        # Resource-budget cap — "how long am I willing to wait" — NOT
        # part of the convergence-target staging.  Switching stages
        # MUST NOT halve / double this value silently.
        "workflow_group": "budget",
        "label": "MaxSCFIterations (SCF cycles per geometry step)",
        "engine_key":  'MaxSCFIterations',
        "range": (10, 5000),
        "tier":  "advanced",
        "help":  "INNER loop: max self-consistency cycles SIESTA "
                 "runs inside each geometry step.  A geometry "
                 "optimisation runs at most relax_steps OUTER steps, "
                 "and each outer step runs at most max_scf_iter inner "
                 "SCF cycles (until DM.Tolerance is met).  500 is "
                 "generous; bump higher if SCF is oscillating.",
    })
    electronic_temperature: float = field(default=300.0, metadata={
        "section": "SCF",
        # System characteristic — high for metallic surfaces (Fermi
        # smearing helps), low for insulators / organics.  Not stage-
        # dependent.
        "workflow_group": "profile",
        "label": "ElectronicTemperature", "unit": "K",
        "engine_key":  'ElectronicTemperature',
        "id_suffix": "temperature",
        "range": (0.0, 5000.0),
        "tier":  "advanced",
        "help":  "electronic temperature for Fermi-Dirac smearing (K)",
    })

    # k-grid -- Tuple field with custom CLI parsing; not auto-generated
    # by add_dataclass_options (the bridge handles only scalar types).
    # In the schema-driven form this renders as three side-by-side int
    # inputs (kx / ky / kz) under id sub-suffixes "x", "y", "z".
    kgrid: Tuple[int, int, int] = field(default=(1, 1, 1), metadata={
        "section": "k-grid (Monkhorst-Pack)",
        "label": "kgrid_Monkhorst_Pack",
        "engine_key":  '%block kgrid_Monkhorst_Pack',
        "id_suffix": "k",
        "triple_labels": ("x", "y", "z"),
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
        "section": "Relaxation",
        "label": "MD.TypeOfRun",
        "engine_key":  'MD.TypeOfRun',
        "id_suffix": "relax",
        "choices": ("CG", "Broyden", "FIRE", "Verlet", "Nose", "none"),
        "help": "MD/relax algorithm: CG / Broyden / FIRE / Verlet / Nose / none",
    })
    relax_steps: int = field(default=200, metadata={
        "section": "Relaxation",
        # Resource-budget cap — same as max_scf_iter, this is "how
        # many outer steps am I willing to wait for", not a
        # convergence target.  Scales with system size, not stage.
        # For ~230-atom Au junctions bump to 500+; the cluster-context
        # closed-shell argument doesn't help convergence speed.
        "workflow_group": "budget",
        "label": "MD.Num*Steps (max geometry-optimisation steps)",
        "engine_key":  'MD.NumCGsteps / MD.NumBroydenSteps / MD.NumFIRESteps (per relax_type)',
        "range": (1, 10000),
        "tier":  "advanced",
        "help":  "OUTER loop: max geometry steps the optimiser is "
                 "allowed (each step runs a full SCF and computes "
                 "forces, then moves atoms).  Maps to the SIESTA "
                 "keyword that matches relax_type: "
                 "MD.NumCGsteps for CG, MD.NumBroydenSteps for "
                 "Broyden, MD.NumFIRESteps for FIRE, "
                 "MD.FinalTimeStep for Verlet/Nose dynamics.  "
                 "Tight final stages need MORE steps (small "
                 "displacement cap = slow descent), not fewer.",
    })
    relax_force_tol: float = field(default=0.02, metadata={
        "section": "Relaxation",
        "workflow_group": "stage",
        "label": "MD.MaxForceTol", "unit": "eV/Å",
        "engine_key":  'MD.MaxForceTol',
        "id_suffix": "force-tol",
        "range": (0.001, 0.5),
        "tier":  "advanced",
        "help":  "force-tol stop criterion (CG/Broyden/FIRE only; ignored in Verlet/Nose)",
    })
    relax_max_displ: float = field(default=0.05, metadata={
        "section": "Relaxation",
        "workflow_group": "stage",
        "label": "MD max-displ", "unit": "Å",
        "engine_key":  'MD.MaxCGDispl / MD.MaxDispl (per relax_type)',
        "id_suffix": "max-displ",
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
    # The three MD knobs below are MEANINGFUL ONLY for Verlet / Nose
    # dynamics (NOT for CG / Broyden / FIRE geometry relaxation), but
    # SIESTA SILENTLY uses these defaults when the user picks Verlet
    # or Nose from the form -- so the user gets a 300 K / 1 fs / 0 K
    # target-temperature run with no UI hint that they could change
    # them.  Adding ``section`` here promotes them into the form so
    # the user at least SEES them on the page; their help text marks
    # them as ignored-for-CG so the form doesn't mislead non-MD users.
    md_initial_temperature: float = field(default=300.0, metadata={
        "section": "Relaxation",
        "label": "MD.InitialTemperature", "unit": "K",
        "engine_key":  'MD.InitialTemperature',
        "range": (0.0, 5000.0),
        "tier":  "advanced",
        "help":  ("initial-velocity-seed temperature for Verlet/Nose "
                  "molecular dynamics (K).  IGNORED by CG / Broyden / "
                  "FIRE geometry relaxation -- those don't have "
                  "velocities to seed."),
    })
    md_target_temperature: Optional[float] = field(default=None, metadata={
        "section": "Relaxation",
        "label": "MD.TargetTemperature", "unit": "K",
        "engine_key":  'MD.TargetTemperature',
        "null_label": "(use MD.InitialTemperature)",
        "range":      (0.0, 5000.0),       # mirror md_initial_temperature
        "tier":  "advanced",
        "help":  ("Nose-Hoover NVT target temperature (K).  Used ONLY "
                  "by Nose dynamics; CG / Broyden / FIRE / Verlet "
                  "ignore this.  Defaults to md_initial_temperature "
                  "when unset."),
    })
    md_length_timestep: float = field(default=1.0, metadata={
        "section": "Relaxation",
        "label": "MD.LengthTimeStep", "unit": "fs",
        "engine_key":  'MD.LengthTimeStep',
        "range": (0.1, 5.0),
        "tier":  "advanced",
        "help":  ("integration timestep for Verlet/Nose dynamics (fs).  "
                  "1.0 fs is SIESTA's default and works for systems "
                  "without H; bonded H typically needs 0.5 fs for "
                  "stable energy conservation.  IGNORED by CG / "
                  "Broyden / FIRE geometry relaxation."),
    })

    # SCF / MD continuation flags (free insurance for restartable jobs)
    use_save_dm: bool = field(default=True, metadata={
        "help": "read .DM from a prior run if present (free warm-start)",
            "engine_key":  'DM.UseSaveDM',
    })
    use_save_cg: bool = field(default=True, metadata={
        "help": "read .CG from a prior CG relaxation if present",
            "engine_key":  'MD.UseSaveCG',
    })
    use_save_xv: bool = field(default=True, metadata={
        "help": "read .XV (final geometry/velocities) from a prior run if present",
            "engine_key":  'MD.UseSaveXV',
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
        "section": "Output & positioning",
        "label": "Wrap atoms into cell",
        "engine_key":  '(molbuilder: pre-emission positioning)',
        "help": "fold atoms with fractional coords outside [0,1) back into the cell",
    })
    center_in_vacuum: bool = field(default=True, metadata={
        "section": "Output & positioning",
        "label": "Center in vacuum cell",
        "engine_key":  '(molbuilder: pre-emission positioning)',
        "help": "centre the molecule in the auto-vacuum cell (auto-cell case)",
    })

    # When True, every section in the emitted FDF carries inline tuning
    # hints (parameter ranges, what to change when SCF / CG misbehave,
    # etc.) plus a "Troubleshooting" block at the end.
    verbose_comments: bool = field(default=True, metadata={
        "section": "Output & positioning",
        "label": "Verbose inline comments",
        "engine_key":  '(molbuilder: .fdf comment-block control)',
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
        "engine_key":  '(molbuilder: filename suffix + log naming)',
        "help":  "stage marker (1/2/3) for the preview .molwatch.log "
                 "filename; None keeps the unsuffixed name",
        "range": (1, 3),
    })

    # Output flags
    write_forces: bool = field(default=True, metadata={
        "help": "write forces to the .FA file (required for relaxation)",
            "engine_key":  'WriteForces',
    })
    write_coor_step: bool = field(default=True, metadata={
        "help": "write coordinates at every MD step in the main .out",
            "engine_key":  'WriteCoorStep',
    })
    write_coor_xmol: bool = field(default=True, metadata={
        "section": "Output & positioning",
        "label": "Write XMOL .xyz per step",
        "engine_key":  'WriteCoorXmol',
        "help": "write .xyz of every relaxation step (movie viewer)",
    })
    write_md_history: bool = field(default=True, metadata={
        "section": "Output & positioning",
        "label": "Write .ANI trajectory",
        "engine_key":  'WriteMDhistory',
        "help": "write the .ANI trajectory file (xcrysden / vmd / OVITO)",
    })
    write_hs: bool = field(default=False, metadata={
        "section": "Output & positioning",
        "label": "Write H+S matrices",
        "engine_key":  'SaveHS / WriteHS',
        "help": "write H + S matrices (TranSIESTA / DOS / transport)",
    })
    write_molwatch_log: bool = field(default=True, metadata={
        "help": "write <job>.molwatch.log preview (lets molwatch render before SIESTA does)",
            "engine_key":  '(molbuilder: writes <basename>.molwatch.log preview)',
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
    # MPI rank count for ``mpirun -np N siesta`` -- exposed on the
    # form so the user can pick the rank count alongside the other
    # parallel-execution knobs.  The run.sh wrapper reads this from
    # the form params.  None / 0 / 1 -> single-process (no mpirun).
    # Don't confuse with parallel_block_size (BlockSize for ScaLAPACK
    # within a rank); rank count is the OUTER parallelism.
    mpi_np: Optional[int] = field(default=None, metadata={
        "section":    "Parallel execution",
        "label":      "MPI ranks (np)",
        "engine_key":  '(molbuilder: .run.sh ``mpirun -np N`` only; not in .fdf)',
        "null_label": "(single-process)",
        "range":      (1, 1024),
        "help":       "MPI rank count baked into ``mpirun -np N siesta`` "
                      "in the generated run-wrapper.  Pick based on your "
                      "host: typically N = physical cores; for "
                      "memory-bound jobs N = sockets x cores_per_socket "
                      "/ 2 is a common rule of thumb.  Cluster schedulers "
                      "(Slurm / PBS) usually set this for you; on a "
                      "workstation you pick it manually.  Leave blank for "
                      "the auto default (physical_cores).\n\n"
                      "RUNTIME OVERRIDE: this value bakes a DEFAULT into "
                      "the wrapper but is not final.  The wrapper accepts "
                      "``bash run.sh -np N`` and ``MB_NP=N bash run.sh``, "
                      "so you can experiment with different rank counts "
                      "WITHOUT regenerating.  This matters because SIESTA "
                      "can crash with ``propor: ERROR: IMAX = 0`` for "
                      "certain mpi_np / molecule combinations -- the "
                      "crash depends on the ProcessorY x ProcessorX grid "
                      "SIESTA auto-picks for that rank count, which is "
                      "hard to predict.  If you hit propor, retry with "
                      "smaller -np (powers of 2 are usually safe).  The "
                      "wrapper prints a focused diagnostic on the propor "
                      "crash with specific suggestions.",
        "skip_cli":   True,
    })

    parallel_block_size: Optional[int] = field(default=None, metadata={
        "section": "Parallel execution",
        "label": "BlockSize",
        "engine_key":  'BlockSize',
        "id_suffix": "block-size",
        "null_label": "(auto)",
        "help": "ScaLAPACK BlockSize for the diagonaliser's orbital "
                "distribution; affects cache efficiency at moderate "
                "rank counts.  None = auto: largest power of 2 that "
                "leaves >= 2 blocks per rank.  NOTE: BlockSize does "
                "NOT fix the ``propor: ERROR: IMAX = 0`` crash -- that "
                "was the previous (incorrect) claim and is contradicted "
                "by direct empirical sweep (BS = 1, 2, 4 all crash at "
                "the same mpi_np).  Propor is a vector-proportionality "
                "check in matel_table's MPI-deduplication step; lower "
                "mpi_np (use the wrapper's ``-np`` override) to clear "
                "that crash.  Set BlockSize explicitly only for the "
                "rare ScaLAPACK perf-tuning case (>1000 atoms on "
                ">= 16 ranks, where 16 or 32 helps).",
        "skip_cli": True,
    })
    parallel_over_k: Optional[bool] = field(default=None, metadata={
        "section": "Parallel execution",
        "label": "ParallelOverK",
        "engine_key":  'Diag.ParallelOverK',
        "help": "MPI parallelise over k-points; None=auto from kgrid",
        "skip_cli": True,
    })
    # OpenMP threads per MPI rank.  Controls the run-wrapper's
    # ``export OMP_NUM_THREADS=<N>`` line (see molbuilder/runwrap.py).
    # Default None -> auto: physical cores // n_mpi_ranks if MPI is
    # used (set by runwrap), else physical cores.  The wrapper also
    # pins BLAS to 1 thread per rank so OMP * BLAS doesn't
    # oversubscribe -- canonical anti-oversubscription recipe shared
    # with the PySCF / spectra scripts.
    omp_threads: Optional[int] = field(default=None, metadata={
        "section":    "Parallel execution",
        "label":      "OMP threads per rank",
        # Not a SIESTA fdf keyword.  Emits ``export OMP_NUM_THREADS=N``
        # into .run.sh AND ``# runtime.omp_threads_requested: N`` comment
        # into the .fdf (so the .out parser can recover the requested
        # value when reading the run back via runtime_info).
        "engine_key":  '(molbuilder: .run.sh OMP_NUM_THREADS + .fdf runtime_info comment)',
        "null_label": "(auto: physical cores)",
        "help":       "OpenMP threads per MPI process.  Default (blank) "
                      "auto-detects physical cores at run time (divided "
                      "by N_MPI when applicable).  Set explicitly to "
                      "bench or leave cores free for other jobs.  The "
                      "emitted run-wrapper pins BLAS to 1 thread per "
                      "rank so OMP*BLAS doesn't oversubscribe -- the "
                      "canonical recipe shared with /spectra + Build PySCF.",
        "skip_cli":   True,
    })
    # SIESTA SystemMemory directive: MB cap for the SCF/diag working
    # set.  Not auto-set in the .fdf today; if set here, runtime_info
    # records it so the /results trajectory inspector shows the cap.
    max_memory_mb: Optional[int] = field(default=None, metadata={
        "section":    "Parallel execution",
        "label":      "Max memory (per rank)",
        # Not a SIESTA fdf keyword.  Emits ``ulimit -v`` into .run.sh
        # AND ``# runtime.max_memory_mb: N`` into the .fdf so the .out
        # parser can recover the cap via runtime_info.
        "engine_key":  '(molbuilder: .run.sh ulimit -v + .fdf runtime_info comment)',
        "unit":       "MB",
        "null_label": "(no cap)",
        "help":       "MB cap per MPI rank.  Emits a SystemMemory hint "
                      "into the .fdf when set; left blank, SIESTA uses "
                      "whatever the OS allows.  Recorded in the run's "
                      "runtime_info so the /results display shows it.",
        "skip_cli":   True,
    })

    # Pseudopotentials -- psml_lib uses click.Path() in the CLI so it's
    # hand-rolled there; species_order needs comma-string parsing on
    # the CLI side, also hand-rolled.
    psml_lib: Optional[str] = field(default=None, metadata={
        "section":    "System",
        # Run-profile identity — which pseudopotential library this
        # run uses is fixed per-project, set alongside SystemLabel
        # and the spin/charge knobs.
        "workflow_group": "profile",
        "label":      "Pseudopotential directory (.psml)",
        "engine_key":  '(molbuilder: stages .psml files next to .fdf; SIESTA reads them by element basename)',
        "null_label": "(none)",
        "help":       "Path to a directory of .psml pseudopotential "
                      "files (one per element).  Accepts an absolute "
                      "path, ``~/...``, OR a path relative to "
                      "``projects/`` (so just ``pseudopotential`` "
                      "resolves to ``projects/pseudopotential/`` --"
                      " the conventional shared location).  Tip: use "
                      "the file-picker button next to this field to "
                      "browse and avoid typing the path by hand.  "
                      "SIESTA pseudos are "
                      "NOT bundled with molbuilder -- you have to "
                      "download them.  RECOMMENDED SOURCE: "
                      "PseudoDojo (http://www.pseudo-dojo.org) -- "
                      "well-tested, peer-reviewed, free.  WHICH SET "
                      "TO PICK from PseudoDojo:\n"
                      " * Format = PSML (NOT PSP8 -- that's for "
                      "ABINIT only; PSML is SIESTA's native format).\n"
                      " * Functional MUST match cfg.xc_authors -- "
                      "pick the SAME family (PBE-SR for PBE / GGA, "
                      "PBEsol-SR for PBEsol, PW for LDA, etc.).\n"
                      " * Relativistic level: SR (scalar-relativistic) "
                      "for almost everything.  FR (fully-relativistic, "
                      "with spin-orbit) only when you actually need "
                      "spin-orbit coupling (heavy-element spectroscopy, "
                      "topological insulators).  SR is the safe default.\n"
                      " * NC vs PAW: PseudoDojo only ships NC (norm-"
                      "conserving) -- right for SIESTA (PAW is for "
                      "ABINIT / VASP / Quantum ESPRESSO).\n"
                      " * Standard vs Stringent: 'standard' is "
                      "production-quality + smaller mesh cutoff (300-"
                      "400 Ry); 'stringent' is for benchmarking / "
                      "publication + needs MeshCutoff >= 500 Ry.\n"
                      "Download recipe for hemeC-like systems: grab "
                      "the 'PBE-SR / standard / PSML' set for {C, H, "
                      "N, O, S, Fe}, unzip into one directory, point "
                      "this field at it.",
        "skip_cli":   True,
    })
    copy_psml: bool = field(default=True, metadata={
        "help": "copy psml files into the output directory (alongside the FDF)",
            "engine_key":  '(molbuilder: triggers .psml staging step)',
    })
    species_order: Optional[Sequence[str]] = field(default=None, metadata={
        "help": "comma-separated species order (e.g. 'C,H,S,Au')",
        "skip_cli": True,
            "engine_key":  '(molbuilder: ChemicalSpeciesLabel block ordering)',
    })

    # Net charge.  When None (default), render_fdf auto-detects from the
    # phosphate protonation state via formal_charge_from_phosphates.
    net_charge: Optional[int] = field(default=None, metadata={
        "section": "System",
        # Run-profile identity — molecule's charge state is a
        # fundamental property of WHAT you're computing.
        "workflow_group": "profile",
        "label": "Net charge",
        "engine_key":  'NetCharge',
        "null_label": "(auto-detect from phosphates)",
        "range": (-10, 10),
        "tier": "basic",
        "help": ("Net charge of the system, in units of |e|.  Default "
                  "(blank) auto-detects from phosphate protonation -- "
                  "fine for DNA/RNA from tleap.  Set EXPLICITLY for "
                  "charged side chains the heuristic doesn't see: "
                  "carboxylates (Asp/Glu), protonated amines (Lys/Arg/"
                  "His+), sulfonates.  Sign convention: -1 = one extra "
                  "electron; +1 = one missing electron."),
    })

    # Spin polarisation.  Default off (closed-shell DFT).
    spin_polarized: bool = field(default=False, metadata={
        "section":     "Spin",
        # System characteristic — depends on chemistry (open-shell
        # metals / radicals require it), not on stage.
        "workflow_group": "profile",
        "label":       "Spin polarized",
        # Emits ``SpinPolarized .true.`` (v4 form) NOT the v5 single-
        # line ``Spin polarized``: SIESTA 5.4.2's v5 parser path does
        # not subsequently read Spin.Fix / Spin.Total, so open-shell
        # metals abort at propor.  See siesta/input.py emission site.
        "engine_key":  "SpinPolarized",
        "help":        ("open-shell DFT (collinear); required for "
                          "radicals / transition metals / triplet "
                          "systems.  Emits ``SpinPolarized .true.`` "
                          "in the .fdf."),
    })
    spin_total: Optional[float] = field(default=None, metadata={
        "section":     "Spin",
        "workflow_group": "profile",
        "label":       "Target spin moment",
        "null_label":  "(default)",
        # Emits TWO keys: Spin.Fix .true. + Spin.Total <v>.  Either
        # alone is silently ignored by SIESTA (Spin.Fix without a
        # value to fix; Spin.Total without Spin.Fix to gate the
        # constraint).
        "engine_key":  "Spin.Fix + Spin.Total",
        "help":        ("target total spin moment in mu_B (= number "
                          "of unpaired electrons).  Emits BOTH "
                          "``Spin.Fix .true.`` and ``Spin.Total <v>`` "
                          "in the .fdf.  Only emitted when --spin-"
                          "polarized."),
    })


# Backwards-compatible alias.  External code that imports `Config` from
# molbuilder.siesta or molbuilder.config.siesta keeps working; new code
# should prefer `SiestaConfig` so it can coexist with PySCFConfig /
# future engine configs in the same module.
Config = SiestaConfig


__all__ = ["SiestaConfig", "Config"]
