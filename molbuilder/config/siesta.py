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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _validate_kgrid(value):
    """Per-component range check for SiestaConfig.kgrid (a
    Tuple[int,int,int]).  Used as the ``validate`` callable on the
    kgrid field metadata so the scalar ``range`` check in
    ``_validate_config_metadata`` doesn't fire (and silently
    swallow a TypeError) on this tuple-typed field.

    Returns a list of Issue (empty = OK).  Range (1, 64) per the
    accepted SIESTA sampling density: 1 is the gamma-only floor;
    anything > 32 in any direction is wasteful for real-space
    integration and very rarely justified.  64 leaves some headroom
    for the periodic-1D / 2D cases without un-bounding the field.
    """
    from ..issues import Issue
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return [Issue(
            "error",
            f"kgrid must be a 3-tuple of ints; got {value!r}",
            "config.kgrid",
        )]
    out = []
    for i, v in enumerate(value):
        if not isinstance(v, int) or isinstance(v, bool):
            out.append(Issue(
                "error",
                f"kgrid[{i}] = {v!r} must be an int (1..64)",
                "config.kgrid",
            ))
        elif v < 1 or v > 64:
            out.append(Issue(
                "warn",
                f"kgrid[{i}] = {v} is outside the recommended "
                f"range [1, 64]",
                "config.kgrid",
            ))
    return out


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
    # 2026-06-15 second restructure: merged "Relaxation" + "Parallel
    # execution" into a single "Compute & budget" section.  Reasoning:
    # both sections covered "how the run proceeds" -- the optimization
    # algorithm + its budget on one hand, the MPI/OMP resources on the
    # other.  Splitting them across two sections forced the user to
    # scroll past unrelated cards (Output) between two semantically
    # connected groups.  Merging keeps the physics axis (System ->
    # Basis -> XC -> SCF -> Spin -> Output) compact and gathers all
    # the "execution strategy + resources" knobs in one section.
    # The workflow-group cards INSIDE the new section split the merged
    # fields cleanly:
    #   * Profile card -- relax_type + MD physics (room/temp/dt)
    #   * Stage card   -- force_tol + max_displ (convergence targets)
    #   * Budget card  -- relax_steps + mpi_np + omp_threads +
    #                     BlockSize + ParallelOverK + max_memory_mb +
    #                     diag_algorithm + enable_gpu
    _form_section_order = (
        "System",
        "Basis & grid",
        "Exchange-correlation",
        "SCF",
        "Spin",
        "Output & positioning",
        "Compute & budget",   # ← optimization algo + resources, sits right above Generate
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

    # NOTE: the vacuum box is NOT a SiestaConfig knob.  Vacuum comes with the
    # STRUCTURE (Structure.vacuum, per-side gap) -- the single source of truth for
    # lattice/vacuum (structure-periodicity.md).  render_fdf derives the auto-cell
    # from ``struct.resolve_cell()``; there is no cell_padding / center_in_vacuum.

    # Basis
    basis_size: str = field(default="DZP", metadata={
        "section": "Basis & grid",
        # Workflow-group tag (2026-06-15): joined the Stage card so it
        # sits alongside ``mesh_cutoff``, ``pao_energy_shift``, and
        # ``kgrid`` — all of which are "how finely we sample the
        # calculation" knobs that scale with the convergence target.
        # Previously this field rendered as a one-control "Base" card
        # bare in the Basis & grid section, separated by visual gap
        # from the actual basis-relevant knobs in the Stage card next
        # to it (user complaint 2026-06-15).
        #
        # Tagging ``stage`` puts basis_size in the same workflow-group
        # card as the other "sampling-fidelity" knobs.  It is NOT in
        # STAGE_PRESETS in viewer.js, so switching the relaxation
        # stage (coarse / medium / tight) does NOT silently rewrite
        # the basis size -- that stays the user's choice (which is
        # what people expect: basis is part of the run's identity, not
        # part of the stage refinement schedule).
        "workflow_group": "stage",
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
        "help":  (
            "Real-space integration grid (Ry).  Sets the spacing of "
            "the 3D mesh SIESTA uses for Hartree + XC potentials, via "
            "the plane-wave-equivalent kinetic-energy cutoff.\n"
            "Per-tier: screening 150, loose preopt 200-250, "
            "publishable 350, tight (vib/phonons) 500 (600 for "
            "first-row elements).\n"
            "Below 150 Ry the forces / energies are noticeably wrong "
            "on organic / biomolecule systems; the validator warns "
            "below that floor.  Egg-box noise sets the floor for "
            "vibrational work — test by varying ±50 Ry.  See "
            "docs/engines/optimization-tuning.md § 2.6."
        ),
    })

    # XC
    xc_functional: str = field(default="GGA", metadata={
        "section": "Exchange-correlation",
        "workflow_group": "profile",
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
        "workflow_group": "profile",
        "label":   "XC.Authors",
        "engine_key":  'XC.authors',
        # Choices feed the validator's authors->family map for the
        # pseudopotential coverage check (see
        # molbuilder/validation/siesta.py::_check_siesta_pseudo_coverage).
        # Free-text was needed historically (unusual functionals);
        # dropdown covers the 99% case and the user can still set
        # unusual values via the Python API.
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
        # Profile-level: SCF solver family is a system-level
        # decision (diagon / OMM / TranSIESTA), set once with XC +
        # basis.  Switching stages MUST NOT rewrite this.
        "workflow_group": "profile",
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
        # Profile-level: DIIS history depth pairs with mixing_weight
        # (also profile) — both are SCF-stability tuning that
        # depends on what the system IS, not on the stage.
        "workflow_group": "profile",
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
        "help":  (
            "Density-matrix element convergence threshold for the "
            "inner SCF loop.  Forces are derived from the converged "
            "density -- sloppy SCF -> noisy forces -> optimizer "
            "thrashes.\n"
            "Per-tier (dimensionless): screening 1e-3, loose preopt "
            "1e-4, publishable 1e-4, tight (vib/IR) 1e-5.\n"
            "Rule of thumb: keep SCF tol ~10x tighter than the "
            "force-precision target you want at convergence.  See "
            "docs/engines/optimization-tuning.md § 2.5."
        ),
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
        # 2026-06-13 fold: k-grid (Monkhorst-Pack) is reciprocal-space
        # sampling; basis_size / mesh_cutoff / pao_energy_shift are
        # real-space sampling.  Same conceptual family — all are
        # "how finely we sample the calculation."  Folding into
        # "Basis & grid" so the form stops having a one-field section.
        "section": "Basis & grid",
        # Folded into the Stage card (a convergence knob: more
        # k-points → tighter sampling → more cost) per the
        # web-ui-coherence Rule 2 attachment pass on 2026-06-13.
        "workflow_group": "stage",
        "label": "kgrid_Monkhorst_Pack",
        "engine_key":  '%block kgrid_Monkhorst_Pack',
        "id_suffix": "k",
        "triple_labels": ("x", "y", "z"),
        "tier":  "basic",
        "help":  ("Monkhorst-Pack k-point mesh: the sampling resolution of "
                  "reciprocal space for the DFT calculation. More k-points = "
                  "finer Brillouin-zone sampling = better convergence at more "
                  "cost. e.g. 4x4x1 in CLI or [4,4,1] in code."),
        "skip_cli": True,
        # 2026-06-14 G5: per-component validator so the metadata
        # range check actually runs on a Tuple-typed field.  Without
        # this, ``_validate_config_metadata`` would TypeError on the
        # scalar comparison and silently skip — a future ``kgrid =
        # (0, 0, 0)`` (illegal: SIESTA requires ≥ 1) would slip
        # through.  Range (1, 64) per the engine's accepted sampling
        # density (anything > 32 in any direction is wasteful for
        # a real-space integration).
        "validate": (lambda value, cfg: _validate_kgrid(value)),
    })

    # Relaxation; relax_type="none" disables the MD block entirely.
    # SIESTA 5.4.2 step-count + max-displacement mapping (see
    # siesta/input.py:render_fdf for the full emission code):
    #   CG / Broyden / FIRE -> MD.NumCGsteps + MD.MaxCGDispl
    #     (UNIVERSAL despite CG-prefixed names -- pre-2026-06-23 we
    #      wrongly emitted MD.NumBroydenSteps + MD.MaxDispl, which
    #      SIESTA silently dropped + ran as Single-point; see
    #      decision-log 2026-06-23 in design.md)
    #   Verlet / Nose -> MD.FinalTimeStep + MD.InitialTemperature
    # The labels below are generic; per-engine help text lives in the
    # FDF's verbose comments.
    relax_type: str = field(default="CG", metadata={
        "section": "Compute & budget",
        # Profile-level: relax/MD algorithm family (CG / Broyden /
        # FIRE / Verlet / Nose) is a run-shape identity choice,
        # parallel to PySCF's ``optimizer`` (also profile).
        "workflow_group": "profile",
        "label": "MD.TypeOfRun",
        "engine_key":  'MD.TypeOfRun',
        "id_suffix": "relax",
        "choices": ("CG", "Broyden", "FIRE", "Verlet", "Nose", "none"),
        "help": (
            "MD/relax algorithm.  Per-tier guidance:\n"
            "  • CG       — robust for loose warm-up stages (far from "
            "minimum, large forces).  No memory cost.  Oscillates near "
            "a minimum on stiff / coupled systems (metals, interfaces, "
            "vdW stacks).\n"
            "  • Broyden  — quasi-Newton; best for publishable / tight "
            "stages on organic-on-metal interfaces, surfaces, "
            "anything where CG oscillates.\n"
            "  • FIRE     — MD-inspired; robust on rough landscapes "
            "(random builder guesses).\n"
            "  • Verlet/Nose — NOT geometry relax; finite-T MD only.\n"
            "  • none     — single-point (skip MD block entirely).\n"
            "Recommended workflow: stage 1 CG (warm-up) → stage 2 "
            "Broyden (refine).  See docs/engines/optimization-tuning.md "
            "§ 2.1 for full algorithm comparison + citations."
        ),
    })
    relax_steps: int = field(default=200, metadata={
        "section": "Compute & budget",
        # Resource-budget cap — same as max_scf_iter, this is "how
        # many outer steps am I willing to wait for", not a
        # convergence target.  Scales with system size, not stage.
        # For ~230-atom Au junctions bump to 500+; the cluster-context
        # closed-shell argument doesn't help convergence speed.
        "workflow_group": "budget",
        "label": "MD.NumCGsteps (max geometry-optimisation steps)",
        "engine_key":  'MD.NumCGsteps (universal for CG / Broyden / FIRE) | MD.FinalTimeStep (Verlet / Nose)',
        "range": (1, 10000),
        "tier":  "advanced",
        "help":  (
            "OUTER loop: max geometry steps the optimiser is allowed "
            "(each step = full SCF + forces + atom move).  In SIESTA "
            "5.4.2 ``MD.NumCGsteps`` is the universal step-count "
            "keyword for CG, Broyden, AND FIRE despite the CG-prefixed "
            "name; Verlet/Nose use MD.FinalTimeStep instead.\n"
            "Per-tier: loose warm-up ~50, publishable ~200, tight "
            "(vib/IR) ~100 (small displacement cap = slow but few "
            "steps from a publishable-converged starting geometry).  "
            "See docs/engines/optimization-tuning.md § 2.10."
        ),
    })
    relax_force_tol: float = field(default=0.02, metadata={
        "section": "Compute & budget",
        "workflow_group": "stage",
        "label": "MD.MaxForceTol", "unit": "eV/Å",
        "engine_key":  'MD.MaxForceTol',
        "id_suffix": "force-tol",
        "range": (0.001, 0.5),
        "tier":  "advanced",
        "help":  (
            "Force-tol stop criterion: max unconstrained atomic force "
            "below which the relaxation declares success.  Ignored by "
            "Verlet/Nose (those are MD, not relax).\n"
            "Per-tier (eV/Å): screening 0.10, loose preopt 0.05, "
            "publishable 0.04 (Gaussian-OPT default), tight (vib/IR) "
            "0.01, very-tight (NEB barrier) 0.001.\n"
            "SIESTA only checks max force.  See docs/engines/"
            "optimization-tuning.md § 2.3 for the 5-criteria "
            "geomeTRIC/Gaussian convention + citations."
        ),
    })
    relax_max_displ: float = field(default=0.05, metadata={
        "section": "Compute & budget",
        "workflow_group": "stage",
        "label": "MD.MaxCGDispl", "unit": "Å",
        "engine_key":  'MD.MaxCGDispl (universal for CG / Broyden / FIRE)',
        "id_suffix": "max-displ",
        "range": (0.001, 0.5),
        "tier":  "advanced",
        "help":  (
            "Displacement cap per optimiser step (Å).  In SIESTA 5.4.2 "
            "``MD.MaxCGDispl`` is the universal displacement-cap keyword "
            "across CG, Broyden, AND FIRE despite the CG-prefixed name.  "
            "Hard ceiling that catches line-search over-shoot.\n"
            "Per-tier (Å): screening 0.30, loose preopt 0.20, "
            "publishable 0.05, tight (vib/IR) 0.02.\n"
            "Symptom of too-large cap: max-force oscillates rather "
            "than descends (e.g. 0.09 → 0.44 → 0.13 → 0.31 → ...).  "
            "Halve the cap.  See docs/engines/optimization-tuning.md "
            "§ 2.2 + the BDT/Au worked example in § 6."
        ),
    })

    # Multi-stage relaxation ladder (#542).  Mirrors PySCFConfig.stages:
    # an ordered list of SiestaStageSpec, each carrying its own per-stage
    # relax_type / relax_steps / relax_force_tol / relax_max_displ +
    # on_nonconvergence policy.  When the form / CLI uses ``cfg.stages``,
    # the generator emits ONE .fdf per enabled stage with ``_<name>``
    # suffixes + a bash runner that loops through them in order with
    # warm-restart via SIESTA's auto ``.XV`` read.  The legacy single-
    # stage relax_* fields above stay -- they're still used by the
    # ``--stage {1,2,3}`` overlay (single-stage CLI) and by any
    # workflow that doesn't multi-stage (energy, spectra, transport).
    # The multi-stage path is opt-in via the form's stage-table widget
    # (#5) or via ``--stages-json`` (#3).  skip_cli on the field so the
    # CLI doesn't auto-generate a flag from it (the dedicated flags in
    # commit #3 handle that with proper JSON parsing).
    stages: List[SiestaStageSpec] = field(
        default_factory=lambda: _default_siesta_stages(),
        metadata={
            # Lives in the "Compute & budget" Stage card -- mirrors the PySCF
            # `stages` field + the design.md 2026-06-22 decision ("cfg.stages
            # lives in Compute & budget's Stage card as a stage-table widget").
            # Was "Optimization"/"budget" -- an unlisted section appended after
            # the schema order, i.e. a form-order regression vs that decision.
            "section":        "Compute & budget",
            "workflow_group": "stage",
            "label":          "Per-stage relaxation ladder",
            "engine_key":     "(molbuilder: cfg.stages -> generator's "
                              "STAGES literal in the emitted fdf-runner "
                              "bash script)",
            "tier":           "advanced",
            "help":           "Ordered list of SiestaStageSpec.  When "
                              "used (form's stage-table widget or "
                              "--stages-json), the generator emits one "
                              ".fdf per enabled stage + a bash runner "
                              "that loops through them, with warm-restart "
                              "via SIESTA's auto .XV read.  Defaults "
                              "match the --stage {1,2,3} overlay's tier "
                              "values so single-stage and multi-stage "
                              "paths produce identical per-stage .fdfs.",
            "skip_cli":       True,
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
        "section": "Compute & budget",
        # Profile-level: MD ensemble identity (initial-velocity-
        # seed temperature for Verlet/Nose); set with the run, not
        # tightened stage-to-stage.
        "workflow_group": "profile",
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
        "section": "Compute & budget",
        # Profile-level: NVT target temperature is MD ensemble
        # identity (Nose-Hoover thermostat target).
        "workflow_group": "profile",
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
        "section": "Compute & budget",
        # Profile-level: MD integration timestep depends on system
        # composition (bonded H needs ~0.5 fs, heavier systems 1
        # fs); chosen with the run, not tightened stage-to-stage.
        "workflow_group": "profile",
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
    # Output + positioning flags (2026-06-13): all of these are set
    # once per project and don't change between stages — tag them as
    # workflow_group="profile" so they fold into the Run profile card
    # alongside SystemLabel / pseudo / spin.  Kills the "Output &
    # positioning" section as a separate untagged surface (the user
    # was hunting for these knobs at the bottom of the form).
    wrap_into_cell: bool = field(default=True, metadata={
        "section": "Output & positioning",
        "workflow_group": "profile",
        "label": "Wrap atoms into cell",
        "engine_key":  '(molbuilder: pre-emission positioning)',
        "help": "fold atoms with fractional coords outside [0,1) back into the cell",
    })
    # (center_in_vacuum removed: centring is intrinsic to the structure-derived
    # vacuum box -- render_fdf centres the molecule via resolve_cell_origin.)

    # When True, every section in the emitted FDF carries inline tuning
    # hints (parameter ranges, what to change when SCF / CG misbehave,
    # etc.) plus a "Troubleshooting" block at the end.
    verbose_comments: bool = field(default=True, metadata={
        "section": "Output & positioning",
        "workflow_group": "profile",
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
        # skip_cli: the CLI's ``--stage`` flag (cli.py::cmd_fdf) is
        # the richer entry point -- it sets cfg.stage AND applies the
        # tier-aligned overlay (relax_type / steps / force_tol /
        # max_displ) via apply_siesta_stage().  Auto-generating a
        # CLI option from this field would collide with that.  Python
        # API + form still set cfg.stage directly.
        "skip_cli":   True,
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
        "workflow_group": "profile",
        "label": "Write XMOL .xyz per step",
        "engine_key":  'WriteCoorXmol',
        "help": "write .xyz of every relaxation step (movie viewer)",
    })
    write_md_history: bool = field(default=True, metadata={
        "section": "Output & positioning",
        "workflow_group": "profile",
        "label": "Write .ANI trajectory",
        "engine_key":  'WriteMDhistory',
        "help": "write the .ANI trajectory file (xcrysden / vmd / OVITO)",
    })
    write_hs: bool = field(default=False, metadata={
        "section": "Output & positioning",
        "workflow_group": "profile",
        "label": "Write H+S matrices",
        "engine_key":  'SaveHS',
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
    # All five Parallel-execution fields tagged workflow_group="budget"
    # (2026-06-13).  Compute layout (MPI ranks, OMP threads, BlockSize,
    # parallel-over-k, memory cap) is "how much compute am I willing
    # to spend on this run" — same category as MaxSCFIterations and
    # MD.NumCGsteps.  Folds the Parallel-execution section into the
    # Compute & budget workflow-group card.
    mpi_np: Optional[int] = field(default=None, metadata={
        "section": "Compute & budget",
        "workflow_group": "budget",
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
        "section": "Compute & budget",
        "workflow_group": "budget",
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
        "section": "Compute & budget",
        "workflow_group": "budget",
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
        "section": "Compute & budget",
        "workflow_group": "budget",
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
        "section": "Compute & budget",
        "workflow_group": "budget",
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
    enable_gpu: bool = field(default=False, metadata={
        "section": "Compute & budget",
        "workflow_group": "budget",
        "label":     "Use GPU (NVIDIA, via ELPA-CUDA)",
        # OPTIONAL accelerator on top of an ELPA ``diag_algorithm``
        # (engines/siesta.md § 13).  It does NOT select ELPA -- that's
        # the ``diag_algorithm`` field.  enable_gpu only decides where an
        # already-chosen ELPA solve runs:
        #   * ON  -> ``Diag.ELPA.GPU .true.``  (GPU-only, no CPU fallback)
        #   * OFF -> ``Diag.ELPA.GPU .false.`` (CPU-ELPA -- explicit
        #            .false. is load-bearing: the source ELPA defaults to
        #            the GPU codepath, so an omitted flag crashes a CPU run;
        #            verified Sol job 57852378).
        # Only meaningful with an ELPA algorithm; GPU + ScaLAPACK is
        # rejected at render time.  Keyword Src/diag_option.F90:138-139
        # (``Diag.ELPA.GPU`` / older ``Diag.ELPA.UseGPU``; we emit the
        # modern form).  Routing to ``molbuilder-siesta-gpu`` is driven by
        # the ELPA *algorithm* choice, not by this toggle (CPU-ELPA still
        # needs the ELPA build).
        "engine_key":  'Diag.ELPA.GPU',
        "id_suffix": "enable-gpu",
        "help":      "OPTIONAL: run the ELPA diagonalization on an NVIDIA "
                     "CUDA GPU.  This does NOT turn ELPA on -- pick the "
                     "solver in ``Diagonalizer`` (ELPA-1STAGE / -2STAGE). "
                     "This toggle only moves that ELPA solve onto the GPU "
                     "(GPU-only, no CPU fallback).  Requires an ELPA "
                     "diagonalizer (GPU + ScaLAPACK is rejected) and an "
                     "NVIDIA GPU on the run machine.  Off = the chosen ELPA "
                     "solver runs on CPU (or ScaLAPACK if that's selected). "
                     "Either ELPA choice routes the job to "
                     "``molbuilder-siesta-gpu`` (the ELPA-linked build); "
                     "the wrapper refuses to emit if that env is missing. "
                     "Affinity hint: GPU favors 1STAGE.",
    })
    diag_algorithm: str = field(default="ScaLAPACK", metadata={
        "section": "Compute & budget",
        "workflow_group": "budget",
        "label":     "Diagonalizer",
        # The EIGENSOLVER choice -- independent of hardware (engines/
        # siesta.md § 13, rewritten 2026-06-29).  ELPA runs on CPU AND
        # GPU; ``enable_gpu`` only moves an ELPA solve onto the GPU.
        #   * ScaLAPACK -> emit NOTHING (SIESTA's built-in Divide-and-
        #     Conquer default); runs in the precompiled ``molbuilder-siesta``.
        #   * ELPA-1STAGE / ELPA-2STAGE (Src/diag_option.F90:264-273) ->
        #     emit ``Diag.Algorithm`` + ``Diag.ELPA.GPU .true./.false.``;
        #     routes to ``molbuilder-siesta-gpu`` (the only ELPA-linked
        #     build) whether or not the GPU is used.
        # Default ScaLAPACK = works out of the box on the precompiled
        # CPU env; ELPA is a freely selectable upgrade (NOT GPU-gated).
        "engine_key":  'Diag.Algorithm',
        "id_suffix": "diag-algorithm",
        "choices":   ("ScaLAPACK", "ELPA-1STAGE", "ELPA-2STAGE"),
        "tier":      "advanced",
        "help":      "Eigensolver for the SCF diagonalization.  ELPA works "
                     "on CPU AND GPU -- it is NOT GPU-only.  Hardware "
                     "affinity (a performance hint, not a restriction): "
                     "GPU favors 1STAGE, CPU favors 2STAGE.\n"
                     "  * ScaLAPACK: SIESTA's built-in Divide-and-Conquer. "
                     "Runs in the precompiled ``molbuilder-siesta`` env "
                     "(no ELPA needed).  The safe default.\n"
                     "  * ELPA-1STAGE: direct tridiagonalisation in one "
                     "step.  Faster on NVIDIA GPUs (arXiv:2502.02460 "
                     "reports ~3x over 2-stage on A100).  Best for GPU.\n"
                     "  * ELPA-2STAGE: tridiagonalise via a banded form; "
                     "the band-reduction exposes more BLAS-3 work, so it "
                     "is typically faster on CPU.  Best for CPU.\n"
                     "Both ELPA variants ship in the SAME library (ELPA "
                     "2024.05.001) -- these are algorithmic strategies, "
                     "not versions.  Selecting either ELPA routes the job "
                     "to ``molbuilder-siesta-gpu`` (the ELPA-linked build); "
                     "turn ``Use GPU`` on to run that ELPA solve on the GPU "
                     "(GPU-only, no CPU fallback).",
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


# --------------------------------------------------------------------- #
#  SIESTA stage presets (minimum-viable per-stage defaults)             #
#                                                                       #
#  Anchors the 3-stage workflow ("stage1 CG warm-up -> stage2 Broyden   #
#  publishable -> stage3 Broyden tight crystal-practical") into a       #
#  single overlay applied via ``--stage {1,2,3}`` on the CLI.  Tier     #
#  values match docs/engines/optimization-tuning.md sect. 2.3.1's      #
#  system-type-aware framework.                                         #
#                                                                       #
#  This is the precursor to #542 SIESTA staged-opt (full staged ladder #
#  + per-stage SiestaStageSpec dataclass + form widget); for now it    #
#  ships as a CLI-only overlay so the user can generate stage2 /       #
#  stage3 fdfs with one flag instead of remembering 4 separate         #
#  --relax-* overrides.                                                 #
# --------------------------------------------------------------------- #


# Per-stage value overlay.  Each entry is a partial dict of SiestaConfig
# field overrides; the overlay leaves other fields (basis, mesh_cutoff,
# psml_lib, etc.) untouched so the user's other choices ride through.
#
# Stage rationale (per optimization-tuning.md sect. 2.3.1):
#   stage1 = loose preopt:    CG, ~0.05 eV/A, 0.2 A displacement cap
#   stage2 = publishable:     Broyden, ~0.04 eV/A (Gaussian-OPT default),
#                                      0.05 A displacement cap
#   stage3 = tight crystal:   Broyden, ~0.01 eV/A (VASP EDIFFG=-0.01),
#                                      0.02 A displacement cap, fewer
#                                      max-steps (publishable->tight on
#                                      the same warm-started geom needs
#                                      fewer outer iters)
#
# All three preset CG/Broyden choices align with SIESTA's recommended
# workflow per the optimization-tuning.md sect. 2.1 algorithm comparison
# table: CG only for stage 1 (no memory / robust far from minimum),
# Broyden for any production-tier work (quasi-Newton + best near minimum).
#
# RELATIONSHIP TO #542 (SIESTA staged-opt full data model):
# This overlay is the minimum-viable precursor.  When ``SiestaStageSpec``
# lands as the parallel to PySCFConfig.stages, it should inherit these
# same per-stage tier values as its defaults.  The CLI overlay below
# REMAINS USEFUL after #542 as a one-flag fast path for users who want
# a single-stage fdf (the common case for hand-edited stage1/stage2/
# stage3 fdfs from one command, as opposed to the per-stage ladder
# emitted by #542's full multi-stage script).  Do NOT delete the
# overlay during the #542 refactor; instead use these values as the
# SiestaStageSpec defaults so the two paths stay aligned.
SIESTA_STAGE_PRESETS: Dict[int, Dict[str, Any]] = {
    1: {
        "relax_type":      "CG",
        "relax_steps":     600,
        "relax_force_tol": 0.05,
        "relax_max_displ": 0.20,
    },
    2: {
        "relax_type":      "Broyden",
        "relax_steps":     200,
        "relax_force_tol": 0.04,
        "relax_max_displ": 0.05,
    },
    3: {
        "relax_type":      "Broyden",
        "relax_steps":     100,
        "relax_force_tol": 0.01,
        "relax_max_displ": 0.02,
    },
}


def apply_siesta_stage(cfg: SiestaConfig, stage: int) -> SiestaConfig:
    """Return a copy of *cfg* with the per-stage tier-aligned values
    overlaid for ``stage`` (1, 2, or 3).

    Values overlaid: ``relax_type``, ``relax_steps``, ``relax_force_tol``,
    ``relax_max_displ``.  Every other field is preserved verbatim from
    the input config -- the overlay is intentionally narrow so user
    choices on basis / mesh_cutoff / psml_lib / spin / k-grid ride
    through unchanged.

    The overlay is applied AFTER the user's explicit CLI / form values,
    NOT before.  An explicit ``--relax-force-tol 0.003`` followed by
    ``--stage 3`` would still get the stage-3 value (0.01); if the
    user wants their override to win they should pass ``--stage 3``
    first then their per-knob override last.  The CLI handler at
    ``cli.py::cmd_siesta`` enforces this ordering by applying the
    stage overlay AFTER constructing the cfg from ``--relax-*`` flags.

    Note: this function overlays the FOUR fields above only.  The CLI
    handler at ``cli.py::cmd_siesta`` ALSO sets ``cfg.stage`` (the
    molwatch-log filename suffix marker) when the user passes
    ``--stage N`` -- so the user-facing semantics of the flag are
    "overlay tier knobs AND mark the stage in the filename".  Calling
    this helper directly from Python gives only the four-field
    overlay; set ``cfg.stage`` separately if you want the filename
    behavior too.

    Raises ``ValueError`` for stages outside {1, 2, 3}.
    """
    import dataclasses as _dc
    if stage not in SIESTA_STAGE_PRESETS:
        valid = ", ".join(map(str, sorted(SIESTA_STAGE_PRESETS)))
        raise ValueError(
            f"unknown SIESTA stage {stage!r}; choose from: {valid}")
    overlay = SIESTA_STAGE_PRESETS[stage]
    return _dc.replace(cfg, **overlay)


# --------------------------------------------------------------------- #
#  SiestaStageSpec -- multi-stage SIESTA optimization (#542)             #
#                                                                       #
#  Cross-engine consistent with PySCFConfig.stages from #534:           #
#                                                                       #
#    * Same generic stage NAMES (stage1 / stage2 / stage3) -- the user  #
#      sees one mental model across engines.                            #
#    * Engine-specific per-stage VALUES -- SIESTA's knobs               #
#      (relax_type / relax_steps / relax_force_tol / relax_max_displ)   #
#      are different fields from PySCF's geomeTRIC convergence targets, #
#      but the TIER meaning lines up:                                   #
#                                                                       #
#        stage1 = screening / loose preopt                              #
#        stage2 = publishable (Gaussian-OPT default)                    #
#        stage3 = tight crystal-practical (default DISABLED)            #
#                                                                       #
#    * Warm-start is implicit and ON by default.  Between stages SIESTA #
#      auto-reads ``.XV`` (and ``.DM`` / ``.LWF`` / ``.CG``) based on   #
#      SystemLabel -- no explicit ``read_charge_density`` field needed. #
#      The generated bash runner emits a defensive check at stage 2+    #
#      entry if ``<basename>.XV`` is missing but other ``*.XV`` files    #
#      exist (catches mid-pipeline project renames).  See #2 generator. #
#                                                                       #
#  The SIESTA_STAGE_PRESETS dict above stays as-is -- it's the          #
#  single-stage CLI overlay (--stage {1,2,3}) that ships today.  We     #
#  use those values as the ``_default_stages()`` baseline so the two    #
#  paths produce identical .fdf content per stage.                      #
# --------------------------------------------------------------------- #


@dataclass
class SiestaStageSpec:
    """One stage of the multi-stage SIESTA relaxation pipeline.

    Mirrors :class:`molbuilder.config.pyscf.StageSpec`'s shape -- same
    ``name`` / ``enabled`` / ``on_nonconvergence`` / ``continue_retries``
    cross-engine concepts, engine-specific convergence knobs.

    Defaults match :data:`SIESTA_STAGE_PRESETS` (the existing single-
    stage CLI overlay) so ``--stage 2`` and "stages[1]" produce the
    same SIESTA configure block.

    Per-field ``metadata`` is read by the web layer's stage-table
    schema emitter (#5) -- same convention as PySCFConfig.stages.
    """
    name: str = field(default="stage1", metadata={
        "label":      "Name",
        "help":       "filename suffix (``<JOB>_<name>.fdf``); "
                      "must match [A-Za-z0-9_]+",
        "pattern":    r"^[A-Za-z0-9_]+$",
    })
    enabled: bool = field(default=True, metadata={
        "label":      "Run",
        "help":       "run this stage; uncheck to skip + carry the "
                      "prior stage's geometry (via auto .XV read) forward",
    })
    relax_type: str = field(default="CG", metadata={
        "label":      "Algorithm",
        "choices":    ("CG", "Broyden", "FIRE"),
        "engine_key": "MD.TypeOfRun",
        "help":       "relaxation algorithm.  CG = stable warm-up; "
                      "Broyden = faster for medium/large systems "
                      "(publishable + tight); FIRE = robust on flexible "
                      "geometries.",
    })
    relax_steps: int = field(default=600, metadata={
        "label":      "Max steps", "range": (1, 10000),
        "engine_key": "MD.NumCGsteps",
        "help":       "max relaxation iterations in this stage.  Universal "
                      "across CG / Broyden / FIRE (SIESTA 5.4.2 uses "
                      "MD.NumCGsteps for all three algorithms despite "
                      "the CG-prefixed name).",
    })
    relax_force_tol: float = field(default=0.05, metadata={
        "label":      "Force tol", "unit": "eV/Å", "step": "any",
        "engine_key": "MD.MaxForceTol",
        "help":       "max-force convergence in this stage (eV/Å).  Tier "
                      "ladder: stage1 ~0.05 (loose), stage2 ~0.04 "
                      "(publishable), stage3 ~0.01 (crystal-tight; "
                      "matches VASP EDIFFG=-0.01).",
    })
    relax_max_displ: float = field(default=0.20, metadata={
        "label":      "Δx max", "unit": "Å", "step": "any",
        "engine_key": "MD.MaxCGDispl",
        "help":       "per-step atom-displacement cap (Å).  Universal "
                      "across CG / Broyden / FIRE (SIESTA 5.4.2 uses "
                      "MD.MaxCGDispl for all three).",
    })
    on_nonconvergence: str = field(default="halt", metadata={
        "label":      "If max_steps runs out",
        "choices":    ("proceed", "continue", "halt"),
        "engine_key": "(molbuilder: per-stage non-convergence policy)",
        "help":       "what to do when relax_steps runs out without the "
                      "force tolerance being met: proceed (move on with "
                      "the partial geometry), continue (re-enter SIESTA "
                      "for another relax_steps batch from the current "
                      ".XV), halt (raise + stop the runner).  The LAST "
                      "enabled stage's value is forced to halt at render "
                      "time -- the final tier must succeed or fail loud.",
    })
    continue_retries: int = field(default=1, metadata={
        "label":      "Continue retries", "range": (1, 5),
        "engine_key": "(molbuilder: max SIESTA re-entries when "
                      "on_nonconvergence=continue)",
        "help":       "only meaningful when on_nonconvergence='continue': "
                      "how many additional relax_steps batches before "
                      "falling through to halt.  Total step budget = "
                      "relax_steps * (1 + continue_retries).",
    })


def _default_siesta_stages() -> List["SiestaStageSpec"]:
    """Three-stage cross-engine-consistent default ladder.

    Matches :data:`SIESTA_STAGE_PRESETS` value-for-value so the
    ``--stage {1,2,3}`` overlay and the multi-stage pipeline produce
    identical per-stage .fdfs.  on_nonconvergence defaults mirror
    PySCF's ladder (proceed on warm-up; halt on the publishable +
    tight tiers).
    """
    return [
        SiestaStageSpec(
            name="stage1", enabled=True,
            relax_type="CG", relax_steps=600,
            relax_force_tol=0.05, relax_max_displ=0.20,
            on_nonconvergence="proceed",
        ),
        SiestaStageSpec(
            name="stage2", enabled=True,
            relax_type="Broyden", relax_steps=200,
            relax_force_tol=0.04, relax_max_displ=0.05,
            on_nonconvergence="halt",
        ),
        SiestaStageSpec(
            name="stage3", enabled=False,
            relax_type="Broyden", relax_steps=100,
            relax_force_tol=0.01, relax_max_displ=0.02,
            on_nonconvergence="halt",
        ),
    ]


def validate_siesta_stages(stages: List[SiestaStageSpec]) -> List[str]:
    """Return human-readable error strings for any stage that fails
    field-level checks.  Empty list means valid.

    Mirrors :func:`molbuilder.config.pyscf.validate_stages`'s shape so
    the CLI's per-stage error reporting reads the same across engines.
    """
    errors: List[str] = []
    name_re = re.compile(r"^[A-Za-z0-9_]+$")
    valid_relax = {"CG", "Broyden", "FIRE"}
    valid_policy = {"proceed", "continue", "halt"}
    # Structural invariants the generator can't recover from (parity with
    # pyscf.validate_stages): an empty / all-disabled ladder, and -- the
    # fatal one -- duplicate names, since per-stage fdfs are keyed
    # ``<basename>_<name>.fdf`` and a collision silently overwrites a stage
    # in render_siesta_stage_fdfs.
    if not stages:
        errors.append("stages: list is empty; need at least 1 stage")
        return errors
    if not any(s.enabled for s in stages):
        errors.append(
            "stages: no stage is enabled; either enable one or set "
            "relax_type='none' to skip the optimization loop entirely")
    seen_names: set = set()
    for i, s in enumerate(stages):
        prefix = f"stages[{i}]"
        if not isinstance(s.name, str) or not name_re.match(s.name or ""):
            errors.append(
                f"{prefix}.name = {s.name!r}: must match [A-Za-z0-9_]+")
        elif s.name in seen_names:
            errors.append(
                f"{prefix}.name = {s.name!r}: duplicate; per-stage output "
                f"files would collide (one stage silently overwrites another)")
        else:
            seen_names.add(s.name)
        if s.relax_type not in valid_relax:
            errors.append(
                f"{prefix}.relax_type = {s.relax_type!r}: must be one of "
                f"{sorted(valid_relax)}")
        if not isinstance(s.relax_steps, int) or s.relax_steps < 1:
            errors.append(
                f"{prefix}.relax_steps = {s.relax_steps!r}: must be a "
                f"positive integer")
        if not isinstance(s.relax_force_tol, (int, float)) or s.relax_force_tol <= 0:
            errors.append(
                f"{prefix}.relax_force_tol = {s.relax_force_tol!r}: "
                f"must be a positive number (eV/Å)")
        if not isinstance(s.relax_max_displ, (int, float)) or s.relax_max_displ <= 0:
            errors.append(
                f"{prefix}.relax_max_displ = {s.relax_max_displ!r}: "
                f"must be a positive number (Å)")
        if s.on_nonconvergence not in valid_policy:
            errors.append(
                f"{prefix}.on_nonconvergence = {s.on_nonconvergence!r}: "
                f"must be one of {sorted(valid_policy)}")
        if (not isinstance(s.continue_retries, int)
                or s.continue_retries < 1 or s.continue_retries > 5):
            errors.append(
                f"{prefix}.continue_retries = {s.continue_retries!r}: "
                f"must be an integer in [1, 5]")
    return errors


# Mirrors PySCF's STAGE_STRATEGY_PRESETS shape exactly.  Same preset
# names, same semantic meanings; only difference is which engine the
# enabled-mask gates.  Keep aligned with web/static/lib/form-schema.js
# ``SIESTA_STAGE_STRATEGY_PRESETS`` (added in #6 commit) -- a regex-
# parse drift-guard test will fire if the two ever diverge.
SIESTA_STAGE_STRATEGY_PRESETS: Dict[str, Tuple[bool, ...]] = {
    "publishable": (True,  True,  False),   # stage1 loose + stage2 publishable
    "loose-only":  (True,  False, False),   # stage1 only (CG warm-up)
    "vib-quality": (True,  True,  True),    # all three (TIGHT for vib/IR)
}


def apply_siesta_stage_strategy(
    stages: List[SiestaStageSpec], strategy: str,
) -> List[SiestaStageSpec]:
    """Return a copy of *stages* with ``enabled`` flags overlaid from
    the named preset in :data:`SIESTA_STAGE_STRATEGY_PRESETS`.

    Raises ``ValueError`` on an unknown strategy.  Non-enabled fields
    (relax_*, on_nonconvergence, ...) are preserved verbatim -- the
    preset only chooses which stages run, not how they're tuned.
    """
    import dataclasses as _dc
    if strategy not in SIESTA_STAGE_STRATEGY_PRESETS:
        valid = ", ".join(sorted(SIESTA_STAGE_STRATEGY_PRESETS))
        raise ValueError(
            f"unknown SIESTA stage strategy {strategy!r}; "
            f"choose from: {valid}")
    enables = SIESTA_STAGE_STRATEGY_PRESETS[strategy]
    out: List[SiestaStageSpec] = []
    for i, s in enumerate(stages):
        if i < len(enables):
            out.append(_dc.replace(s, enabled=enables[i]))
        else:
            out.append(_dc.replace(s))
    return out


def siesta_stages_from_dicts(payload: Any) -> List[SiestaStageSpec]:
    """Coerce a list-of-dicts (e.g. parsed from ``--stages-json``) into
    a ``List[SiestaStageSpec]``.

    Unknown keys silently ignored (consistent with PySCF's
    stages_from_dicts).  Missing keys fall back to SiestaStageSpec's
    field defaults.  Type coercion permissive so JSON-from-shell
    payloads round-trip cleanly.
    """
    import dataclasses as _dc
    # TypeError on shape errors -- parity with pyscf.stages_from_dicts and
    # the web _shared coercion boundary (so callers catch one type).
    if not isinstance(payload, list):
        raise TypeError(
            f"stages payload must be a list, got {type(payload).__name__}")
    spec_fields = {f.name: f for f in _dc.fields(SiestaStageSpec)}
    out: List[SiestaStageSpec] = []
    for i, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise TypeError(
                f"stages[{i}] must be a dict, got "
                f"{type(entry).__name__}")
        coerced: Dict[str, Any] = {}
        for k, v in entry.items():
            if k not in spec_fields:
                continue
            f = spec_fields[k]
            if f.type in ("bool", bool):
                if isinstance(v, str):
                    coerced[k] = v.strip().lower() in ("true", "1", "yes", "y")
                else:
                    coerced[k] = bool(v)
            elif f.type in ("int", int):
                coerced[k] = int(v) if not isinstance(v, bool) else v
            elif f.type in ("float", float):
                coerced[k] = float(v)
            else:
                coerced[k] = v
        out.append(SiestaStageSpec(**coerced))
    return out


__all__ = [
    "SiestaConfig",
    "Config",
    "SIESTA_STAGE_PRESETS",
    "apply_siesta_stage",
    "SiestaStageSpec",
    "SIESTA_STAGE_STRATEGY_PRESETS",
    "apply_siesta_stage_strategy",
    "validate_siesta_stages",
    "siesta_stages_from_dicts",
    "_default_siesta_stages",
]
