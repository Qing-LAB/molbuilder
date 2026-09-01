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
    * DM tolerance 1e-5 plus a redundant DM.EnergyTolerance 1e-4 eV
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
from typing import Any, Dict, List, Optional, Tuple

from ..identity import RestartGroup

# Job-layout v1 protocol (docs/execution/job-contracts.md): the basename
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


def _validate_block_size(value):
    """``BlockSize`` is a COUNT of orbitals, so it starts at 1.

    Two states (tuning.md § 2.11): unset is *auto* -- the keyword is not
    emitted and SIESTA uses its own automatic -- or a positive integer,
    honoured verbatim.  ``0`` used to be a third state meaning *"omit the
    keyword"*, which auto now covers; left unrefused it would be written
    into the deck as ``BlockSize 0``, a distribution block holding no
    orbitals.  Refused rather than quietly re-read as auto, because the two
    asks are different and only the user knows which was meant.
    """
    from ..issues import Issue
    if value is None:
        return []                       # auto -- the keyword is omitted
    if not isinstance(value, int) or isinstance(value, bool):
        return [Issue("error",
                      f"block_size = {value!r} must be an integer "
                      f"number of orbitals, or unset for (auto)",
                      "config.block_size")]
    if value < 1:
        return [Issue(
            "error",
            f"block_size = {value} is not a block size -- it is a "
            f"count of orbitals per rank, so the smallest meaningful value "
            f"is 1.  Leave it unset for (auto), which omits the keyword and "
            f"lets SIESTA choose; 0 used to mean that and no longer does "
            f"(tuning.md 2.11)",
            "config.block_size")]
    return []


def _validate_kgrid_displacement(value, cfg=None):
    """Per-component check for SiestaConfig.kgrid_displacement (SIESTA's
    ``displ(3)``, the k-grid origin in grid-vector coordinates).

    Two things are worth saying and one is not.  The shape is an error: a
    two-tuple or a string cannot become the block's fourth column.  A
    component outside [0, 1) is a *warn* -- the displacement is periodic in
    one mesh spacing, so 1.5 names the same point as 0.5 and the user
    probably meant something else.

    The one that matters scientifically: **a shift on an axis sampled at a
    single k-point moves that point off Gamma**, to the zone boundary.  For
    the 1x1x1 default -- a molecule in a box, which is what molbuilder ships
    -- that is simply the wrong point, and nothing downstream would say so.
    See `docs/archive/2026-08-14-template-execution-review.md` § 54.2.

    Not warned: 0.5 on an ODD mesh.  It is a legitimate (if unusual)
    sampling choice, not a mistake, and the ``help`` text already says which
    parity wants which value.
    """
    from ..issues import Issue
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return [Issue(
            "error",
            f"kgrid_displacement must be a 3-tuple of floats; got {value!r}",
            "config.kgrid_displacement",
        )]
    out = []
    # ``cfg`` is None only when a caller checks a value on its own; the
    # cross-field warning below needs the mesh and is skipped without it.
    kgrid = cfg.kgrid if cfg is not None else None
    for i, v in enumerate(value):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            out.append(Issue(
                "error",
                f"kgrid_displacement[{i}] = {v!r} must be a number",
                "config.kgrid_displacement",
            ))
            continue
        if v < 0.0 or v >= 1.0:
            out.append(Issue(
                "warn",
                f"kgrid_displacement[{i}] = {v} is outside [0, 1); the "
                f"displacement is in units of one mesh spacing and wraps, so "
                f"this names the same k-point as {v % 1.0}",
                "config.kgrid_displacement",
            ))
        if (v != 0.0 and isinstance(kgrid, (tuple, list))
                and len(kgrid) == 3 and kgrid[i] == 1):
            out.append(Issue(
                "warn",
                f"kgrid_displacement[{i}] = {v} shifts an axis sampled at a "
                f"SINGLE k-point (kgrid[{i}] = 1), which moves that point "
                f"off Gamma to the zone boundary.  For an isolated molecule "
                f"only Gamma is meaningful -- set this component to 0.",
                "config.kgrid_displacement",
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
                    "docs/execution/job-contracts.md."
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
    #   * Budget card  -- relax_steps + BlockSize + ParallelOverK +
    #                     diag_algorithm
    #   * Staging card -- mpi_np + omp_threads + gpu_count + use_gpu +
    #                     max_memory_mb + continue_retries (the
    #                     workflow_group="staging" members)
    #
    # ``use_gpu`` moved rows here on 2026-08-23 -- on paper only: the
    # field's own metadata 1200 lines below has said `workflow_group:
    # "staging"` all along, and so does the catalogue, which is what the
    # form actually reads.  This comment had it on the Budget card, so the
    # one place a person looks for the layout disagreed with the two places
    # that produce it (`execution/gpu.md` C2).
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
        "category": ("system", "procedure"),
        # `system` FIRST (2026-08-15): the label is the identity of the
        # calculation, and within the Setup card the primary category is
        # what orders the fields -- filed under `procedure` it rendered
        # BELOW the pseudopotential directory, which reads backwards for
        # the first thing a user types.
        "section":  "System",
        # Run-profile identity — what the run IS named.  Lives in the
        # Run profile workflow-group card alongside the system-character
        # knobs (mixing weight, electronic temperature, spin) because
        # the user sets these together at the start of a run and
        # rarely revisits.
        "workflow_group": "setup",
        "label":    "System label (output prefix)",
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
        "category": ("method", "accuracy"),
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
        "label": "Basis size",
        "engine_key":  'PAO.BasisSize',
        "choices": ("SZ", "SZP", "DZ", "DZP", "DZDP",
                    "TZ", "TZP", "TZDP", "TZTP"),
        "help": "Size of the numerical-orbital basis, roughest to tightest.  S/D/T = single / double / triple zeta (radial functions per valence orbital); P/DP/TP = one / two / three polarisation shells.\nDZP is the production default and what most published SIESTA work uses.  SZ and SZP are screening-grade only.  Go to TZP or beyond for vibrational frequencies, weak interactions and anything where basis-set superposition error matters.\nCost grows steeply -- each step up roughly multiplies the orbital count, and diagonalisation scales as its cube.  All nine are accepted by SIESTA 5.4.2 (basis_specs.f::size_name); the manual's option list mentions only four, which is why four of these were missing here until 2026-08-15.",
    })
    pao_energy_shift: float = field(default=0.01, metadata={
        "category": ("method", "accuracy"),
        "section": "Basis & grid",
        "workflow_group": "stage",
        "label": "Orbital confinement (energy shift)", "unit": "Ry",
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
        "help":  """How diffuse the PAO orbitals are, in Ry: the energy rise that defines each orbital's cutoff radius.  Smaller = more diffuse = more accurate = slower.
Per-tier (Ry):
  0.05    fast screening only
  0.01    production (this project's default)
  0.005   accuracy-critical: band gaps, weak interactions, vdW
DEVIATION: SIESTA's own default is 0.02 Ry.  This catalogue starts at 0.01, because its targets are molecules and metal-molecule junctions, where the more diffuse tails are what set adsorption geometry and level alignment.""",
    })

    # Mesh cutoff lives in the "Basis & grid" section in the form
    # (next to the basis-size dropdown) even though it's strictly a
    # real-space-grid parameter; SIESTA users think of "basis + grid"
    # together when sizing their run.
    mesh_cutoff: float = field(default=300.0, metadata={
        "category": ("accuracy",),
        "section": "Basis & grid",
        # Workflow-group tag (2026-06-13): "stage" means switching the
        # relaxation-stage preset MAY rewrite this field.  Three
        # tag values exist (system / stage / budget) — see docs/web/
        # results.md and viewer.js STAGE_PRESETS for the
        # design rationale.  Untagged fields render bare (outside any
        # workflow-group card) and STAGE_PRESETS never touches them.
        "workflow_group": "stage",
        "label": "Real-space grid cutoff", "unit": "Ry",
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
        "help":  """Real-space integration grid (Ry).  Sets the spacing of the 3D mesh SIESTA uses for Hartree + XC potentials, via the plane-wave-equivalent kinetic-energy cutoff.
Per-tier (Ry):
  150      screening (sanity-check only)
  200-250  loose preopt
  350      publishable -- forces stable to < 0.01 eV/Ang on organic + Au
  500+     tight / vibrational -- egg-box noise below 0.001 eV/Ang
           (600 for first-row elements)
Below 150 Ry the forces / energies are noticeably wrong on organic / biomolecule systems; the validator warns below that floor.  Egg-box noise sets the floor for vibrational work — test by varying ±50 Ry.  See docs/engines/tuning.md § 2.6.""",
    })

    # XC
    xc_functional: str = field(default="GGA", metadata={
        "category": ("method",),
        "section": "Exchange-correlation",
        "workflow_group": "profile",
        "label":   "XC functional family",
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
                   "for PBE / PBEsol / LDA -- pick the matching set.\n"
                   "DEVIATION: SIESTA's own default is LDA; this "
                   "catalogue starts at GGA.  The reason is the over-binding above -- "
                   "LDA's systematic error is large enough that essentially "
                   "no current published work on molecules or biomolecules "
                   "uses it for production geometries.  GGA/PBE has been the "
                   "baseline since Perdew, Burke & Ernzerhof, Phys. Rev. "
                   "Lett. 77, 3865 (1996).  LDA remains here because it is "
                   "cheap and legitimate for screening.",
    })
    xc_authors: str = field(default="PBE", metadata={
        "category": ("method",),
        "section": "Exchange-correlation",
        "workflow_group": "profile",
        "label":   "XC parameterisation",
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
                   "PseudoDojo organises downloads by this name.\n"
                   "DEVIATION: SIESTA's own default is PZ, which is the LDA "
                   "parameterisation that goes with its LDA default family.  "
                   "This catalogue starts at PBE because it ships GGA -- the two "
                   "move together, and a family and a parameterisation that "
                   "do not belong to each other is not a configuration "
                   "SIESTA implements.  Change one and check the other.",
    })

    # SCF
    solution_method: str = field(default="diagon", metadata={
        "category": ("method", "convergence"),
        "section": "SCF",
        # Profile-level: SCF solver family is a system-level
        # decision (diagon / OMM / TranSIESTA), set once with XC +
        # basis.  Switching stages MUST NOT rewrite this.
        "workflow_group": "profile",
        "label": "Solution method",
        "engine_key":  'SolutionMethod',
        "choices": ("diagon", "OMM", "transiesta"),
        "help": """Which solver produces the density matrix each SCF step.
  diagon      standard diagonalisation, O(N^3) -- the default, and right for almost everything this project runs
  OMM         order-N; worth it only for systems beyond ~500 atoms
  transiesta  non-equilibrium transport; requires the TranSIESTA build""",
    })
    mixing_weight: float = field(default=0.02, metadata={
        "category": ("convergence",),
        "section": "SCF",
        # System characteristic — depends on what the system IS
        # (metallic / organic / open-shell), NOT on the stage.
        # Switching stages MUST NOT rewrite this.
        "workflow_group": "profile",
        "label": "SCF mixing weight",
        "engine_key":  'SCF.Mixer.Weight',
        "range": (0.001, 0.5),
        "tier":  "advanced",
        "help":  'How much of each new SCF solution is mixed in.  SIESTA\'s own default is 0.25; this catalogue starts at 0.02, deliberately, because the systems it targets (metal junctions, open-shell metals) leave the convergence basin at high weights.  The manual backs the direction: "a low value ... is more likely to converge", at the cost of more SCF steps, and the value is "heavily system dependent".\nFIRST THING TO TRY when the SCF oscillates -- lower this before touching anything else (manual: "experimentation with the mixing weight is preferred as a first resort").\nOrganic molecules with no metal converge happily at 0.1-0.25 and will run in far fewer steps; raise it if your system is well-behaved.',
    })
    pulay_history: int = field(default=8, metadata={
        "category": ("convergence",),
        "section": "SCF",
        # Profile-level: DIIS history depth pairs with mixing_weight
        # (also profile) — both are SCF-stability tuning that
        # depends on what the system IS, not on the stage.
        "workflow_group": "profile",
        "label": "Pulay history depth",
        "engine_key":  'SCF.Mixer.History',
        "range": (0, 20),
        "tier":  "advanced",
        "help":  'How many previous SCF steps the mixer uses to predict the next one.  Higher = steadier convergence, at a few vectors of memory.\n8 because SIESTA\'s manual puts 2-6 in a band where "a too low value (say 2-6) might change the convergence properties a lot", advises "around 6 or above", and notes that two different high values barely differ -- so 8 buys the manual\'s advice with margin, at a cost the manual says is negligible.\nDEVIATION: SIESTA\'s own default is 2.\nRaise to 12-20 if the SCF still oscillates AFTER lowering the mixing weight -- the weight is the first thing to try (manual: "experimentation with the mixing weight is preferred as a first resort").\n',
    })
    dm_tolerance: float = field(default=1e-5, metadata={
        "category": ("accuracy",),
        "section": "SCF",
        "workflow_group": "stage",
        "label": "Density-matrix tolerance",
        "engine_key":  'DM.Tolerance',
        "range": (1e-8, 1e-3),
        "tier":  "advanced",
        "help":  """Density-matrix element convergence threshold for the inner SCF loop.  Forces are derived from the converged density -- sloppy SCF -> noisy forces -> optimizer thrashes.
Per-tier (dimensionless):
  1e-3   screening (sanity-check only)
  1e-4   loose preopt / publishable
  1e-5   tight (vib / IR / accurate forces)
  1e-6   very-tight (band structure, phonons)
DEVIATION: SIESTA's own default is 1e-4; this catalogue starts at 1e-5, one decade tighter.  The work this project is built for is relaxations and vibrational analysis, where the forces come out of the converged density and a loose SCF shows up as force noise the optimiser then chases -- which costs more geometry steps than the extra SCF cycles cost.  For single-point screening, 1e-4 is the engine's own answer and is enough.
Rule of thumb: keep SCF tol ~10x tighter than the force-precision target you want at convergence.  See docs/engines/tuning.md § 2.5.""",
    })
    dm_energy_tolerance: float = field(default=1e-4, metadata={
        "category": ("accuracy",),
        "section": "SCF",
        "workflow_group": "stage",
        "label": "SCF free-energy tolerance", "unit": "eV",
        "engine_key":  'DM.EnergyTolerance',
        "range": (1e-8, 1e-1),
        "tier":  "advanced",
        "help": "How little the total FREE energy must change between SCF cycles before that cycle counts as settled.\nONLY HAS AN EFFECT WHEN THE SWITCH BELOW IS ON.  SIESTA reads this value either way and then ignores it: read_options.F90 loads it into tolerance_FreeE, and siesta_forces.F90 installs it as a criterion only `if (converge_FreeE)`.  Until 2026-08-15 molbuilder wrote this line and never wrote the switch, so the control looked live and could not change any result.\n1e-4 eV is SIESTA's own default, and pairs sensibly with a 1e-5 density-matrix tolerance -- the intent is that the energy test is not the thing that stops you first.",
    })
    # PAIRED WITH THE TOLERANCE ABOVE, and adjacent on purpose: the tolerance
    # does nothing without this switch, and a user meeting one without the
    # other cannot tell that (user, 2026-08-15 -- "placed next to each other
    # and their relation explained").  Same `category` and `group`, declared
    # here, so the form renders them side by side without special-casing.
    scf_energy_converge: bool = field(default=False, metadata={
        "category": ("accuracy",),
        "section": "SCF",
        "workflow_group": "stage",
        "label": 'Also require the free energy to settle',
        "engine_key":  'SCF.FreeE.Converge',
        "tier":  "advanced",
        "help": 'Whether the SCF must ALSO see the free energy settle, not just the density matrix.  Off by default, as in SIESTA.\nHOW SIESTA DECIDES AN SCF IS CONVERGED: it checks several things each cycle and requires ALL THE ENABLED ONES to pass -- a plain AND (scfconvergence_test.F).  Density-matrix change, Hamiltonian change and energy-density-matrix change are ON by default; free-energy and Harris-energy convergence are OFF.  So turning this on can only make the SCF stop LATER.  It never makes a result wrong; it refuses to accept one early.\nWHEN IT EARNS ITS COST: systems with many electronic states near the Fermi level -- metals, metal-molecule junctions, and large periodic cells where the spectrum is dense -- especially at a raised ELECTRONIC temperature (the ElectronicTemperature smearing, not the MD temperature).  There the free energy carries an entropy term (F = E - TS) large enough that the density-matrix criterion can go quiet while the energy is still drifting.  For a molecule with a clear HOMO-LUMO gap this changes nothing but the runtime.\nTHE COST LANDS WHERE THE BENEFIT DOES: those are the same systems with the most expensive SCF cycles, so budget for more of them and check max_scf_iter before turning this on.\nSOURCES: the AND-combination, the per-criterion defaults and the gating are from SIESTA 5.4.2\'s source and manual (see engines/template.md 10b).  The F = E - TS argument and the dense-spectrum reasoning are standard DFT, not statements the manual makes; the manual says only that the smearing temperature is "useful specially for metals".',
    })
    max_scf_iter: int = field(default=1000, metadata={
        "category": ("convergence",),
        "section": "SCF",
        # Resource-budget cap — "how long am I willing to wait" — NOT
        # part of the convergence-target staging.  Switching stages
        # MUST NOT halve / double this value silently.
        "workflow_group": "budget",
        "label": "Max SCF cycles per geometry step",
        "engine_key":  'MaxSCFIterations',
        "range": (10, 5000),
        "tier":  "advanced",
        "help": 'INNER loop: the most self-consistency cycles SIESTA will run inside ONE geometry step.  A relaxation runs at most relax_steps outer steps, and each of those runs at most this many inner cycles (or until DM.Tolerance is met).\n1000 is SIESTA\'s own default, and it is the right guard HERE because this catalogue ships mixing_weight 0.02 against SIESTA\'s 0.25: the manual says a low weight "may result in high number of SCF steps but is more likely to converge", so a run that is converging normally needs more cycles than a stock one.\nThis is a RUNAWAY GUARD, not a budget -- the budget is wall time and continue_retries.  The two failure modes are not symmetric: too high wastes some CPU on a run that was going to fail anyway, while too low KILLS A CONVERGING RUN at the cap and throws away that whole geometry step, which compounds over 200 outer steps.  With a 0.02 mixing weight, several hundred cycles is normal for a metal junction -- do not read this number as a target.',
    })
    # A MEASUREMENT'S SWITCH, adjacent to the cap it modifies: what SIESTA
    # does when max_scf_iter is hit without convergence -- abort (its own
    # default) or accept the unconverged density and continue with a
    # warning.  Optional and unset for ordinary work (the abort protects
    # the budget); the bench pins set False so a capped trial ends cleanly
    # as the single-point measurement it is (project-layout.md 3.2).  This
    # keyword had NO item until 2026-08-19 -- the retired deck-splicer used
    # to invent the line -- so every properly-capped trial ended in
    # ABNORMAL_TERMINATION and no sweep could ever produce a verdict.
    scf_must_converge: Optional[bool] = field(default=None, metadata={
        "category": ("convergence",),
        "section": "SCF",
        "workflow_group": "budget",
        "label": "Abort if the SCF hits its cycle cap",
        "null_label": "(SIESTA default: abort)",
        "optional": True,
        "engine_key":  'SCF.MustConverge',
        "tier":  "advanced",
        "help": "What SIESTA does when a geometry step's SCF loop hits max_scf_iter without meeting its tolerances: abort the run (SIESTA's own default, true), or accept the unconverged density and CONTINUE with a warning (false).\nLeave it unset for ordinary work -- an unconverged density means the forces are noise, and a relaxation that keeps walking on noise wastes every step after the first bad one.  The abort is protecting your budget, not enforcing bureaucracy.\nTHE ONE ORDINARY REASON TO SET false: a run that is a MEASUREMENT rather than a result.  A benchmark trial deliberately caps the SCF at a few cycles to time an iteration (project-layout.md section 3.2's pins); with the abort left on, every properly-capped trial ends in ABNORMAL_TERMINATION and the timing machinery must read a 'failed' run.  The bench pins set this false so a capped trial ends cleanly as the single-point measurement it is.  Until 2026-08-19 this keyword had no catalogue item at all -- the retired deck-splicer used to invent the line -- so no described trial could say it, and no sweep could ever produce a verdict (its every point classified incomplete).",
    })
    electronic_temperature: float = field(default=300.0, metadata={
        # PRIMARY category `system`, not `accuracy` (2026-08-15, user).  The
        # smearing width answers *what kind of system is this* -- does it
        # have a gap? -- which is the same question as net_charge and
        # spin_treatment, and it is set once from the chemistry rather than
        # tightened by a ladder.  Filed under `accuracy` it also put a
        # SECOND "accuracy" legend inside the Run profile card while the
        # real one lived in Convergence targets, so the same word named two
        # different places.
        "category": ("system", "accuracy"),
        "section": "SCF",
        "workflow_group": "profile",
        "label": "Electronic temperature (smearing)", "unit": "K",
        "engine_key":  'ElectronicTemperature',
        "id_suffix": "temperature",
        "range": (0.0, 5000.0),
        "tier":  "advanced",
        "help":  "How sharply the electronic states fill at the Fermi "
                 "level -- the width of the Fermi-Dirac smearing.\n"
                 "THIS IS NOT THE TEMPERATURE OF YOUR SIMULATION.  It is a "
                 "property of the ELECTRONS and it applies to every run "
                 "type, including a 0 K geometry relaxation with no atomic "
                 "motion at all.  Do not confuse it with 'Initial "
                 "temperature' / 'Target temperature', which are about how "
                 "fast the ATOMS move and are read only by Verlet / Nose "
                 "molecular dynamics.  The two are independent: a metal "
                 "needs smearing whether or not its atoms are moving.\n"
                 "WHY IT EXISTS: in a metal, states sit right at the Fermi "
                 "level and swap occupancy between SCF cycles, so the "
                 "density oscillates and never settles.  Smearing lets "
                 "states be partially occupied, which damps that.  A "
                 "molecule with a clear HOMO-LUMO gap does not need it, "
                 "and for such a system the value barely matters.\n"
                 "RAISE IT (1000-2000 K) for a metal or a metal-molecule "
                 "junction whose SCF will not converge -- after lowering "
                 "the mixing weight, which is the first thing to try.  "
                 "LOWER IT toward 0 only for an insulator where you want "
                 "strictly integer occupations.\n"
                 "COST: a raised smearing temperature adds an entropy term "
                 "to the free energy (F = E - TS), which is what makes the "
                 "free-energy convergence criterion worth turning on for "
                 "exactly these systems -- see 'Also require the free "
                 "energy to settle'.  300 K is SIESTA's own default.",
    })

    # k-grid -- Tuple field with custom CLI parsing; not auto-generated
    # by add_dataclass_options (the bridge handles only scalar types).
    # In the schema-driven form this renders as three side-by-side int
    # inputs (kx / ky / kz) under id sub-suffixes "x", "y", "z".
    kgrid: Tuple[int, int, int] = field(default=(1, 1, 1), metadata={
        "category": ("accuracy",),
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
        "label": "k-point mesh",
        "engine_key":  '%block kgrid_Monkhorst_Pack',
        "id_suffix": "k",
        "triple_labels": ("x", "y", "z"),
        "tier":  "basic",
        "help":  ("Monkhorst-Pack sampling of the Brillouin zone, one count "
                  "per axis.\n"
                  "  1x1x1          an isolated molecule -- only Gamma "
                  "matters\n"
                  "  4x4x4 - 8x8x8  a periodic 3D crystal\n"
                  "  n x n x 1      a slab; no sampling along the vacuum "
                  "axis\n"
                  "\n"
                  "Cost scales linearly with the number of k-points.  "
                  "Converge by raising the density ~1.5x per axis: the total "
                  "energy should move less than 1 meV/atom.\n"
                  "\n"
                  "SIESTA reports an EQUIVALENT CUTOFF for whatever mesh you "
                  "give -- a length that says how dense the sampling really "
                  "is.  That number, not the counts, is what makes two "
                  "DIFFERENT cells comparable: the same 4x4x4 on a small and "
                  "a large cell samples them differently."),
        "skip_cli": True,
        # Bounds PER COMPONENT (validation/metadata.py); the form puts
        # them on each of the three inputs so 0 or -4 cannot be typed.
        "range": (1, 64),
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

    # The FOURTH column of that same %block — SIESTA's ``displ(3)``, the
    # k-grid ORIGIN in grid-vector coordinates (``kgridinit.F``: "origin(ix)
    # = sum_j gridk(ix,j)*displ(j)").  Its own item rather than three more
    # numbers on ``kgrid``: it is a separate scientific decision (WHERE the
    # mesh sits, not how fine it is), and a stage may vary one without the
    # other.  molbuilder wrote a hard-coded 0.0 here until 2026-08-14 and
    # could not express the shift SIESTA's own manual example uses.
    kgrid_displacement: Tuple[float, float, float] = field(
        default=(0.0, 0.0, 0.0), metadata={
            "category": ("accuracy",),
            "workflow_group": "stage",
            "label": "k-grid displacement",
            # Same block as kgrid; the note says which column, and
            # ``_bare_anchor`` keeps only the keyword.
            "engine_key": '%block kgrid_Monkhorst_Pack  (displ column)',
            # NO ``section`` / ``id_suffix`` / ``triple_labels`` / ``tier``.
            # Those four are the OLD Build form's keys, and ``section`` is the
            # opt-in that puts a field on it -- retired at `@2` in favour of
            # ``category`` (`engines/template.md` § 5).  The UI is to be rebuilt
            # FROM the template; a new field does not join the form that is
            # being replaced (user, 2026-08-14).  ``category`` is what a surface
            # groups by, and it is here.
            # No scalar ``range``: ``_validate_config_metadata`` refuses one on
            # a tuple-valued field (it cannot compare a 3-tuple against two
            # numbers) and says so as a programmer bug.  ``kgrid`` has the same
            # shape and the same omission.  The [0, 1) bound is per component,
            # so it lives in the validator below.
            "help": (
                "Shifts the k-point mesh off Gamma, one value per axis, in "
                "units of the mesh spacing.  [0,0,0] is Gamma-centred.\n"
                "\n"
                "  0.0    Gamma-centred.  Required for a 1x1x1 mesh (an "
                "isolated molecule), and the safe choice for ODD meshes, "
                "which already contain Gamma.\n"
                "  0.5    The classic Monkhorst-Pack shift.  Use on EVEN "
                "meshes -- it samples better than a Gamma-centred grid of "
                "the same size, and matters most for metals.\n"
                "\n"
                "Axes are independent: [0.5, 0.5, 0.0] shifts two and leaves "
                "the third on Gamma -- which is what a slab wants when the "
                "third axis is vacuum.\n"
                "\n"
                "TRANSPORT: SIESTA forces this to 0 along the transport "
                "direction, whatever you set, because that direction is "
                "sampled at one k-point."),
            "skip_cli": True,
            # Per component.  ADVISORY and inclusive, so the browser box is
            # [0, 1]; the exact half-open rule ([0, 1) and the 1-point-axis
            # case) stays in the callable, which is where refusals live.
            "range": (0.0, 1.0),
            "validate": (lambda value, cfg:
                         _validate_kgrid_displacement(value, cfg)),
        })

    # Relaxation; relax_type="none" disables the MD block entirely.
    # SIESTA 5.4.2 step-count + max-displacement mapping (see
    # siesta/input.py:render_fdf for the full emission code):
    #   CG / Broyden / FIRE -> MD.Steps + MD.MaxDispl
    #     (UNIVERSAL despite CG-prefixed names -- pre-2026-06-23 we
    #      wrongly emitted MD.NumBroydenSteps + MD.MaxDispl, which
    #      SIESTA silently dropped + ran as Single-point; see
    #      decision-log 2026-06-23 in design.md)
    #   Verlet / Nose -> MD.FinalTimeStep + MD.InitialTemperature
    # The labels below are generic; per-engine help text lives in the
    # FDF's verbose comments.
    relax_type: str = field(default="CG", metadata={
        "category": ("procedure",),
        "section": "Compute & budget",
        # 2026-08-07: was ``workflow_group="profile"``, on the reasoning that
        # the relax/MD algorithm family is a run-shape identity choice.  It is
        # not: a LADDER CHANGES THE OPTIMIZER ON PURPOSE -- CG to warm up, then
        # Broyden once the geometry is close -- so the `profile` card's own
        # claim, "doesn't change between stages", is false for this field.
        # Retagged `stage`, which also puts its "vary per stage" box among the
        # ones ticked by default (engines/stages.md § 1.3).
        "workflow_group": "stage",
        "label": "Relaxation / MD algorithm",
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
            "Broyden (refine).  See docs/engines/tuning.md "
            "§ 2.1 for full algorithm comparison + citations."
        ),
    })
    relax_steps: int = field(default=200, metadata={
        "category": ("procedure",),
        "section": "Compute & budget",
        "item_kind":  "deck",
        "expands":    ['MD.Steps', 'MD.FinalTimeStep'],
        # Resource-budget cap — same as max_scf_iter, this is "how
        # many outer steps am I willing to wait for", not a
        # convergence target.  Scales with system size, not stage.
        # For ~230-atom Au junctions bump to 500+; the cluster-context
        # closed-shell argument doesn't help convergence speed.
        "workflow_group": "budget",
        "label": "Max geometry-optimisation steps",
        "engine_key":  'MD.Steps (CG / Broyden / FIRE) | MD.FinalTimeStep (Verlet / Nose)',
        "range": (1, 10000),
        "tier":  "advanced",
        "help": (
            "OUTER loop: max geometry steps the optimiser is allowed "
            "(each step = full SCF + forces + atom move). Which keyword "
            "carries it is decided by MD.TypeOfRun: CG / Broyden / FIRE "
            "relax and bound the loop with ``MD.Steps``; Verlet / Nose "
            "integrate and bound it with ``MD.FinalTimeStep`` instead "
            "(SIESTA 5.4.2, siesta_init.F -- idyn 0 uses MD.Steps, idyn "
            "1-5 use MD.InitialTimeStep..MD.FinalTimeStep). ``MD.Steps`` "
            "DEPRECATES the older ``MD.NumCGsteps``, whose CG-prefixed "
            "name hid that it was never CG-only. Per-tier: loose warm-up "
            "~50, publishable ~200, tight (vib/IR) ~100 (small "
            "displacement cap = slow but few steps from a "
            "publishable-converged starting geometry). See "
            "docs/engines/tuning.md § 2.10. A well-behaved relaxation "
            "converges in 30-150 steps, so 200+ is a safety cap rather "
            "than a target -- it is there to stop a run that is not going "
            "to converge, not to describe one that is. For molecular "
            "dynamics the count is not a cap but a duration: steps x "
            "timestep is the timescale you actually sample, so pick it "
            "from the physics you want to see."
        ),
    })
    relax_force_tol: float = field(default=0.02, metadata={
        "category": ("accuracy",),
        "section": "Compute & budget",
        "workflow_group": "stage",
        "label": "Force convergence threshold", "unit": "eV/Å",
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
            "DEVIATION: SIESTA's own default is 0.04 eV/Å -- the "
            "'publishable' row above.  This catalogue starts at 0.02, twice as "
            "tight, because a relaxation that stops at the loose end leaves "
            "residual forces big enough to contaminate a frequency "
            "calculation run on top of it, and re-relaxing afterwards costs "
            "more than the extra steps did.  For a single-point or a "
            "screening pass, 0.04 is the engine's own answer.\n"
            "SIESTA only checks max force.  See docs/engines/"
            "tuning.md § 2.3 for the 5-criteria "
            "geomeTRIC/Gaussian convention + citations."
        ),
    })
    relax_max_displ: float = field(default=0.05, metadata={
        "category": ("procedure", "convergence"),
        "section": "Compute & budget",
        "workflow_group": "stage",
        "label": "Max displacement per step", "unit": "Å",
        "engine_key":  'MD.MaxDispl (CG / Broyden / FIRE)',
        "id_suffix": "max-displ",
        "range": (0.001, 0.5),
        "tier":  "advanced",
        "help": (
            "Displacement cap per optimiser step (Å). Applies across CG, "
            "Broyden AND FIRE. Hard ceiling that catches line-search "
            "over-shoot. ``MD.MaxDispl`` DEPRECATES the older "
            "``MD.MaxCGDispl`` (SIESTA 5.4.2); same meaning, same 0.2 "
            "Bohr default, and the CG-prefixed name was never CG-only. "
            "DEVIATION: that engine default of 0.2 Bohr is 0.106 Å; this "
            "catalogue starts at 0.05 Å, about half. The cap only ever "
            "LIMITS a step, so a smaller one costs steps and never "
            "accuracy -- and the oscillation below is what a too-large "
            "cap looks like. Raise it back toward 0.2 Å for a cheap first "
            "pass on a structure that starts far from its minimum. "
            "Per-tier (Å): screening 0.30, loose preopt 0.20, publishable "
            "0.05, tight (vib/IR) 0.02. Symptom of too-large cap: "
            "max-force oscillates rather than descends (e.g. 0.09 → 0.44 "
            "→ 0.13 → 0.31 → ...). Halve the cap. See "
            "docs/engines/tuning.md § 2.2 + the BDT/Au worked example in "
            "§ 6. It is a hard ceiling that catches line-search "
            "over-shoot, not a target. The symptom of one set too large "
            "is a maximum force that oscillates instead of descending "
            "(0.09 -> 0.44 -> 0.13 -> 0.31 ...); halve the cap and "
            "continue."
        ),
    })

    # ``continue_retries`` -- the warm-retry budget.  It arrived here when
    # ``SiestaStageSpec`` was deleted (P2 unit 2/3), and engines/stages.md § 3
    # is why it is a SHARED field rather than a stage one: it passes both of
    # § 3's questions -- it survives without a scheduler (running-a-job.md
    # § 3.5: a SINGLE run's wrapper re-enters SIESTA with --continue), and a
    # single run can mean it.  What made it look like a stage property is only
    # where it LANDS, which § 3 says is never the test.
    #
    # It reaches the wrapper via ``jobset.Resources.continue_retries``
    # (job-contracts.md § 6.2's translation table, decided 2026-08-07): the
    # same road ``mpi_np`` and ``omp_threads`` already ride, rather than a
    # second hand-maintained mapping from a stage to its wrapper.  Unlike
    # those two it becomes NO sbatch flag -- it is baked in at install time.
    continue_retries: int = field(default=1, metadata={
        "category": ("execution",),
        "section":        "Compute & budget",
        "item_kind":  "wrapper",
        # MOVED to the staging surface 2026-08-15 (user): it is spent OUTSIDE
        # the engine call.  Nothing here reaches the .fdf -- the wrapper
        # decides, after SIESTA has exited, whether to launch it again from
        # the geometry it reached.  That is a property of how the stage is
        # RUN, which is the staging surface's question, and it sat in the
        # budget card only because a retry costs compute.
        "workflow_group": "staging",
        "label":          "Warm-retry budget",
        # 0 IS A REAL ANSWER: "run once, whatever happens".  The lower bound
        # was 1, so there was no way to say it -- and a BENCHMARK TRIAL is
        # exactly the run that must not retry.  A trial is capped at a few SCF
        # cycles on purpose (3 since 2026-08-21), so it never converges, so the wrapper retried it
        # every time and `summarize` timed the SECOND run.  The wrapper has
        # always handled 0 (`continue_retries and > 0` gates the whole loop);
        # only this bound refused to express it.
        "range":          (0, 5),
        "engine_key":     "(molbuilder: baked into the run wrapper at "
                          "install time; never an .fdf line and never an "
                          "sbatch flag)",
        "tier":           "advanced",
        "help":           "How many extra relaxation batches the wrapper "
                          "may run when a job hits its step cap without "
                          "converging: it re-enters SIESTA with --continue "
                          "from the current .XV.  Total step budget = "
                          "relax_steps x (1 + continue_retries).",
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
        "category": ("procedure",),
        "section": "Compute & budget",
        # Profile-level: MD ensemble identity (initial-velocity-
        # seed temperature for Verlet/Nose); set with the run, not
        # tightened stage-to-stage.
        "workflow_group": "profile",
        "label": "Initial temperature", "unit": "K",
        "engine_key":  'MD.InitialTemperature',
        "range": (0.0, 5000.0),
        "tier":  "advanced",
        "help":  ("Temperature the initial atomic velocities are drawn "
                  "for, in Verlet / Nose molecular dynamics.  IGNORED by "
                  "CG / Broyden / FIRE geometry relaxation -- those don't "
                  "have velocities to seed.\n"
                  "THIS IS ABOUT THE ATOMS, not the electrons.  It is a "
                  "different quantity from 'Electronic temperature "
                  "(smearing)', which shares the word and the unit and "
                  "nothing else: that one sets how sharply electronic "
                  "states fill at the Fermi level and applies to every "
                  "run type, including this one.  Setting them to match "
                  "means nothing -- an MD at 0 K still needs electronic "
                  "smearing if the system is metallic.\n"
                  "DEVIATION: SIESTA's own default is 0 K, i.e. start from "
                  "rest.  This catalogue starts at 300 K because a run seeded at "
                  "0 K spends its opening picoseconds simply acquiring "
                  "thermal motion, and 300 K is both room temperature and "
                  "the condition most reported simulations are run at.  Set "
                  "it to 0 deliberately if you want the cold start."),
    })
    md_target_temperature: Optional[float] = field(default=None, metadata={
        "category": ("procedure",),
        "section": "Compute & budget",
        # Profile-level: NVT target temperature is MD ensemble
        # identity (Nose-Hoover thermostat target).
        "workflow_group": "profile",
        "label": "Target temperature (NVT)", "unit": "K",
        "engine_key":  'MD.TargetTemperature',
        "null_label": "(use MD.InitialTemperature)",
        "range":      (0.0, 5000.0),       # mirror md_initial_temperature
        "tier":  "advanced",
        "help":  """Nose-Hoover NVT target temperature (K).  Used ONLY by Nose dynamics; CG / Broyden / FIRE / Verlet ignore it.  Defaults to md_initial_temperature when unset.
REQUIRED for the thermostat: without it SIESTA defaults the target to 0 K and the run QUENCHES instead of equilibrating.""",
    })
    md_length_timestep: float = field(default=1.0, metadata={
        "category": ("procedure",),
        "section": "Compute & budget",
        # Profile-level: MD integration timestep depends on system
        # composition (bonded H needs ~0.5 fs, heavier systems 1
        # fs); chosen with the run, not tightened stage-to-stage.
        "workflow_group": "profile",
        "label": "MD timestep", "unit": "fs",
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
    # ``restart`` is the ONE field a user sets; the three ``use_save_*``
    # flags below are what it expands into (docs/execution/run-identity.md
    # § 4).  Nobody is asked to keep three engine keys in step -- they state
    # the intent once and the generator does the rest.
    #
    # It is a shared-schema field, not a stage field: engines/stages.md § 3
    # ("One field arrives") -- a SINGLE run can mean "continue from what is
    # in this folder" too, which is question 2's test.  A stage may promote
    # it like any other field, and the stage table draws it as the
    # "start from" row (web/task-setup-plan.md § 6).
    restart: str = field(default="continue", metadata={
        "category": ("convergence", "execution"),
        "section": "Compute & budget",
        "item_kind":  "deck",
        "expands":    ['DM.UseSaveDM', 'MD.UseSaveXV', 'MD.UseSaveCG'],
        # MOVED to the staging surface 2026-08-15 (user).  It is not a
        # convergence target -- it is a LINK between two runs, and the other
        # half of that link (`prep --from <attempt>`) is named on the staging
        # side.  Set here, the two could disagree: 'continue' with no --from
        # copies nothing, and --from onto a 'clean' stage places files whose
        # deck omits MD.UseSave* and leaves them unread (run-identity.md § 4,
        # "present but not honoured").  One surface owns both halves.
        "workflow_group": "staging",
        "label": "Start from",
        "choices": ("clean", "continue"),
        "id_suffix": "restart",
        "tier": "advanced",
        "engine_key": ("(molbuilder: one field, one mechanism per "
                       "engine -- SIESTA expands it to DM.UseSaveDM / "
                       "MD.UseSaveXV / MD.UseSaveCG; PySCF emits control "
                       "flow that reads <JOB>.chk and "
                       "<JOB>_optimized.xyz.  Not a single engine key on "
                       "either)"),
        "help": (
            'Whether this run starts from what is already in the folder.\n'
            '  continue  -- read it (the default).  Nothing there is not an '
            "error: the engine starts from the deck's own coordinates.\n"
            '  clean     -- ignore it and start over, OVERWRITING what is there '
            'as the run proceeds.\n'
            "Which files that means is the engine's own: SIESTA reads .XV / .DM "
            '/ .CG, PySCF reads <JOB>_optimized.xyz and <JOB>.chk.\n'
            'Continuing is the default because a run you start in a folder that '
            'already holds a result is a run you started after looking at that '
            'result.  To keep the old state, save it first with `molbuilder '
            'checkpoint save` -- the launcher warns before a clean run overwrites '
            'anything and stops unless you pass --force.'
        ),
    })

    # ``use_save_dm`` / ``use_save_cg`` / ``use_save_xv`` are DELETED here
    # (P3 unit 4, 2026-08-08).  They were the group's members carried
    # individually, which run-identity.md § 4 rule 2 forbids in as many
    # words: "no description can carry its members individually and disagree
    # with itself".  They also made ``restart`` inert -- the renderer read
    # the three booleans and never the field, so `--restart clean` emitted
    # all three flags as .true. and a stage told to start clean continued.
    # Their absence is the proof; SIESTA_RESTART_GROUP below is what
    # replaced them.

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
        "category": ("procedure",),
        "section": "Output & positioning",
        "item_kind":  "produce",
        "workflow_group": "profile",
        "label": "Wrap atoms into cell",
        "engine_key":  '(molbuilder: pre-emission positioning)',
        "help": "Move any atom that sits outside the cell box back inside "
                "it, by shifting it a whole number of cell vectors.\n"
                "MOLBUILDER DOES THIS, NOT SIESTA.  It rewrites the "
                "coordinates that get written into the .fdf, so what you "
                "see in the deck is what ran.  In a periodic crystal it "
                "changes nothing physical: shifting an atom by a whole "
                "lattice vector lands it on an identical position, so the "
                "energy and forces are the same.  It just keeps the "
                "numbers tidy and comparable between runs.\n"
                "WHEN TO TURN IT OFF: a MOLECULE that straddles a cell "
                "face.  Wrapping moves only the atoms that stuck out, so "
                "the molecule is split -- half at one edge of the box, "
                "half at the opposite edge.  The physics is still right "
                "for a periodic calculation, but the structure LOOKS torn "
                "in the viewer, and anything that measures geometry "
                "directly from the coordinates (bond lengths, a centre of "
                "mass, an RMSD against another frame) reads the "
                "box-crossing distance instead of the real one.  If your "
                "system is one molecule in a vacuum box, or a slab with an "
                "adsorbate near an edge, turn this off.\n"
                "It has no effect when the cell was built from the "
                "structure itself: that path already centres the atoms in "
                "the box, so nothing is outside to fold back.",
    })
    # (center_in_vacuum removed: centring is intrinsic to the structure-derived
    # vacuum box -- render_fdf centres the molecule via resolve_cell_origin.)

    # When True, every section in the emitted FDF carries inline tuning
    # hints (parameter ranges, what to change when SCF / CG misbehave,
    # etc.) plus a "Troubleshooting" block at the end.
    verbose_comments: bool = field(default=True, metadata={
        "category": ("procedure",),
        "section": "Output & positioning",
        "item_kind":  "produce",
        "workflow_group": "output",
        "label": "Verbose inline comments",
        "engine_key":  '(molbuilder: comment-block control in the generated input)',
        "help": (
        "Emit inline tuning hints and a Troubleshooting block in the "
        "generated script, in whatever comment syntax that engine uses."),
    })

    # The ``stage`` FIELD left this schema 2026-08-12 (C7): a stage's
    # artifact token ``<NN>_<name>`` is a RENDER ARGUMENT
    # (``render_fdf(..., stage_token=)``), carried by `prep` -- which holds
    # the StageRef -- to the emitter, never stored on the config.  "The
    # emitter that reads it never learns the word" (engines/stages.md
    # § 1.1): a config states WHAT to compute; which rung of a ladder it is
    # belongs to the description and the call.  SystemLabel stays identical
    # across stages, so SIESTA's .XV / .DM / .CG transfer untouched
    # (decision 26); the token's own rules live at decision 27 /
    # ``identity.stage_token``.

    # Output flags
    write_forces: bool = field(default=True, metadata={
        "workflow_group": "output",
        "category": ("procedure",),
        "label": "Write forces each step",
        "help": 'Write the atomic forces into the main .out at every '
                'relaxation or MD step, so the force history can be read '
                'back from the log.\n'
                'DEVIATION: SIESTA ships this off; we ship it on.  A '
                'relaxation whose force history was not recorded cannot be '
                'diagnosed afterwards -- "did it descend or oscillate?" is '
                'the first question about any run that did not converge, and '
                'the answer costs a few lines of text per step.\n'
                'It does NOT control the .FA file.  This help said "write '
                'forces to the .FA file (required for relaxation)" until '
                '2026-08-15 and both halves were wrong: the manual says the '
                'last step\'s forces "can be found in the file .FA" whatever '
                'this flag is set to, and a relaxation runs perfectly well '
                'without either.',
            "engine_key":  'WriteForces',
    })
    write_coor_step: bool = field(default=True, metadata={
        "workflow_group": "output",
        "category": ("procedure",),
        "label": "Write coordinates each step",
        "help": 'Write the atomic coordinates into the main .out at every '
                'relaxation or MD step.\n'
                'DEVIATION: SIESTA defaults this to LongOutput (off unless '
                'you asked for verbose output); we ship it on, for the same '
                'reason as the force history -- the .out is the one file that '
                'always survives, so the trajectory should be recoverable '
                'from it alone.\n'
                'CAUTION -- IT HAS A SIDE EFFECT ON ANOTHER KEYWORD: '
                'WriteMDXmol (the .ANI animation file) defaults to '
                '`.not. WriteCoorStep` (read_options.F90), so turning this ON '
                'turns .ANI OFF unless WriteMDXmol is set explicitly.',
            "engine_key":  'WriteCoorStep',
    })
    write_coor_xmol: bool = field(default=True, metadata={
        "category": ("procedure",),
        "section": "Output & positioning",
        "workflow_group": "output",
        "label": "Write XMOL .xyz",
        "engine_key":  'WriteCoorXmol',
        "help": 'Write an extra <label>.xyz holding the FINAL atomic '
                'coordinates, in Angstrom whatever input format was used, '
                'readable by XMol / JMol / Molden.\n'
                'DEVIATION: SIESTA ships this off; we ship it on, because a '
                'finished relaxation whose result is only inside the .out '
                'has to be re-extracted before anything else can open it.\n'
                'ONE STRUCTURE, NOT A MOVIE.  This help promised ".xyz of '
                'every relaxation step (movie viewer)" until 2026-08-15; the '
                'manual is explicit that the file holds the final '
                'coordinates.  The per-step animation file is .ANI, and it '
                'comes from a different keyword (WriteMDXmol).',
    })
    write_md_history: bool = field(default=True, metadata={
        "category": ("procedure",),
        "section": "Output & positioning",
        "workflow_group": "output",
        "label": "Write MD history (.MD/.MDE)",
        "engine_key":  'WriteMDhistory',
        "help": 'Accumulate the trajectory into <label>.MD -- positions and '
                'velocities (and cell, for a variable cell) at every step, '
                'written UNFORMATTED for post-processing -- plus <label>.MDE, '
                'a short per-step line of energy, temperature and the like.  '
                'Both are appended across runs, so a restarted job extends '
                'them rather than replacing them.\n'
                'DEVIATION: SIESTA ships this off; we ship it on, because it '
                'is the only complete record of what the trajectory did.\n'
                'IT DOES NOT WRITE .ANI.  The label and this help said '
                '"Write .ANI trajectory ... (xcrysden / vmd / OVITO)" until '
                '2026-08-15 and that was wrong: read_options.F90 binds this '
                'keyword to `writmd`, which write_md_record.F routes to '
                '`iomd` (the .MD file).  .ANI is written by `pixmol` under '
                '`writpx`, which is the separate WriteMDXmol keyword.\n'
                'NOTE the .MD file is unformatted, so it is not something a '
                'viewer opens directly.',
    })
    # THE .ANI FILE, which molbuilder silently switched off for two years.
    # Added 2026-08-15 (user) after the deviation sweep traced why no run
    # ever produced one.  Declared next to write_md_history because a reader
    # who wants "the trajectory file" lands on that one first and needs to
    # see, in the same place, that the animation file is a different switch.
    write_md_xmol: bool = field(default=True, metadata={
        "category": ("procedure",),
        "section": "Output & positioning",
        "workflow_group": "output",
        "label": "Write XMOL animation (.ANI)",
        "engine_key":  'WriteMDXmol',
        "help": 'Accumulate every step\'s coordinates into <label>.ANI, a '
                'plain-text multi-frame .xyz in Angstrom that xcrysden, VMD '
                'and OVITO open directly as an animation.  Appended across '
                'runs, so a restart extends it.\n'
                'WHY THIS IS A SWITCH AND NOT JUST ON: SIESTA defaults it to '
                '`.not. WriteCoorStep` (read_options.F90) -- the two keywords '
                'are coupled, and nothing in the form said so.  Because this '
                'project ships WriteCoorStep on, that default resolved to OFF '
                'and no molbuilder run ever wrote a .ANI, while the form '
                'advertised one on a different control (fixed the same day).  '
                'Setting it explicitly is the only way to stop one keyword '
                'silently deciding another.\n'
                'MOLBUILDER DOES NOT READ THIS FILE.  Trajectory coordinates '
                'come from <label>.MD.nc, which carries full double precision '
                'rather than text; .ANI is listed among a run\'s files but '
                'never parsed.  So it is purely for opening the trajectory in '
                'an external viewer -- turn it off if disk matters and you do '
                'not need that, and nothing inside molbuilder changes.',
    })
    write_hs: bool = field(default=True, metadata={
        "category": ("procedure",),
        "section": "Output & positioning",
        "workflow_group": "output",
        "label": "Write H+S matrices",
        "engine_key":  'SaveHS',
        "help": 'Write the Hamiltonian and overlap matrices to <label>.HSX.  The manual: it "contains all relevant information to construct the Brillouin zone Hamiltonian and can thus be used for subsequent density of states calculations" -- and it is what TranSIESTA / TBtrans read for transport.\nDEFAULT CHANGED false -> true on 2026-08-15, to match SIESTA\'s own default (read_options.F90: fdf_get(\'SaveHS\', .true.)).  Shipping it off meant a finished relaxation had no .HSX, so wanting bands, DOS or transport afterwards meant re-running the SCF.  The cost of having it is disk; the cost of not having it is a repeat run.',
    })
    write_molwatch_log: bool = field(default=True, metadata={
        "workflow_group": "output",
        "category": ("procedure",),
        "label": "Write the molwatch trajectory log",
        "help": (
        "Write <job>.molwatch.log alongside the run: per-step "
        "coordinates, energy and forces, in one additive file the Watch "
        "tab reads. It exists so a trajectory can be followed while the "
        "engine is still running."),
            "engine_key":  '(molbuilder: writes <basename>.molwatch.log for the live viewer)',
        # Consumed by the GENERATOR, not the deck: § 7's kind, stated
        # because the engine_key is a molbuilder note (U16).
        "item_kind": "produce",
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
    # Don't confuse with block_size (BlockSize for ScaLAPACK
    # within a rank); rank count is the OUTER parallelism.
    # The parallel-execution family (MPI ranks, OMP threads, GPU count,
    # BlockSize, parallel-over-k, memory cap; the machine-answered ones
    # moved to workflow_group="staging" on 2026-08-15).  Compute layout
    # is "how much compute am I willing to spend on this run" — same category as MaxSCFIterations and
    # MD.Steps.  Folds the Parallel-execution section into the
    # Compute & budget workflow-group card.
    mpi_np: Optional[int] = field(default=None, metadata={
        "category": ("execution",),
        "section": "Compute & budget",
        # NOT a template item: a machine fact, which floor 2 must never
        # name (engines/template.md 7).  It arrives as the ALLOCATION at
        # `prep`, on the machine that will run it (project-layout.md M4).
        # The `section` stays so the Build form can still offer it until
        # the web has a prep surface of its own (P10/P11) -- exposure to a
        # surface and membership of the template are different questions.
        "allocation": True,
        "item_kind":  "wrapper",
        "workflow_group": "staging",
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

    gpu_count: Optional[int] = field(default=None, metadata={
        "category": ("execution",),
        # A machine fact like ``mpi_np`` -- an ALLOCATION ask, never a
        # template value.  As a bench axis it is EXPLICIT (user,
        # 2026-08-21: "explicit is what we need"): declare the device
        # counts to try and the grid enumerates exactly those, one shelf
        # each; absent, the machine proposes the divisors of ``mpi_np``
        # bounded by the recorded device count (generator.md 4.3a).
        "allocation": True,
        "item_kind":  "wrapper",
        "workflow_group": "staging",
        "label":      "GPUs per trial (G)",
        "engine_key":  "(molbuilder: scheduler ``--gres=gpu:<type>:G``; "
                       "not in .fdf)",
        "null_label": "(machine proposes)",
        "range":      (1, 16),
        "help":       "How many GPU devices one trial (or run) asks the "
                      "scheduler for.  Ranks per device follow as "
                      "mpi_np / G, and the split must be EVEN -- ELPA's "
                      "own rule is the same rank count on every device "
                      "(tuning.md 2.12) -- so a bench cell whose mpi_np "
                      "does not divide by G is dropped by name at prep.  "
                      "Declared in the bench block it is an axis: "
                      "gpu_count = [1, 2, 4] measures exactly those "
                      "device counts.  Leave it out and prep proposes "
                      "the divisors of each declared rank count.",
        "skip_cli":   True,
    })

    block_size: Optional[int] = field(default=None, metadata={
        "category": ("execution",),
        # A PLAIN INT.  It was ``decl_type: "pow2"`` until 2026-08-15, and
        # `pow2` does not merely check -- ``template._shape`` SNAPS the value
        # down to the nearest power of two, so a benchmarked 24 silently
        # became 16.  The power-of-two rule is real but it is not this
        # keyword's: the manual states it for ``Diag.BlockSize``, only under
        # a GPU-enabled ELPA, and breaking it is not an error there either
        # (ELPA falls back to the CPU).  `pow2` survives where it belongs --
        # BENCH-MARKS, a constraint the benchmark puts on its own sweep.
        "validate": (lambda value, cfg: _validate_block_size(value)),
        "section": "Compute & budget",
        "workflow_group": "budget",
        "label": "ScaLAPACK block size",
        "engine_key":  'BlockSize',
        "id_suffix": "block-size",
        "null_label": "(auto)",
        "help": "How many consecutive orbitals go to one MPI rank before "
                "the distribution moves to the next -- the ScaLAPACK "
                "block.  It cannot change the answer, only how evenly the "
                "ranks are fed and therefore how long the run takes.\n"
                "TWO STATES.  Left as (auto) the keyword is NOT WRITTEN and "
                "SIESTA uses its own automatic, which is what its manual "
                "declares as the default.  Set to a number, that number is "
                "written verbatim -- which is what you want after "
                "benchmarking, and a benchmark is the only thing that "
                "really answers this: the best block depends on the matrix "
                "size, the rank count, the interconnect and the node's "
                "memory layout at once.\n"
                "molbuilder DERIVED a value here until 2026-08-15, and it "
                "should not have: the guess went into the deck as if it "
                "were a decision, and below four atoms it wrote "
                "BlockSize 1 -- legal, and the opposite of the cache "
                "blocking the parameter exists for.\n"
                "GUIDANCE if you set one by hand: powers of two (16, 32, "
                "64, 128); smaller for few orbitals, larger for thousands. "
                "Stay under n_orbitals / ranks or some rank gets no block "
                "at all.\n"
                "GPU: with an ELPA diagonaliser on the GPU the block must "
                "be a power of two or ELPA silently runs on the CPU. `prep` "
                "realigns it there -- that is the layer that knows the GPU "
                "flag and the rank count (tuning.md 2.11).\n"
                "It does NOT fix ``propor: ERROR: IMAX = 0``.  That claim "
                "was disproved by direct sweep (BS = 1, 2, 4 all crash at "
                "the same mpi_np); propor is a vector-proportionality check "
                "in matel_table's MPI-deduplication step.  Lower the rank "
                "count to clear it.",
        "skip_cli": True,
    })
    parallel_over_k: Optional[bool] = field(default=None, metadata={
        "category": ("execution",),
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
        "category": ("execution",),
        "section": "Compute & budget",
        # NOT a template item: a machine fact, which floor 2 must never
        # name (engines/template.md 7).  It arrives as the ALLOCATION at
        # `prep`, on the machine that will run it (project-layout.md M4).
        # The `section` stays so the Build form can still offer it until
        # the web has a prep surface of its own (P10/P11) -- exposure to a
        # surface and membership of the template are different questions.
        "allocation": True,
        "item_kind":  "wrapper",
        "workflow_group": "staging",
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
        "category": ("execution",),
        "section": "Compute & budget",
        # NOT a template item: a machine fact, which floor 2 must never
        # name (engines/template.md 7).  It arrives as the ALLOCATION at
        # `prep`, on the machine that will run it (project-layout.md M4).
        # The `section` stays so the Build form can still offer it until
        # the web has a prep surface of its own (P10/P11) -- exposure to a
        # surface and membership of the template are different questions.
        "allocation": True,
        "item_kind":  "wrapper",
        "workflow_group": "staging",
        "label":      "Max memory",
        # Not a SIESTA fdf keyword.  Emits ``ulimit -v`` into .run.sh
        # AND ``# runtime.max_memory_mb: N`` into the .fdf so the .out
        # parser can recover the cap via runtime_info.
        "engine_key":  '(molbuilder: memory cap for the run -- ulimit -v in .run.sh / mol.max_memory)',
        "unit":       "MB",
        # Advisory bounds for a surface offering a cap.  Added 2026-08-14 to
        # match PySCF's: the two engines declare ONE item (template.md § 6.3)
        # and a merged item cannot carry two answers.  Advisory only -- the
        # normal state is unset, which means the node's maximum.
        "range":      (100, 1_000_000),
        "null_label": "(no cap)",
        "help":       (
        "How much memory this run may use. Left blank -- the normal "
        "state -- it is the machine's maximum, resolved at prep on the "
        "node that granted it; set a number only when you need a "
        "ceiling. Each engine applies it its own way: SIESTA emits a "
        "SystemMemory hint into the deck and caps the wrapper, PySCF "
        "passes it to mol.max_memory, which is what it consults to "
        "choose in-core versus out-of-core."),
        "skip_cli":   True,
    })
    use_gpu: bool = field(default=False, metadata={
        "category": ("execution",),
        "section": "Compute & budget",
        "workflow_group": "staging",
        "label":     "Use GPU (NVIDIA)",
        # OPTIONAL accelerator on top of an ELPA ``diag_algorithm``
        # (engines/siesta.md § 7).  It does NOT select ELPA -- that's
        # the ``diag_algorithm`` field.  use_gpu only decides where an
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
        # § 6.1: the WRAPPER derives from this value too, and for a
        # DIFFERENT question than ``diag_algorithm``'s.  That one decides
        # which env to activate; this one decides the GPU RUNTIME -- the
        # gres ask, MPS, the NUMA pin, and the rank/thread budget
        # (``runwrap._fdf_requests_gpu`` has eight call sites, only one of
        # them the env route).  Declared 2026-08-13, when T8's walk found
        # the wrapper scanning TWO deck keywords while only one was
        # declared: trusting the declarations alone would have dropped
        # every GPU runtime fact silently.
        "read_by": ("wrapper",),
        # MERGED with PySCF's item 2026-08-23 (ruled 2026-08-13).  One
        # question -- does this run use a GPU -- so one item, `kind="deck"`,
        # each engine's writer rendering its own reach.  `net_charge` is the
        # worked example (`engines/template.md` § 6.3).
        "item_kind":   "deck",
        "expands":     ("Diag.ELPA.GPU",),
        "engine_key":  "Diag.ELPA.GPU (SIESTA) | mf = mf.to_gpu() (PySCF)",
        "id_suffix": "use-gpu",
        "help":      "OPTIONAL: run the ELPA diagonalization on an NVIDIA CUDA GPU.  This does NOT turn ELPA on -- pick the solver in ``Diagonalizer`` (ELPA-1STAGE / -2STAGE). This toggle only moves that ELPA solve onto the GPU (GPU-only, no CPU fallback).  Requires an ELPA diagonalizer (GPU + ScaLAPACK is rejected) and an NVIDIA GPU on the run machine.  Off = the chosen ELPA solver runs on CPU (or ScaLAPACK if that's selected). This toggle is the ONLY thing that needs the source-built ``molbuilder-siesta-gpu`` env: the packaged SIESTA runs ELPA on CPU perfectly well, but its ELPA is built without the GPU entry.  The wrapper refuses to emit if that env is missing. Affinity hint: GPU favors 1STAGE.\n\nPySCF: run the SCF (and geom-opt forces) on an NVIDIA GPU via the gpu4pyscf extension.  The recipe for ``molbuilder-pySCF`` installs the matching ``cupy-cudaNx[ctk]`` + ``gpu4pyscf-cudaNx`` wheels for the project's pinned CUDA toolkit (see ``molbuilder envs doctor molbuilder-pySCF``); the script probes gpu4pyscf at run start and STOPS with an actionable message if the package isn't importable or the GPU is missing / too old (compute capability < 7.0) -- there is no silent CPU fallback: a run that changed where it executed would report a CPU time under a GPU label.",
    })
    diag_algorithm: str = field(default="ScaLAPACK", metadata={
        "category": ("execution",),
        # NO ``read_by``, and that is the finding rather than an omission.
        # It carried ``read_by = ("wrapper",)`` from 2026-08-11 until
        # 2026-08-13, on the belief that an ELPA deck must run in
        # molbuilder-siesta-gpu.  Measured: the packaged SIESTA runs both
        # ELPA stages on CPU (ELPA is compiled in through ELSI), so the
        # solver choice decides no environment and the wrapper derives
        # NOTHING from this value.  ``use_gpu`` is the one item the
        # wrapper reads -- see its declaration above.
        #
        # Declaring ``read_by`` here anyway would be the same defect the
        # key exists to remove, pointing the other way: a dependency
        # asserted where none exists makes the wrapper look like it
        # consults a value it never opens.
        "section": "Compute & budget",
        "workflow_group": "budget",
        "label":     "Diagonalizer",
        # The EIGENSOLVER choice -- independent of hardware (engines/
        # siesta.md § 7, rewritten 2026-06-29).  ELPA runs on CPU AND
        # GPU; ``use_gpu`` only moves an ELPA solve onto the GPU.
        #   * ScaLAPACK -> emit NOTHING (SIESTA's built-in Divide-and-
        #     Conquer default); runs in the precompiled ``molbuilder-siesta``.
        #   * ELPA-1STAGE / ELPA-2STAGE (Src/diag_option.F90:264-273) ->
        #     emit ``Diag.Algorithm`` + ``Diag.ELPA.GPU .true./.false.``.
        #     Runs in the PACKAGED env too: conda-forge's SIESTA carries
        #     ELPA through ELSI and both stages work on CPU (measured
        #     2026-08-13).  No environment follows from this choice.
        # Default ScaLAPACK = SIESTA's own default; ELPA is a freely
        # selectable upgrade, gated by neither GPU nor a source build.
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
                     "Both ELPA variants are algorithmic strategies, not "
                     "versions -- one library ships both.  Either runs on "
                     "CPU in the packaged ``molbuilder-siesta`` env, so "
                     "this choice needs no particular environment.  Turn "
                     "``Use GPU`` on to run that ELPA solve on the GPU "
                     "(GPU-only, no CPU fallback) -- THAT is what needs "
                     "the source-built ``molbuilder-siesta-gpu``.",
    })

    # Pseudopotentials -- psml_lib uses click.Path() in the CLI so it's
    # hand-rolled there; species_order needs comma-string parsing on
    # the CLI side, also hand-rolled.
    psml_lib: Optional[str] = field(default=None, metadata={
        "category": ("method",),
        "section":    "System",
        "item_kind":  "produce",
        # Run-profile identity — which pseudopotential library this
        # run uses is fixed per-project, set alongside SystemLabel
        # and the spin/charge knobs.
        "workflow_group": "setup",
        "label":      "Pseudopotential directory (.psml)",
        "engine_key":  '(molbuilder: stages .psml files next to .fdf; SIESTA reads them by element basename)',
        "null_label": "(none)",
        "help":       "Path to a directory of .psml pseudopotential "
                      "files (one per element).  A path INSIDE the "
                      "projects tree, measured from the tree root: the "
                      "convention is ``pseudopotential``.  (An absolute "
                      "path is accepted if it lies inside the tree; "
                      "``./`` spellings are retired -- pseudos already "
                      "beside the calculation are used without this "
                      "field.)  Do NOT write the ``projects/`` prefix: "
                      "paths are measured from the tree root already.  "
                      "Tip: use the fil"
                      "e-picker button next to this field to browse and avoid typing "
                      "the path by hand.  "
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
        "workflow_group": "output",
        "category": ("procedure",),
        "label": "Stage pseudopotential files",
        "help": "copy psml files into the output directory (alongside the FDF)",
            "engine_key":  '(molbuilder: triggers .psml staging step)',
        "item_kind": "produce",
    })
    # ``List`` and not ``Sequence`` since U16: the template grammar names
    # ``strlist`` for ``List[str]``, and this field is an ITEM -- it orders
    # the ChemicalSpeciesLabel block, which run-identity § 6a calls
    # identity-sensitive, so a template that omitted it did not pin the
    # deck it claims to describe.
    species_order: Optional[List[str]] = field(default=None, metadata={
        "workflow_group": "profile",
        "category": ("system",),
        "label": "Species order",
        "help": "comma-separated species order (e.g. 'C,H,S,Au')",
        "skip_cli": True,
            "engine_key":  '(molbuilder: ChemicalSpeciesLabel block ordering)',
        "item_kind": "produce",
    })

    # Net charge.  When None (default), render_fdf auto-detects from the
    # phosphate protonation state via formal_charge_from_phosphates.
    net_charge: Optional[int] = field(default=None, metadata={
        "category": ("system",),
        "section": "System",
        # Run-profile identity — molecule's charge state is a
        # fundamental property of WHAT you're computing.
        "workflow_group": "profile",
        "label": "Net charge",
        # MERGED with PySCF's `charge` 2026-08-19 -- one question, one name.
        # The engine_key names both spellings because the item now belongs to
        # both engines and neither spelling is THE answer (`template.md` § 6.3).
        "engine_key":  "NetCharge (SIESTA) | gto.M(charge=...) (PySCF)",
        "item_kind": "deck",
        "expands": ("NetCharge", "gto.M"),
        "null_label": "(auto-detect from phosphates)",
        "range": (-10, 10),
        "tier": "basic",
        "help": ("Net charge of the system, in units of |e|.  Default "
                 "(blank) auto-detects from phosphate protonation -- one "
                 "negative charge per backbone phosphate, which is right "
                 "for DNA/RNA from tleap.  For everything else (peptide, "
                 "SMILES, PDB load) auto resolves to 0, so set this "
                 "EXPLICITLY for a charged species: carboxylates "
                 "(Asp/Glu), protonated amines (Lys/Arg/His+), "
                 "sulfonates.  Sign convention: -1 = one extra electron; "
                 "+1 = one missing electron."),
    })

    # HOW SPIN IS TREATED.  Four states, not a boolean (2026-08-15).
    #
    # SIESTA 5.4.2 consolidated three booleans -- ``SpinPolarized``,
    # ``NonCollinearSpin`` and ``SpinOrbit`` -- into ONE keyword taking one
    # of four words, and deprecated all three (``spin_subs.F90``:
    # ``fdf_deprecated('SpinPolarized','Spin')``).  Three independent
    # booleans could contradict each other; one enum cannot.
    #
    # WHY WE NO LONGER EMIT THE v4 FORM.  This field carried a comment
    # saying SIESTA 5.4.2's ``Spin`` path "does not subsequently read
    # Spin.Fix / Spin.Total, so open-shell metals abort at propor"
    # (2026-05-24).  That is NOT true of 5.4.2 and the source says so
    # plainly: ``spin_subs.F90`` reads the deprecated flags into ``opt_old``
    # and then does ``opt = fdf_get('Spin', opt_old)`` -- one variable, the
    # new spelling merely winning -- while ``Spin.Fix`` / ``Spin.Total`` are
    # read in a DIFFERENT file (``read_options.F90``) gated only on
    # ``nspin == 2``, which both spellings produce identically.  Whatever
    # caused ``propor: ERROR: IMAX = 0`` in May, this mechanism is not it
    # here.  Re-verified against the 5.4.2 source 2026-08-15.
    #
    # ``spin_total`` is only meaningful for ``polarized``: SIESTA DIES with
    # *"You can only fix the spin of the system for collinear spin
    # polarized calculations"* if ``Spin.Fix`` is set under non-colinear or
    # spin-orbit.  The validator refuses that combination rather than
    # letting the run reach the queue and abort.
    spin_treatment: str = field(default="non-polarized", metadata={
        "category": ("system",),
        # Still carried for `dataclass_to_form_schema`, which Spectra and
        # Transport still use and which gates visibility on `section`
        # (`web/form-schema.md` 1a).  The Build tab reads the catalogue and
        # ignores this.
        "section":     "Spin",
        # System characteristic — depends on chemistry (open-shell
        # metals / radicals require it), not on stage.
        "workflow_group": "profile",
        "label":       "Spin treatment",
        "choices":     ("non-polarized", "polarized",
                        "non-colinear", "spin-orbit"),
        "engine_key":  "Spin",
        "help":        ("How spin is treated.  non-polarized: spin-"
                        "degenerate, the cheap default.  polarized: "
                        "collinear open-shell, required for radicals / "
                        "transition metals / triplets.  non-colinear: "
                        "moments may point in any direction (canted or "
                        "spiral magnetic order).  spin-orbit: couples spin "
                        "to orbital motion -- matters for heavy elements "
                        "(Au, Bi, Pt) and REQUIRES fully-relativistic "
                        "pseudopotentials.  Only 'polarized' can carry a "
                        "fixed total spin."),
    })
    spin_total: Optional[float] = field(default=None, metadata={
        "category": ("system",),
        "section":     "Spin",
        "item_kind":  "deck",
        "expands":    ['Spin.Fix', 'Spin.Total'],
        "workflow_group": "profile",
        "label":       "Target spin moment",
        "null_label":  "(default)",
        # Emits TWO keys: Spin.Fix .true. + Spin.Total <v>.  Either
        # alone is silently ignored by SIESTA (Spin.Fix without a
        # value to fix; Spin.Total without Spin.Fix to gate the
        # constraint).
        "engine_key":  "Spin.Fix + Spin.Total",
        "help":        """Target total spin moment in mu_B (= the number of unpaired electrons).  Emits BOTH `Spin.Fix .true.` and `Spin.Total <v>`: the first is required or the second is silently ignored, which is why one item writes two keywords.  Only emitted with a polarized spin treatment.
NOTE 0.0 with a POLARIZED treatment asks for a constrained singlet via open-shell DFT (broken-symmetry capable).  Most users wanting a singlet are better served by the non-polarized treatment -- the spin-restricted formalism is cheaper and gives the same answer.  Set 0.0 here only when you specifically want an anti-ferromagnetic / broken-symmetry singlet.""",
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
#  values match docs/engines/tuning.md sect. 2.3.1's      #
#  system-type-aware framework.                                         #
#                                                                       #
#  These values are the ladder's SCIENCE, and they outlive every        #
#  mechanism that has carried them: read as an overlay by ``--stage N`` #
#  for a one-shot deck, and read again by                               #
#  ``siesta/stages.py::default_siesta_stages`` as each stage's          #
#  ``overrides`` in the shipped ladder.  One table, two readers.        #
# --------------------------------------------------------------------- #


# Per-stage value overlay.  Each entry is a partial dict of SiestaConfig
# field overrides; the overlay leaves other fields (basis, mesh_cutoff,
# psml_lib, etc.) untouched so the user's other choices ride through.
#
# Stage rationale (per tuning.md sect. 2.3.1):
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
# workflow per the tuning.md sect. 2.1 algorithm comparison
# table: CG only for stage 1 (no memory / robust far from minimum),
# Broyden for any production-tier work (quasi-Newton + best near minimum).
#
# WHY THE ONE-SHOT OVERLAY STAYS (``apply_siesta_stage``, below).  It is a
# one-flag fast path to a SINGLE tier-N deck, which is a different request
# from "run the ladder" -- and since both readers take their values from
# this one table, the deck ``--stage 2`` writes and the deck stage2 of the
# ladder writes cannot drift apart.
#
# ``SiestaStageSpec`` used to copy these values into its own field defaults,
# which is the copy this table now replaces: a stage carries ``overrides``,
# and the shipped ladder's overrides ARE these dicts (engines/stages.md
# § 1.1 -- an engine config carries no stage list).
#: § 4 rule 1 — SIESTA's identity group, declared in one place.
#:
#: The literal is what every warm file is keyed by (`job-contracts.md § 4.1`);
#: the three keys are what SIESTA reads those files *only* when set (§ 4.2).
#: Both halves are needed, and stating one without the other is how a deck
#: comes to say it resumed while the engine started cold.
#:
#: ``MD.UseSaveCG`` is emitted only for CG relaxations — Broyden, FIRE and the
#: dynamics modes ignore it. That conditionality lives in the renderer beside
#: the optimizer it depends on, not here: this declares what the group *is*,
#: and the emitter decides which members are meaningful for a given run.
#: WHICH KEYWORDS -- spelled here, and PROVEN equal to the catalogue.
#:
#: This is `identity.OUR_FILE_PATTERNS`' arrangement and for the same reason:
#: this module is **L1** and the catalogue reader is **L2**, so importing it
#: here is the violation `tests/test_layering.py` catches (tried 2026-08-18 and
#: reverted).  The fact still has ONE authority -- `[item.restart].expands` --
#: and every PRODUCTION reader goes there through `script_emit.parameter`; this
#: tuple has no production reader left at all.
#:
#: What keeps it honest is a gate, not discipline:
#: `test_the_restart_group_object_is_not_a_second_declaration` asserts identity
#: with the catalogue rather than naming the keywords again, so a tuple that
#: drifts fails rather than quietly becoming a fourth spelling -- which is what
#: it was, in its own order, until 2026-08-18.
SIESTA_RESTART_GROUP = RestartGroup(
    literal="SystemLabel",
    keys=("DM.UseSaveDM", "MD.UseSaveXV", "MD.UseSaveCG"),
    # MEASURED, not assumed (2026-08-18).  This said "SIESTA reads .DM/.CG/.XV
    # only when set", and a deck carrying NONE of these keys, with a `.DM`
    # beside it, printed:
    #     Attempting to read DM from file... Succeeded...
    #     DM from file: <dSpData2D:IO-DM: bdt-e2e-K1C1.DM
    # -- so the read is not gated on the key being present.  Every member is
    # therefore written for BOTH answers: `.true.` to continue and `.false.`
    # to start clean.  Omission is not a refusal, and a design that expressed
    # "clean" by leaving the keys out was expressing nothing.
    #
    # This is the same lesson `Diag.ELPA.GPU` records one file away -- *the
    # explicit `.false.` is load-bearing* -- learned twice, for the same
    # reason: what a keyword does when ABSENT is the engine's business, and
    # the only way to state an intention is to state it.
    mechanism="declared .fdf keys, written for both answers; SIESTA reads "
              ".DM/.CG/.XV unless told .false.",
    field="system_label",
)


#: What each tier is CALLED.  Decision 27 (2026-08-10) put the ordinal in the
#: artifact token (``01_coarse``), which forces these to be descriptive rather
#: than positional: ``bdt_au_01_stage1.fdf`` says the number twice and the
#: science none.  These are the names the browser's preset dropdown has always
#: shown (``index.html`` -- *Stage 1 — Coarse (fast descent)*) and the ones
#: every worked example in ``engines/stages.md`` uses.
#:
#: ONE table, read by both doors: ``default_siesta_stages`` builds the ladder
#: from it and the CLI's ``--stage N`` resolves through it, so the deck
#: ``--stage 2`` writes and the deck tier 2 of the ladder writes cannot drift.
SIESTA_STAGE_NAMES: Dict[int, str] = {1: "coarse", 2: "medium", 3: "tight"}


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
    This function overlays the FOUR fields above only.  (The ``--stage``
    CLI flag that used to drive it went with ``molbuilder fdf``,
    2026-08-11; the filename suffix it also set is a RENDER argument now —
    ``render_fdf(..., stage_token=)``, C7, 2026-08-12.)  The tier values
    remain the shipped ladder's defaults: ``default_siesta_stages`` reads
    the same presets table into ``Stage.overrides``.

    Raises ``ValueError`` for stages outside {1, 2, 3}.
    """
    import dataclasses as _dc
    if stage not in SIESTA_STAGE_PRESETS:
        valid = ", ".join(map(str, sorted(SIESTA_STAGE_PRESETS)))
        raise ValueError(
            f"unknown SIESTA stage {stage!r}; choose from: {valid}")
    overlay = SIESTA_STAGE_PRESETS[stage]
    return _dc.replace(cfg, **overlay)


# Which of the three tiers a named strategy runs.  Pure enable-mask data:
# it says nothing about how a stage is tuned, only which ones are in the
# ladder.  ``siesta/stages.py::default_siesta_stages`` reads it together with
# SIESTA_STAGE_PRESETS above to build the shipped ladder; keep aligned with
# config/pyscf.py's STAGE_STRATEGY_PRESETS (the drift-guard test fires if
# the two engines ever diverge; the JS third copy retired 2026-08-22 with
# the stage-table widget).
SIESTA_STAGE_STRATEGY_PRESETS: Dict[str, Tuple[bool, ...]] = {
    "publishable": (True,  True,  False),   # stage1 loose + stage2 publishable
    "loose-only":  (True,  False, False),   # stage1 only (CG warm-up)
    "vib-quality": (True,  True,  True),    # all three (TIGHT for vib/IR)
}


__all__ = [
    "SiestaConfig",
    "Config",
    "SIESTA_STAGE_NAMES",
    "SIESTA_STAGE_PRESETS",
    "apply_siesta_stage",
    "SIESTA_STAGE_STRATEGY_PRESETS",
]
