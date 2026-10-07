"""SiestaConfig -- every parameter the SIESTA .fdf generator emits.

L1 dataclass.  Its field metadata is read by the validation pass
(``molbuilder/validation/``; the readers are `web/form-schema.md` § 1a's)
-- the forms are drawn from the catalogue; the SIESTA deck's spec,
``molbuilder/siesta/input.py:spec_for``, renders the configured values.

Defaults follow current SIESTA best-practice for a small / medium
organic-or-inorganic system that's about to be relaxed:

    * MeshCutoff 300 Ry, PAO.BasisSize DZP, GGA-PBE.
    * DM mixing weight 0.02 (SIESTA tutorials
      recommend this for relaxation; the older default of 0.01 is
      stable but slow, the v5 default of 0.25 is too aggressive
      without the v5 mixing scheme).
    * DM tolerance 1e-5 plus a redundant DM.EnergyTolerance 1e-4 eV
      guard.
    * MaxSCFIterations 1000 -- typical relaxation runs need < 100
      per geometry, but a generous limit avoids stalls on the first
      step where the DM is fresh.
    * Force tol 0.02 eV/Ang and CG max-displ 0.05 Ang -- tighter
      than SIESTA's defaults (0.04 / 0.20 Bohr) but appropriate for
      structures destined for property calculations afterwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from ..identity import RestartGroup
from . import state as _state

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
    """``kgrid``'s SHAPE: three whole counts -- the one thing its bounds
    cannot say.  The bounds are declared, not checked here: the recommended
    (1, 64) is the metadata range, warned per component
    (`validation/metadata.py`, ``outside_range``) -- 1 is the Gamma-only
    floor, more than 32 along any axis is rarely justified, 64 leaves
    headroom for 1-D and 2-D periodic cells -- and 0 or below is the
    catalogue's hard limit (``above``, `engines/template.md` § 5.3), refused
    on every door.  What a rung writes along each axis is the k-point
    mesh's (`kmesh.py`, `engines/siesta.md` § 6.1).
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
                f"kgrid[{i}] = {v!r} must be a whole count of k-points",
                "config.kgrid",
            ))
    return out


def _validate_block_size(value):
    """``BlockSize`` is a whole COUNT of orbitals.

    Two states (tuning.md § 2.11): unset is *auto* -- the keyword is not
    emitted and SIESTA uses its own automatic -- or a positive integer,
    honoured verbatim.  ``0`` is refused by the catalogue's hard limit
    (``above``), so this checks only that the value is an integer.
    """
    from ..issues import Issue
    if value is None:
        return []                       # auto -- the keyword is omitted
    if not isinstance(value, int) or isinstance(value, bool):
        return [Issue("error",
                      f"block_size = {value!r} must be an integer "
                      f"number of orbitals, or unset for (auto)",
                      "config.block_size")]
    # 0 or below is the catalogue's hard limit (``above``,
    # `engines/template.md` § 5.3), refused on every door with one
    # message.
    return []


def _validate_kgrid_displacement(value):
    """``kgrid_displacement``'s SHAPE: three numbers, the block's fourth
    column (SIESTA's ``displ(3)``, the k-grid origin in grid-vector
    coordinates) -- a two-tuple or a string cannot become it.

    Its BOUNDS are declared, not checked here: the metadata range [0, 1],
    warned per component (`validation/metadata.py`, ``outside_range``) --
    the offset is periodic in one mesh spacing, so 1.5 names the point 0.5
    does.  **A shift on an axis sampled at a single k-point** -- which moves that
    point off Gamma to the zone boundary -- is the k-point mesh's check
    (`kmesh.check`, `engines/siesta.md` § 6.1), on the mesh the deck writes.

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
    return [Issue("error",
                  f"kgrid_displacement[{i}] = {v!r} must be a number",
                  "config.kgrid_displacement")
            for i, v in enumerate(value)
            if isinstance(v, bool) or not isinstance(v, (int, float))]


def _validate_basename(label: str):
    r"""Return a validate callable for SiestaConfig.system_label /
    PySCFConfig.job_name.  Used as ``metadata["validate"]`` -- the
    validation pass surfaces a clean error-severity ``Issue`` instead
    of letting a malformed basename reach the filesystem.

    **ONE VALIDATOR, TWO ENGINES, AND THAT IS THE WHOLE OF WHAT THEY SHARE.**
    A SIESTA ``SystemLabel`` and a PySCF ``JOB`` are the same KIND of thing --
    the basename every output file of a run is prefixed with -- so what a name
    may be is one rule, stated here and in `execution/job-contracts.md` § 2.5b.
    They are not the same MECHANISM and must not be read as if they were: a
    ``SystemLabel`` is an fdf directive holding a bare token, a ``JOB`` is a
    Python string literal where the **quotes are syntax, not value**.

    **What this rule guarantees a reader.** ``[A-Za-z0-9_-]+`` admits no quote,
    no dot, no space, so a name that reaches disk cannot carry one.  A reader
    recovering one therefore never needs to strip quotes from a SIESTA deck --
    a quoted ``SystemLabel`` is a hand-edit this validator would have refused
    -- while a PySCF reader must always strip them, because there they are how
    Python spells a string.  That asymmetry is the format's, not an
    inconsistency between the two readers, and it is written down here because
    it has been re-derived from the two regexes more than once.

    **Verifying beats extracting.**  `pyscf/layout.py` does not parse the name
    out of a deck; it takes the identity the deck was written FOR and checks
    the deck states it (``^JOB\s*=\s*(['"])<label>\1``), which tolerates
    both emitted spellings by construction and answers the question that
    matters -- *did we get the right result* -- rather than *what token is in
    there*.  Prefer that shape.
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
    # System
    system_label: str = field(default="siesta", metadata={
        "category": ("system", "procedure"),
        # `system` FIRST: the label is the identity of the calculation, and
        # within the Setup card the primary category is what orders the
        # fields.
        "workflow_group": "setup",
        "label":    "System label (output prefix)",
        "engine_key":  'SystemLabel',
        "pattern":  r"^[A-Za-z0-9_\-]+$",
        "validate": _validate_basename("system_label"),
    })

    # NOTE: the vacuum box is NOT a SiestaConfig knob.  Vacuum comes with the
    # STRUCTURE (Structure.vacuum, per-side gap) -- the single source of truth for
    # lattice/vacuum (structure-periodicity.md); the deck's cell is the
    # structure's (`cell.to_engine`).

    # Basis
    basis_size: str = field(default="DZP", metadata={
        "category": ("method", "accuracy"),
        # In the Stage card beside ``mesh_cutoff``, ``pao_energy_shift``
        # and ``kgrid`` -- all "how finely we sample the calculation" knobs
        # that scale with the convergence target.
        "workflow_group": "stage",
        "label": "Basis size",
        "engine_key":  'PAO.BasisSize',
        "choices": ("SZ", "SZP", "DZ", "DZP", "DZDP",
                    "TZ", "TZP", "TZDP", "TZTP"),
    })
    pao_energy_shift: float = field(default=0.01, metadata={
        "category": ("method", "accuracy"),
        "workflow_group": "stage",
        "label": "Orbital confinement (energy shift)", "unit": "Ry",
        "engine_key":  'PAO.EnergyShift',
        # Upper bound 0.05: 0.1 Ry contracts PAO
        # cutoff radii to ~3 Bohr, putting bond energies hundreds of
        # meV off -- well outside any defensible production window.
        "range": (0.001, 0.05),
        "tier":  "advanced",
        # Default 0.01 Ry.  SIESTA's
        # own internal default (0.02) is fine for screening / quick
        # scans but produces under-converged PAO tails for production
        # work; the SIESTA manual itself recommends 0.001-0.01 Ry for
        # "well-converged" calculations.  0.01 is the production-side
        # of "well-converged" -- ~2x slower than 0.02 but bond
        # energies converge to within a few meV instead of tens.
        # Loosen back to 0.02 only for screening; tighten to 0.005
        # for phonon / vibrational work.
    })

    mesh_cutoff: float = field(default=300.0, metadata={
        "category": ("accuracy",),
        "workflow_group": "stage",
        "label": "Real-space grid cutoff", "unit": "Ry",
        "engine_key":  'MeshCutoff',
        # Lower bound 100 Ry: 50 Ry is a screening-grade value that produces
        # noticeably wrong forces / energies for any production work;
        # letting it sit at the slider floor invited silent garbage.
        # 100 Ry is still a reasonable "I'm doing a quick estimate"
        # floor; the validation pass warns at < 150 Ry separately
        # (see _check_siesta_mesh_cutoff in validation/siesta.py) so users
        # picking a low-but-not-tiny value see a soft nudge.
        "range": (100.0, 1000.0),
        "tier":  "basic",
    })

    # XC
    xc_functional: str = field(default="GGA", metadata={
        "category": ("method",),
        "workflow_group": "profile",
        "label":   "XC functional family",
        "engine_key":  'XC.functional',
        "choices": ("LDA", "GGA", "VDW"),
    })
    xc_authors: str = field(default="PBE", metadata={
        "category": ("method",),
        "workflow_group": "profile",
        "label":   "XC parameterisation",
        "engine_key":  'XC.authors',
        # Choices feed the validator's authors->family map for the
        # pseudopotential coverage check (see
        # molbuilder/validation/siesta.py::_check_siesta_pseudo_coverage).
        "choices": ("PBE", "PBEsol", "revPBE", "RPBE", "BLYP",
                    "CA", "PZ", "PW", "DRSLL", "LMKLL"),
    })

    # SCF
    solution_method: str = field(default="diagon", metadata={
        "category": ("method", "convergence"),
        # Profile-level: SCF solver family is a system-level
        # decision (diagon / OMM / TranSIESTA), set once with XC +
        # basis.  Switching stages MUST NOT rewrite this.
        "workflow_group": "profile",
        "label": "Solution method",
        "engine_key":  'SolutionMethod',
        "choices": ("diagon", "OMM", "transiesta"),
    })
    mixing_weight: float = field(default=0.02, metadata={
        "category": ("convergence",),
        # System characteristic — depends on what the system IS
        # (metallic / organic / open-shell), NOT on the stage.
        # Switching stages MUST NOT rewrite this.
        "workflow_group": "profile",
        "label": "SCF mixing weight",
        "engine_key":  'SCF.Mixer.Weight',
        "range": (0.001, 0.5),
        "tier":  "advanced",
    })
    pulay_history: int = field(default=8, metadata={
        "category": ("convergence",),
        # Profile-level: DIIS history depth pairs with mixing_weight
        # (also profile) — both are SCF-stability tuning that
        # depends on what the system IS, not on the stage.
        "workflow_group": "profile",
        "label": "Pulay history depth",
        "engine_key":  'SCF.Mixer.History',
        "range": (0, 20),
        "tier":  "advanced",
    })
    dm_tolerance: float = field(default=1e-5, metadata={
        "category": ("accuracy",),
        "workflow_group": "stage",
        "label": "Density-matrix tolerance",
        "engine_key":  'DM.Tolerance',
        "range": (1e-8, 1e-3),
        "tier":  "advanced",
    })
    dm_energy_tolerance: float = field(default=1e-4, metadata={
        "category": ("accuracy",),
        "workflow_group": "stage",
        "label": "SCF free-energy tolerance", "unit": "eV",
        "engine_key":  'DM.EnergyTolerance',
        "range": (1e-8, 1e-1),
        "tier":  "advanced",
    })
    # PAIRED WITH THE TOLERANCE ABOVE, and adjacent on purpose: the tolerance
    # does nothing without this switch, and a user meeting one without the
    # other cannot tell that (user, 2026-08-15 -- "placed next to each other
    # and their relation explained").  Same `category` and `group`, declared
    # here, so the form renders them side by side without special-casing.
    scf_energy_converge: bool = field(default=False, metadata={
        "category": ("accuracy",),
        "workflow_group": "stage",
        "label": 'Also require the free energy to settle',
        "engine_key":  'SCF.FreeE.Converge',
        "tier":  "advanced",
    })
    max_scf_iter: int = field(default=1000, metadata={
        "category": ("convergence",),
        # Resource-budget cap — "how long am I willing to wait" — NOT
        # part of the convergence-target staging.  Switching stages
        # MUST NOT halve / double this value silently.
        "workflow_group": "budget",
        "label": "Max SCF cycles per geometry step",
        "engine_key":  'MaxSCFIterations',
        "range": (10, 5000),
        "tier":  "advanced",
    })
    # A MEASUREMENT'S SWITCH, adjacent to the cap it modifies: what SIESTA
    # does when max_scf_iter is hit without convergence -- abort (its own
    # default) or accept the unconverged density and continue with a
    # warning.  Optional and unset for ordinary work (the abort protects
    # the budget); the bench pins set False so a capped trial ends cleanly
    # as the single-point measurement it is (project-layout.md 3.2).
    scf_must_converge: Optional[bool] = field(default=None, metadata={
        "category": ("convergence",),
        "workflow_group": "budget",
        "label": "Abort if the SCF hits its cycle cap",
        "null_label": "(SIESTA default: abort)",
        "engine_key":  'SCF.MustConverge',
        "tier":  "advanced",
    })
    electronic_temperature: float = field(default=300.0, metadata={
        # PRIMARY category `system`, not `accuracy` (2026-08-15, user).  The
        # smearing width answers *what kind of system is this* -- does it
        # have a gap? -- which is the same question as net_charge and
        # spin_treatment, and it is set once from the chemistry rather than
        # tightened by a ladder.
        "category": ("system", "accuracy"),
        "workflow_group": "profile",
        "label": "Electronic temperature (smearing)", "unit": "K",
        "engine_key":  'ElectronicTemperature',
        "range": (0.0, 5000.0),
        "tier":  "advanced",
    })

    # k-grid -- three integers; the form renders them as three
    # side-by-side inputs (kx / ky / kz).
    kgrid: Tuple[int, int, int] = field(default=(1, 1, 1), metadata={
        "category": ("accuracy",),
        # In the Stage card (a convergence knob: more
        # k-points → tighter sampling → more cost).
        "workflow_group": "stage",
        "label": "k-point mesh",
        "engine_key":  '%block kgrid_Monkhorst_Pack',
        "tier":  "basic",
        # Bounds PER COMPONENT, a recommendation (validation/metadata.py's
        # `outside_range`); the form puts them on each of the three inputs.
        # 0 or below is the catalogue's hard limit, refused on every door.
        "range": (1, 64),
        # The SHAPE -- three whole counts -- which bounds cannot say.
        "validate": (lambda value, cfg: _validate_kgrid(value)),
    })

    # The FOURTH column of that same %block — SIESTA's ``displ(3)``, the
    # k-grid ORIGIN in grid-vector coordinates (``kgridinit.F``: "origin(ix)
    # = sum_j gridk(ix,j)*displ(j)").  Its own item rather than three more
    # numbers on ``kgrid``: it is a separate scientific decision (WHERE the
    # mesh sits, not how fine it is), and a stage may vary one without the
    # other.
    kgrid_displacement: Tuple[float, float, float] = field(
        default=(0.0, 0.0, 0.0), metadata={
            "category": ("accuracy",),
            "workflow_group": "stage",
            "label": "k-grid displacement",
            # Same block as kgrid; the note says which column, and
            # ``_bare_anchor`` keeps only the keyword.
            "engine_key": '%block kgrid_Monkhorst_Pack  (displ column)',
            # Per component, a recommendation, inclusive: the browser box is
            # [0, 1], and the metadata pass warns outside it
            # (`outside_range`).  The offset wraps, so 1.0 names Gamma again.
            "range": (0.0, 1.0),
            "validate": (lambda value, cfg:
                         _validate_kgrid_displacement(value)),
        })

    # Relaxation; relax_type="none" disables the MD block entirely.
    # SIESTA 5.4.2 step-count + max-displacement mapping (see
    # siesta/input.py:_relaxation_facts for the full emission code):
    #   CG / Broyden / FIRE -> MD.Steps + MD.MaxDispl, one pair for all three
    #   Verlet / Nose -> MD.FinalTimeStep + MD.InitialTemperature
    # The labels below are generic; per-engine help text lives in the
    # FDF's verbose comments.
    relax_type: str = field(default="CG", metadata={
        "category": ("procedure",),
        # A `stage` item, its "vary per stage" box ticked by default
        # (engines/stages.md § 1.3): a LADDER CHANGES THE OPTIMIZER ON
        # PURPOSE -- CG to warm up, then Broyden once the geometry is close.
        "workflow_group": "stage",
        "label": "Relaxation / MD algorithm",
        "engine_key":  'MD.TypeOfRun',
        "choices": ("CG", "Broyden", "FIRE", "Verlet", "Nose", "none"),
    })
    relax_steps: int = field(default=200, metadata={
        "category": ("procedure",),
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
    })
    relax_force_tol: float = field(default=0.02, metadata={
        "category": ("accuracy",),
        "workflow_group": "stage",
        "label": "Force convergence threshold", "unit": "eV/Å",
        "engine_key":  'MD.MaxForceTol',
        "range": (0.001, 0.5),
        "tier":  "advanced",
    })
    # ----- The vibration kind on this engine: the force-constant run. -----
    # ONE knob.  Which atoms are nudged is the structure's (the free atoms,
    # sorted into one run by prep -- siesta/vibration_deck.py), and the run
    # type is fixed by the kind; only the nudge size is a person's choice.
    fc_displacement: float = field(default=0.04, metadata={
        "category": ("accuracy",),
        "workflow_group": "stage",
        "label": "Force-constant displacement", "unit": "Bohr",
        "engine_key":  'FC.Displacement',
        "range": (0.005, 0.2),
        "tier":  "advanced",
    })
    # THE PERSON'S EXPLICIT CHOICE (`engines/vibration.md` § 2.2): unticked,
    # the ladder relaxes first (a `relax` stage before `freq`); ticked, the
    # force constants are taken at the geometry as given and the job's finish
    # answers the statement with the reference-step forces.  Never a
    # refusal.
    already_relaxed: bool = field(default=False, metadata={
        "category": ("procedure",),
        "workflow_group": "profile",
        "label": 'Structure is already relaxed',
        "tier": "basic",
        # `produce`, not `deck`: on this engine the item puts no keyword in
        # the deck (template.md § 6 -- the `restart` precedent); the gate
        # and the job's finish are what consume it.
        "item_kind": "produce",
        "engine_key": "(molbuilder: no engine keyword -- the ladder or the deck decides)",
    })
    # THE THERMOCHEMISTRY'S TEMPERATURE, one meaning on both engines
    # (`engines/vibration.md` § 3.1, § 4.7): the job's finish sums the
    # harmonic vibrational part at it, read from the deck's `vibration`
    # block (§ 5.3).  No pressure beside it -- no SIESTA result has the
    # gas-phase translational term one would enter.
    temperature_K: float = field(default=298.15, metadata={
        "category": ("procedure",),
        "workflow_group": "profile",
        "label": "Thermochemistry temperature", "unit": "K",
        # `produce`, as `already_relaxed` beside it: no SIESTA keyword --
        # the deck's `vibration` block carries it to the finish.
        "item_kind": "produce",
        "engine_key": "(molbuilder: the thermochemistry's temperature -- PySCF's thermo.thermo(temperature=) for a free molecule, the harmonic vibrational sums otherwise and in SIESTA's finish)",
        "range": (1.0, 5000.0),
        "tier":  "advanced",
    })
    relax_max_displ: float = field(default=0.05, metadata={
        "category": ("procedure", "convergence"),
        "workflow_group": "stage",
        "label": "Max displacement per step", "unit": "Å",
        "engine_key":  'MD.MaxDispl (CG / Broyden / FIRE)',
        "range": (0.001, 0.5),
        "tier":  "advanced",
    })

    # ``continue_retries`` -- the warm-retry budget.  engines/stages.md § 3
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
        "item_kind":  "wrapper",
        # On the staging surface (user, 2026-08-15): it is spent OUTSIDE
        # the engine call.  Nothing here reaches the .fdf -- the wrapper
        # decides, after SIESTA has exited, whether to launch it again from
        # the geometry it reached.  That is a property of how the stage is
        # RUN, which is the staging surface's question.
        "workflow_group": "staging",
        "label":          "Warm-retry budget",
        # 0 IS A REAL ANSWER: "run once, whatever happens" -- what a
        # BENCHMARK TRIAL, capped at a few SCF cycles so it never converges,
        # must say.
        "range":          (0, 5),
        "engine_key":     "(molbuilder: baked into the run wrapper at "
                          "install time; never an .fdf line and never an "
                          "sbatch flag)",
        "tier":           "advanced",
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
    # them.  They are on the form so the user at least SEES them on
    # the page; their help text marks
    # them as ignored-for-CG so the form doesn't mislead non-MD users.
    md_initial_temperature: float = field(default=300.0, metadata={
        "category": ("procedure",),
        # Profile-level: MD ensemble identity (initial-velocity-
        # seed temperature for Verlet/Nose); set with the run, not
        # tightened stage-to-stage.
        "workflow_group": "profile",
        "label": "Initial temperature", "unit": "K",
        "engine_key":  'MD.InitialTemperature',
        "range": (0.0, 5000.0),
        "tier":  "advanced",
    })
    md_target_temperature: Optional[float] = field(default=None, metadata={
        "category": ("procedure",),
        # Profile-level: NVT target temperature is MD ensemble
        # identity (Nose-Hoover thermostat target).
        "workflow_group": "profile",
        "label": "Target temperature (NVT)", "unit": "K",
        "engine_key":  'MD.TargetTemperature',
        "null_label": "(use MD.InitialTemperature)",
        "range":      (0.0, 5000.0),       # mirror md_initial_temperature
        "tier":  "advanced",
    })
    md_length_timestep: float = field(default=1.0, metadata={
        "category": ("procedure",),
        # Profile-level: MD integration timestep depends on system
        # composition (bonded H needs ~0.5 fs, heavier systems 1
        # fs); chosen with the run, not tightened stage-to-stage.
        "workflow_group": "profile",
        "label": "MD timestep", "unit": "fs",
        "engine_key":  'MD.LengthTimeStep',
        "range": (0.1, 5.0),
        "tier":  "advanced",
    })

    # SCF / MD continuation
    # ``restart`` is the ONE field a user sets; it expands into the keys of
    # SIESTA_RESTART_GROUP below (docs/execution/run-identity.md § 4).  Nobody is asked to keep three engine keys in step -- they state
    # the intent once and the generator does the rest.
    #
    # It is a shared-schema field, not a stage field: engines/stages.md § 3
    # ("One field arrives") -- a SINGLE run can mean "continue from what is
    # in this folder" too, which is question 2's test.  A stage may promote
    # it like any other field, and the stage table draws it as the
    # "start from" row.
    restart: str = field(default="continue", metadata={
        "category": ("convergence", "execution"),
        "item_kind":  "deck",
        "expands":    ['DM.UseSaveDM', 'MD.UseSaveXV', 'MD.UseSaveCG'],
        # On the staging surface (user, 2026-08-15): it is not a
        # convergence target -- it is a LINK between two runs, and the other
        # half of that link (`prep --from <attempt>`) is named on the staging
        # side.  Set here, the two could disagree: 'continue' with no --from
        # copies nothing, and --from onto a 'clean' stage places files whose
        # deck answers its restart group `.false.` and leaves them unread
        # (run-identity.md § 4,
        # "present but not honoured").  One surface owns both halves.
        "workflow_group": "staging",
        "label": "Start from",
        "choices": ("clean", "continue"),
        "tier": "advanced",
        "engine_key": ("(molbuilder: one field, one mechanism per "
                       "engine -- SIESTA expands it to DM.UseSaveDM / "
                       "MD.UseSaveXV / MD.UseSaveCG; PySCF emits control "
                       "flow that reads <JOB>.chk and "
                       "<JOB>_optimized.xyz.  Not a single engine key on "
                       "either)"),
    })

    # When True, every section in the emitted FDF carries inline tuning
    # hints (parameter ranges, what to change when SCF / CG misbehave,
    # etc.) plus a "Troubleshooting" block at the end.
    verbose_comments: bool = field(default=True, metadata={
        "category": ("procedure",),
        "item_kind":  "produce",
        "workflow_group": "output",
        "label": "Verbose inline comments",
        "engine_key":  '(molbuilder: comment-block control in the generated input)',
    })

    # A stage's artifact token ``<NN>_<name>`` is a RENDER ARGUMENT
    # (``spec_for(..., names=)``, the stage's names), carried by `prep` -- which holds
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
            "engine_key":  'WriteForces',
    })
    write_coor_step: bool = field(default=True, metadata={
        "workflow_group": "output",
        "category": ("procedure",),
        "label": "Write coordinates each step",
            "engine_key":  'WriteCoorStep',
    })
    write_coor_xmol: bool = field(default=True, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "label": "Write XMOL .xyz",
        "engine_key":  'WriteCoorXmol',
    })
    write_md_history: bool = field(default=True, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "label": "Write MD history (.MD/.MDE)",
        "engine_key":  'WriteMDhistory',
    })
    # THE .ANI FILE.  Declared next to write_md_history because a reader
    # who wants "the trajectory file" lands on that one first and needs to
    # see, in the same place, that the animation file is a different switch.
    write_md_xmol: bool = field(default=True, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "label": "Write XMOL animation (.ANI)",
        "engine_key":  'WriteMDXmol',
    })
    write_hs: bool = field(default=True, metadata={
        "category": ("procedure",),
        "workflow_group": "output",
        "label": "Write H+S matrices",
        "engine_key":  'SaveHS',
    })
    write_molwatch_log: bool = field(default=True, metadata={
        "workflow_group": "output",
        "category": ("procedure",),
        "label": "Write the molwatch trajectory log",
            "engine_key":  '(molbuilder: writes <basename>.molwatch.log for the live viewer)',
        # Consumed by the GENERATOR, not the deck: § 7's kind, stated
        # because the engine_key is a molbuilder note (U16).
        "item_kind": "produce",
    })

    # ---------------- Parallel execution (MPI) ----------------
    # Only matter when running `mpirun -np N siesta`; single-rank runs
    # ignore them.
    # MPI rank count for ``mpirun -np N siesta``.
    # Don't confuse with block_size (BlockSize for ScaLAPACK
    # within a rank); rank count is the OUTER parallelism.
    # The parallel-execution family (MPI ranks, OMP threads, GPU count,
    # BlockSize, parallel-over-k, memory cap): how much compute this
    # run is given.
    mpi_np: Optional[int] = field(default=None, metadata={
        "category": ("execution",),
        # NOT a template item: a machine fact, which floor 2 must never
        # name (engines/template.md 7).  It is stated for the run -- its
        # run card or a prep flag (template.md § 6.4, `allocation = true`).
        "allocation": True,
        "item_kind":  "wrapper",
        "workflow_group": "staging",
        "label":      "MPI ranks (np)",
        "engine_key":  '(molbuilder: .run.sh ``mpirun -np N`` only; not in .fdf)',
        "null_label": "(not stated: prep refuses)",
        "range":      (1, 1024),
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
        "label":      "GPUs (G)",
        "engine_key":  "(molbuilder: scheduler ``--gres=gpu:G``; "
                       "not in the deck)",
        "null_label": "(machine proposes)",
        "range":      (1, 16),
    })

    block_size: Optional[int] = field(default=None, metadata={
        "category": ("execution",),
        # A PLAIN INT, written verbatim.  Under a GPU-enabled ELPA, SIESTA
        # itself rounds the diagonaliser's block down to a power of two
        # (`Src/diag_option.F90`, ``elpa_gpu_block_size``).
        "validate": (lambda value, cfg: _validate_block_size(value)),
        "workflow_group": "budget",
        "label": "ScaLAPACK block size",
        "engine_key":  'BlockSize',
        "null_label": "(auto)",
    })
    parallel_over_k: Optional[bool] = field(default=None, metadata={
        "category": ("execution",),
        "workflow_group": "budget",
        "label": "ParallelOverK",
        "engine_key":  'Diag.ParallelOverK',
    })
    # OpenMP threads per MPI rank -- the cores per rank a run states
    # (`cpus_per_task` on `Resources`, translated at `resolve`).  The run
    # script exports it as OMP_NUM_THREADS; None is unstated, which prep
    # refuses (`architecture.md` § 5.2).  The run script pins BLAS to 1 thread per rank so
    # OMP * BLAS doesn't oversubscribe.
    omp_threads: Optional[int] = field(default=None, metadata={
        "category": ("execution",),
        # NOT a template item: a machine fact, which floor 2 must never
        # name (engines/template.md 7).  It is stated for the run -- its
        # run card or a prep flag (template.md § 6.4, `allocation = true`).
        "allocation": True,
        "item_kind":  "wrapper",
        "workflow_group": "staging",
        "label":      "OMP threads per rank",
        # Not a SIESTA fdf keyword.  Emits ``export OMP_NUM_THREADS=N``
        # into .run.sh AND ``# runtime.omp_threads_requested: N`` comment
        # into the .fdf (so the .out parser can recover the requested
        # value when reading the run back via runtime_info).
        "engine_key":  '(molbuilder: .run.sh OMP_NUM_THREADS + .fdf runtime_info comment)',
        "null_label": "(not stated: prep refuses)",
    })
    max_memory_mb: Optional[int] = field(default=None, metadata={
        "category": ("execution",),
        # NOT a template item: a machine fact, which floor 2 must never
        # name (engines/template.md 7).  It is stated for the run -- its
        # run card or a prep flag (template.md § 6.4, `allocation = true`).
        "allocation": True,
        "item_kind":  "wrapper",
        "workflow_group": "staging",
        "label":      "Max memory",
        # Not a SIESTA fdf keyword.  Emits ``ulimit -v`` into .run.sh
        # AND ``# runtime.max_memory_mb: N`` into the .fdf so the .out
        # parser can recover the cap via runtime_info.
        "engine_key":  '(molbuilder: memory cap for the run -- ulimit -v in .run.sh / mol.max_memory)',
        "unit":       "MB",
        # Advisory bounds for a surface offering a cap: the two engines
        # declare ONE item (template.md § 6.3) and a merged item cannot carry
        # two answers.  Advisory only -- the normal state is unset, which
        # means no cap.
        "range":      (100, 1_000_000),
        "null_label": "(no cap)",
    })
    use_gpu: bool = field(default=False, metadata={
        "category": ("execution",),
        "workflow_group": "staging",
        "label":     "Use GPU (NVIDIA)",
        # OPTIONAL accelerator on top of an ELPA ``diag_algorithm``
        # (engines/siesta.md § 7).  It does NOT select ELPA -- that's
        # the ``diag_algorithm`` field.  use_gpu only decides where an
        # already-chosen ELPA solve runs:
        #   * ON  -> ``Diag.ELPA.GPU .true.``  (GPU-only, no CPU fallback)
        #   * OFF -> ``Diag.ELPA.GPU .false.`` (CPU-ELPA, written
        #            explicitly: a CPU-ELPA run with the flag omitted
        #            crashed on Sol, job 57852378).
        # Only meaningful with an ELPA algorithm; GPU + ScaLAPACK is
        # rejected at render time.  Keyword Src/diag_option.F90:138-139
        # (``Diag.ELPA.GPU`` / older ``Diag.ELPA.UseGPU``; we emit the
        # modern form).
        # § 6.1: the WRAPPER derives from this value too -- the env (the
        # GPU build of SIESTA; the solver decides none, `execution/gpu.md`
        # G3) and the GPU runtime: the gres ask, MPS, the NUMA pin.  It is
        # TOLD the value, never reads the deck for it: `resolve` carries it
        # onto the job's resources and every reader asks the job's GPU
        # request (`jobset.model.gpu_request`).
        "read_by": ("wrapper",),
        # ONE item with PySCF's (ruled 2026-08-13): one question -- does
        # this run use a GPU -- so one item, `kind="deck"`,
        # each engine's writer rendering its own reach.  `net_charge` is the
        # worked example (`engines/template.md` § 6.3).
        "item_kind":   "deck",
        "expands":     ("Diag.ELPA.GPU",),
        "engine_key":  "Diag.ELPA.GPU (SIESTA) | mf = mf.to_gpu() (PySCF)",
    })
    diag_algorithm: str = field(default="ScaLAPACK", metadata={
        "category": ("execution",),
        # NO ``read_by``, and that is the finding rather than an omission.
        # Measured: the packaged SIESTA runs both ELPA stages on CPU (ELPA is compiled in through ELSI), so the
        # solver choice decides no environment and the wrapper derives
        # NOTHING from this value.  ``use_gpu`` is the one item the
        # wrapper reads -- see its declaration above.
        #
        # Declaring ``read_by`` here anyway would be the same defect the
        # key exists to remove, pointing the other way: a dependency
        # asserted where none exists makes the wrapper look like it
        # consults a value it never opens.
        "workflow_group": "budget",
        "label":     "Diagonalizer",
        # The EIGENSOLVER choice -- independent of hardware (engines/
        # siesta.md § 7).  ELPA runs on CPU AND
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
        "choices":   ("ScaLAPACK", "ELPA-1STAGE", "ELPA-2STAGE"),
        "tier":      "advanced",
    })

    # Pseudopotentials
    psml_lib: Optional[str] = field(default=None, metadata={
        "category": ("method",),
        "item_kind":  "produce",
        # Run-profile identity — which pseudopotential library this
        # run uses is fixed per-project, set alongside SystemLabel
        # and the spin/charge knobs.
        "workflow_group": "setup",
        "label":      "Pseudopotential directory (.psml)",
        "engine_key":  '(molbuilder: stages .psml files next to .fdf; SIESTA reads them by element basename)',
        "null_label": "(none)",
    })
    copy_psml: bool = field(default=True, metadata={
        "workflow_group": "output",
        "category": ("procedure",),
        "label": "Stage pseudopotential files",
            "engine_key":  '(molbuilder: triggers .psml staging step)',
        "item_kind": "produce",
    })
    # ``List``: the template grammar names ``strlist`` for ``List[str]``,
    # and this field is an ITEM -- it orders the ChemicalSpeciesLabel block,
    # which run-identity § 6a calls identity-sensitive.
    species_order: Optional[List[str]] = field(default=None, metadata={
        "workflow_group": "profile",
        "category": ("system",),
        "label": "Species order",
            "engine_key":  '(molbuilder: ChemicalSpeciesLabel block ordering)',
        "item_kind": "produce",
    })

    # THE ELECTRONIC STATE (`science/chemistry-correctness.md` § 2a) -- the
    # three merged items, declared once in `config/state.py` for both
    # engines.  A blank means *work it out*; `electronic_state` does, for
    # the form, the checks and the deck alike.  SIESTA has no `method`: it
    # is a density-functional code.
    net_charge: Optional[int] = _state.net_charge()
    spin_treatment: Optional[str] = _state.spin_treatment()
    unpaired_electrons: Optional[Union[int, str]] = _state.unpaired_electrons()

    # ================================================================== #
    #  The TRANSPORT kind's parameters                                    #
    #                                                                     #
    #  They live on SiestaConfig and not on a class of their own for the  #
    #  reason `engines/transport.md` 3.2 measures: every transport stage  #
    #  IS a SIESTA run -- the seed is a plain SCF, each electrode is an   #
    #  SCF, the device is an SCF with open boundaries -- so transport is  #
    #  the siesta base minus the relaxation driver plus these.            #
    #                                                                     #
    #  The seven ELECTRONIC-CONTRACT parameters are not here: they are    #
    #  the shared rows (basis_size, mesh_cutoff, kgrid, pao_energy_shift, #
    #  electronic_temperature, xc_functional, xc_authors) tagged          #
    #  `citation = ["transport"]` where they already live, because for    #
    #  this kind the cited run answers them (template.md 6.4).            #
    # ================================================================== #

    transmission_emin_ev: float = field(default=-2.0, metadata={
        "category": ("accuracy", ),
        "item_kind":  "deck",
        "expands":    ['%block TBT.Contour.window'],
        "workflow_group": "stage",
        "label":       "Transmission window, lower edge",
        "engine_key":  "%block TBT.Contour.window",
        "unit":        "eV",
        "range":       (-20.0, 0.0),
        "tier":        "basic",
    })

    transmission_emax_ev: float = field(default=2.0, metadata={
        "category": ("accuracy", ),
        "item_kind":  "deck",
        "expands":    ['%block TBT.Contour.window'],
        "workflow_group": "stage",
        "label":       "Transmission window, upper edge",
        "engine_key":  "%block TBT.Contour.window",
        "unit":        "eV",
        "range":       (0.0, 20.0),
        "tier":        "basic",
    })

    transmission_n_points: int = field(default=401, metadata={
        "category": ("accuracy", ),
        "item_kind":  "deck",
        "expands":    ['%block TBT.Contour.window'],
        "workflow_group": "stage",
        "label":       "Transmission energy points",
        "engine_key":  "%block TBT.Contour.window",
        "range":       (11, 20001),
        "tier":        "basic",
    })

    tbt_k_grid: Tuple[int, int, int] = field(default=(1, 1, 1), metadata={
        "category": ("accuracy", ),
        "item_kind":  "engine",
        "workflow_group": "stage",
        "label":       "Transverse k-grid for T(E)",
        "engine_key":  "TBT.k",
        "range":       (1, 64),
        "tier":        "basic",
    })

    tbt_spin: int = field(default=0, metadata={
        "category": ("system", ),
        "item_kind":  "engine",
        "workflow_group": "profile",
        "label":       "Spin channel for T(E)",
        "engine_key":  "TBT.Spin",
        "range":       (0, 2),
        "tier":        "advanced",
    })

    tbt_elecs_eta_ev: float = field(default=0.001, metadata={
        "category": ("convergence", ),
        "item_kind":  "engine",
        "workflow_group": "stage",
        "label":       "Electrode self-energy broadening",
        "engine_key":  "TBT.Elecs.Eta",
        "unit":        "eV",
        "range":       (0.0, 1.0),
        "tier":        "advanced",
    })

    tbt_contours_eta_ev: float = field(default=0.0, metadata={
        "category": ("convergence", ),
        "item_kind":  "engine",
        "workflow_group": "stage",
        "label":       "Device Green-function broadening",
        "engine_key":  "TBT.Contours.Eta",
        "unit":        "eV",
        "range":       (0.0, 1.0),
        "tier":        "advanced",
    })

    tbt_dos_gf: bool = field(default=True, metadata={
        "category": ("procedure", ),
        "item_kind":  "engine",
        "workflow_group": "output",
        "label":       "Write the Green-function DOS",
        "engine_key":  "TBT.DOS.Gf",
        "tier":        "advanced",
    })

    tbt_dos_a: bool = field(default=True, metadata={
        "category": ("procedure", ),
        "item_kind":  "engine",
        "workflow_group": "output",
        "label":       "Write the spectral DOS per electrode",
        "engine_key":  "TBT.DOS.A",
        "tier":        "advanced",
    })

    tbt_dos_elecs: bool = field(default=True, metadata={
        "category": ("procedure", ),
        "item_kind":  "engine",
        "workflow_group": "output",
        "label":       "Write the bulk electrode DOS",
        "engine_key":  "TBT.DOS.Elecs",
        "tier":        "advanced",
    })

    tbt_t_eig: int = field(default=4, metadata={
        "category": ("procedure", ),
        "item_kind":  "engine",
        "workflow_group": "output",
        "label":       "Transmission eigenchannels",
        "engine_key":  "TBT.T.Eig",
        "range":       (0, 20),
        "tier":        "advanced",
    })

    tbt_t_bulk: bool = field(default=True, metadata={
        "category": ("procedure", ),
        "item_kind":  "engine",
        "workflow_group": "output",
        "label":       "Write the bulk transmission",
        "engine_key":  "TBT.T.Bulk",
        "tier":        "advanced",
    })

    tbt_t_all: bool = field(default=False, metadata={
        "category": ("procedure", ),
        "item_kind":  "engine",
        "workflow_group": "output",
        "label":       "Write all electrode pairs",
        "engine_key":  "TBT.T.All",
        "tier":        "advanced",
    })

    tbt_verbosity: int = field(default=5, metadata={
        "category": ("procedure", ),
        "item_kind":  "engine",
        "workflow_group": "output",
        "label":       "How much tbtrans reports",
        "engine_key":  "TBT.Verbosity",
        "range":       (0, 10),
        "tier":        "advanced",
    })

    negf_eq_pole_ev: float = field(default=10.0, metadata={
        "category": ("convergence", ),
        "item_kind":  "engine",
        "workflow_group": "stage",
        "label":       "Equilibrium pole energy",
        "engine_key":  "TS.Contours.Eq.Pole",
        "unit":        "eV",
        "range":       (1.0, 40.0),
        "tier":        "advanced",
    })

    bias_voltage_v: float = field(default=0.0, metadata={
        "category": ("system", ),
        "item_kind":  "engine",
        "workflow_group": "stage",
        "label":       "Bias voltage",
        "engine_key":  "TS.Voltage",
        "unit":        "V",
        "range":       (-5.0, 5.0),
        "tier":        "basic",
    })

    ts_hs_save: bool = field(default=True, metadata={
        "category": ("procedure", ),
        "item_kind":  "engine",
        "workflow_group": "output",
        "label":       "Write the TranSIESTA .TSHS",
        "engine_key":  "TS.HS.Save",
        "tier":        "advanced",
    })

    negf_neq_eta_ev: float = field(default=0.0, metadata={
        "category": ("convergence", ),
        "item_kind":  "engine",
        "workflow_group": "stage",
        "label":       "Non-equilibrium broadening",
        "engine_key":  "TS.Contours.nEq.Eta",
        "unit":        "eV",
        "range":       (0.0, 1.0),
        "tier":        "advanced",
    })

    electrodes_bulk: bool = field(default=True, metadata={
        "category": ("method", ),
        "item_kind":  "engine",
        "workflow_group": "stage",
        "label":       "Use the electrodes' own bulk Hamiltonian",
        "engine_key":  "TS.Elecs.Bulk",
        "tier":        "advanced",
    })

    electrode_kz: int = field(default=40, metadata={
        "category": ("accuracy", ),
        "item_kind":  "engine",
        "workflow_group": "stage",
        "label":       "Electrode k-points along transport",
        "engine_key":  "%block kgrid_Monkhorst_Pack",
        # A RECOMMENDATION from 20 (warned below); 1 and under is the item's
        # hard limit, refused on every door (`above`, engines/siesta.md 6.1).
        "range":       (20, 200),
        "tier":        "basic",
    })


#: § 4 rule 1 — SIESTA's identity group, declared in one place.
#:
#: The literal is what every warm file is keyed by (`job-contracts.md § 4.1`);
#: the three keys tell SIESTA whether to read those files -- `.true.` to
#: resume, `.false.` to start clean; it reads a .DM it finds unless told
#: `.false.`, and an .XV or .CG only when told `.true.` (§ 4.2,
#: ``mechanism`` below).
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
#: here is the upward import review refuses (`process/code-audit.md` § 1c (e)).
#: The fact still has ONE authority -- `[item.restart].expands` --
#: and every PRODUCTION reader goes there through `script_emit.parameter`; this
#: tuple has no production reader left at all.
#:
#: What keeps it honest is a gate, not discipline:
#: `test_the_restart_group_object_is_not_a_second_declaration` asserts identity
#: with the catalogue rather than naming the keywords again, so a tuple that
#: drifts fails rather than quietly becoming a fourth spelling.
SIESTA_RESTART_GROUP = RestartGroup(
    literal="SystemLabel",
    keys=("DM.UseSaveDM", "MD.UseSaveXV", "MD.UseSaveCG"),
    # MEASURED, not assumed (2026-08-18): a deck carrying NONE of these
    # keys, with a `.DM` beside it, printed:
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
              "a .DM unless told .false., an .XV or .CG only when told .true.",
    field="system_label",
)


#: What each tier is CALLED.  Decision 27 (2026-08-10) put the ordinal in the
#: artifact token (``01_coarse``), which forces these to be descriptive rather
#: than positional: ``bdt_au_01_stage1.fdf`` says the number twice and the
#: science none.  These are the names every worked example in
#: ``engines/stages.md`` uses.
#:
#: ONE table: the ladder builders, the build route and ``jobset init`` name
#: tiers from it.
SIESTA_STAGE_NAMES: Dict[int, str] = {1: "coarse", 2: "medium", 3: "tight"}


# --------------------------------------------------------------------- #
#  SIESTA stage presets (minimum-viable per-stage defaults)             #
#                                                                       #
#  The ladder's SCIENCE: ``siesta/stages.py::default_siesta_stages``    #
#  reads each tier as a stage's ``overrides`` in the shipped ladder     #
#  (engines/stages.md § 1.1 -- an engine config carries no stage list), #
#  and the build route lists them.  Tier values follow                  #
#  docs/engines/tuning.md's tier framework.                             #
# --------------------------------------------------------------------- #
#
# Each entry is a partial dict of SiestaConfig field overrides; other
# fields (basis, mesh_cutoff, psml_lib, etc.) ride through untouched.
#
# Stage rationale (per tuning.md § 2):
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
# workflow per the tuning.md § 2.1 algorithm comparison
# table: CG only for stage 1 (no memory / robust far from minimum),
# Broyden for any production-tier work (quasi-Newton + best near minimum).
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


# THE SHARED TABLE (`config/stages.py`): a strategy says which TIERS run,
# which is not an engine's property.
from .stages import STAGE_STRATEGY_PRESETS


__all__ = [
    "SiestaConfig",
    "SIESTA_STAGE_NAMES",
    "SIESTA_STAGE_PRESETS",
    "STAGE_STRATEGY_PRESETS",
]
