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
from ..issues import Issue

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


def _validate_parallel_over_k(value, cfg):
    """``Diag.ParallelOverK`` on with an ELPA solver: SIESTA's manual says
    ELPA *"is not compatible with Diag.ParallelOverK"* (§ Diagonalization
    options), so prep says so -- a warning: the value is written as set and
    SIESTA decides what runs, which its output states.  Unset, the automatic
    answer is off for ELPA (`siesta.input`)."""
    algo = str(getattr(cfg, "diag_algorithm", None) or "").strip().upper()
    if value is True and algo.startswith("ELPA"):
        return [Issue("warn",
                      f"parallel_over_k = true with diag_algorithm = "
                      f"{cfg.diag_algorithm!r}: SIESTA's manual says ELPA "
                      f"is not compatible with Diag.ParallelOverK.  It is "
                      f"written as you set it; SIESTA's output says which "
                      f"split it used.",
                      "config.parallel_over_k")]
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
    system_label: str = field(
        default="siesta",
        metadata={"validate": _validate_basename("system_label")})

    # NOTE: the vacuum box is NOT a SiestaConfig knob.  Vacuum comes with the
    # STRUCTURE (Structure.vacuum, per-side gap) -- the single source of truth for
    # lattice/vacuum (structure-periodicity.md); the deck's cell is the
    # structure's (`cell.to_engine`).

    # Basis
    basis_size: str = "DZP"
    pao_energy_shift: float = 0.01

    mesh_cutoff: float = 300.0

    # XC
    xc_functional: str = "GGA"
    xc_authors: str = "PBE"

    # SCF
    solution_method: str = "diagon"
    mixing_weight: float = 0.02
    pulay_history: int = 8
    dm_tolerance: float = 1e-5
    h_tolerance: float = 1e-3
    dm_energy_tolerance: float = 1e-4
    # PAIRED WITH THE TOLERANCE ABOVE, and adjacent on purpose: the tolerance
    # does nothing without this switch, and a user meeting one without the
    # other cannot tell that (user, 2026-08-15 -- "placed next to each other
    # and their relation explained").  Same `category` and `group`, declared
    # here, so the form renders them side by side without special-casing.
    scf_energy_converge: bool = False
    max_scf_iter: int = 1000
    # A MEASUREMENT'S SWITCH, adjacent to the cap it modifies: what SIESTA
    # does when max_scf_iter is hit without convergence -- abort (its own
    # default) or accept the unconverged density and continue with a
    # warning.  Optional and unset for ordinary work (the abort protects
    # the budget); the bench pins set False so a capped trial ends cleanly
    # as the single-point measurement it is (project-layout.md 3.2).
    scf_must_converge: Optional[bool] = None
    electronic_temperature: float = 300.0

    # the k-point grid -- three integers; the form renders them as three
    # side-by-side inputs (kx / ky / kz).
    kgrid: Tuple[int, int, int] = field(
        default=(1, 1, 1),
        metadata={"validate": lambda value, cfg: _validate_kgrid(value)})

    # The FOURTH column of that same %block — SIESTA's ``displ(3)``, the
    # k-grid ORIGIN in grid-vector coordinates (``kgridinit.F``: "origin(ix)
    # = sum_j gridk(ix,j)*displ(j)").  Its own item rather than three more
    # numbers on ``kgrid``: it is a separate scientific decision (WHERE the
    # mesh sits, not how fine it is), and a stage may vary one without the
    # other.
    kgrid_displacement: Tuple[float, float, float] = field(
        default=(0.0, 0.0, 0.0),
        metadata={"validate": (lambda value, cfg:
                               _validate_kgrid_displacement(value))})

    # Relaxation; relax_type="none" disables the relaxation block entirely.
    # SIESTA 5.4.2 step-count + max-displacement mapping (see
    # siesta/input.py:_relaxation_facts for the full emission code):
    #   CG / Broyden / FIRE -> MD.Steps + MD.MaxDispl, one pair for all three
    # The labels below are generic; per-engine help text lives in the
    # FDF's verbose comments.
    relax_type: str = "CG"
    relax_steps: int = 200
    relax_force_tol: float = 0.02
    # ----- The vibration kind on this engine: the force-constant run. -----
    # ONE knob.  Which atoms are nudged is the structure's (the free atoms,
    # sorted into one run by prep -- siesta/vibration_deck.py), and the run
    # type is fixed by the kind; only the nudge size is a person's choice.
    fc_displacement: float = 0.04
    # THE PERSON'S EXPLICIT CHOICE (`engines/vibration.md` § 2.2): unticked,
    # the ladder relaxes first (a `relax` stage before `freq`); ticked, the
    # force constants are taken at the geometry as given and the job's finish
    # answers the statement with the reference-step forces.  Never a
    # refusal.
    already_relaxed: bool = False
    # THE THERMOCHEMISTRY'S TEMPERATURE, one meaning on both engines
    # (`engines/vibration.md` § 3.1, § 4.7): the job's finish sums the
    # harmonic vibrational part at it, read from the deck's `vibration`
    # block (§ 5.3).  No pressure beside it -- no SIESTA result has the
    # gas-phase translational term one would enter.
    temperature_K: float = 298.15
    relax_max_displ: float = 0.05

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
    continue_retries: int = 1
    #: ``wall_slack`` -- how many seconds before its wall the engine stops
    #: itself, saving its restart state whole (`running-a-job.md` § 3.6).
    #: Rides to the wrapper as ``continue_retries`` does; SIESTA takes it
    #: on its command line (``MaxWalltime.Slack``), never in the deck.
    wall_slack: int = 600

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
    restart: str = "continue"

    # When True, every section in the emitted FDF carries inline tuning
    # hints (parameter ranges, what to change when SCF / CG misbehave,
    # etc.) plus a "Troubleshooting" block at the end.
    verbose_comments: bool = True

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
    write_forces: bool = True
    write_coor_step: bool = True
    write_coor_xmol: bool = True
    write_md_history: bool = True
    # THE .ANI FILE.  Declared next to write_md_history because a reader
    # who wants "the trajectory file" lands on that one first and needs to
    # see, in the same place, that the animation file is a different switch.
    write_md_xmol: bool = True
    write_hs: bool = True
    write_molwatch_log: bool = True

    # ---------------- Parallel execution (MPI) ----------------
    # Only matter when running `mpirun -np N siesta`; single-rank runs
    # ignore them.
    # MPI rank count for ``mpirun -np N siesta``.
    # Don't confuse with block_size (BlockSize for ScaLAPACK
    # within a rank); rank count is the OUTER parallelism.
    # The parallel-execution family (MPI ranks, OMP threads, GPU count,
    # BlockSize, parallel-over-k, memory cap): how much compute this
    # run is given.
    mpi_np: Optional[int] = None

    gpu_count: Optional[int] = None

    block_size: Optional[int] = field(
        default=None,
        metadata={"validate": lambda value, cfg: _validate_block_size(value)})
    parallel_over_k: Optional[bool] = field(
        default=None, metadata={"validate": _validate_parallel_over_k})
    # OpenMP threads per MPI rank -- the cores per rank a run states
    # (`cpus_per_task` on `Resources`, translated at `resolve`).  The run
    # script exports it as OMP_NUM_THREADS; None is unstated, which prep
    # refuses (`architecture.md` § 5.2).  The run script pins BLAS to 1 thread per rank so
    # OMP * BLAS doesn't oversubscribe.
    omp_threads: Optional[int] = None
    max_memory_mb: Optional[int] = None
    use_gpu: bool = False
    diag_algorithm: str = "ELPA-1STAGE"

    # Pseudopotentials
    psml_lib: Optional[str] = None
    # ``List``: the template grammar names ``strlist`` for ``List[str]``,
    # and this field is an ITEM -- it orders the ChemicalSpeciesLabel block,
    # which run-identity § 6a calls identity-sensitive.
    species_order: Optional[List[str]] = None

    # THE ELECTRONIC STATE (`science/chemistry-correctness.md` § 2a) -- the
    # three merged items, declared once in the catalogue for both engines.
    # A blank means *work it out*; `electronic_state` does, for the form,
    # the checks and the deck alike.  SIESTA has no `method`: it is a
    # density-functional code.
    # THE ELECTRONIC STATE'S ITEMS -- `net_charge`, `spin_treatment`,
    # `unpaired_electrons` -- one question asked of both engines, declared
    # once in the catalogue (`engines/template.md` § 6.3); a blank means
    # *work it out* (`electronic_state.electronic_state`).
    net_charge: Optional[int] = None
    spin_treatment: Optional[str] = None
    unpaired_electrons: Optional[Union[int, str]] = None

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

    # THE DEFAULT WINDOW PUTS ITS ENERGIES ON THE 0.01 eV LATTICE, E_F among
    # them -- -2.005…2.005 eV at 401 points (`engines/transport.md` § 2a.16;
    # the user, 2026-10-22).
    transmission_emin_ev: float = -2.005

    transmission_emax_ev: float = 2.005

    transmission_n_points: int = 401

    # BLANK IS THE RULE'S: three times the shared k-point grid in the
    # electrode plane (x, y), worked out when the deck is written
    # (`kmesh.mesh_for`), so it follows `kgrid` until the person sets it
    # (`engines/transport.md` § 0.3b).
    tbt_k_grid: Optional[Tuple[int, int, int]] = None

    tbt_spin: int = 0

    ts_elecs_eta_ev: float = 0.001

    tbt_elecs_eta_ev: float = 0.001

    # BLANK IS NOT 0 (2.1.0): unset leaves tbtrans's own rule, a tenth of
    # the leads' broadening; 0 is no device broadening (the catalogue's help).
    tbt_contours_eta_ev: Optional[float] = None

    tbt_dos_gf: bool = True

    tbt_dos_a: bool = True

    tbt_dos_elecs: bool = True

    tbt_t_eig: int = 4

    tbt_t_bulk: bool = True

    tbt_t_all: bool = False

    tbt_verbosity: int = 5

    tbt_progress: float = 5.0

    negf_eq_pole_ev: float = 10.0

    bias_voltage_v: float = 0.0

    ts_hs_save: bool = True

    ts_solution_method: str = "BTD"

    tbt_solution_method: str = "BTD"

    ts_hartree_fix: str = "C"

    negf_neq_fermi_cutoff_kt: float = 5.0

    # THE BIAS WINDOW'S LINE and the leads' files (`engines/transport.md`
    # § 2a.11, *The levers*): TranSIESTA's own defaults, written.
    negf_neq_spacing_ev: float = 0.01

    ts_elecs_out_of_core: bool = True

    ts_scf_dq_converge: bool = True

    # BLANK IS NOT SET: TranSIESTA's own tolerance, a thousandth of the
    # electrons on the atoms that are not buffer atoms (the catalogue's help).
    ts_scf_dq_tolerance: Optional[float] = None

    # BLANK IS NOT 0 (2.1.0): unset leaves TranSIESTA's own rule, a tenth of
    # the leads' broadening; 0 is no device broadening (the catalogue's help).
    negf_neq_eta_ev: Optional[float] = None

    electrodes_bulk: bool = True

    electrode_kz: int = 100


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
#: ONE table: the ladder builders, the build route and ``molbuilder describe`` name
#: tiers from it.



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


__all__ = [
    "SiestaConfig",
    "SIESTA_STAGE_PRESETS",
]
