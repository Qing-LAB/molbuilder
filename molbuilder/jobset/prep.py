"""Prep — THE CONDUCTOR of `docs/execution/script-preparation.md`.

It walks floors 1 -> 4 in order and owns no decision of its own: the five
steps, and the eleven sub-steps inside step 3, are that document's (§ 4);
what each engine supplies at each step is its § 5.  **It may call, but it
may never decide** -- a value settled here is a value no floor owns, which
is the shape of the "stomp" bugs (§ 3.3).

:func:`prep_task` (a task's groups) and :func:`prep_bench` (one stage's
benchmark) are the verb's two entries, and both doors call them with the
calculation they opened once (`calculation.open`, A16), checked here
(:func:`checked`), planning each stage with the one member planner
(:func:`plan_stage`), then saves,
writes and records.  The planner hands its answer to
:func:`prep_calculation` -- the five steps, on the described route: a
description plus its template in, one rendered deck and wrapper **per
element** of the resolved :class:`~molbuilder.resolve.ParameterSet` out.
:func:`prep_jobset` is steps 4–5, the tail of those.  Neither reads or
decides anything of its own.

§ 2.3.1a: *benchmarking is `prep` whose parameters are a set rather than a
point* — the five steps are general, the grid is the specialisation.

Wrappers render **once per job, in the JOB'S OWN DIRECTORY** (L2, roadmap
7.10).  Every element renders its own deck, so each wrapper carries its own
element's resources.
"""

from __future__ import annotations

from .model import KIND_LADDER, KIND_SWEEP

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from typing import (TYPE_CHECKING, Callable, Dict, List, NamedTuple, Set,
                    Optional, Sequence, Tuple)

from .. import calcdirs
from .. import script_emit as _sc
from ..paths import Shape
from .materialize import materialize
from .naming import (job_dir_names, ladder_homes, open_container,
                     run_names, stage_home)
from ..warmfiles import warm_list
from ..issues import calling as _calling
from .model import FILENAME as JOBSET_FILENAME, Job, JobSet, Resources
from ..parse.fdf import read_deck
from .plan import FILENAME as _PLAN_FILE
from . import prepared_with
from ..runfiles import RunNames
from .errors import PrepError, as_prep_error, step_errors, user_error_as_prep
from .machine import machine_record, require_activation, set_machine
from .engines import EngineSeam, engine_seam, siesta_provide_pseudos
from .gather import composed_for_prep
from .transport_rung import (at_frame, at_point, scratch_dir_of,
                             provide_pseudos, transport_parts,
                             transport_sources)

if TYPE_CHECKING:                      # annotations only
    from ..calculation import Calculation
    from ..resolve import ParameterSet
    from ..scheduler.record import Environment
    from ..task import Task


def prep_jobset(jobset: JobSet, base_dir, *, plan, shape, provenance,
                machine_record, env: str = None,
                emit_sbatch: bool = True, record_dir=None,
                log=None, render=None, points=None,
                this_prep=None, notes: Optional[list] = None) -> List[Path]:
    """Render launchers + lay out the per-job tree under ``base_dir`` --
    steps 4 and 5 of :func:`prep_calculation`, its one caller, which hands
    over what the prep entry read and decided.  ``notes`` is the answer's
    list: what a wrapper's rendering says and does not refuse goes there
    (`runwrap.write_run_wrapper`).

    ``plan`` (`jobset.planned.Plan`) receives what this writes, and is read
    for the decks it already holds (`job-system.md` § 5.0).

    ``shape`` is the layout as the entry read it from the description -- a
    layer below takes it (`Shape.named(task.shape)`, A16).  ``this_prep`` is what
    this prep's hand-over takes, which `STAGE-PLAN.md` says under its table
    (`plan.render_plan`).  ``provenance`` is which config files answered,
    as the prep entry read them with the machine's record at its
    checkpoint 4 (:func:`checked`); ``machine_record`` is that record.

    ``points`` names, per job, the folders that hold a copy of its deck
    written at another point -- a sweep's (`Rung.points`) -- each with how
    its run script is rendered beyond the job's own facts
    (`Rung.wrapper_at`: a vibration phase's `resumes`, its Python steps,
    its monitor as a member); each gets the job's wrapper beside it,
    rendered by the one call the job's own folder gets.

    ``render`` names the jobs whose run scripts this prep writes -- the
    stage it preps.  The plan it is handed is the calculation's, merged
    (`_merge_run_jobset`), and every other stage in it was prepared by a
    prep before, or planned before it in this one: its run script and
    monitor stay as that stage's own prep wrote them,
    the copy its attempt ran with (`project-layout.md` § 1.6.2, § 2.6).  ``None``
    renders every job: a benchmark's plan is its own, every trial this
    prep's.

    Steps, in order:
      1. render each job's ``.run.sh`` (and ``.sbatch`` when
         ``emit_sbatch`` and a scheduler is configured) **in that job's own
         directory**, beside the deck it launches — reusing
         ``runwrap.write_run_wrapper`` (no reinvention).  Every job has a
         deck of its own: prep renders one per element.
      2. ``materialize`` — the shared package, as real copies into each job
         directory (what a stage continues from is copied when its attempt
         is opened: ``materialize.prepare_attempt``).
      3. emit ``STAGE-PLAN.md`` beside the job-set it describes.

    Returns the per-job directories.  Raises :class:`PrepError` on an
    invalid JobSet or a script that is not in its job's directory.
    """
    from ..runwrap import write_run_wrapper

    # The allocation is NOT a parameter here (U2): an allocation is an input
    # to *prep* (`project-layout.md` M4), but it enters ONCE, at resolve,
    # where each element folds it into its own resources (generator.md § 5);
    # by this floor every job already carries the answer.
    errs = jobset.validate()
    if errs:
        raise PrepError(
            "cannot prep an invalid JobSet:\n  - " + "\n  - ".join(errs))
    base = Path(base_dir).resolve()

    # ---- 1. render each job's wrapper, IN THE JOB DIR ------------------ #
    # The machine was resolved at step 1 of `prep_calculation`, from the
    # record the entry read (`set_machine`, once per calculation).
    # Nothing rendered lives at the bundle root (user, 2026-08-24;
    # `project-layout.md` § 1.0).  The deck was born in its directory by
    # `prep_calculation`, and the wrapper is written beside the deck it
    # launches.
    if log is not None:
        log.phase("STEP 4 · WRAPPERS — how each deck is launched")
    _sh = shape
    _dir_of = job_dir_names(jobset, _sh)
    for job in jobset.jobs:
        if render is not None and job.name not in render:
            continue
        # THE SAME QUESTION `prep_calculation` AND `materialize` ASK.  A
        # trial's files live in its attempt when the shape keeps them
        # (`project-layout.md` § 1.5a).
        _jd = base / _dir_of[job.name]
        # THE NAMES OF THIS JOB'S FILES (`naming.run_names`) -- what its
        # run script names every file of its runs by.
        _names = run_names(jobset, job, _sh)
        if jobset.kind == KIND_SWEEP:
            # A TRIAL'S FOLDER, opened by the one opener (`open_run`).
            from .materialize import open_run, trial_work_dir
            _jd = open_run(base, trial_work_dir(_jd, _sh, _names), plan)
        plan.folder(_jd)
        script_path = _jd / job.script
        # A SWEPT RUNG'S DECKS ARE ITS POINTS' -- none at the stage level
        # (`engines/transport.md` § 2a.11) -- and its one header is its
        # walk's (`prep_calculation`): a point is never sent alone.
        _pts = (points or {}).get(job.name, ())
        _decks = ([(Path(p_) / job.script, dict(x)) for p_, x in _pts]
                  or [(script_path, {})])
        for _deck, _x in _decks:
            if not plan.is_file(_deck):
                raise PrepError(
                    f"job {job.name!r}: script {job.script!r} not in "
                    f"{_deck.parent} (render the inputs before prep).")
        def _wrap(script_path, point_facts):
            # The ALLOCATION, whole (architecture.md § 3.1, rule A8): passed
            # as the object, no field of it can be dropped by a hand-copied
            # argument list.
            write_run_wrapper(
                script_path,
                # THE NAMES TRAVEL (`gpu.md` G7): the job's own -- its label
                # (a trial's carries its point, `paths.trial_label`), its
                # stage and whether its runs share a folder -- so the wrapper
                # is TOLD every name it writes, and composes none.
                names=_names,
                resources=job.resources,
                program=job.program,
                env=env,
                emit_sbatch=emit_sbatch and not _pts,
                # The BUNDLE'S scope, explicitly: the script is born in its
                # job directory now, and the renderer's parent-derived
                # fallback would read the record one level below the
                # bundle's environment.json (roadmap 7.10 M1).
                project_dir=base,
                # WHICH MACHINE THIS IS FOR, carried rather than re-derived.
                # The record always travels, so the wrapper reads that
                # machine's own activation off it instead of this machine's
                # config.
                machine_record=machine_record,
                # THE QUEUE IT WAS PLACED ON at prep, as its job records it
                # (`job-system.md` § 6.0): the header renders it and binds no
                # name a second time.
                domain_pq=((job.placement["partition"], job.placement["qos"])
                           if job.placement else None),
                # THE RESTART FILES IN EFFECT for this calculation -- its own
                # list first (`warmfiles.warm_list`, `job-contracts.md`
                # § 4.2a): written into the script here, never read by it at
                # import (plan W36 ⑧).
                warm=warm_list(jobset.engine, None, base).suffixes,
                # THE JOB'S LAST STEP, when its engine leaves no result
                # (`Job.finish`, `engines/vibration.md` § 5.5): the wrapper
                # runs it, and its bundle is written beside the deck.
                finish=job.finish,
                # WHETHER A RE-RUN CONTINUES (`Job.resumes`): the wrapper
                # says a retry of a run that cannot resume repeats it
                # (`running-a-job.md` § 3.5) -- a point's own where the
                # rung renders each point its own (`Rung.wrapper_at`: a
                # vibration phase's, with its two Python steps and its
                # monitor a member of the run, `engines/vibration.md`
                # § 5.11 § 3, § 7a).
                resumes=point_facts.get("resumes", job.resumes),
                steps=point_facts.get("steps"),
                member=point_facts.get("member"),
                plan=plan,
                # WHAT THE RENDERING SAYS (`gpu.md` G6), onto the answer.
                notes=notes,
            )
        with user_error_as_prep():
            for _deck, _x in _decks:
                _wrap(_deck, _x)
                # A HEADER THIS PREP DOES NOT WRITE is taken away: one an
                # earlier prep left -- for a queue since dropped, or a point
                # -- would otherwise be sent as this one's (`job-system.md`
                # § 6.0, *every header is prep's*).
                _left = _deck.parent / _names.name(".sbatch")
                if not plan.decides(_left) and _left.is_file():
                    plan.remove(_left)
        if log is not None:
            log.received(job.script, _flat_resources(job.resources)
                         + (f", env={env}" if env else ""))
            for _w in (_names.name(".run.sh"), _names.name(".sbatch")):
                if plan.is_file(_decks[0][0].parent / _w):
                    log.produced(_w, f"{len(plan.read_text(_decks[0][0].parent / _w).splitlines())}"
                                     f" lines")
                elif _w.endswith(".sbatch"):
                    log.note(f"{_w}: not written "
                             + ("(emit_sbatch off)" if not emit_sbatch
                                else "(a run of points: its walk's header "
                                     "is the job's)" if _pts
                                else "(no scheduler configured)"))

    # ---- 2. the shared package, copied --------------------------------- #
    # ONLY THE JOBS THIS PREP RENDERS: every other stage in the merged plan
    # was prepared already, and its folder is its own prep's.
    dirs = materialize(jobset, base, plan, shape=_sh, only=render)
    if log is not None:
        log.phase("STEP 5 · RUN DIRECTORY — where each job will be launched")
        log.received("shape", str(_sh))
        log.received("shared package",
                     ", ".join(jobset.shared) or "nothing (W5)")
        for _d in dirs:
            log.produced(_d.name if _d.resolve() != base.resolve() else ".",
                         "the bundle root — flat runs here, nothing is linked"
                         if _d.resolve() == base.resolve() else str(_d))

    # ---- 3. emit STAGE-PLAN.md (§ 5 D3; mirrors bench's BENCH-PLAN.md) --- #
    # The plan lands BESIDE the job-set it describes: the run's at the
    # root, a bench's inside its stage's bench/ container -- so a bench
    # prep can never overwrite the run's reviewable plan (U1, 2026-08-12).
    plan_dir = Path(record_dir) if record_dir is not None else base
    plan.text(plan_dir / _PLAN_FILE,
              stage_plan_text(jobset, this_prep, base=base,
                              provenance=provenance, plan=plan))
    if log is not None:
        log.produced(_PLAN_FILE, str(plan_dir / _PLAN_FILE))
    return dirs


def stage_plan_text(jobset: JobSet, this_prep, *, base, provenance,
                    plan) -> str:
    """`STAGE-PLAN.md` -- the reviewable plan a prep leaves (§ 5 D3;
    bench's BENCH-PLAN.md mirrors it): the job set's table
    (`plan.render_plan`), what this prep's stages take at their first
    launch (``this_prep``, one line each), and the CONFIG PROVENANCE --
    which files supplied the effective execution settings, so a behaviour
    difference between two machines is explained by the bundle itself (user
    request 2026-08-12; secrets excluded by construction) -- a file the prep
    writes itself, as ``plan`` holds it, said so."""
    from ..runtime_config import format_provenance
    from .plan import render_plan
    # WHICH file supplied the warm-file vocabulary (U6a provenance): the
    # engine's own, or this calculation's fine-tuned copy -- a surprising
    # carry must be debuggable from the plan alone (§ 4.2a).
    _in_effect = warm_list(jobset.engine, None, base)
    _vocab = (f"warm-files: {_in_effect.path}"
              + (" (this calculation's own)" if _in_effect.own else "")
              + "\n")
    return (render_plan(jobset, this_prep) + "\n\n" + _vocab
            + format_provenance(provenance, written=plan.writes()) + "\n")


def this_prep_lines(stage: str, takes) -> List[str]:
    """What ``stage``'s first launch takes, as `STAGE-PLAN.md` says it under
    its table -- the stage, then each line both prep doors print."""
    return ([f"`{stage}` {t['line']}" for t in takes]
            or [f"`{stage}` takes nothing from another run"])


# --------------------------------------------------------------------- #
#  The five steps, entire — `project-layout.md` § 2.3.1                  #
# --------------------------------------------------------------------- #

def _vibration_block(stage: str, cfg, *, criterion, criterion_from: str,
                     struct) -> dict:
    """A SIESTA force-constant deck's `vibration` block, format v4: the
    facts the run's finish reads and no SIESTA keyword states
    (`engines/vibration.md` § 5.3, § 5.11 § 4), built here because only
    `prep` holds all of them -- the stage, its resolved config, the
    molbuilder rendering the deck -- and the mode selection the optical
    phase applies (`select_modes`), the one fact of that phase no keyword
    states.  ``criterion`` is the relaxation phase's own ``relax_force_tol``
    -- the force the reference geometry is relaxed to, and judged by.
    ``struct`` is the structure the deck is written from, whose atoms'
    masses the block states (`chemistry.atomic_mass`, I3).  The relaxation
    record is the `relax` call's, read from the run's index at the finish
    (§ 5.11 § 3), never written here."""
    from .. import __version__ as _mb_version
    from ..chemistry import atomic_mass
    from ..spectra.siesta_vibration import vibration_record
    from ..structure import IDENTITY_FIELDS
    return vibration_record(
        stage=stage,
        force_criterion_ev_ang=criterion,
        force_criterion_from=criterion_from,
        already_relaxed=bool(getattr(cfg, "already_relaxed", False)),
        relaxation=None,
        temperature_K=float(cfg.temperature_K),
        masses_amu=[atomic_mass(el) for el in struct.elements],
        molbuilder_version=str(_mb_version),
        selection={"es_mode_selection": str(cfg.es_mode_selection),
                   "es_explicit_indices": str(cfg.es_explicit_indices),
                   "freq_min_cm1": cfg.freq_min_cm1,
                   "freq_max_cm1": cfg.freq_max_cm1},
        # WHAT NO OTHER BLOCK OF THE DECK CARRIES of the structure the result
        # states (engines/vibration.md 6.10.1).
        title=str(struct.title or ""),
        identity={k: getattr(struct, k) for k in IDENTITY_FIELDS})


def structure_path(task, base: Path) -> Optional[Path]:
    """Where the structure this calculation is *of* is -- beside the
    calculation first, at the recorded path second (`stages.md` § 6.3); the
    suffix-corrected candidate is the codec's own naming rule (a .pdb source
    travels as <stem>.xyz).  ``None`` when it is in neither place, and for
    a calculation of no structure of its own -- a transport calculation,
    which cites its junction."""
    from ..workingcopy_structure import StructureCodec
    if task.structure is None:
        return None
    src = Path(task.structure.source)
    for candidate in (base / src.name,
                      (base / src.name).with_suffix(
                          StructureCodec.GEOMETRY_SUFFIX),
                      src):
        if candidate.is_file():
            return candidate
    return None


def _order_said(task, base: Path) -> list:
    """WHAT THE PAIR THIS CALCULATION IS OF STATES OF ITS ORDER, when it
    states none (`model/overview.md` § 2.2b, F4): prep takes it as the user
    order of this calculation, the warning said -- its structure's pair, or
    a transport calculation's cited pair (:func:`_cited_pair_path`), the
    codec's read notice from its one read (`StructureCodec.load`).  A pair
    that does not read says nothing here: the read that builds refuses
    it."""
    from ..workingcopy_structure import ORDER_NOT_STATED, StructureCodec
    path = structure_path(task, base) or _cited_pair_path(task, base)
    if path is None:
        return []
    said: list = []
    try:
        StructureCodec().load(path, frames=True, said_out=said)
    except (OSError, ValueError):
        return []
    return [n for n in said if n.where == ORDER_NOT_STATED]


def _cited_pair_path(task, base: Path) -> Optional[Path]:
    """A transport calculation's cited structure pair (`engines/transport.md`
    § 3.1), fenced to the projects tree -- ``None`` for a cited run's
    folder, and for any other kind of calculation."""
    citation = ((task.slots or {}).get("junction")
                if task.calculation == "transport" else None)
    if not citation:
        return None
    from ..projects import OutsideRoot, contain, find_projects_root
    root = Path(find_projects_root(base))
    try:
        path = contain(root / citation, root)
    except OutsideRoot:
        return None
    return path if path.is_file() else None


def _structure_for(task, base: Path, pset):
    """The structure this calculation is *of*, from the reference in the
    description (`stages.md` § 6.3 — a reference plus a witness, never a copy).

    Looked for beside the calculation first and at the recorded path second, so
    a folder carried to a cluster with its structure alongside resolves without
    the original tree existing there.

    **The witness is checked and the mismatch is loud**: a description opened
    against a structure that has since changed would otherwise build a
    different calculation under the same id — § 1's second failure mode.
    """
    from ..structure_pair import GeometryChangedError
    from ..workingcopy_structure import StructureCodec
    codec = StructureCodec()
    # The codec, not the bare loader: describe writes the structure as the
    # pair (--vacuum lives NOWHERE a bare .xyz can put it), and the codec
    # applies the .molstruct.json and compares its pin with the file
    # (`structure-molstruct.md` § 3).
    candidate = structure_path(task, base)
    if candidate is not None:
        try:
            struct = codec.load(candidate)
        except GeometryChangedError as exc:
            # THE PIN (`structure-molstruct.md` § 3): the atoms or the
            # cell moved after the pair was written -- a deck rendered
            # now would put the old labels on them.
            raise PrepError(f"the structure has changed since this "
                            f"calculation was described: {exc}") from None
    else:
        raise PrepError(
            f"the structure this calculation describes is not here: "
            f"{task.structure.source!r}. `task.json` records a REFERENCE to it "
            f"(engines/stages.md § 6.3), so `prep` needs the file to be either "
            f"beside the calculation or still at that path.")
    if task.structure.fingerprint and codec.fingerprint_on_disk(
            candidate) != task.structure.fingerprint:
        # THE WHOLE PAIR IS THE ONE IT WAS DESCRIBED WITH (`engines/stages.md`
        # § 6.3): every stage of a calculation is of one structure, and an
        # edit after its description -- a label, a held atom, a coordinate,
        # the box -- is refused, never worked around.
        raise PrepError(
            f"the structure has changed since this calculation was "
            f"described: {candidate.name} is not the pair its description "
            f"records (task.json's structure fingerprint) -- an atom, a "
            f"label, a held atom or the box was edited after it.  Every "
            f"stage of a calculation is of the one structure it was "
            f"described with: describe it again to build on the edit.")
    if struct.formula != task.structure.formula or \
            struct.n_atoms != task.structure.atoms:
        raise PrepError(
            f"the structure has changed since this calculation was described: "
            f"the description witnesses {task.structure.formula} with "
            f"{task.structure.atoms} atoms, and {candidate} now holds "
            f"{struct.formula} with {struct.n_atoms}.\n"
            f"  Describing again is the honest fix -- rendering this deck would "
            f"build a different calculation under the same id.")
    # THE STRUCTURE ENTERS THIS CALCULATION (`model/structure-periodicity.md`
    # § 6.0): an engine's stated origin is not carried in, so the one rule
    # places it -- unless the kind keeps the engine's frame: a vibration
    # whose input is stated relaxed, as THIS stage resolves it (``pset``,
    # the stage's own values over the template's -- the door every other
    # reader of `already_relaxed` asks).
    from ..cell import entering, keeps_engine_frame
    relaxed = bool(getattr(pset[0].render_config(), "already_relaxed", False))
    return entering(struct, keep_engine_frame=keeps_engine_frame(
        task.calculation, relaxed))


# --------------------------------------------------------------------- #
#  The rung -- what a kind adds to the one table of steps               #
#  (`script-preparation.md` § 3.0)                                       #
# --------------------------------------------------------------------- #

def _at_no_point(cfg, point):
    return cfg


@dataclass
class Rung:
    """What one rung's decks are written from -- the structure step's
    answer, read by every step after it (`script-preparation.md` § 3.0).

    A kind with steps of its own builds one (:data:`RUNGS`); every other
    kind is a calculation of its described structure, one deck per element
    (:func:`_described_rung`).  The conductor (:func:`prep_calculation`)
    reads these and never asks which kind it is in."""
    #: The structure the decks describe.
    struct: object
    #: The program each deck is -- `spec_for`'s ``calculation``: the
    #: calculation's own (a SIESTA vibration's phases are its parts,
    #: `spec_extra`'s ``part``).
    render_kind: str
    #: WHAT THE KIND SAYS OF THE RUN IT PREPARES, beside its request -- a
    #: SIESTA vibration's count and sizes (`engines/vibration.md` § 5.11
    #: § 9); prep's notes carry them to both doors.  Nothing judges the
    #: wall: it stays the person's.
    said: Tuple[str, ...] = ()
    #: ``cfg -> cfg``: the config an element renders from -- a lead's own
    #: label, the junction's electronic state folded in.
    configure: Callable = lambda cfg: cfg
    #: ``(element, cfg, point) -> {keyword: value}``: what else `spec_for`
    #: is told -- a vibration's block and its phase's part, the cell; the
    #: junction's electronic state and the deck's point.
    spec_extra: Callable = lambda element, cfg, point: {}
    #: ``((folder, point), ...)``: the folders each holding a deck written at
    #: that point -- a frame and a voltage (`transport.stages.Point`,
    #: `engines/transport.md` § 2a.9-11) -- each with its own wrapper and
    #: attempt ladder.  Empty: one deck, in the element's folder.
    points: tuple = ()
    #: ``(cfg, point) -> cfg``: a deck's config at its point -- the rung
    #: answers the bias, the one value the catalogue does not hold
    #: (`engines/template.md` § 6.4).
    at_point: Callable = _at_no_point
    #: ``point -> struct``: the structure a point's deck describes -- its
    #: frame of the cited set; ``None``, every deck describes ``struct``.
    struct_at: Optional[Callable] = None
    #: ``(plan) -> None``: the data files, when the kind's come from
    #: somewhere other than the engine's library (transport: the citation);
    #: ``None`` is the engine's own step (`EngineSeam.provide_data`).
    provide_data: Optional[Callable] = None
    #: ``job -> job``: what the rung's job carries forward and runs, where
    #: the kind answers it per rung -- a transport rung's restart files, the
    #: transmission's program.
    job_facts: Callable = lambda job: job
    #: ``element -> ((item, who), ...)``: who set each value the rung's decks
    #: are rendered from -- `resolve`'s rows, and the rung's own words for a
    #: value its steps set after them (a transport rung's label;
    #: `job-contracts.md` § 3.2).
    sources: Callable = lambda element: element.deck_sources
    #: ``point -> {resumes, steps, member}``: how a point's run script is
    #: rendered beyond the job's own facts -- a vibration phase's `resumes`,
    #: its two Python steps and its monitor as a member of the run
    #: (`engines/vibration.md` § 5.11 § 3, § 7a); ``{}`` renders the job's.
    wrapper_at: Callable = lambda point: {}
    #: ``point -> bool``: whether a point's deck relaxes -- its progress
    #: log's targets (`_seed_trajectory_log`); by default the render kind's.
    relaxes_at: Optional[Callable] = None
    #: ``(point, spec, folder, plan) -> None``: what the rung writes BESIDE
    #: a deck before the deck is checked -- a vibration phase's
    #: `coordinates.fdf`, which the deck reads as its coordinate block
    #: (`engines/vibration.md` § 5.11 § 4); ``None`` for a kind whose deck
    #: holds everything.
    beside: Optional[Callable] = None


def _described_rung(calc, pset, *, seam, sweep, log, plan,
                    contours=None) -> Rung:
    """A calculation of its described structure, one deck per element --
    every kind with no steps of its own."""
    return Rung(struct=_structure_for(calc.task, calc.base, pset),
                render_kind=calc.task.calculation)


def _siesta_vibration_rung(calc, pset, *, seam, sweep, log, plan,
                           contours=None) -> Rung:
    """A SIESTA vibration's rung -- ONE RUN IN PHASES (`engines/vibration.md`
    § 5.11): the relaxation, the force constants and the optical calls,
    each phase a point of the run with its own deck, written for its part
    by the one deck writer (§ 5.11 § 4) from the structure as given -- the
    atoms in the order the force-constant run needs, the held ones first,
    on a COPY whose order record its deck states and the job's finish
    inverts (`model/overview.md` § 2.2a, I5); the user order never reaches
    the engine and the sorted order never reaches a person.  Every phase's
    deck is written at the structure's coordinates; what a call computes
    at is its own `coordinates.fdf`, written by the run's plan step
    (§ 5.11 § 3).  Which phases run is the stage's config's: each `PHASES`
    row's switch (`already_relaxed` turns the relaxation off,
    `es_mode_selection` skip the optical calls).  Each phase's run script
    is rendered with its own `resumes` and its two Python steps
    (:attr:`Rung.wrapper_at`)."""
    from ..paths import Shape
    from ..siesta.vibration_phases import (FC, RELAX, Call, phase, phases_on,
                                           run_size_words)
    from ..transport.sort import deck_sort_key, sort_by
    from ..warmfiles import warm_list
    from .naming import stage_home
    base, task = calc.base, calc.task
    # THE SHAPE IS HIERARCHICAL ONLY (§ 5.11): three SIESTA calls in one
    # directory would write over each other's .DM and .XV, and every stage
    # its .FC and spectrum.
    if not Shape.named(task.shape).keeps_attempts_as_directories:
        raise PrepError(
            "a SIESTA vibration runs as one run in phases -- its "
            "relaxation, its force constants and its optical calls each a "
            "SIESTA call in a folder of its own -- which the flat layout "
            "cannot hold: in one directory the calls would write over each "
            "other's .DM and .XV (engines/vibration.md 5.11).  Describe it "
            "with --shape hierarchical.")
    # THE GEOMETRY IN THE USER ORDER, then sorted as every prep sorts
    # (`model/overview.md` § 2.2a, I3-I5): the structure as given, every
    # phase's deck at it.  THE ORDER RECORD RIDES THE COPY into its deck's
    # ATOM-METADATA block (I5): the deck states the way back, beside the job.
    struct = _structure_for(task, base, pset)
    key = deck_sort_key(task.engine, task.calculation, pset.stage)
    struct = sort_by(struct, key).structure
    log.step(f"the atom order the engine needs -- {key}, {struct.n_atoms} "
             f"atoms; the deck states its order record")
    cfg0 = pset[0].render_config()
    # THE PHASES THIS STAGE RUNS, by the table's switches (§ 5.11 § 1) --
    # each a point of the run: its folder below the stage, holding the
    # phase's deck and run script every call of it copies (§ 5.11 § 2).
    on = phases_on(cfg0)
    stage_dir = stage_home(base, task, pset.stage).dir
    points = tuple((stage_dir / p.name, Call(p.name, p.name)) for p in on)
    log.step("the run's phases: " + ", ".join(p.name for p in on))
    # THE FINISH'S FORCE CRITERION is the relaxation phase's own tolerance
    # (`engines/vibration.md` § 5.5): the one yardstick the run relaxes to
    # and judges the reference geometry by -- said with where it came from.
    criterion = getattr(cfg0, "relax_force_tol", None)
    criterion_from = "the stage's relax_force_tol"

    def spec_extra(element, cfg, point):
        # THE PART THE DECK IS -- the point's phase; a benchmark trial's
        # the force constants' (a trial measures the one call that costs).
        part = point.phase if point is not None else FC
        return {"part": part,
                # The finish's facts ride the force-constant deck alone --
                # never a benchmark trial's: a spectrum of capped SCFs is a
                # spectrum of nothing (§ 5.5).
                **({"vibration": _vibration_block(
                        pset.stage, cfg, criterion=criterion,
                        criterion_from=criterion_from, struct=struct)}
                   if part == FC and not element.is_trial else {})}

    def wrapper_at(point):
        # EACH PHASE'S RUN SCRIPT (§ 5.11 § 3): the phase's `resumes` from
        # its warm-files section, its two Python steps, and its monitor a
        # member of the run's (§ 7a).
        rule = phase(point.phase)
        return {"resumes": warm_list("siesta", rule.warm_kind, base).resumes,
                "steps": (("plan", point.phase), ("record", point.phase)),
                # ...its way up to the run a CALL's (§ 7a): a planned
                # phase's calls lie below a container named after it.
                "member": (Call(rule.name, "call") if rule.planned
                           else point).to_run}

    def beside(point, spec, folder, plan):
        # THE STRUCTURE'S COORDINATES, beside every phase's deck (§ 5.11
        # § 4): the rows the deck writer derived, as the deck reads them.
        from ..runfiles import COORDINATES_FILE
        plan.text(Path(folder) / COORDINATES_FILE,
                  spec.derived["coordinates_file"])

    # THE COUNT AND THE SIZES, beside the wall (§ 5.11 § 9): the free atoms
    # are those the held label leaves; the modes the run reports, the rank
    # rule's -- the one count prep's check states too.
    from ..template import engine_name
    from ..validation.spectra import modes_of
    n_free, _n_rigid, n_modes = modes_of(struct, engine_name(type(cfg0)))
    return Rung(struct=struct, render_kind="vibration",
                said=(run_size_words(cfg0, n_free, n_modes),),
                spec_extra=spec_extra, points=points, wrapper_at=wrapper_at,
                beside=beside,
                relaxes_at=lambda pt: pt is not None and pt.phase == RELAX,
                # THE RUN'S PHASES, on its job (`Job.phases`): what launch
                # opens and walks, what the run's own python follows.
                job_facts=lambda job: dataclasses.replace(
                    job, phases=[p.name for p in on]))


def _transport_rung_of(calc, pset, *, seam, sweep, log, plan,
                       contours=None) -> Rung:
    """A transport rung (`engines/transport.md`): its four steps of its own.

    **compose** -- the junction and its leads, composed from the cited
    relaxation, or read from the record that travelled beside ``task.json``
    (:func:`composed_for_prep`); the rung's structure is the junction or
    the lead taken out of it (:func:`transport_parts`).  **electronic
    state** -- decided once, on the junction, folded into every rung.
    **points** -- a sweep's rung writes one deck per bias point, each in its
    own folder (`transport.stages.point_folders`).  Its data files come
    from the citation, and its job carries the rung's own restart files and,
    for the transmission, TBtrans's program.  The fifth, **gather** -- each
    rung's inputs from the runs upstream -- is its first launch's
    (:func:`gather_sources`, `job-system.md` § 6.0)."""
    from .reuse import point_key
    base, task, stage = calc.base, calc.task, pset.stage
    if sweep is not None:
        raise PrepError(
            "a transport calculation takes no parameter sweep or "
            "translation.  Its parameters come from its own template and "
            "each rung's run card, and its one axis is the bias -- a list "
            "in task.json, rendered as one deck per point "
            "(engines/transport.md 2a.10: single bias is the degenerate "
            "case of that axis, one point at zero).")
    composed = composed_for_prep(calc, plan)
    struct, configure, state = transport_parts(task, stage, composed,
                                                pset[0].render_config())
    from ..transport.stages import point_folders, warm_declaration
    # THE RUNG'S POINTS (`transport.stages.rung_points`): a frame of the
    # cited set and a voltage each, the frames the junction composed here.
    points = tuple(point_folders(base, task, stage,
                                 frames=composed.junction.n_frames))
    citation = task.slots["junction"]

    def provide_data(plan):
        # The pseudopotentials: a cited run's travel with it, a cited pair's
        # come from the library the template names (`engines/transport.md`
        # § 3.1) -- screened against the config the rung's decks render
        # from: what it checks is each file's XC family against the
        # functional the run will ask for.
        cfg = configure(pset[0].render_config())
        if composed.provenance["kind"] == "pair":
            siesta_provide_pseudos(composed.junction, cfg, base,
                                    plan=plan)
        else:
            provide_pseudos(composed.junction, cfg, base,
                                       citation, plan=plan)

    def job_facts(job):
        # The rung's own restart files (`transport.stages.warm_declaration`:
        # the seed's and the device's; the leads and the transmission carry
        # none) -- and TBtrans post-processes the device run from its own
        # deck, its binary riding the allocation road into the wrapper
        # (`model.Job.program`, `engines/transport.md` § 6.1b).
        return dataclasses.replace(
            job, warm=warm_declaration(stage, task.label, base),
            # THE RUN'S SCRATCH FOLDER (`engines/transport.md` § 2a.11): the
            # device's, the vocabulary's name.
            scratch_dir=scratch_dir_of(base) if stage == "device" else None,
            program=("tbtrans" if stage == "transmission" else None))
    return Rung(struct=struct, render_kind="transport", configure=configure,
                sources=transport_sources,
                # THE DECK'S POINT reaches the spec: the device names its
                # run's shared files the way its point reaches them
                # (`engines/transport.md` § 2a.11).
                spec_extra=lambda element, cfg, point: {
                    "state": state, "point": point,
                    # WHERE THE DEVICE KEEPS THE LEADS' FILES: the
                    # vocabulary's folder (`engines/transport.md` § 2a.11).
                    "scratch_dir": scratch_dir_of(base),
                    # WHAT A TRANSMISSION POINT COMPUTES where it is not its
                    # window whole -- the prep's decision (§ 2a.16); a rung
                    # that runs once, its one point.
                    "contours": (contours or {}).get(point_key(point))},
                points=points, at_point=at_point,
                struct_at=lambda pt: at_frame(struct, pt),
                provide_data=provide_data, job_facts=job_facts)


#: THE KINDS WITH STEPS OF THEIR OWN, keyed by ``(engine, kind)``, each with
#: the builder of its rung (`script-preparation.md` § 3.0) -- data the one
#: conductor reads.  Every other kind is :func:`_described_rung`.
RUNGS = {
    ("siesta", "vibration"): _siesta_vibration_rung,
    ("siesta", "transport"): _transport_rung_of,
}


@dataclass(frozen=True)
class Resolved:
    """Steps 1 and 2 answered (`script-preparation.md` § 3.0), for every step
    after them: the description, its template and the machine's record --
    each READ ONCE by the prep that holds this -- the allocation folded once
    and the stage's elements resolved once.  The steps after them read these
    and none of the three files."""
    task: "Task"
    template: Path
    template_text: str
    #: the machine's record, its activation checked
    environment: "Environment"
    allocation: Resources
    pset: "ParameterSet"
    #: the stage's token, by the one stage-number door (`stage_home`)
    token: str = ""
    #: the label the rung's files are named on, by the same door
    #: (`StageHome.label`: the calculation's, or a transport lead's own stem)
    label: str = ""
    #: the machine named at this prep (``--target``), or ``None``
    target: Optional[str] = None
    #: a benchmark's grid, or ``None``: a run
    sweep: object = None
    #: which config files answered, and what each supplied -- built once,
    #: from the record read (`runtime_config.config_provenance`): what
    #: `STAGE-PLAN.md`, the pipeline log and the ledger's *prepared* line say
    provenance: Optional[dict] = None
    #: where each value of ``allocation`` came from -- ``flag``, ``run
    #: card`` or ``allocation`` (`prep_inputs.fold_allocation`)
    sources: Optional[dict] = None
    #: the placement at checkpoint 4 on the queue it names -- what its jobs
    #: record, a benchmark's every trial, and its header renders
    #: (`job-system.md` § 6.0); ``None`` with no header written (a machine
    #: with no scheduler, `--no-sbatch`)
    placement: Optional[dict] = None
    #: a run's fit on that queue, the line prep says of it and records
    #: nowhere (`scheduler.md` R6); ``""`` for a benchmark, whose shelves
    #: say theirs
    fit: str = ""

    @property
    def shape(self):
        """The layout, as the description states it -- handed to every layer
        that asks (`Shape.named(task.shape)`, the one door)."""
        from ..paths import Shape
        return Shape.named(self.task.shape)

    def record_dir(self, base) -> Path:
        """Where this prep's record lives -- its job-set, `STAGE-PLAN.md` and
        pipeline log: the calculation's folder for a run, the stage's bench
        container for a benchmark (`job-contracts.md` § 6.3)."""
        from ..paths import Shape, bench_container
        return (Path(base) / bench_container(Shape.named(self.task.shape),
                                             self.token)
                if self.sweep is not None else Path(base))

    def log_path(self, base) -> Path:
        """The pipeline log this prep writes (`script-preparation.md`
        § 4.5): on the rung's own stem -- the deck's -- the one place its
        path is composed."""
        from ..pipeline_log import log_name
        t = self.task
        return self.record_dir(base) / log_name(self.label, self.token,
                                                t.engine, t.shape)


def _resolve_stage(stage, *, task, request, template, template_text,
                   environment, allocation=None, chosen=None, sweep=None,
                   pins=None, run_settings=None, translation=None, home,
                   target=None, provenance=None) -> Resolved:
    """Step 2, once (`script-preparation.md` § 3.0): the allocation folded --
    the caller's ask over the job request's, and its reporting policy -- and
    the stage's elements resolved from the template, the stage, the sweep,
    the pins and the machine's record.

    Asked by the prep entry at its checkpoint 4, where the job it makes is
    placed (`job-system.md` § 5.0).  ``template`` ``None`` is refused: the
    folder is a template PLUS a description.  ``home`` is the stage's
    (`naming.stage_home`): its token and the label its files are
    named on."""
    from ..resolve import ResolveError, resolve
    from .prep_inputs import fold_allocation
    # THE JOB REQUEST'S OWN ALLOCATION, under the caller's (2026-08-24).
    # `job-request.json` carries the queue, the wall and the memory a person
    # chose for this calculation (`job_request.Allocation`), so a prepared
    # bundle needs
    # no flag to know them -- and an explicit flag still WINS, because a
    # person typing `--mem` now is answering about now.  Field by field,
    # not object by object: `--np 8` alone must not erase the file's
    # memory ask, which a whole-object override would do silently.
    # AND THE SHAPE IT DECIDES, when it decides one.  It arrives as an
    # argument because its producer, `prep_inputs.declared_run_shape` -- a
    # DIRECT MAP of the condition's machine items, never the grid
    # enumerator -- runs with the rest of the inputs before the five steps
    # (A12), so `prep` folds what it is handed and owns no second
    # translation (`generator.md` § 2).
    allocation, sources = fold_allocation(allocation, request.allocation,
                                          chosen)
    # AND THE JOB REQUEST'S REPORTING POLICY, at the same seam and for the
    # same reason: the file says when this calculation should speak up, so a
    # prepared bundle needs no flag to know it.  A separate line rather than a
    # third field on the helper above -- that one is about the ALLOCATION's
    # precedence against a CLI flag, and notify has no flag to lose to.
    allocation = _with_notify(allocation, request.notify)
    try:
        pset = resolve(template_text, task, engine_seam(task.engine).config_cls,
                       allocation=(allocation or Resources()), stage=stage,
                       sweep=sweep, pins=pins, run_settings=run_settings,
                       launch_from=sources, translation=translation,
                       environment=environment)
    except ResolveError as exc:
        raise PrepError(str(exc)) from exc
    return Resolved(task=task, template=Path(template),
                    template_text=template_text, environment=environment,
                    allocation=allocation, pset=pset, token=home.token,
                    label=home.label, target=target, sweep=sweep,
                    provenance=provenance,
                    sources=sources)


def prep_calculation(calc, stage: Optional[str], *,
                     resolved: "Resolved", plan,
                     env: str = None,
                     emit_sbatch: bool = True,
                     opened: Optional[list] = None,
                     findings: Optional[list] = None,
                     walks: Optional[dict] = None,
                     notes: Optional[list] = None,
                     copied: bool = False,
                     contours: Optional[dict] = None) -> List[Path]:
    """**`prep`, entire** — the five steps of `project-layout.md` § 2.3.1, in
    the order it calls *forced rather than chosen*.  ``notes`` is the
    planner's list, handed down to step 4 for what a rendering says.

    The stage's first run is opened with its own files alone: what any
    stage takes from another stage's run is its first launch's
    (`job-system.md` § 6.0).

    1. **resolve the machine** — READ its record, persist the snapshot as
       ``environment.json``; refuse when no record answers (§ 3.1);
    2. **resolve the parameters** — the description ⊕ this stage ⊕ the sweep ⊕
       the pins, into a :class:`~molbuilder.resolve.ParameterSet`;

    -- each file read once, and the stage resolved once (:class:`Resolved`):
    ``resolved`` is the entry's, answered at its own checkpoints and handed
    over (:func:`plan_stage`, `job-system.md` § 5.0, checkpoint 4).  The planner is
    its one caller;
    3. **render the deck(s)** — one per element of that set;
    4. **render the wrapper**;
    5. **build the run directory**.

    Every prep writes the PIPELINE LOG -- what each step received, decided
    and produced -- beside this prep's ``STAGE-PLAN.md``: an observer of the
    pipeline, never a step in it (`script-preparation.md` § 4.5).

    ``opened``, when given, receives the :class:`~molbuilder.jobset.materialize.Attempt`
    reports of the attempts this prep opened.  A caller reporting on the
    attempt reads its freshness from here: opening it a second time finds it
    already there, unlaunched, and calls it reused.

    ``walks``, when given, receives the fit of each job this prep writes
    that walks several members -- a run of points, a benchmark's shelf --
    keyed by its header (:func:`walk_header_of`).

    ``copied`` -- a member of a group copied in from the finished
    calculation a transport calculation reuses (`jobset.reuse`): its files
    planned as any stage's, no run opened -- its runs are the copies, which
    the entry plans after it (`engines/transport.md` § 2a.16).  ``contours``
    is what each transmission point computes where it is not its window
    whole, by the point's key (`reuse.point_key`, § 2a.16).

    ``findings``, when given, receives what each deck's checks said
    (`script_emit.prepare_deck`) -- the terminal also reads them on stderr as
    each deck renders; the Task setup tab reads them from here, through the
    one prep entry's answer.

    ``plan`` (`jobset.planned.Plan`) receives everything this prep writes,
    and is carried out by the entry, which saves the folder's state between
    the two (`job-system.md` § 5.0): nothing is written until
    every step has passed, so a refusal writes nothing.
    While the steps plan, every reader of the calculation's pseudopotentials
    reads the folder as the plan will leave it (`pseudos.PLANNED`).

    Returns the per-job directories. Raises :class:`PrepError`.
    """
    from ..pseudos import PLANNED
    scope = PLANNED.set(plan)
    try:
        return _plan_calculation(
            calc, stage, resolved, env=env, emit_sbatch=emit_sbatch,
            opened=opened, findings=findings,
            plan=plan, walks=walks,
            notes=notes, copied=copied, contours=contours)
    finally:
        PLANNED.reset(scope)


def _plan_calculation(calc, stage: Optional[str], resolved: "Resolved",
                      *, env: str = None,
                      emit_sbatch: bool = True,
                      opened: Optional[list] = None,
                      findings: Optional[list] = None,
                      plan,
                      walks: Optional[dict] = None,
                      notes: Optional[list] = None,
                      copied: bool = False,
                      contours: Optional[dict] = None) -> List[Path]:
    """The steps of :func:`prep_calculation`, planned into ``plan``: nothing
    is written here.  Steps 1 and 2 arrive answered (``resolved``): this
    snapshots the machine's record it was handed and plans the rest -- it
    reads none of the description, the template or the record again.
    ``calc`` is the calculation as this prep holds it, its plan as the
    prep's overlay leaves it (`Calculation.as_planned`)."""
    from ..pipeline_log import PipelineLog, config_rows
    from ..task import FILENAME as TASK_FILENAME
    base = Path(calc.base).resolve()

    task, environment = resolved.task, resolved.environment
    template_path, pset = resolved.template, resolved.pset
    allocation, sweep, target = (resolved.allocation, resolved.sweep,
                                 resolved.target)
    # WHERE THIS PREP'S RECORD LIVES -- its job-set, STAGE-PLAN.md and
    # pipeline log -- by the one rule the entry asks too (`Resolved.
    # record_dir`): the sweep's record and its trials' container are one
    # place by construction (A-1/A-2).
    record_dir = resolved.record_dir(base)

    # ---- 1. the machine: its record, read and checked at the entry's ------ #
    # checkpoint 4 -- snapshotted here, the calculation's copy of the record
    # it was handed.
    set_machine(base, target, environment=environment, plan=plan)

    # THE LOG OPENS HERE, and not before: its NAME carries the stage token, and
    # the token needs the description.  Everything step 1 did is still in hand,
    # so nothing is lost by writing that phase a moment later than it ran.
    # It is HELD, and written with the rest of the plan once every step has
    # passed (`script-preparation.md` § 4.5): a refused prep writes no log.
    log = PipelineLog.open(resolved.log_path(base), label=resolved.label,
                           token=resolved.token, engine=task.engine,
                           shape=task.shape)
    log.phase("STEP 1 · MACHINE — where this job will run")
    log.received("calculation", str(base))
    log.received(TASK_FILENAME, f"{task.label} · {task.engine} · "
                                f"{task.shape} · "
                                f"{len(task.stages or ())} stage(s)")
    for _group, _line in _environment_rows(environment):
        log.produced(_group, _line)
    # WHICH FILE supplied each execution setting -- through the ONE
    # formatter that already answers this, not a second table of the same
    # facts.  It is also the security boundary: `config_provenance`
    # publishes only the sections marked provenance-safe
    # (`configuration.md` § 4), so nothing else may be printed here.
    # Read once, with the record (`Resolved.provenance`): STAGE-PLAN.md and
    # the ledger's *prepared* line say the same.
    from ..runtime_config import format_provenance
    log.produced("config", "which file supplied each setting")
    log.text(format_provenance(resolved.provenance))
    seam = engine_seam(task.engine)
    log.phase("STEP 2 · RESOLVE — the values for this rung")
    log.received(template_path.name, f"{len(pset[0].provenance)} fields")
    log.received("stage", pset.stage)
    log.received("allocation", _flat_resources(allocation or Resources()))
    # EACH VALUE'S SOURCE IN THE ONE VOCABULARY, the words `resolve`
    # composed for the deck's table (`ResolvedConfig.said`, `engines/
    # template.md` § 5.4).
    for _el in pset:
        # THE KIND'S VALUES, the ones `resolve` says who set.
        _rows = [r for r in config_rows(_el.render_config(), _el.provenance)
                 if r[0] in _el.said]
        _decided = [r for r in _rows if r[2] not in ("template", "default")]
        log.step(f"{_el.label} — what this rung decided")
        for _n, _v, _s in _decided:
            log.chose(_n, _v, _el.said[_n])
        log.step(f"{_el.label} — the rest, as the template and the "
                 f"catalogue declare")
        for _n, _v, _s in _rows[len(_decided):]:
            log.chose(_n, _v, _el.said[_n])
    log.produced("ParameterSet", f"{len(pset)} element(s) -> spec_for")

    # ---- 3. the structure: the kind's rung ----------------------------- #
    # WHAT THE DECKS DESCRIBE, and what the kind adds to the steps after
    # this one, from its one record (`script-preparation.md` § 3.0): the
    # structure as described, unless the kind composes or reorders it -- a
    # transport junction and its leads; a SIESTA vibration's held-first
    # copy.
    rung = RUNGS.get((str(task.engine), str(task.calculation)),
                     _described_rung)(calc, pset, seam=seam,
                                      sweep=sweep, log=log, plan=plan,
                                      contours=contours)
    if notes is not None:
        notes.extend(rung.said)
    # The DATA FILES the engine will open, before any deck is written: a
    # missing pseudopotential is a run that cannot start, and finding that out
    # here costs a second (project-layout.md § 2.6).  Idempotent -- what is
    # already in the folder is left alone.  The elements come from the
    # structure, which has just been checked against the description's witness.
    if rung.provide_data is not None:
        with user_error_as_prep(), _calling(
                "provide_data", engine=task.engine):
            rung.provide_data(plan)
    elif seam.provide_data is not None:
        with user_error_as_prep(), _calling(
                "provide_data", engine=task.engine):
            seam.provide_data(rung.struct, pset[0].render_config(), base,
                              plan=plan)
    token = resolved.token
    jobs: List[Job] = []
    from .materialize import trial_work_dir
    from .naming import trial_dir
    from ..paths import Shape as _Shape
    _shape = _Shape.named(task.shape)
    # ONE line per unique finding, however many trials repeat it (user,
    # 2026-08-28, O5).  The gate still FIRES per deck -- every trial is
    # validated and its own <deck>.validation.txt carries its findings --
    # but sixteen identical thin-vacuum warnings on one terminal is
    # noise wearing a safety vest.  Scoped to THIS loop (not a module
    # global) so a long-lived server process cannot quietly swallow a
    # later prep's warnings -- and to this THREAD: the gate's report reads
    # its stream from `validation.REPORT_STREAM`, a context variable.
    import io as _io
    import sys as _sys
    from ..validation import REPORT_STREAM

    class _OncePerLine(_io.TextIOBase):
        def __init__(self, wrapped):
            self._w, self._seen, self.dropped = wrapped, set(), 0

        def write(self, text):
            for line in text.splitlines(keepends=True):
                key = line.strip()
                if key and (key.startswith("warn") or key.startswith("info")):
                    if key in self._seen:
                        self.dropped += 1
                        continue
                    self._seen.add(key)
                self._w.write(line)
            return len(text)

        def flush(self):
            self._w.flush()

    _once = _OncePerLine(_sys.stderr)
    _scope = REPORT_STREAM.set(_once)
    try:
        for element in pset:
            # THE NAMES OF THIS ELEMENT'S FILES (`runfiles.RunNames`): its
            # own label -- a trial's carries its point -- and stage, in the
            # calculation's shape.  Its deck, its run script and its seed are
            # all named by them, so none can spell another's name its way.
            _names = RunNames.of(element.label, token, _shape.name,
                                 trial=element.is_trial)
            script = _names.name(seam.suffix)
            # WHERE THIS ELEMENT'S FILES GO -- its own directory, never the
            # bundle root (user, 2026-08-24; `project-layout.md` § 1.0: "only
            # rendered files and copies go down to where the engine
            # runs").  The directory is the
            # same one `job_dir_names` will answer for this job, computed from
            # the same two facts (token + trial-ness), so the deck is born
            # where the launch will look for it.
            if element.is_trial:
                # ONE RULE, asked -- not composed again from the same two
                # facts.  `naming.trial_dir` is what `job_dir_names`
                # itself uses, so the deck is born where the launch looks by
                # CONSTRUCTION rather than by a comment promising it.
                #
                # The dir is named by the JOB's name -- the sweep coordinate,
                # `_job_for`'s own first rule -- never by the element's
                # RELABELLED SystemLabel (`<calc-label>-<coord>`): the two
                # differ by the calculation prefix, and every reader asks
                # `job_dir_names`.
                from ..resolve import point_token as _pt
                _trial = _pt(element.point)
                _jdir = trial_work_dir(
                    base / trial_dir(_shape, token, _trial), _shape, _names)
            else:
                _trial = None
                _sd = _shape.stage_dir(token)
                _jdir = base if _sd == "." else base / _sd
            plan.folder(_jdir)
            # The deck is rendered from values ⊕ THIS element's allocation, so it
            # records the rank count it actually assumed.  Rendering from the
            # values alone emits `mpi_np auto` and the launch check then refuses a
            # deck that `prep` itself just made.
            # THE CONFIG THIS ELEMENT RENDERS FROM, as the kind's rung says
            # (a lead's own label, the junction's state) -- identity for
            # every kind with no steps of its own.
            cfg = rung.configure(element.render_config())
            # WHO SET EACH VALUE THE DECKS WRITE, the rung's own steps
            # included (`job-contracts.md` § 3.2).
            from ..resolve import SET_AFTER, said_by
            _sources = rung.sources(element)
            if element.is_trial:
                # The deck's OWN identity line carries the trial label -- this,
                # not the filename, is what keys SIESTA's warm files away from
                # the real run's (project-layout.md § 2.3.2).
                with _calling("relabel", engine=task.engine,
                              where=element.label):
                    cfg = seam.relabel(cfg, element.label)
                _sources = said_by(_sources, "system_label",
                                   SET_AFTER["trial_label"])
                # The forced cold has ONE setter -- the measurement pin
                # (`template.TRIAL_PINS`, resolved with provenance "pin") --
                # and ONE verifier, the submission door
                # (`agreement.check_trial_starts_cold`; user-settled
                # 2026-08-21: prep bakes the intent, submission determines
                # the actual state).
            # The render's refusals (missing pseudos above all) are user-fixable
            # and translate to PrepError -- see user_error_as_prep
            # (2026-08-12 plan A8).
            # STEP 3, WHOLE, IN ONE CALL: validate the settings, render the deck,
            # write it through the one writer (which keeps the reader's USER-CUSTOM
            # block), then read the file back and refuse one that does not say what
            # it was meant to say.  The conductor says WHEN; the framework owns the
            # order (`script-preparation.md` § 4.3).
            log.phase(f"STEP 3 · DECK — {script}")
            log.received("config", f"{type(cfg).__name__}"
                                   + f", stage_token={token}"
                                   + (f", trial {element.label}"
                                      if element.is_trial else ""))
            # EVERY DECK THIS ELEMENT WRITES: its own folder's -- or, for a
            # swept rung, one in each point's folder, written at its point
            # and none at the stage level (`engines/transport.md` § 2a.11).
            spec = None
            # EACH POINT'S OWN SPEC: a frame set's points are its frames,
            # each placed by the engine where its own deck puts it.  A
            # BENCHMARK TRIAL IS ONE DECK in its own folder: it measures the
            # one call that costs, never a run of points.
            spec_at: dict = {}
            _points_of = [] if element.is_trial else list(rung.points)
            for _out, _point in (_points_of or [(_jdir, None)]):
                # A POINT'S FOLDER BELOW ITS STAGE says what it is, as every
                # folder this tree makes below the root does
                # (`project-layout.md` invariant 6b): a container, holding
                # the point's deck and run script every run copies.
                if _point is not None:
                    open_container(base, _out, plan)
                else:
                    plan.folder(_out)
                _at = rung.at_point(cfg, _point)
                _struct = (rung.struct_at(_point)
                           if rung.struct_at is not None
                           and _point is not None else rung.struct)
                with user_error_as_prep():
                    with _calling("spec_for", engine=task.engine,
                                  where=script):
                        _spec = seam.spec_for(
                            _struct, _at, names=_names,
                            calculation=rung.render_kind,
                            **rung.spec_extra(element, cfg, _point),
                            # A TRIAL'S DECK NAMES ITS OWN LAUNCH (plan § 5w
                            # K12) -- the bench lane is SIESTA's alone.
                            **({"trial": _trial} if _trial else {}))
                    # WHAT THE RUNG WRITES BESIDE THE DECK, before the deck
                    # is checked as the engine will read it (`Rung.beside`).
                    if rung.beside is not None:
                        rung.beside(_point, _spec, _out, plan)
                    _sc.prepare_deck(_spec, _struct, _at, _out / script,
                                     log=log, dest_dir=base,
                                     findings=findings, plan=plan,
                                     sources=_sources)
                spec = spec or _spec
                spec_at[_out] = _spec
            if seam.sibling_artifacts is not None:
                with _calling("sibling_artifacts", engine=task.engine,
                              where=script):
                    seam.sibling_artifacts(rung.struct, cfg, _jdir / script,
                                           kind=rung.render_kind,
                                           names=_names, plan=plan)
            # THE PROGRESS LOG IS SEEDED BESIDE THE DECK IT FOLLOWS -- the
            # element's own, or for a sweep each point's -- and moved into the
            # run that writes it (`_move_progress_channel_into`), never left
            # in a folder nothing writes to again.  AND NAMED FOR ITS DECK
            # (`job-contracts.md` § 6.3: the trajectory log takes the deck's
            # basename): the element's label, the deck's own.
            for _seed_dir, _point in (_points_of or [(_jdir, None)]):
                _seed_trajectory_log((rung.struct_at(_point)
                                      if rung.struct_at is not None
                                      and _point is not None
                                      else rung.struct),
                                     cfg, _seed_dir,
                                     engine=task.engine,
                                     kind=task.calculation, names=_names,
                                     frame=spec_at[_seed_dir].engine_frame,
                                     relaxes=(rung.relaxes_at(_point)
                                              if rung.relaxes_at is not None
                                              else rung.render_kind
                                              == "optimization"),
                                     plan=plan)
            log.step("what this deck's text PROMISES, kept")
            log.produced("sibling_artifacts",
                         "written" if seam.sibling_artifacts is not None
                         else "nothing (W5)")
            log.produced("trajectory log",
                         "seeded" if getattr(cfg, "write_molwatch_log", False)
                         else "not asked for")
            # A BENCHMARK TRIAL IS NOT FINISHED: it measures how long a
            # setting takes under capped SCFs, and modes derived from those
            # would be a spectrum of nothing (`engines/vibration.md` § 5.5).
            jobs.append(rung.job_facts(_job_for(
                element, script, task, pset.stage, seam, base,
                finish=(None if element.is_trial else spec.finish),
                # WHERE IT WAS PLACED (`job-system.md` § 6.0): a run's
                # queue, and a benchmark's one queue on each trial, recorded
                # on its job -- what launch sends to.
                placement=resolved.placement)))
    finally:
        REPORT_STREAM.reset(_scope)
    if _once.dropped:
        # TRUE OF A PREVIEW TOO, which writes nothing: the report is the
        # deck's, written with it when the prep is.
        print(f"  (each warning shown once; {_once.dropped} repeat(s) "
              f"across this prep's other decks suppressed -- each deck's "
              f"own .validation.txt, written with it when the prep is, "
              f"carries its full findings)", file=_sys.stderr)

    # ---- 4 + 5, and the record floor 3 leaves behind -------------------- #
    # ``kind`` is INTENT, not length (review 2026-08-12: a one-point grid is
    # still a benchmark).  A sweep's whole record — its job-set, its plan,
    # later its verdict — lives in the stage's ``bench/`` container
    # (job-contracts.md § 6.3's Directories row, the cross-layer authority),
    # so two stages' benchmarks can never collide.  The ROOT job-set.json
    # is the RUN's plan and MERGES per stage, so preparing `tight` keeps
    # `coarse` — status, and the cross-stage ``--from`` carry whose
    # pair rule needs the source job on file, read the whole ladder.
    kind = KIND_SWEEP if sweep is not None else KIND_LADDER
    js = JobSet(name=task.label, engine=task.engine, kind=kind,
                shared=_shared_for(base, seam, engine=task.engine,
                                   plan=plan),
                jobs=jobs)
    if kind == KIND_SWEEP:
        plan.folder(record_dir)
    else:
        # A ROW WHOSE STAGE THE DESCRIPTION NO LONGER HOLDS leaves the plan
        # here, never launched -- and the header of the job it was in with
        # it, named while every member's row is still on file
        # (`group.header_of`).  A LAUNCHED one stays, with its job's header:
        # its runs' facts are read from its job (§ 5.0, 2a), and a stage
        # described again finds it launched.
        _old = calc.jobset
        _keep = {s.name for s in task.stages}
        if _old is not None:
            from ..identity import stage_key
            from ..runrecord import LaunchRecordError
            from .group import header_of
            from .materialize import ever_launched
            _held = {stage_key(s.name) for s in task.stages}
            for _j in _old.jobs:
                if stage_key(_j.name) in _held:
                    continue
                try:
                    _ran = ever_launched(calc, _j)
                except LaunchRecordError as exc:
                    raise PrepError(str(exc)) from None
                if _ran:
                    _keep.add(_j.name)
                    continue
                _h = header_of(calc, _j)
                if _h is not None and plan.is_file(_h):
                    plan.remove(_h)
        js = _merge_run_jobset(
            _old, js,
            # The CURRENT ladder bounds what the merge keeps -- and the
            # launched stages it no longer holds.
            ladder=frozenset(_keep))
    # WHAT ITS FIRST LAUNCH WILL TAKE, said under `STAGE-PLAN.md`'s table
    # (`job-system.md` § 5.4) -- the line both prep doors print.
    this_prep = None
    if kind == KIND_LADDER and stage:
        from .ready import at_launch
        this_prep = this_prep_lines(stage, at_launch(
            calc.with_plan(js), stage, job=js.job(stage)))
    log.phase("FLOOR 3 · THE JOB-SET — what was declared to the runner")
    log.received("kind", kind)
    for _j in js.jobs:
        log.produced(_j.name, f"{_j.script}  "
                              f"{_flat_resources(_j.resources)}")
    log.produced(JOBSET_FILENAME, str(JobSet.path_of(record_dir)))
    # The allocation is NOT passed on: every job already carries its own
    # resolved resources, per element (generator.md § 5).
    # ONE FRAMEWORK, LOCAL OR REMOTE (user, 2026-08-24: *"the jobset probe
    # should do its job whether it's running on the local machine or a
    # remote HPC environment.  Either way, it should provide the only set
    # of data the script generator would need to generate a fully
    # self-contained script"*).
    #
    # So the record ALWAYS travels to the generator -- step 1 resolved one
    # either way, and `--target` only decides WHICH machine it describes.
    # There is no second road for the local case: a machine that has been
    # probed states its own activation, and the generator reads it from the
    # same field whoever it is for.
    #
    # A record that does not state it was refused at step 1, before anything
    # was written (`machine.require_activation`) -- whichever machine it describes.
    # A SCAN'S POINTS hold a deck each, so each gets the job's wrapper too.
    # EACH POINT'S FOLDER, and how its run script is rendered beyond the
    # job's own facts (`Rung.wrapper_at`).
    _points = [(d, rung.wrapper_at(pt)) for d, pt in rung.points]
    dirs = prep_jobset(js, base, env=env, emit_sbatch=emit_sbatch,
                       record_dir=record_dir, log=log,
                       machine_record=environment,
                       render=(None if kind == KIND_SWEEP
                               else {j.name for j in jobs}),
                       points={j.name: _points for j in jobs}, plan=plan,
                       shape=resolved.shape, this_prep=this_prep,
                       provenance=resolved.provenance, notes=notes)

    # ---- THE ATTEMPT, because PREP is what sets a stage up to run ------- #
    # Opening the attempt is design, none of it spending a queue slot;
    # `launch` starts what prep set up, and refuses a hierarchical stage with
    # no attempt open (C5) because opening one is not its job.  A bench
    # point's deck is rendered straight into its attempt (`trial_work_dir`
    # above).
    if kind == KIND_LADDER and stage and _points:
        # A SWEPT RUNG'S RUN (`engines/transport.md` § 2a.11): `run-<n>/`,
        # a folder per point inside it holding copies of the point's
        # prepared files -- the points of the junction composed in THIS
        # plan, whose record may not be on disk yet.
        from ..paths import bench_container
        from ..runfiles import LAUNCH_DIR
        from .materialize import (empty_old_points, ever_launched,
                                  open_sweep_run)
        # A POINT THE DESCRIPTION NO LONGER HOLDS leaves no folder below the
        # stage (§ 5.0, 2a: its files rendered again).
        home = stage_home(base, task, stage)
        gone = empty_old_points(
            home.dir, [pt.rel for _d, pt in rung.points], plan,
            keep=(LAUNCH_DIR, Path(bench_container(
                resolved.shape, home.token)).name))
        planned = calc.with_plan(js)
        if copied or ever_launched(planned, planned.job(stage)):
            # A LAUNCHED RUNG OF POINTS, prepared again for its job request
            # (`execution/job-request.md` § 5): no run opened -- its next
            # launch opens its next one -- and its points the ones its runs
            # ran, as its last prep left them below the stage: a point added
            # or taken away is another calculation.  A rung COPIED IN opens
            # none either: its runs are the copies, of exactly its points.
            added = [str(pt.rel) for d, pt in rung.points
                     if not Path(d).is_dir()]
            if not copied and (gone or added):
                from .commands import redo_launched
                raise PrepError(
                    f"stage {stage!r} was launched, and its points now are "
                    f"not the ones its runs ran -- "
                    + "; ".join(([f"added: {', '.join(added)}"] if added
                                 else [])
                                + ([f"taken away: {', '.join(gone)}"]
                                   if gone else []))
                    + ".  A launched stage is prepared again for its job "
                      "request alone (execution/job-request.md § 5); to "
                      "change what it computes, "
                    + redo_launched(stage, calc=calc))
            for d, _point in rung.points:
                _drop_progress_channel(Path(d), plan)
        else:
            sweep = open_sweep_run(planned, stage, plan=plan)
            # EACH POINT'S LOG, from the point's folder below the stage,
            # where it was seeded beside the point's deck, into the run's
            # point.
            seeded = {pt.rel: d for d, pt in rung.points}
            for pdir, point in sweep.points:
                _move_progress_channel_into(pdir, plan,
                                            seeded=seeded[point.rel])
            if opened is not None:
                opened.append(sweep)
        # THE WALK'S ONE HEADER, written and placed here as a group's is
        # (`job-system.md` § 6.0): one allocation walking every point, their
        # walls summed (`group.envelope`) -- or the stage's one wall, where
        # its points share it (`jobset.points`) -- launch sends it as it is.
        from ..runfiles import GroupNames
        from .group import envelope
        from .points import walls_of
        job = js.job(stage)
        walk = walk_header_of(
            home.dir, GroupNames.of_points(
                run_names(js, job, resolved.shape).stem),
            envelope([job] * walls_of(task, len(_points))), environment,
            engine=task.engine, jobs=[job], emit_sbatch=emit_sbatch,
            what=f"stage {stage!r}, "
                 f"{walk_words(planned, stage, read=plan.read_text)}")
        if walk is not None:
            open_container(base, home.dir / LAUNCH_DIR, plan)
            plan.text(walk.path, walk.text)
            if walks is not None:
                walks[walk.path] = walk.fit
        else:
            # NO HEADER THIS TIME, so none an earlier prep wrote is left to
            # be sent as this one's.
            _left = (home.dir / LAUNCH_DIR / GroupNames.of_points(
                run_names(js, job, resolved.shape).stem).name(".sbatch"))
            if _left.is_file():
                plan.remove(_left)
    elif kind == KIND_LADDER and stage:
        reports = ([] if copied else
                   _open_attempts(calc.with_plan(js), stage, plan=plan,
                                  shape=resolved.shape))
        if opened is not None:
            opened.extend(reports)
    elif kind == KIND_SWEEP:
        # EACH SHELF'S ONE HEADER, written and placed here on the
        # benchmark's one queue (`generator.md` § 4.3a, `job-system.md`
        # § 6.0): its trials' shared request -- launch walks the trials
        # still to run and sends it as it is.
        from ..runfiles import LAUNCH_DIR
        from .shelves import ShelfError, shelf_request, shelves_of
        try:
            shelves = [(names, shelf_request(trials), trials)
                       for _side, names, trials in shelves_of(js.jobs)]
        except ShelfError as exc:
            raise PrepError(str(exc)) from None
        for names, request, trials in shelves:
            got = walk_header_of(
                record_dir, names, request, environment, engine=task.engine,
                jobs=trials, emit_sbatch=emit_sbatch,
                what=(f"the benchmark's shelf {names.stem} ({len(trials)} "
                      f"trial(s), one after another)"))
            if got is not None:
                open_container(base, record_dir / LAUNCH_DIR, plan)
                plan.text(got.path, got.text)
                if walks is not None:
                    walks[got.path] = got.fit

    # THE LOG, then THE PLAN'S ROW LAST -- the moment the stage is prepared
    # (`job-system.md` § 5.0, checkpoint 6): a prep that stops before this
    # line has not prepared the stage.
    log.close()
    plan.text(log.path, (plan.read_text(log.path) if plan.is_file(log.path)
                         else "") + log.held_text())
    js.write(record_dir, plan=plan)
    return dirs


def as_written(path, data: bytes) -> bool:
    """Whether the file at ``path`` holds ``data``, the stamps of both
    masked (when and by which build each was written,
    `deck_record.without_stamps`) -- the one test of *what prep writes now is
    what is there*, for a stage's own files and a group's header, asked by
    prep of a group prepared and never launched."""
    from ..deck_record import without_stamps
    path = Path(path)
    return path.is_file() and (without_stamps(path.read_bytes())
                               == without_stamps(data))


def prepared_differs(calc, plan, prepared, planned) -> List[str]:
    """THE PREPARED STAGE AGAINST WHAT ``plan`` WRITES FOR IT NOW:
    every file its run takes, as written (:func:`as_written`) -- by the
    catalogue, an input its run reads or a file derived for the stage or
    its run: its deck, its scripts and headers, the bundles beside them
    (:func:`_taken_by_its_run`); for a stage never launched, each deck of
    its first run -- each point's -- the same calculation as the one prep
    renders now (`deck_record.same_calculation`: a launch that took a
    finished point over put that point's own deck there, and filling a run
    anew is the next launch's, R-L25), and none removed; and its row in
    `job-set.json`, its members aside -- the run settings every script
    states.  Each that differs, named from the calculation; none when the
    plan writes the stage as it stands.  The comparison a prep of a group
    prepared and never launched asks (`job-system.md`, *The task*, 2);
    launch compares nothing (§ 6.0).  ``prepared`` is the
    stage's row on disk, ``planned`` the row the plan holds for it
    (``None`` when it planned none)."""
    from ..deck_record import same_calculation
    from ..paths import attempt_index
    from .materialize import ever_launched
    base, task = Path(calc.base), calc.task
    names = run_names(calc.jobset, prepared, Shape.named(task.shape))
    deck = Path(prepared.script).name
    home = stage_home(base, task, prepared.name).dir
    runs = ever_launched(calc, prepared)

    def in_a_run(path) -> bool:
        return any(attempt_index(part) is not None
                   for part in Path(path).relative_to(base).parts)

    def taken(path) -> bool:
        return _taken_by_its_run(path, names, home)

    differ = []
    for p in plan.writes():
        if not taken(p):
            continue
        rel = str(p.relative_to(base))
        if not in_a_run(p):
            if not as_written(p, plan.read_bytes(p)):
                differ.append(rel)
        elif runs:
            continue                  # a launched stage's runs are its own
        elif p.name == deck:
            if not p.is_file() or same_calculation(
                    str(task.engine), read_deck(p),
                    read_deck(p, read=plan.read_text)):
                differ.append(rel)
        elif not as_written(p, plan.read_bytes(p)):
            differ.append(rel)
    differ += [f"{p.relative_to(base)} (removed)" for p in plan.removes()
               if taken(p) and (not in_a_run(p) or not runs)]

    def row(j):
        # ITS JOB'S MEMBERS are the group's to check, by membership (R-L8).
        return {k: v for k, v in j.to_dict().items() if k != "members"}
    if planned is None or row(planned) != row(prepared):
        differ.append(f"its row in {JOBSET_FILENAME}")
    return differ


def _taken_by_its_run(path, names, home) -> bool:
    """Whether ``path``, a file a plan writes, is one the run of the stage
    ``names`` names TAKES: in the stage's home, named for it or for no
    stage, and by the catalogue (`runfiles.WRITTEN`'s ``kind``, what losing
    it costs) an input its run reads, or a file derived for the stage or
    its run -- its deck, its scripts and headers, the bundles beside them --
    or the walk header of its points; never a record, nor a file of the
    calculation's own (its plan, its job set), which a prep writes for every
    stage.  In a calculation whose stages share one folder, the name is what
    tells one stage's file from another's (`runfiles.parse`)."""
    from ..runfiles import GroupNames, parse, row_for
    p = Path(path)
    try:
        p.relative_to(home)
    except ValueError:
        return False
    rec = parse(p.name, names.label)
    if rec is not None and rec.stage not in (None, names.stage):
        return False
    if p.name == GroupNames.of_points(names.stem).name(".sbatch"):
        return True
    row = row_for(p, names.label)
    return row is not None and (
        row.kind == "input"
        or (row.kind == "derived" and row.level in ("stage", "run")))


def _computes_else(base, engine: str, plan, answer) -> Optional[str]:
    """Every difference between what a launched stage's decks -- ``answer``'s,
    as ``plan`` writes them, the stage's and each point's -- compute and what
    the decks they replace computed, through the one door
    (`deck_record.same_calculation`), each named by its folder; ``None``
    when every one is the same calculation."""
    from ..deck_record import same_calculation
    if not answer.deck:
        return None
    said: List[str] = []
    for path in plan.writes():
        if path.name != answer.deck or not path.is_file():
            continue
        diffs = same_calculation(engine, read_deck(path),
                                 read_deck(path, read=plan.read_text))
        if diffs:
            said.append(f"{Path(path).parent.relative_to(base)}: "
                        + "; ".join(diffs))
    return " -- ".join(said) or None


def _open_attempts(calc, stage: str, *, plan, shape) -> List:
    """Open this stage's first run -- **step 6** -- with its own files, and
    return its :class:`~molbuilder.jobset.materialize.Attempt` report in a
    list, or ``[]`` when the shape keeps no run folders (flat: its container
    IS the run, § 1.5a).  A swept rung's run is
    :func:`materialize.open_sweep_run`'s.  A stage prepared again, never
    launched, reuses its ``run-0``, emptied of anything put there before
    (`materialize.empty_run`, `job-system.md` § 5.0, 2a); what it takes
    from another run is its first launch's (§ 6.0)."""
    from .materialize import ever_launched, prepare_attempt

    if shape is None or not shape.keeps_attempts_as_directories:
        return []
    # A LAUNCHED STAGE, prepared again for its job request, opens no run:
    # its next launch opens the next one (`execution/job-request.md` § 5),
    # and the progress log seeded beside its deck has no run to go to.
    job = calc.job(stage)
    if job is not None and ever_launched(calc, job):
        _drop_progress_channel(stage_home(calc.base, calc.task, stage).dir,
                               plan)
        return []
    rep = prepare_attempt(calc, stage, plan=plan)
    _move_progress_channel_into(rep.dir, plan)
    return [rep]


def walk_words(calc, stage: str, *, read=None) -> str:
    """A RUN OF POINTS' ONE JOB, in words -- the one composer both doors
    print and a placement refusal names it by: its points in the kind's
    word, counted as its declaration knows them at prep
    (`PointsOf.rung_points`) -- and where the stage's job plans more in a
    container below the run (`PointsOf.containers`, a vibration's optical
    calls), the rest planned in the job -- and the walls it asks: the
    stage's one where its points share it (`PointsOf.one_wall`), else
    their sum.  ``read`` reads a record the prep's plan holds and the disk
    does not yet -- a junction's, composed by this prep -- as the
    declaration reads it at the run's opening."""
    from .points import points_of
    pts = points_of(calc.task)
    known = len(pts.rung_points(calc, stage, read=read))
    unit = pts.unit
    planned = pts.planned_in_job and bool(pts.containers(calc, stage))
    count = (f"its {unit}s -- {known} known at prep, the rest planned in "
             f"the job" if planned else
             f"its {known} {unit}{'s' if known != 1 else ''}")
    return (f"one job walking {count} ("
            + ("the stage's one wall" if pts.one_wall
               else "their walls summed")
            + ")")


def _launch_as_written(plan, header, script, kind) -> dict:
    """What the job will be launched with, as the header its launch sends
    and its run script carry it -- A13's end point
    (`execution/architecture.md` § 5.2): the header's directive lines --
    ``kind`` the target's scheduler kind -- and the run script's stated
    counts, read from the text the plan holds by each writer's own reader,
    never worked out again."""
    from ..runwrap import stated_counts
    from ..scheduler.emit import Directives
    lines = (Directives.lines_of(plan.read_text(header), kind)
             if plan.is_file(header) else [])
    return {"header": lines,
            "said": Directives.said(lines) if lines else "",
            "run_script": (stated_counts(plan.read_text(script))
                           if plan.is_file(script) else [])}


def _walk_said(walk, kind) -> dict:
    """A group's one job as its answer carries it -- its header's directive
    lines, them in one line, and its fit (`scheduler.md` R6); empty with no
    header written."""
    from ..scheduler.emit import Directives
    if walk is None:
        return {"header": [], "said": "", "fit": ""}
    lines = Directives.lines_of(walk.text, kind)
    return {"header": lines, "said": Directives.said(lines), "fit": walk.fit}


def _move_progress_channel_into(attempt: Path, plan, *,
                                seeded: Optional[Path] = None) -> None:
    """Put the seeded progress log where the run will write it -- from
    ``seeded``, the folder it was seeded in (the run's container, or for a
    point of a run of points the point's folder below the stage).

    **The progress channel is a PRODUCT, so it belongs in the run and nowhere
    else** (`project-layout.md` § 1.0: the run directory "holds everything it
    produces", and § 1.4a's container/run split is what makes "nowhere else"
    checkable).  `prep` seeds it beside the deck it renders — which in the
    FLAT shape is already the run directory, and in the hierarchy is the
    stage CONTAINER, where nothing ever writes to it again.

    Measured 2026-09-19 on a finished Raman run: a 971-byte stub frozen at
    prep time with no ``# concluded:`` footer sat in ``01_raman/``, beside the
    real 1763-byte concluded log the run wrote in ``01_raman/run-0/``.
    Different inodes, one name, and an unconcluded log is how every reader
    tells a run is still going — so the stage directory reported a finished
    calculation as **Running**, for ever, and offered the stub to open.

    Moved rather than copied, and moved rather than left: a second copy is
    what created the problem.  The deck and wrapper legitimately exist at
    both levels (materialize: rendered files are born in the stage directory
    and copied down) because they are INPUTS; this is not one.

    THE ROLE IS NOT SPELLED HERE (R-RO1).  The catalogue's ``output`` column
    says which artifacts are the progress channel, and asking it is what keeps
    this true if a second engine ever seeds one.
    """
    stage_dir = attempt.parent if seeded is None else Path(seeded)
    if stage_dir == attempt:
        return
    for log in _seeded_progress_channel(stage_dir, plan):
        plan.move(log, attempt / log.name)


def _seeded_progress_channel(folder: Path, plan) -> List[Path]:
    """The progress logs in ``folder`` as the plan leaves it -- read as
    `runfiles.find_by_role` reads a folder on disk, each name's own role,
    the catalogue's ``output`` column saying which roles are the progress
    channel (R-RO1)."""
    from ..runfiles import progress_roles, role_of
    roles = set(progress_roles())
    return [f for f in plan.glob(folder, "*") if role_of(f.name) in roles]


def _drop_progress_channel(folder: Path, plan) -> None:
    """A LAUNCHED STAGE PREPARED AGAIN opens no run, so the progress log
    seeded beside its deck in ``folder`` -- the stage's, or a point's below
    it -- has no run to go to: taken out, as :func:`_move_progress_channel_into`
    takes it out by moving it, so no stub with no ``# concluded:`` footer is
    left to read as a run still going."""
    for log in _seeded_progress_channel(folder, plan):
        plan.remove(log)


def _merge_run_jobset(old: Optional[JobSet], new: JobSet,
                      ladder: Optional[frozenset] = None) -> JobSet:
    """The root ``job-set.json`` is the RUN's whole plan: each stage's prep
    updates its OWN row and leaves the others standing -- ``old`` as the
    plan leaves it (`JobSet.of`).

    The status rollup and the ``--from`` pair rule read the source job's
    row (`project-layout.md` § 2.3.4 row 3).

    ``ladder`` names the rows kept standing: the CURRENT task's stages, and
    the launched ones it no longer holds, whose runs are read through their
    rows.
    """
    if old is None:
        return new
    # A STAGE PREPARED AGAIN -- never launched -- is its new row; the old
    # one is replaced, never doubled.
    # ONE KEY for a stage's name, as every reader of the plan matches it
    # (`identity.stage_key`).
    from ..identity import stage_key
    fresh = {stage_key(j.name) for j in new.jobs}
    standing = (None if ladder is None
                else {stage_key(n) for n in ladder})
    kept = [j for j in old.jobs if stage_key(j.name) not in fresh
            and (standing is None or stage_key(j.name) in standing)]
    # THE SHARED PACKAGE IS THE ENGINE'S ANSWER for the folder as this prep
    # leaves it, the same for every job (`_shared_for`) -- never carried from
    # an older plan, which may list a file molbuilder no longer writes.
    merged = dataclasses.replace(new, jobs=kept + list(new.jobs))
    # The plan's order is the LADDER's, not the order stages were prepared
    # in: `tight` prepared before `medium` is still listed after it.  The seq token is zero-padded (§ 6.3) so it sorts as it reads.
    from .naming import stage_refs
    refs = stage_refs(merged)
    return dataclasses.replace(
        merged, jobs=sorted(merged.jobs,
                            key=lambda j: (refs[j.name].token or "", j.name)))


def _flat_resources(resources) -> str:
    """One allocation as one line — the fields it actually carries.

    ``None`` means *not asked for* and is left out rather than printed as a
    null: a line of ``domain=None, time=None, gres=None`` is noise in a file
    whose whole value is that a person can read it.
    """
    def _val(v):
        # A SEQUENCE FIELD PRINTS AS ITS VALUE, not as its repr.  `()` is a
        # real answer meaning *nowhere*, so it gets a word.
        if isinstance(v, tuple):
            return ",".join(v) if v else "(none)"
        return v

    return ", ".join(f"{k}={_val(v)}" for k, v in
                     dataclasses.asdict(resources).items() if v is not None) \
        or "(nothing asked for)"


def _environment_rows(environment) -> "List[tuple]":
    """Floor 1's answer as ``[(group, one line)]``.

    ONE ROW PER GROUP, not one line for the whole record: the probe's answer
    nests (topology, site, source), and flattening it produced a single line
    of nested dict reprs -- unreadable, in the one file whose whole claim is
    that a person can read it.
    """
    if environment is None:
        return [("environment", "none — this machine was not probed")]
    raw = (dataclasses.asdict(environment)
           if dataclasses.is_dataclass(environment)
           else environment if isinstance(environment, dict)
           else {"environment": environment})
    scalars, rows = [], []
    for k, v in raw.items():
        if v is None or v == {} or v == []:
            continue
        if isinstance(v, dict):
            # A group whose every member is None was PROBED and came back
            # empty (a workstation has no partition, no QOS, no account).
            # A bare heading with nothing after it says less than no row.
            line = ", ".join(f"{a}={b}" for a, b in v.items() if b is not None)
            rows.append((k, line or "nothing — probed, and this machine "
                                    "has none"))
        elif isinstance(v, (list, tuple)):
            rows.append((k, ", ".join(str(x) for x in v)))
        else:
            scalars.append(f"{k}={v}")
    return ([("environment", ", ".join(scalars) or "(no scalar facts)")]
            + sorted(rows))


def _seed_trajectory_log(struct, cfg, base: Path, *, engine: str,
                         kind: str, names, frame=None,
                         relaxes: bool = True, plan=None) -> None:
    """Write the one-block preview the Watch tab discovers before a run starts.

    The deck NAMES its trajectory log; something has to CREATE it, or the tab
    has nothing to find until the engine writes its first step.

    ``engine`` comes through the caller from the :class:`EngineSeam`.
    ``names`` are the deck's
    stage's (`runfiles.RunNames`): the seed is the FIRST run's log, named as
    that run names it -- in a shared folder with its number, ``-run0``.
    Its header states its atoms' indices (`model/overview.md` § 2.2b): the
    deck's structure listed by engine index, both columns from its order
    record -- the engine column alone for a structure of its own (an
    electrode cell, `transport.sort.deck_own_structure`).
    """
    if not getattr(cfg, "write_molwatch_log", False):
        return
    from ..runfiles import FIRST_ATTEMPT
    from ..trajectory_log import write_initial_preview
    # The stage is the names': nothing here re-derives one.
    # The stage's own convergence targets travel with its log, so the Watch
    # tab's threshold line is THIS stage's and not the ladder's first.  They
    # come from the RESOLVED config, which is the whole point of resolving
    # before rendering: `coarse` and `tight` disagree about both of these.
    # THE KEY NAMES ARE THE LOG'S OWN (`trajectory_log.keys`, one home
    # for every writer and reader), and the block is nested by this rung's
    # token as every writer writes it (`engines/pyscf.md` § 4; one shape,
    # 2026-10-18).
    # ONLY A DECK THAT RELAXES HAS TARGETS -- the viewer draws "the targets
    # the run was chasing" (`web/trajectory.md` § 3), and a force-constant run
    # chases none: a threshold line over its steps called 115 nudges a
    # relaxation that never settles.  ``relaxes`` is the rung's render kind.
    from ..trajectory_log import keys as _K
    leaf = {}
    for key, attr in (() if not relaxes else
                      ((_K.MAX_FORCE_TOL, "relax_force_tol"),
                       (_K.MAX_GEOM_ITER, "relax_steps"))):
        value = getattr(cfg, attr, None)
        if value is not None:
            leaf[key] = value
    targets = {names.stage: leaf} if leaf else {}
    write_initial_preview(
        struct,
        base / names.name(".molwatch.log", FIRST_ATTEMPT),
        job=names.label, engine=engine,
        convergence_targets=(targets or None),
        frame=frame,
        frozen_atoms=list(getattr(struct, "frozen_atoms", []) or []),
        plan=plan, atom_index_map=_deck_atoms(struct, engine, kind,
                                              names.stage).to_dict())


def _deck_atoms(struct, engine: str, kind: str, stage):
    """THE ATOM INDEX MAP of the files a deck of ``struct`` makes its run
    write (`model/overview.md` § 2.2b, F2): listed by engine index, both
    columns from the structure's order record, the identity where it
    carries none; the engine column alone for a structure of its own."""
    from ..atom_permutation import AtomIndexMap
    from ..transport.sort import deck_own_structure
    own = deck_own_structure(engine, kind, stage)
    if own:
        return AtomIndexMap.of_own_structure(struct.n_atoms, own)
    return AtomIndexMap.of_engine_record(struct.atom_order, struct.n_atoms)


def _job_for(element, script: str, task, stage_name: Optional[str],
             seam: EngineSeam, base_dir=None,
             finish: Optional[str] = None,
             placement: Optional[dict] = None) -> Job:
    """One element of the parameter set as one :class:`Job`.

    ``resources`` is **copied from the element**, never re-derived: the element
    resolved it once, from the allocation, and a second derivation here is the
    habit `generator.md` § 5 exists to end.

    The **name** answers *which job is this*: a trial is named by its sweep
    coordinate, a rung of a ladder by its stage.
    """
    from ..resolve import point_token

    name = point_token(element.point) if element.point else stage_name

    # THE CALCULATION'S KIND answers what it carries and whether a re-run of
    # it resumes (`job-contracts.md` § 4.2a); a run in phases renders each
    # phase's run script with that phase's own section (`Rung.wrapper_at`).
    kind = str(task.calculation)
    with _calling("warm_for", engine=task.engine, where=name):
        warm = seam.warm_for(element.label, element.values, kind, base_dir)
    resumes = warm_list(str(task.engine), kind, base_dir).resumes
    with _calling("traits_for", engine=task.engine, where=name):
        traits = seam.traits_for(element.values)
    # ``finish`` is the deck's own statement (`DeckSpec.finish`): the bundle
    # its run is finished by, when the engine alone leaves no result
    # (`engines/vibration.md` § 5.5).
    return Job(name=name, script=script, resources=element.resources,
               warm=warm, traits=traits, point=dict(element.point),
               finish=finish, resumes=resumes, placement=placement)


def _with_notify(flags, declared) -> "Resources":
    """The job request's `notify` block, onto the allocation that reaches
    the wrapper.

    Not an allocation and not a scheduler flag -- it rides ``Resources``
    because that is the road from a job to its wrapper, the one
    ``continue_retries`` already rides (`jobset/model.Resources`).  There is
    no ``--notify-*`` flag today, so in practice the description is the only
    voice.

    **FIELD BY FIELD ANYWAY**, for the reason the function above it states:
    a whole-object copy makes a description that sets only the period erase
    a caller's SCF trigger, which is the same silent loss as `--np 8`
    erasing a memory ask.  Nothing sets these from outside yet; the point
    is that the shape cannot start losing values the day something does.

    No block leaves the fields ``None``, which the wrapper renders as no
    flags at all -- and no flag is nothing sent, the start and the end
    included: reports stay off for everyone who has not asked for them.
    """
    out = flags or Resources()
    # NO BLOCK IS ``None`` (`job_request.Notify`): a block whose values are all off
    # still reports the start and the end.
    if declared is None:
        return out
    import dataclasses as _dc
    patch = {}
    if declared.on_scf_converged and out.notify_on_scf is None:
        patch["notify_on_scf"] = True
    # A period, or ``None`` for never (`job-request.md` § 3.5).
    if declared.every_hours is not None and out.notify_every_hours is None:
        patch["notify_every_hours"] = declared.every_hours
    # `is not None`, NOT truthiness: an empty tuple says "send this
    # calculation nowhere", and a truthiness guard here would drop it and
    # hand the job every channel on the machine instead (`run-reports.md`
    # 3.0).
    if out.notify_channels is None:
        # EVERY CHANNEL (`["*"]`, ``None`` here) is every channel of the
        # machine that runs the job, which only that machine knows -- so it
        # travels as the marker, resolved there (`run-reports.md` § 3.0).
        # No block at all is not here: it returned above, and the wrapper
        # renders no flag, which is nothing sent.
        from ..config_dir import ALL_CHANNELS
        patch["notify_channels"] = (declared.channels
                                    if declared.channels is not None
                                    else (ALL_CHANNELS,))
    # SAME RULE, SAME REASON: `()` here means "the summary line and no field
    # grid", which is a real answer and not an absent one (`stages.md` § 6.9).
    if declared.report is not None and out.notify_report is None:
        patch["notify_report"] = declared.report
    return _dc.replace(out, **patch) if patch else out


def _shared_for(base: Path, seam: "EngineSeam" = None, *, engine: str = "",
                plan=None) -> List[str]:
    """The static package every job links (`project-layout.md` § 2.1).

    **Asked of the engine, not guessed from the folder.**  An engine that puts
    no data files in has an empty package, and that is an answer rather than an
    accident of which suffix the glob happened to name.  It names the folder
    as ``plan`` will leave it -- the files the data-files step put in place.
    """
    if seam is None or seam.shared_package is None:
        return []
    with _calling("shared_package", engine=engine):
        return list(seam.shared_package(base, plan))


# --------------------------------------------------------------------- #
#  The prep verb's ONE entry (`job-system.md` § 5.3; plan W38 F7)       #
# --------------------------------------------------------------------- #
#
# `prep` has two doors -- `molbuilder jobset prep` and the Task setup tab's
# Prep buttons.  This is the act, once; it prints nothing and asks nothing, and returns what it
# found and decided as data for each door to show in its own way.


@dataclass
class PrepAnswer:
    """What one prep found and decided -- the whole of what either door shows
    (`job-system.md` § 5.3's table)."""
    kind: str
    stage: Optional[str]
    #: WITH NO GROUP NAMED, what the calculation's own record took: a
    #: junction whose sidecar states no order record, its order found and
    #: written, in one sentence (`engines/transport.md` § 1.1, *No group left
    #: to prepare*); ``None`` for every answer about a stage.
    recorded: Optional[str] = None
    #: A STAGE OF A GROUP ALREADY AS THIS PREP PLANS IT -- prepared, never
    #: launched, its decks, scripts and row what prep writes now: the
    #: sentence both doors say, nothing written for it (`job-system.md`,
    #: *The task*, 2); ``None`` for a stage this prep prepares.
    unchanged: Optional[str] = None
    #: A STAGE PREPARED, NEVER LAUNCHED, AND PREPARED AGAIN: the files and
    #: the row of its that differ from what this prep writes now, named
    #: from the calculation (:func:`prepared_differs`) -- what made it
    #: prepared again; empty for a stage prepared here for the first time.
    differs: List[str] = dataclasses.field(default_factory=list)
    #: The description's preflight notes (an error refuses instead).
    findings: list = dataclasses.field(default_factory=list)
    #: What the inputs said: a bench's grid -- enumerated, left out,
    #: kept (`prep_inputs.bench_inputs`); a run's GPUs over its ranks
    #: (`ask.gpu_share_notes`); what a rendering says and does not refuse
    #: -- a GPU build the target's record lacks (`gpu.md` G6).
    notes: List[str] = dataclasses.field(default_factory=list)
    #: The folder's state saved before the five steps wrote, or the one it
    #: stood at -- `checkpoint.Kept.said` (`checkpointing.md` § 9).
    saved: Optional[str] = None
    dirs: List[Path] = dataclasses.field(default_factory=list)
    #: Which config files answered, as the prep read them at its checkpoint
    #: 4 (`runtime_config.config_provenance`) -- on a preview too.
    provenance: Optional[dict] = None
    #: The machine it is prepared for, by name: the one named at its first
    #: prep, or the one its copy of the record names (`configuration.md`
    #: M-3).
    machine: Optional[str] = None
    #: A flat run: its wrappers are rendered and there is no attempt to open.
    flat: bool = False
    #: A launched stage prepared again for its job request: no attempt is
    #: opened, its next launch opens its next run
    #: (`execution/job-request.md` § 5).
    launched: bool = False
    #: The attempt opened or reused (`materialize.Attempt`), with the
    #: stage's own files.
    attempt: Optional[object] = None
    #: A rung that runs per point: ``(point folder, Point)`` each
    #: (`engines/transport.md` § 2a.11).
    points: List[tuple] = dataclasses.field(default_factory=list)
    #: WHAT ITS FIRST LAUNCH WILL TAKE, and from which stage --
    #: ``[{"from", "files", "line"}]`` (`ready.at_launch`): nothing
    #: is copied at prep (`job-system.md` § 5.0).
    takes: List[dict] = dataclasses.field(default_factory=list)
    #: What each deck's checks said (`script_emit.prepare_deck`), one of each
    #: -- the terminal read them on stderr as the decks rendered.
    deck_findings: list = dataclasses.field(default_factory=list)
    resources: Optional[dict] = None
    #: The stage's deck, by file name.
    deck: Optional[str] = None
    pipeline_log: Optional[Path] = None
    #: A LINKED stage -- its kind gives its rungs roles (`paths.KIND_ROLES`)
    #: -- which builds on what its kind says, never on the stage before it:
    #: when it takes nothing from another run, both doors say so without
    #: "the first stage, or one that starts clean".
    linked: bool = False
    #: WHERE IT WAS PLACED -- a run's placement, as its job records it
    #: (`job-system.md` § 6.0), or ``None`` with no queue.
    placement: Optional[dict] = None
    #: ITS FIT there, the job it sends asked of the queue's machines --
    #: a run's own, a run of points' walk (their walls summed) -- said and
    #: recorded nowhere (`scheduler.md` R6); ``""`` for a benchmark, whose
    #: shelves say theirs, and for a member of a group of several, whose
    #: group's one job says its own (``group``).
    fit: str = ""
    #: THE MEMORY THE STAGE NEEDS at the ranks it is prepared with -- at most
    #: so much, and the run that held it, or that nothing has measured it
    #: (`jobset.memory.Need`; `execution/job-request.md` § 3.1a): a hint
    #: said after its placement and fit, refusing nothing.  ``None`` for a
    #: benchmark.
    need: Optional[object] = None
    #: A13 -- what the job will be launched with, as its header and its run
    #: script carry it (:func:`_launch_as_written`).
    launch: Optional[dict] = None
    #: A MEMBER OF A GROUP OF SEVERAL: the group's one job, the same on
    #: every member -- ``number``, ``members`` in order, ``header`` (its
    #: file in ``launch/``, ``None`` with no scheduler), ``launch`` (A13:
    #: that header's lines, which a first launch sends, not a member's
    #: own) and ``writes`` (the files the group adds) -- as
    #: :func:`_prep_groups` planned it.  ``None`` for a group of one.
    group: Optional[dict] = None
    #: THE LAST ANSWER OF A PREP WITH ``--one-job``: the sequence of groups'
    #: one job (`job-system.md` § 6.0, *A sequence of groups*) --
    #: ``numbers``, ``header`` (its file in ``launch/``, ``None`` with no
    #: scheduler), ``launch`` (its header's lines and fit, as a group's),
    #: ``less`` (each group asking less than the allocation, in words).
    #: ``None`` otherwise.
    sequence: Optional[dict] = None
    #: A CALCULATION THAT REUSES A FINISHED ONE: the manifest -- each group
    #: of the prep copied in, its runs, or prepared here and why
    #: (`jobset.reuse.Reuse`; `engines/transport.md` § 2a.16) -- the same
    #: on every answer of the prep.  ``None`` otherwise.
    copied: Optional[object] = None
    #: A PREVIEW: the plan, stopped before the save, named by its identity
    #: for the Prep that follows (`job-system.md` § 5.0), with what it would
    #: write.
    preview: bool = False
    plan_id: Optional[str] = None
    writes: List[str] = dataclasses.field(default_factory=list)
    #: THE JOB AS PLANNED -- a run's row of `job-set.json` -- what a group's
    #: prep checks its members' shared allocation by, before anything is
    #: written (`group.envelope`).
    job: Optional[object] = None
    #: A PREVIEW'S PLAN, whole (`jobset.planned.Plan`).  Not on the wire.
    plan: Optional[object] = None
    #: WHERE EACH VALUE CAME FROM, the one fold's answer
    #: (`prep_inputs.fold_allocation`) -- what a prep that writes no header
    #: says in place of a placement (`placement.stated_line`).
    sources: Optional[dict] = None
    #: A BENCHMARK'S SHELVES, each as its one header carries it -- what its
    #: launch sends to the queue (`shelves.shelves_of`): ``[{"name",
    #: "header", "said", "fit"}]``, the header's directive lines, them in
    #: one line (`Directives.said`) and the shelf's fit on the queue.
    shelves: List[dict] = dataclasses.field(default_factory=list)

    @property
    def copied_whole(self) -> bool:
        """Every group the prep was asked for copied in, its runs launched
        already -- the next step the ladder's own (`job-system.md`, *The
        next step*; `engines/transport.md` § 2a.16): what both doors that
        say what comes next ask."""
        return (self.copied is not None and not self.preview
                and all(d.copied for d in self.copied.groups))

    def as_dict(self, base) -> dict:
        """The answer as JSON, paths relative to the calculation folder --
        what the Task setup tab's prep route returns (`web/web-api.md`)."""
        from .placement import placement_line, stated_line
        base = Path(base)

        def rel(p):
            try:
                return str(Path(p).resolve().relative_to(base.resolve()))
            except ValueError:
                return str(p)

        a = self.attempt
        return {
            "kind": self.kind, "stage": self.stage,
            "unchanged": self.unchanged,
            "differs": list(self.differs),
            # THE ONE WIRE FORM of a finding (`Issue.to_json`).
            "findings": [i.to_json() for i in self.findings],
            "deck_findings": [i.to_json() for i in self.deck_findings],
            "notes": list(self.notes),
            "saved": self.saved,
            "dirs": [rel(d) for d in self.dirs],
            "provenance": self.provenance,
            "machine": self.machine,
            "flat": self.flat,
            "launched": self.launched,
            "attempt": ({"dir": rel(a.dir), "fresh": a.fresh,
                         "brought": list(a.brought)}
                        if a is not None else None),
            "points": [{"attempt": rel(att), **pt.to_wire()}
                       for att, pt in self.points],
            "takes": [dict(t) for t in self.takes],
            "resources": self.resources,
            "deck": self.deck,
            "pipeline_log": rel(self.pipeline_log) if self.pipeline_log else None,
            "linked": self.linked,
            # THE PLACEMENT, and the line both doors say of it.
            "placement": (dict(self.placement,
                               line=placement_line(self.placement),
                               fit=self.fit)
                          if self.placement else None),
            # NO HEADER: where each value it states came from, the line the
            # terminal prints in place of a placement.
            "stated": ("" if self.placement else
                       stated_line(self.sources)),
            "need": (self.need.to_wire() if self.need is not None
                     else None),
            "launch": self.launch,
            "group": self.group,
            "copied": (self.copied.to_wire(base) if self.copied is not None
                       else None),
            "preview": self.preview,
            "plan_id": self.plan_id,
            "writes": list(self.writes),
            "shelves": [dict(sh) for sh in self.shelves],
            "sources": dict(self.sources or {}),
        }


def opened_no_run(answers) -> bool:
    """Every group a prep was asked for copied in from a finished
    calculation, already as it plans it, or launched and prepared again for
    its job request -- it opened no run of its own, so what comes next is
    the ladder's own step (`job-system.md`, *The next step*): what both
    doors that say what comes next ask."""
    return bool(answers) and all(a.unchanged is not None or a.copied_whole
                                 or a.launched for a in answers)


def _record_preflight(base, findings, **said) -> None:
    """The description's preflight notes, in the ledger once per prep --
    with its record, or with its refusal (`job-system.md` § 5.0, checkpoint
    7); ``said`` names what the prep was of (a stage, or the groups)."""
    if findings:
        from .ledger import record as ledger
        ledger(base, "prep", "preflight-report", **said,
               notes=[i.message for i in findings])


def _record_prepared(base, answer: "PrepAnswer") -> "PrepAnswer":
    """What a prep writes down once its plan is written (`job-system.md`
    § 5.0, checkpoint 7) -- *prepared* with which config files answered,
    the last line of its record.  What a stage takes from another run is
    recorded by the launch that takes it (`continues`, `gathers`).  The one
    writer of it: a benchmark's prep and every member of a task's."""
    from .ledger import prepared as ledger_prepared
    answer.provenance = ledger_prepared(base, kind=answer.kind,
                                        stage=answer.stage, dirs=answer.dirs,
                                        provenance=answer.provenance)
    answer.preview = False
    return answer


def prepared_already(calc, kind: str, stage: str) -> Optional[str]:
    """That ``stage`` is prepared -- a sentence saying where -- or ``None``
    (`job-system.md` § 5.0, checkpoint 2a).  A benchmark prepared is not
    prepared again (user, 2026-10-02: *"refuse it, redo via rollback"*), so
    its sentence is that refusal; a run prepared and never launched is
    prepared again (`ready.launched_already` says when it may not be).

    PREPARED IS WHAT `status` READS: the stage's job in the calculation's
    plan, ``job-set.json``, matched as `status` matches it
    (`identity.stage_key`) -- for a bench, the sweep its bench folder holds.
    A refused prep writes nothing but its ledger line (:func:`prep_task`),
    so a stage is counted prepared only by a prep that
    finished."""
    from ..paths import Shape
    from .commands import rollback
    from ..paths import bench_container
    from .naming import job_dir_names
    base, task = Path(calc.base), calc.task
    if kind == "bench":
        home = base / bench_container(Shape.named(task.shape),
                                      stage_home(base, task, stage).token)
        if JobSet.of(home) is None:
            return None
        what, where = (f"the benchmark of stage {stage!r}",
                       f"{home.relative_to(base)}/")
    else:
        job = calc.job(stage)
        if job is None:
            return None
        home = job_dir_names(calc.jobset, Shape.named(task.shape)).get(
            job.name, "")
        where = (f"{home}/" if home not in ("", ".")
                 else "in this calculation's folder")
        return f"stage {stage!r} is prepared ({where})"
    return (f"{what} is already prepared ({where}) -- a prepared benchmark "
            f"is not prepared again (job-system.md § 5.0).  To redo it, "
            + rollback("its prep", base=base))


def prepared_stages(calc) -> List[str]:
    """The stages of ``task`` prepared in ``base`` -- as a run or as a
    benchmark -- in the description's order: whether the calculation has
    PRODUCED, which fixes its shape (`web/task-setup.md` § 4).  Each is
    :func:`prepared_already`'s answer, so a refused prep -- which writes
    nothing -- counts for none."""
    return [st.name for st in calc.task.stages
            if any(prepared_already(calc, kind, st.name)
                   for kind in ("task", "bench"))]


def shape_fixed(calc) -> Optional[Dict[str, str]]:
    """The shape ``task`` holds FIXED, and why -- ``{"shape", "why"}`` -- or
    ``None`` while it may still change (`web/task-setup.md` § 4).

    Two reasons, one answer: the calculation's KIND fixes it
    (`task.fixed_by_kind` -- a transport calculation is hierarchical), or the
    calculation has PRODUCED (:func:`prepared_stages`) -- a prepared stage
    lies where the shape put it, and another shape would orphan every deck,
    output and warm file there; the way back is the state saved before its
    first prep (`job-system.md` § 5.0).  Asked by the page that draws the
    shape card and by the save that refuses another shape, so the two say
    the same thing."""
    from ..task import fixed_by_kind
    from ..template import a_kind
    task = calc.task
    kind = fixed_by_kind(task.calculation)
    if kind is not None:
        said = a_kind(task.calculation)
        return {"shape": kind["shape"],
                "why": f"{said[:1].upper()}{said[1:]} is {kind['shape']} -- "
                       f"the kind fixes it, so no other is offered."}
    done = prepared_stages(calc) if task.shape else []
    if not done:
        return None
    from .commands import rollback
    return {"shape": task.shape,
            "why": f"This calculation is {task.shape}, and its shape is "
                   f"fixed once it has produced: {', '.join(done)} "
                   f"{'is' if len(done) == 1 else 'are'} prepared, laid out "
                   f"{task.shape}, and another shape would orphan every deck, "
                   f"output and warm file there (task-setup.md § 4).  To "
                   f"change it: " + rollback("its first prep",
                                             base=calc.base)}


@dataclass
class Described:
    """The calculation as the entry read it, ONCE (W55 B1,
    `calculation.open`, A16): its description, template and plan, then --
    checkpoints 3 and 4 -- what its preflight noted, the machine's record
    it is prepared for and which config files answered.  Every member a
    prep plans reads these, never the files again."""
    calc: "Calculation"
    findings: list = dataclasses.field(default_factory=list)
    environment: "Optional[Environment]" = None
    provenance: Optional[dict] = None

    @property
    def task(self) -> "Task":
        return self.calc.task

    @property
    def template(self) -> Path:
        return self.calc.template.path

    @property
    def template_text(self) -> str:
        return self.calc.template_text


def checked(base, described: Described, *, target) -> Described:
    """CHECKPOINTS 3 AND 4 (`job-system.md` § 5.0), on ``described``:

    3 · § 6.6's PREFLIGHT, at its live moment (R5, 2026-08-12): prep on a
    machine whose molbuilder differs from the description's author, the
    template riding along -- which adds § 6.4/§ 6.6a's sequence warnings.
    An error refuses; the notes are the answer's ``findings``.

    4 · THE MACHINE, READ ONCE (`script-preparation.md` § 3.0, step 1): the
    calculation's copy of its record -- at its first prep, the named
    machine's -- which says how a shell enters an environment there, or the
    prep is refused before anything is written (W52); and WHICH FILES
    ANSWERED, as they were read -- what STAGE-PLAN.md, the pipeline log and
    the ledger's *prepared* line each say (`configuration.md` § 2.2)."""
    from ..runtime_config import config_provenance
    from ..validation.task import preflight
    base = Path(base).resolve()
    issues = preflight(described.task, template_text=described.template_text,
                       request=described.calc.request)
    described.findings[:] = ([i for i in issues if i.severity != "error"]
                             + _order_said(described.task, base))
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise PrepError(
            "the description or its job request fails its own preflight "
            "(engines/stages.md § 6.6; execution/job-request.md § 7):\n  - "
            + "\n  - ".join(i.message for i in errors))
    described.environment = machine_record(base, target)
    require_activation(target, described.environment, base=base)
    described.provenance = config_provenance(
        project_dir=base, target=target, record=described.environment)
    return described


#: What :func:`plan_stage` decides on the calculation as it is when its
#: caller hands it nothing (``contours``).
DECIDE = object()


def plan_stage(base, described: Described, kind: str, stage: str, *,
               target: Optional[str] = None, allocation=None,
               env: Optional[str] = None,
               emit_sbatch: bool = True, plan, notes: Optional[list] = None,
               on_inputs=None, copied: bool = False,
               contours=DECIDE) -> PrepAnswer:
    """**THE MEMBER PLANNER** -- one stage, planned into ``plan``, nothing
    written, saved or recorded (`job-system.md` § 5.0, checkpoints 4–4b):
    its inputs, assembled (`prep_inputs`, A12) -- a benchmark's grid
    enumerated into ``notes``; the stage resolved once (step 2); a run's GPU
    request agreeing with itself (`execution/gpu.md` G5); every launch value
    stated, and a run's whole request placed on the queue it names, where
    a header is written (`placement.placed`: the placement recorded on its
    job, its fit said); then the steps planned
    (:func:`prep_calculation`) and the answer read off the plan.

    Asked by every door that plans a stage: `prep task` for each member of
    the groups it picks, into its one plan; `prep bench`.  The doors save,
    write and record; this never does.
    ``described`` is the calculation as the door opened and checked it
    (`calculation.open`, :func:`checked`); ``on_inputs`` is called
    once the inputs and their checks stand, before anything is rendered --
    so a terminal says a benchmark's grid ahead of what its decks say.
    ``copied`` plans a member of a group copied in (:func:`prep_calculation`):
    its answer is a launched stage's.  ``contours`` -- what each transmission
    point computes where it is not its window whole -- is the prep's
    decision; :data:`DECIDE`, decided on the calculation as it is
    (:func:`_contours_now`)."""
    from .prep_inputs import bench_inputs, prep_run_inputs
    from .placement import launch_refusal, one_process, placed
    from ..runtime_config import routing_of
    base = Path(base).resolve()
    task, environment = described.task, described.environment
    template_text = described.template_text
    notes = [] if notes is None else notes
    deck_findings: list = []

    #     EVERY LAUNCH VALUE IS STATED, OR THE PREP IS REFUSED -- before
    #     anything is written (`architecture.md` § 5.2; user, 2026-10-02:
    #     "explicit job config is the only way allowed").  A run states its
    #     processes; a run or a benchmark that this prep writes a `.sbatch`
    #     for states its queue, wall and memory.  The target's record SHOWS
    #     how an ask fits -- its queues are listed with the fit of ``asked``,
    #     the request the queue list and the placement both read, so one can
    #     be named -- and supplies no value of it.
    header = bool(emit_sbatch and environment.kind.queue)

    def stated_or_refused(allocation, asked, *, shape: bool) -> None:
        why = launch_refusal(
            allocation, engine=task.engine, shape=shape, stage=stage,
            header=header, queues=routing_of(environment), asked=asked,
            base=base, target=target)
        if why:
            raise PrepError(why)

    def bench_asks(allocation):
        # A BENCHMARK'S REQUEST OF ITS ONE QUEUE: its wall and memory --
        # each trial's shape is checked against that queue cell by cell
        # where its grid is enumerated (`prep_inputs.bench_inputs`).
        return dataclasses.replace(allocation, mpi_np=None,
                                   cpus_per_task=None, gres=None,
                                   use_gpu=False)

    #     THE INPUTS -- one assembly per kind (`prep_inputs`, A12).  A bench
    #     measures ONE stage's configuration, and there is always a stage to
    #     name (§ 6.5); its grid is enumerated from the record read, on the
    #     one queue it states -- so the queue, wall and memory are asked of
    #     it first.
    sweep = pins = translation = None
    chosen: dict = {}
    if kind == "bench":
        from .prep_inputs import card_settings, fold_allocation
        ask = fold_allocation(allocation, described.calc.request.allocation)[0]
        stated_or_refused(ask, bench_asks(ask), shape=False)
        sweep, pins, translation = bench_inputs(
            described.calc, target, notes=notes, environment=environment,
            flags=allocation, on_a_queue=header)
        # A TRIAL'S RUN SETTINGS: the calculation's card, never a rung's
        # (`job-request.md` § 3.4) -- under its own pins.
        run_settings = card_settings(described.calc.request, None)
    else:
        allocation, run_settings, chosen = prep_run_inputs(
            described.calc.request, stage, allocation)

    #     STEP 2, ONCE (`script-preparation.md` § 3.0): the allocation folded
    #     and the stage resolved -- what the checks below ask of the job,
    #     and what every step after reads.
    resolved = _resolve_stage(
        stage, task=task, request=described.calc.request,
        template=described.template,
        template_text=template_text, environment=environment,
        allocation=allocation, chosen=chosen, sweep=sweep, pins=pins,
        run_settings=run_settings,
        translation=translation, home=stage_home(base, task, stage),
        target=target, provenance=described.provenance)
    if kind == "task":
        #  ...AND ITS GPU REQUEST AGREES WITH ITSELF (`execution/gpu.md`
        #  G5): a run on the GPU states how many, a run on the CPU none --
        #  asked of the job this prep will write, before anything is.
        from .model import GpuRequestError, gpu_request
        try:
            gpu_request(resolved.pset.elements[0].resources)
        except GpuRequestError as exc:
            raise PrepError(f"stage {stage!r}: {exc}") from None
    #     WHAT THE WORK ASKS OF ITS QUEUE: a run's whole request -- cores,
    #     GPUs, memory and wall -- with the whole assembly in hand.
    asked = (resolved.pset.elements[0].resources if kind == "task"
             else bench_asks(resolved.allocation))
    if kind == "task":
        stated_or_refused(resolved.allocation, asked, shape=True)
    #     ...AND THE WORK IS PLACED ON THE QUEUE IT NAMES, recorded on its
    #     jobs with where each value came from -- the one decision launch
    #     sends (`job-system.md` § 5.0, checkpoint 4; § 6.0; `scheduler.md`
    #     R6, R9); every trial of a benchmark records the queue.  A run's
    #     fit there is said; a benchmark's shelves say their own.
    if header:
        placed_on, fit, why = placed(
            asked, environment, one_process=one_process(task.engine),
            what=f"stage {stage!r}", sources=resolved.sources)
        if why:
            raise PrepError(why)
        resolved = dataclasses.replace(
            resolved, placement=placed_on,
            fit=fit if kind == "task" else "")
    #     ...AND WHAT A RUN'S GPUs MEAN, said here where the count can still
    #     change -- its ranks over its devices (`ask.gpu_share_notes`); a
    #     benchmark's grid says its own (`prep_inputs.bench_inputs`).
    if kind == "task":
        from .ask import gpu_share_notes
        from .model import gpu_request
        _r = resolved.pset.elements[0].resources
        _g = gpu_request(_r)
        if _g.uses:
            notes.extend(line.strip() for line in gpu_share_notes(
                _g.count, _r.mpi_np or 1,
                one_process=one_process(task.engine)))

    # 4a · EVERY STAGE'S HAND-OVER IS FILES, taken at its first launch
    #      (`job-system.md` § 5.0): prep reads no other stage's run.
    # THE CALCULATION AS THIS PREP FINDS IT, its plan read through the
    # overlay once (`Calculation.as_planned`, A16): the rung builders and
    # the plan's merge read this.
    pcalc = described.calc.as_planned(plan)
    if on_inputs is not None:
        on_inputs()
    if contours is DECIDE:
        contours = _contours_now(pcalc, stage) if kind == "task" else None

    # 4b · THE STEPS, PLANNED -- every deck, wrapper and data file, the
    #      plan's row, the run and what it receives, decided with nothing
    #      written (`job-system.md` § 5.0, rule 3; `jobset.planned`).
    had = set(plan.writes())
    opened: list = []
    walks: dict = {}
    dirs = prep_calculation(pcalc, stage, env=env, emit_sbatch=emit_sbatch,
                            opened=opened, findings=deck_findings,
                            plan=plan,
                            resolved=resolved, walks=walks,
                            notes=notes, copied=copied, contours=contours)
    flat = not resolved.shape.keeps_attempts_as_directories

    # THE ANSWER, from the plan -- what a preview shows and a prep records
    # once it is written.
    seen: set = set()
    from ..scheduler.record import LOCAL_TARGET
    out = PrepAnswer(
        kind, stage, findings=described.findings, notes=notes,
        # ONE ENTRY PER FOLDER: on the flat layout every stage's folder is
        # the calculation's one (W52).
        dirs=list(dict.fromkeys(dirs)),
        # ONE OF EACH: a sweep's trials repeat one finding per deck, and the
        # terminal said each once (`prep_calculation`).
        deck_findings=[i for i in deck_findings
                       if not (repr(i.to_json()) in seen
                               or seen.add(repr(i.to_json())))],
        # WHICH FILES ANSWERED, as read at checkpoint 4 -- a preview's too.
        provenance=described.provenance,
        # THE MACHINE, by name: the one named, else the one the
        # calculation's copy of its record names, else this one.
        machine=(target or getattr(environment, "machine", None)
                 or LOCAL_TARGET))
    # THE PIPELINE LOG, which every prep writes (`script-preparation.md`
    # § 4.5): where this one is, by the rule its writer asks.
    out.pipeline_log = resolved.log_path(base)
    out.placement = resolved.placement
    out.fit = resolved.fit
    out.sources = resolved.sources
    out.plan = plan
    out.writes = [str(Path(w).relative_to(base))
                  for w in plan.writes() if w not in had]
    if kind == "bench":
        # EACH SHELF AS ITS HEADER CARRIES IT, read back from the text the
        # plan holds (`Directives.lines_of`), in the order they are sent.
        from ..runfiles import LAUNCH_DIR, role_of
        from ..scheduler.emit import Directives
        shelf_dir = resolved.record_dir(base) / LAUNCH_DIR
        out.shelves = []
        for w in plan.writes():
            if w.parent == shelf_dir and role_of(w.name) == ".sbatch":
                lines = Directives.lines_of(plan.read_text(w),
                                            environment.kind)
                out.shelves.append({"name": w.stem, "header": lines,
                                    "said": Directives.said(lines),
                                    "fit": walks.get(w, "")})
    if kind != "task":
        return out
    # THE PLAN AS THIS PREP LEFT IT -- its own row now in it.
    pcalc = described.calc.as_planned(plan)
    js = pcalc.jobset
    # THE STAGE'S FIRST RUN -- opened with its own files (`_open_attempts`);
    # what it takes from another run is its first launch's.  Flat keeps no
    # attempt directories: the run is the calculation's folder.
    from ..paths import KIND_ROLES
    from .materialize import SweepRun
    out.linked = task.calculation in KIND_ROLES
    if flat:
        out.flat = True
        run_dir = base
    elif opened and isinstance(opened[0], SweepRun):
        # A SWEPT RUNG'S RUN: each point (`engines/transport.md` § 2a.11).
        # EVERY POINT LAUNCHES THE ONE JOB, with one shape: the first point
        # says what it is (A13).
        out.points = list(opened[0].points)
        run_dir = opened[0].points[0][0] if opened[0].points else None
    elif opened:
        out.attempt = opened[0]
        run_dir = opened[0].dir
    else:
        # A LAUNCHED STAGE, prepared again for its job request, opens no
        # run: its files are the stage's own, which its next launch's run
        # takes (`execution/job-request.md` § 5).
        out.launched = True
        run_dir = stage_home(base, task, stage).dir
    # WHAT IT WILL LAUNCH WITH -- prep writes a deck and its launch
    # together, once (`architecture.md` A3), and launch sends what it wrote.
    # A13: the end point, as the header and the run script carry it.
    job = js.job(stage)
    out.job = job
    # WHAT ITS FIRST LAUNCH WILL TAKE, said before any run is read
    # (`job-system.md` § 5.0, *a preview*) -- a launched stage's next
    # launch is a launch again, from its own runs (§ 5.4).
    from .ready import at_launch
    out.takes = ([{"from": stage, "files": [],
                   "line": "at its next launch: launched again, it continues "
                           "from its own latest run or starts over as its "
                           "job says -- --cold starts it over "
                           "(job-system.md § 5.4)"}]
                 if out.launched
                 else at_launch(pcalc, stage, job=job))
    if job is not None and run_dir is not None:
        r = job.resources
        out.resources = {"mpi_np": r.mpi_np,
                         "cpus_per_task": r.cpus_per_task,
                         "max_memory_mb": r.max_memory_mb,
                         "continue_retries": r.continue_retries}
        out.deck = Path(job.script).name
        from ..runfiles import LAUNCH_DIR, GroupNames
        names = run_names(js, job, Shape.named(task.shape))
        # THE HEADER ITS LAUNCH SENDS: a run of points', the walk's one
        # (:func:`prep_calculation`); every other run's, its own.
        # A RUNG OF POINTS -- its points opened, or, launched and prepared
        # again, its walk's header written (`walks`) -- sends its walk.
        walked = bool(out.points or (out.launched and walks))
        sends = (stage_home(base, task, stage).dir / LAUNCH_DIR
                 / GroupNames.of_points(names.stem).name(".sbatch")
                 if walked else Path(run_dir) / names.name(".sbatch"))
        out.launch = _launch_as_written(
            plan, sends, Path(run_dir) / names.name(".run.sh"),
            environment.kind)
        # A RUN OF POINTS' ONE JOB, in the words both doors print beside
        # its header (:func:`walk_words`).
        if out.points:
            out.launch["walk"] = walk_words(pcalc, stage,
                                            read=plan.read_text)
        # ...AND ITS FIT IS THE WALK'S, its points' walls summed: the walk
        # is what is sent, and no point is sent alone.
        if walked:
            out.fit = walks.get(sends, "")
    return out


def _contours_now(calc, stage: str) -> Optional[dict]:
    """What each point of a transmission of a calculation that reuses a
    finished one computes, where it is not its window whole
    (`engines/transport.md` § 2a.16): for a stage launched, as the decks its
    prep wrote record it -- never decided again; else decided on the
    calculation as it is (`reuse.decide`).  ``None`` for any other
    stage."""
    from .materialize import ever_launched
    from .reuse import SHARING_RUNG, decide, point_key
    if stage != SHARING_RUNG or not calc.task.reuses:
        return None
    job = calc.job(stage)
    if job is not None and ever_launched(calc, job):
        from ..transport.energies import Contours
        from ..transport.stages import point_folders
        out = {}
        # A RUNG THAT RUNS ONCE: its one point, the stage's own folder.
        for d, pt in (point_folders(calc.base, calc.task, stage)
                      or [(stage_home(calc.base, calc.task, stage).dir,
                           None)]):
            deck = d / job.script
            c = (Contours.of_deck(read_deck(deck)) if deck.is_file() else None)
            if c is not None:
                out[point_key(pt)] = c
        return out
    reuse = decide(calc, [calc.request.group_of(stage)])
    return reuse.contours_for(stage) if reuse is not None else None


def prep_bench(calc, stage: Optional[str], *, target: Optional[str] = None,
               allocation=None,
               env: Optional[str] = None, emit_sbatch: bool = True,
               on_found=None, preview: bool = False,
               plan_id: Optional[str] = None) -> PrepAnswer:
    """**`prep bench`** -- one stage's benchmark, prepared
    (`job-system.md` § 5.0, § 7): the calculation as the door opened it
    (`calculation.open`); a calculation that takes no benchmark, a
    stage not named, or one prepared as a benchmark already refused --
    a prepared benchmark is not prepared again (its way back, the state
    saved before its prep); the description's preflight and the machine
    (:func:`checked`); the stage planned (:func:`plan_stage`), its grid
    said to ``on_found`` before anything is rendered.  ``preview`` stops
    there; a Prep naming ``plan_id`` is refused when the plan it makes now
    differs.  Then the folder is saved, always, asking nothing; the plan is
    written, deciding nothing; then the record.  A refusal can only come
    before the save, and writes nothing but its line in
    ``jobset-decisions.log`` (§ 5.0, rule 3)."""
    base = Path(calc.base)
    from .ledger import record as ledger
    from .planned import Plan
    from .prep_inputs import bench_refusal
    base = Path(base).resolve()
    described: Optional[Described] = None
    notes: List[str] = []

    def _refused(exc: PrepError) -> PrepError:
        # WHAT WAS FOUND RIDES WITH THE REFUSAL: a sentence that points at
        # "the cells left out above" is honest only if the list is shown
        # -- and the refusal is a decision too, written down in a described
        # calculation (W52); a preview records nothing.
        findings = described.findings if described is not None else []
        if not preview and calcdirs.described(base):
            _record_preflight(base, findings, stage=stage)
            ledger(base, "prep", "refused", kind="bench", stage=stage,
                   reason=str(exc))
        exc.findings, exc.notes = tuple(findings), tuple(notes)
        return exc

    try:
        described = Described(calc)
        task = described.task
        # A CALCULATION THAT HAS NO BENCHMARK says so before it is asked
        # which stage's -- or the stage offered is refused next (W52).
        why = bench_refusal(str(task.engine), str(task.calculation))
        if why:
            raise PrepError(why)
        if stage is None:
            # ITS ONE STAGE, named -- before anything is read of the
            # machine or written (W52).  THE STAGES IT TAKES are offered: one
            # it refuses is not (2a).
            from .commands import name_a_stage
            from .naming import described_refs
            takes = [r for r in described_refs(base, task)
                     if not prepared_already(described.calc, "bench", r.name)]
            raise PrepError(
                "`prep bench` names the stage it measures; "
                + (name_a_stage("prep", "bench", takes, base=base) if takes
                   else "every stage is prepared already, and a prepared "
                        "benchmark is not prepared again (job-system.md "
                        "§ 5.0)."))
        # 2 · THE STAGE GRAMMAR (user-settled 2026-08-21): named by its
        #     NAME, or by `#N` -- the NN of its directory -- through the ONE
        #     resolver, so `prep bench '#2'` and `status #2` cannot disagree.
        from ..identity import StageRef, resolve_stage_ref
        stage = resolve_stage_ref([StageRef(h.seq, h.name)
                                   for h in ladder_homes(base, task)],
                                  stage).name
        # 2a · NOT PREPARED AGAIN (user, 2026-10-02: "refuse it, redo via
        #      rollback"), before anything is read of the machine.
        why = prepared_already(described.calc, "bench", stage)
        if why:
            raise PrepError(why)
        checked(base, described, target=target)
        plan = Plan()
        out = plan_stage(base, described, "bench", stage, target=target,
                         allocation=allocation, env=env,
                         emit_sbatch=emit_sbatch, plan=plan, notes=notes,
                         # THE GRID, said before anything is rendered.
                         on_inputs=(None if on_found is None else
                                    lambda: on_found(described.findings,
                                                     notes)))
        identity = plan.identity()
        if plan_id is not None and plan_id != identity:
            raise PrepError(
                "what prep would write now differs from the plan you "
                "previewed -- something it is made from changed since.  "
                "Preview again (job-system.md § 5.0).")
        if preview:
            out.preview = True
            out.plan_id = identity
            return out
        # 5 · THE SAVE -- always, once the whole plan stands and before
        #     anything is written, the ledger included (`checkpointing.md`
        #     § 9; user, 2026-10-03: "always save through checkpoint, notify
        #     user").  A redo is a rollback, and this is the state it
        #     restores, so a save that fails refuses the prep.
        from ..checkpoint import CheckpointError, save_before
        try:
            # A molbuilder.json that does not read is the person's to fix,
            # said in its own words (the save reads it again).
            with user_error_as_prep():
                kept = save_before(base, f"prep bench {stage}",
                                   engine=str(task.engine))
        except CheckpointError as exc:
            raise PrepError(f"the folder's state could not be saved, so "
                            f"nothing was prepared: {exc}")
        out.saved = kept.said()
        ledger(base, "prep", "saved", stage=stage, state=kept.state.short,
               note=kept.state.note, new=kept.new)
        # 6 · THE PLAN, WRITTEN -- deciding nothing and refusing nothing.
        plan.carry_out()
        # 7 · THE RECORD, after the save -- so the state saved holds no line
        #     of this prep: the description's notes, then the stage's.
        _record_preflight(base, described.findings, stage=stage)
        return _record_prepared(base, out)
    except PrepError as exc:
        raise _refused(exc)
    except step_errors() as exc:
        # WHAT THE PERSON FIXES -- which machine this is for, an answer only
        # they have (`preparing-for-another-machine.md` § 4), and the steps'
        # own refusals (`errors.step_errors`) -- said the same way on both
        # doors.
        raise _refused(as_prep_error(exc)) from exc


def prep_task(calc, kind: str, *, groups: Sequence[int] = (),
              stage: Optional[str] = None,
              target: Optional[str] = None, allocation=None,
              env: Optional[str] = None, emit_sbatch: bool = True,
              on_found=None, preview: bool = False,
              plan_id: Optional[str] = None,
              asked: Optional[dict] = None,
              one_job: bool = False) -> List[PrepAnswer]:
    """**`prep`, the verb** -- what `molbuilder jobset prep` and the Task
    setup tab's Prep both call (`job-system.md` § 5.0, § 5.3): `prep task`
    for the ``groups`` picked -- the description's group numbers, each group
    one job (`engines/stages.md` § 2) -- or `prep bench` for one ``stage``
    (:func:`prep_bench`).  ``asked`` is the question that picked the groups,
    with its answer, written down after the save (§ 5.0, checkpoint 7).
    ``one_job`` -- `--one-job` -- writes too the one header of the groups as
    one job, each launched in turn (§ 6.0, *A sequence of groups*)."""
    if kind == "bench":
        return [prep_bench(calc, stage, target=target,
                           allocation=allocation,
                           env=env, emit_sbatch=emit_sbatch,
                           on_found=on_found, preview=preview,
                           plan_id=plan_id)]
    return _prep_groups(calc, list(groups), target=target,
                        allocation=allocation,
                        env=env, emit_sbatch=emit_sbatch, on_found=on_found,
                        preview=preview, plan_id=plan_id, asked=asked,
                        one_job=one_job)


def _record_the_order(calc, *, refuse, asked, preview,
                      plan_id) -> PrepAnswer:
    """`prep task` with no group named, on a junction whose sidecar states no
    order record (`engines/transport.md` § 1.1, *No group left to prepare*):
    its order found by composing the citation again and comparing it atom
    for atom, and its record written again by the one writer
    (`composed_for_prep`, never composing afresh) -- the folder's state
    saved first, nothing else written.  ``refuse`` and ``asked`` are the
    prep's own (:func:`_prep_groups`)."""
    from ..checkpoint import CheckpointError, save_before
    from .ledger import record as ledger
    from .planned import Plan
    base, task = calc.base, calc.task
    plan = Plan()
    try:
        composed_for_prep(calc, plan, afresh=False)
    except PrepError as exc:
        raise refuse(str(exc)) from None
    written = [str(w.relative_to(base)) for w in plan.writes()]
    answer = PrepAnswer(
        kind="task", stage=None,
        recorded=(f"the composed junction's order recorded -- checked "
                  f"against its citation, atom for atom: "
                  f"{', '.join(written)} written; no group prepared"))
    identity = plan.identity()
    if preview:
        answer.preview, answer.plan_id = True, identity
        return answer
    if plan_id is not None and plan_id != identity:
        raise refuse("what prep would write now differs from the plan you "
                     "previewed -- something it is made from changed "
                     "since.  Preview again (job-system.md § 5.0).")
    try:
        with user_error_as_prep():
            kept = save_before(base, "prep task: the junction's order record",
                               engine=str(task.engine))
    except (CheckpointError, PrepError) as exc:
        raise refuse(f"the folder's state could not be saved, so nothing "
                     f"was written: {exc}") from None
    asked()
    ledger(base, "prep", "saved", groups=[], state=kept.state.short,
           note=kept.state.note, new=kept.new)
    plan.carry_out()
    ledger(base, "prep", "order recorded", writes=written)
    answer.saved = kept.said()
    return answer


class Walk(NamedTuple):
    """The one header of a job that walks several members, where it is
    written, and the members' one request's fit on its queue
    (`scheduler.md` R6)."""
    path: Path
    text: str
    fit: str


def walk_header_of(where, names, request, environment, *, engine: str,
                   what: str, jobs, emit_sbatch: bool = True
                   ) -> Optional[Walk]:
    """THE ONE HEADER OF A JOB THAT WALKS SEVERAL MEMBERS -- a group's
    stages, a run of points, a benchmark's shelf (`job-system.md` § 6.0):
    ``request``, the members' one allocation, placed on the queue it names
    as a run's is (`placement.placed`) and rendered for that queue
    in the `launch/` folder of ``where``, the folder it is sent from -- a
    :class:`Walk`, its fit said, or ``None`` where no header is written (a
    machine with no scheduler, `--no-sbatch`).  Where each value came from
    is read off the members' own placements (``jobs``), each source any
    member states it in; ``what`` names the walk in a refusal for a queue
    stated nowhere or not on the record.  Launch sends it as it is."""
    from ..runfiles import LAUNCH_DIR
    if not (emit_sbatch and environment.kind.queue):
        return None
    from .placement import one_process, placed
    sources: dict = {}
    for j in jobs:
        for field, source in ((j.placement or {}).get("from") or {}).items():
            if source not in sources.setdefault(field, ()):
                sources[field] += (source,)
    placed_on, fit, why = placed(request, environment,
                                 one_process=one_process(engine), what=what,
                                 sources=sources)
    if why:
        raise PrepError(why)
    from ..runwrap import walk_header
    text = walk_header(
        where, names, resources=request, one_process=one_process(engine),
        machine_record=environment,
        domain_pq=(placed_on["partition"], placed_on["qos"]))
    return Walk(Path(where) / LAUNCH_DIR / names.name(".sbatch"), text, fit)


def group_header(base, task, number: int, jobs, environment, *,
                 emit_sbatch: bool = True) -> Optional[Walk]:
    """The :class:`Walk` of group ``number``'s one header -- its members'
    ``jobs`` sharing one allocation (`group.envelope`: the walls summed, the
    largest memory) -- by the one header of a walk (:func:`walk_header_of`),
    or ``None`` where no header is written.  Raises :class:`PrepError`
    naming what the members do not share."""
    from .group import GroupError, envelope, names_for
    base = Path(base)
    names = [j.name for j in jobs]
    try:
        shared = envelope(jobs)
    except GroupError as exc:
        raise PrepError(f"group {number}: {exc}") from None
    return walk_header_of(
        base, names_for(base, task, names), shared, environment,
        engine=task.engine, jobs=jobs, emit_sbatch=emit_sbatch,
        what=(f"group {number}'s one job ({', '.join(names)}: their walls "
              f"summed and the largest memory)"))


def sequence_header(base, task, request, numbers: Sequence[int], jobs_of,
                    points_of, environment, *, emit_sbatch: bool = True
                    ) -> "Tuple[Optional[Walk], List[str]]":
    """A SEQUENCE OF GROUPS' ONE HEADER (`job-system.md` § 6.0, *A sequence
    of groups*) -- ``(its Walk, or None where no header is written; each
    group asking less than it, in words)``: each group's own request
    (`group.own_request`: its jobs ``jobs_of[n]``, a run of points' every
    point -- ``points_of[n]`` of them), their one request
    (`group.sequence_request`: one queue, the largest of each, the walls
    summed), by the one header of a walk (:func:`walk_header_of`).  Asked by
    prep's ``--one-job``.  Raises :class:`PrepError`
    naming what the groups cannot share (R-P22)."""
    from .group import (GroupError, own_request, sequence_names,
                        sequence_request)
    numbers = sorted(numbers)
    requests = [(n, own_request(jobs_of[n], points_of.get(n, 0)))
                for n in numbers]
    words = ", ".join(map(str, numbers))
    try:
        shared = sequence_request(requests)
    except GroupError as exc:
        raise PrepError(f"groups {words} as one job: {exc} "
                        f"(job-system.md R-P22)") from None
    less = []
    for n, r in requests:
        short = [w for w, mine, all_ in (
            ("ranks", r.mpi_np, shared.mpi_np),
            ("cores per rank", r.cpus_per_task, shared.cpus_per_task))
            if mine is not None and all_ is not None and mine < all_]
        from .model import gpu_request
        if gpu_request(shared).uses and not gpu_request(r).uses:
            short.append("no GPU")
        if short:
            less.append(f"group {n} runs with its own card inside it -- "
                        + ", ".join(f"{r.mpi_np} of {shared.mpi_np} ranks"
                                    if w == "ranks" else
                                    f"{r.cpus_per_task} of "
                                    f"{shared.cpus_per_task} cores per rank"
                                    if w == "cores per rank" else w
                                    for w in short)
                        + "; the rest of the allocation idle meanwhile")
    walk = walk_header_of(
        base, sequence_names(base, task, request, numbers), shared, environment,
        engine=task.engine, emit_sbatch=emit_sbatch,
        jobs=[j for n in numbers for j in jobs_of[n]],
        what=(f"groups {words} as one job, each launched when the one "
              f"before has finished (the largest request, the walls "
              f"summed)"))
    return walk, less


def _prep_groups(calc, numbers: List[int], *, target, allocation,
                 env, emit_sbatch, on_found, preview,
                 plan_id, asked=None, one_job: bool = False
                 ) -> List[PrepAnswer]:
    """**The groups picked, prepared** (`job-system.md`, *The task*, and
    § 5.0's checkpoints) -- each one job, in ONE plan.

    The groups named are the description's (checkpoint 2); a pick touching a
    stage prepared in a job whose members are no longer one group -- groups
    are matched by membership, never by number -- holds every group that
    job's members are in now, or it is refused naming them, so every such
    job is formed again in this one prep (nothing prepared unseen); a
    launched member among them is refused first, naming the way back.
    The calculation is the door's, opened once (`calculation.open`) and
    checked here (:func:`checked`), and every member is planned into one
    :class:`~molbuilder.jobset.planned.Plan` by the one member planner
    (:func:`plan_stage`), each seeing what those before it planned; then
    each group's own checks -- the members
    share one allocation (`group.envelope`), and that request, the walls
    summed and the largest memory, is placed on the queue it names, as a
    run's is (`placement.placed`) -- its one header, a header of a job
    formed again taken away, and each member's row naming its group, all in
    that plan.  ``preview`` stops there, each answer carrying the plan,
    named; a Prep naming ``plan_id`` is refused when the plan it makes now
    differs.  Then the folder is saved ONCE, the plan is written as it was
    made -- deciding nothing, refusing nothing (§ 5.0, rule 3) -- and each
    member's record and each group's written down after it."""
    base = Path(calc.base)
    from ..checkpoint import CheckpointError, save_before
    from ..identity import stage_key
    from .ledger import record as ledger
    base = Path(base).resolve()

    def _asked():
        # THE QUESTION THAT PICKED THE GROUPS, written once -- after the
        # save, or with a refusal, which saves nothing.
        if asked is not None and not preview:
            ledger(base, "prep", "question", **asked)

    def _refuse(why: str) -> PrepError:
        # A REFUSAL IS WRITTEN DOWN; a preview records nothing
        # (`job-system.md` § 5.0, rule 3 and *a preview*).
        if not preview and calcdirs.described(base):
            _asked()
            ledger(base, "prep", "refused", kind="task", groups=numbers,
                   reason=why)
        return PrepError(why)

    # 1 · THE DESCRIPTION, as the verb opened it (A16) -- and THE GROUPS
    #     PICKED, its own.
    description = Described(calc)
    task, request = description.task, calc.request
    described = request.numbers
    if not numbers:
        # NO GROUP NAMED: the calculation's own record -- a junction whose
        # sidecar states no order record -- has it recorded, and nothing else
        # (`engines/transport.md` § 1.1, *No group left to prepare*).
        from ..transport.compose import ComposeError, recorded_junction
        try:
            recorded = recorded_junction(base)
        except ComposeError as exc:
            raise _refuse(str(exc)) from None
        if recorded is not None and recorded.order_unstated:
            return [_record_the_order(description.calc, refuse=_refuse,
                                      asked=_asked, preview=preview,
                                      plan_id=plan_id)]
        # THE GROUPS IT MAY PREPARE NOW, the one answer's
        # (`ready.task_state`), and why each other may not.
        from .commands import block, prep_group_lines
        from .ready import task_state
        gs = task_state(description.calc).groups
        offer = [g.number for g in gs if g.preparable]
        held = "".join(f"\n  group {g.number} is not offered: {g.why}"
                       for g in gs if not g.preparable and g.why)
        raise _refuse(
            "`prep task` names the groups it prepares -- a group prepared "
            "and never launched is prepared again by naming it -- and the "
            + (f"groups it may prepare now are "
               f"{', '.join(map(str, offer))}:\n"
               + block(prep_group_lines(offer, base=base))
               if offer else "job request holds no group it may prepare now")
            + held)
    unknown = [n for n in numbers if n not in described]
    if unknown:
        raise _refuse(
            f"group {', '.join(map(str, unknown))}: the job request holds "
            f"groups {', '.join(map(str, described))} "
            f"(execution/job-request.md § 3.3).")
    numbers = sorted(set(numbers))
    # A SEQUENCE OF GROUPS (`--one-job`, `job-system.md` § 6.0): the groups
    # named, and of them the ones launched -- read as prepared, never
    # prepared again -- the rest prepared here as any group is.
    sequence: List[int] = []
    if one_job:
        from ..runrecord import LaunchRecordError
        from .materialize import ever_launched
        sequence = list(numbers)
        if len(sequence) < 2:
            raise _refuse("--one-job sends several groups as one job, and "
                          f"names group {', '.join(map(str, sequence))} "
                          "alone -- name every group it launches "
                          "(job-system.md R-P22).")
        old_rows = description.calc.jobset
        launched_groups = []
        for n in sequence:
            jobs_n = [old_rows.job(st) if old_rows is not None else None
                      for st in request.members(n)]
            try:
                ran = any(j is not None
                          and ever_launched(description.calc, j)
                          for j in jobs_n)
            except LaunchRecordError as exc:
                raise _refuse(str(exc)) from None
            # A LAUNCHED GROUP'S MEMBERS ARE FIXED: the one rule, in its
            # words (`ready.launched_without`, `ready.joined_launched`).
            from .ready import joined_launched, launched_without
            try:
                hit = launched_without(description.calc, n)
            except LaunchRecordError as exc:
                raise _refuse(str(exc)) from None
            if hit:
                raise _refuse(joined_launched(description.calc, n, *hit))
            if ran:
                launched_groups.append(n)
        numbers = [n for n in sequence if n not in launched_groups]
        if not numbers:
            named_ = ", ".join(map(str, sequence))
            raise _refuse(
                f"--one-job: every group named ({named_}) is launched "
                f"already, so nothing is prepared -- send them one by one, "
                f"each launched again (job-system.md R-P22).")
    members = {n: request.members(n) for n in numbers}
    group_of = {stage_key(n): e.group for n, e in request.stages.items()}
    # 1a · THE JOBS PREPARED, matched by membership: a job whose members
    #      are no longer one group is formed again -- every group its
    #      members are in now, in this prep (`group.formed_again`).
    from .group import formed_again, header_of, members_of
    old_js = description.calc.jobset
    prepared = old_js.jobs if old_js is not None else []
    whole = formed_again(request, prepared, numbers)
    # A LAUNCHED MEMBER among them is the refusal that names the way out --
    # a launched group's members are fixed (§ 5.0, 2a) -- before any other.
    from ..runrecord import LaunchRecordError
    from .ready import joined_launched, launched_without
    from .ready import launched_already
    for n in whole:
        # A STAGE PUT INTO A LAUNCHED GROUP: that stage is the change, and
        # a group of its own the way on beside the restore.
        try:
            hit = launched_without(description.calc, n)
        except LaunchRecordError as exc:
            raise _refuse(str(exc)) from None
        if hit:
            raise _refuse(joined_launched(description.calc, n, *hit))
        for st in request.members(n):
            why = launched_already(description.calc, st)
            if why:
                raise _refuse(why)
        # A GROUP A SEQUENCE STILL GOING HOLDS: its job is about to launch
        # the files it was prepared with, and a prep now would rewrite them
        # under it (R-P23; `project-layout.md` § 1.6.3).
        from .ready import held_by_sequence, held_why
        held = held_by_sequence(description.calc, n)
        if held is not None:
            raise _refuse(held_why(n, held))
    missing = [n for n in whole if n not in numbers]
    if missing:
        from .commands import block, prep_group_lines, prep_sequence_lines
        # THE SAME PREP, ONE JOB STILL when it was asked for one.
        again = (prep_sequence_lines if one_job else prep_group_lines)
        raise _refuse(
            f"group {', '.join(map(str, missing))} holds stages prepared in "
            f"one job with the groups picked, and that job's members are no "
            f"longer one group -- every such job is formed again in one "
            f"prep, so pick "
            + ("it" if len(missing) == 1 else "them") + " too:\n"
            + block(again(whole, base=base)))
    # A LAUNCHED MEMBER, prepared again for its job request on the
    # hierarchical layout (`execution/job-request.md` § 5): its deck must
    # stay the calculation its runs ran -- asked of the decks this prep
    # renders, below.
    from .materialize import ever_launched as _ever_launched
    launched_members: Set[str] = set()
    for n in numbers:
        for st in members[n]:
            job_st = description.calc.job(st)
            if job_st is not None and _ever_launched(description.calc,
                                                     job_st):
                launched_members.add(st)
    # 1b · A CALCULATION THAT REUSES A FINISHED ONE (`engines/transport.md`
    #      § 2a.16): each group none of whose members is launched here,
    #      COPIED IN or prepared here -- decided before anything is planned,
    #      in the ladder's order -- the manifest.  A group copied in is in no
    #      sequence of groups: no job launches it.
    from .reuse import decide
    try:
        reuse = decide(description.calc, [
            n for n in numbers
            if not any(st in launched_members for st in members[n])])
    except PrepError as exc:
        raise _refuse(str(exc)) from None
    if reuse is not None and sequence:
        sequence = [n for n in sequence if not reuse.copied(n)]
        if len(sequence) < 2:
            copied_ = [d.number for d in reuse.groups if d.copied]
            raise _refuse(
                f"--one-job sends several groups as one job, and group"
                f"{'s' if len(copied_) > 1 else ''} "
                f"{', '.join(map(str, copied_))} "
                f"{'are' if len(copied_) > 1 else 'is'} copied in from "
                f"{reuse.name} -- launched by no job -- leaving "
                f"{', '.join(map(str, sequence)) or 'none'} to send: prepare "
                f"without --one-job (engines/transport.md § 2a.16; "
                f"job-system.md R-P22).")
    # 2 · THE DESCRIPTION CHECKED AND THE MACHINE READ, once (3, 4); then
    #     EVERY MEMBER, PLANNED INTO ONE PLAN by the one member planner,
    #     nothing written -- each member seeing what those before it planned
    #     (their rows in `job-set.json`).  A refusal is the prep's, written
    #     down as every refusal is -- the description's notes first.
    from .planned import Plan
    the_plan = Plan()
    previews: Dict[int, List[PrepAnswer]] = {}
    try:
        try:
            checked(base, description, target=target)
            if on_found is not None:
                on_found(description.findings, [])
            for n in numbers:
                previews[n] = [plan_stage(
                    base, description, "task", at, target=target,
                    allocation=allocation,
                    env=env, emit_sbatch=emit_sbatch, plan=the_plan,
                    copied=reuse is not None and reuse.copied(n),
                    contours=(reuse.contours_for(at)
                              if reuse is not None
                              and at not in launched_members else DECIDE))
                    for at in members[n]]
        except step_errors() as exc:
            raise as_prep_error(exc) from exc
    except PrepError as exc:
        if not preview:
            _record_preflight(base, description.findings, groups=numbers)
        exc.findings = tuple(description.findings)
        _refuse(str(exc))
        raise
    # 2a · A LAUNCHED MEMBER'S DECKS, THE SAME CALCULATION as the decks its
    #      runs ran (`execution/job-request.md` § 5, § 6): each deck this
    #      prep renders for it, against the one it replaces, through the one
    #      door -- the job request may change, what it computes may not.
    for n in numbers:
        for a in previews[n]:
            if a.stage not in launched_members:
                continue
            differs = _computes_else(base, task.engine, the_plan, a)
            if differs:
                from .commands import redo_launched
                raise _refuse(
                    f"stage {a.stage!r} was launched, and its deck rendered "
                    f"now is not the calculation its runs ran -- {differs}.  "
                    f"A launched stage is prepared again for its job request "
                    f"alone (execution/job-request.md § 5); to change what it "
                    f"computes, " + redo_launched(a.stage,
                                                   calc=description.calc))
    # 2b · EACH GROUP'S OWN FILE AND REQUEST, in the same plan -- its one
    #      header, where the machine has a scheduler, rendered for the
    #      members' shared allocation, which is placed as a run's is; the
    #      header of a job formed again, taken away; and each member's row
    #      naming its group.
    from ..runfiles import LAUNCH_DIR
    headers: Dict[int, Optional[Path]] = {}
    #: THE SEQUENCE'S ONE HEADER this prep writes, or one an earlier prep
    #: wrote that it takes away -- ``None`` for neither.
    seq_header: Optional[Path] = None
    # EACH GROUP OF SEVERAL, as its members' answers say it: the one job,
    # its header and what that header launches it with (A13).
    said: Dict[int, dict] = {}
    environment = description.environment
    try:
        for n in numbers:
            headers[n] = None
            if len(members[n]) < 2:
                continue
            got = group_header(base, task, n, [a.job for a in previews[n]],
                               environment, emit_sbatch=emit_sbatch)
            had = set(the_plan.writes())
            if got is not None:
                open_container(base, base / LAUNCH_DIR, the_plan)
                headers[n] = got.path
                the_plan.text(got.path, got.text)
            said[n] = {
                "number": n, "members": list(members[n]),
                "header": (str(got.path.relative_to(base))
                           if got is not None else None),
                "launch": _walk_said(got, environment.kind),
                "writes": [str(w.relative_to(base))
                           for w in the_plan.writes() if w not in had]}
        # THE SEQUENCE'S ONE HEADER (`--one-job`, `job-system.md` § 6.0):
        # each group's own request -- this prep's members as planned, a
        # launched group's as its rows stand -- together.
        if sequence:
            # A RUN OF POINTS' WALLS, as many as its kind asks
            # (`points.walls_of`): one each, or the stage's one.
            from .points import rung_points_of, walls_of
            jobs_of, points_of = {}, {}
            for n in sequence:
                if n in previews:
                    jobs_of[n] = [a.job for a in previews[n]]
                    points_of[n] = walls_of(
                        task, len(previews[n][0].points or ())
                        if len(previews[n]) == 1 else 0)
                else:
                    jobs_of[n] = [old_js.job(st)
                                  for st in request.members(n)]
                    points_of[n] = walls_of(
                        task, len(rung_points_of(
                            description.calc, jobs_of[n][0].name))
                        if len(jobs_of[n]) == 1 else 0)
            got, less = sequence_header(base, task, request, sequence,
                                        jobs_of, points_of, environment,
                                        emit_sbatch=emit_sbatch)
            if got is not None:
                open_container(base, base / LAUNCH_DIR, the_plan)
                the_plan.text(got.path, got.text)
                seq_header = got.path
            else:
                # NO HEADER THIS TIME (a machine with no queue,
                # `--no-sbatch`), so none an earlier prep wrote is left to be
                # sent as this one's (`job-system.md` § 5.0, checkpoint 6).
                from .group import sequence_names
                _left = base / LAUNCH_DIR / sequence_names(
                    base, task, request, sequence).name(".sbatch")
                if _left.is_file():
                    the_plan.remove(_left)
                    seq_header = _left
            sequence_said = {
                "numbers": list(sequence),
                "header": (str(got.path.relative_to(base))
                           if got is not None else None),
                "launch": _walk_said(got, environment.kind), "less": less}
        # A JOB FORMED AGAIN leaves no header of its old members behind --
        # named by its members' own files, a member the description no
        # longer holds among them (`group.header_of`).
        for j in prepared:
            if any(group_of.get(stage_key(m)) in numbers
                   for m in members_of(j)):
                old = header_of(description.calc, j)
                if (old is not None and old not in headers.values()
                        and old.is_file()):
                    the_plan.remove(old)
    except PrepError as exc:
        raise _refuse(str(exc)) from None
    #      THE MEMORY EACH STAGE NEEDS at the ranks it is prepared with, its
    #      job's the largest of its members' -- said beside the fit, refusing
    #      nothing (`execution/job-request.md` § 3.1a).
    from .memory import largest, need
    for n in numbers:
        for a in previews[n]:
            a.need = need(description.calc, a.stage,
                          (a.resources or {}).get("mpi_np"))
    for n, g in said.items():
        g["need"] = largest([a.need for a in previews[n]]).to_wire()
    if sequence:
        sequence_said["need"] = (largest(
            [a.need for n in sequence for a in previews.get(n, ())]
            + [need(description.calc, j.name, j.resources.mpi_np)
               for n in sequence if n not in previews for j in jobs_of[n]])
            .to_wire())
    # 2d · NOTHING TO WRITE (`job-system.md`, *The task*, 2; checkpoint 2a):
    #      a group prepared, never launched, and planned now as it stands --
    #      each member's decks, scripts and row (`prepared_differs`), its
    #      members, its record of what it was prepared with and the group's
    #      header -- is left out of the save, the
    #      writing and the record, said with its next step; the groups that
    #      differ are prepared without it, a preview's plan the same as its
    #      Prep's.  A sequence of groups (`--one-job`) is one job: nothing to
    #      write when every group of it stands and its one header is as
    #      written, else every group of it is prepared.
    from .group import same_members

    #: EACH STAGE PREPARED AND NEVER LAUNCHED HERE, what of it differs from
    #: this plan -- what makes it prepared again, said in its answer.
    differs: Dict[str, List[str]] = {}

    def _rel(path) -> str:
        return str(Path(path).relative_to(base))

    def _header_stands(path) -> bool:
        return path is None or (as_written(path, the_plan.read_bytes(path))
                                if the_plan.is_file(path)
                                else not path.is_file())

    def _as_it_stands(n: int) -> bool:
        if (any(st in launched_members for st in members[n])
                or (reuse is not None and reuse.copied(n))):
            return False
        same, fresh = True, False
        for a in previews[n]:
            was = description.calc.job(a.stage)
            if was is None:
                fresh = True              # prepared here for the first time
                continue
            differs[a.stage] = prepared_differs(description.calc, the_plan,
                                                was, a.job)
            if not same_members(was.members or [was.name], members[n]):
                differs[a.stage].append("its group's members")
            # ITS RECORD OF WHAT IT WAS PREPARED WITH, as the records give
            # it now (`prepared_with.changes`) -- a group status calls
            # changed since prepared is never one with nothing to write.
            recorded, since = prepared_with.changes(description.calc,
                                                    a.stage)
            if not recorded or since:
                differs[a.stage].append(
                    f"its record of what it was prepared with "
                    f"({prepared_with.FILENAME})")
            same = same and not differs[a.stage]
        if fresh:
            return False
        # ITS HEADER, named too when a member differs.
        header = headers[n] or header_of(description.calc,
                                         description.calc.job(members[n][0]))
        if not _header_stands(header):
            differs.setdefault(members[n][0], []).append(_rel(header))
            same = False
        return same
    stands = [n for n in numbers if _as_it_stands(n)]
    if sequence and (len(stands) < len(numbers)
                     or not _header_stands(seq_header)):
        # ONE JOB, prepared whole: its header named when it is what
        # differs, else each group that stands prepared again with the
        # others, saying so.
        if len(stands) == len(numbers):
            differs.setdefault(members[numbers[0]][0], []).append(
                _rel(seq_header))
        for n in stands if len(stands) < len(numbers) else ():
            differs.setdefault(members[n][0], []).append(
                f"the one job of groups {', '.join(map(str, numbers))} it "
                f"is sent in, prepared whole")
        stands = []
    for n in numbers:
        for a in previews[n]:
            if n not in stands:
                a.differs = differs.get(a.stage, [])
    if stands:
        # ...ITS DESCRIPTION'S NOTES KEPT, as every answer carries them.
        kept_as_is = [PrepAnswer(
            kind="task", stage=a.stage, machine=a.machine, preview=preview,
            findings=list(a.findings),
            unchanged=(f"{a.stage} ({stage_home(base, task, a.stage).dir.name}"
                       f"): prepared, never launched, and already as this "
                       f"prep would prepare it now -- nothing to write "
                       f"(group {n})"))
            for n in stands for a in previews[n]]
        rest = [n for n in numbers if n not in stands]
        if not rest:
            # A SEQUENCE STANDING WHOLE is still sent as one job.
            if sequence:
                kept_as_is[-1].sequence = sequence_said
            return kept_as_is
        # THE REST, prepared as any pick.
        again = _prep_groups(
            calc, rest, target=target, allocation=allocation,
            env=env, emit_sbatch=emit_sbatch, on_found=on_found,
            preview=preview, plan_id=plan_id, asked=asked)
        for a in again:
            a.differs = differs.get(a.stage, [])
        # IN THE LADDER'S ORDER, as every answer of a prep is.
        order = {st.name: i for i, st in enumerate(task.stages)}
        return sorted(again + kept_as_is,
                      key=lambda a: order.get(a.stage, len(order)))
    # 2c · EVERY RUN OF EACH GROUP COPIED IN, into the same plan
    #      (`engines/transport.md` § 2a.16): a run of a member prepared here
    #      and never launched removed first, each run's launch record last.
    if reuse is not None:
        from .reuse import copy_in
        for d in reuse.groups:
            if d.copied:
                copy_in(description.calc, d, the_plan)
    #      EACH GROUP ON ITS MEMBERS' ROWS -- the members in order, ``None``
    #      for a group of one.
    js = JobSet.of(base, plan=the_plan)
    for j in js.jobs:
        g = group_of.get(stage_key(j.name))
        if g in numbers:
            j.members = list(members[g]) if len(members[g]) > 1 else None
    js.write(base, plan=the_plan)
    answers = [a for n in numbers for a in previews[n]]
    #      WHAT EACH WAS PREPARED WITH, for `status` to say when its records
    #      give otherwise (`job-system.md`, *Changed since prepared*).
    prepared_with.record(description.calc.with_plan(js),
                         [a.stage for a in answers], plan=the_plan)
    if sequence:
        answers[-1].sequence = sequence_said
    #      THE REVIEWABLE PLAN, ONCE: the job set as this prep leaves it,
    #      and what every member's first launch takes.
    the_plan.text(base / _PLAN_FILE, stage_plan_text(
        js, [ln for a in answers for ln in this_prep_lines(a.stage, a.takes)]
        + (reuse.lines(base) if reuse is not None else []),
        base=base, provenance=answers[0].provenance, plan=the_plan))
    for a in answers:
        a.copied = reuse
    for n in numbers:
        for a in previews[n]:
            a.group = said.get(n)
            # A MEMBER OF A GROUP OF SEVERAL is sent in the group's one job,
            # whose fit is said after its last member: its own is not what
            # is sent.
            if a.group is not None:
                a.fit = ""
    identity = the_plan.identity()
    if preview:
        for a in answers:
            a.preview, a.plan_id = True, identity
        return answers
    if plan_id is not None and plan_id != identity:
        raise _refuse("what prep would write now differs from the plan you "
                      "previewed -- something it is made from changed "
                      "since.  Preview again (job-system.md § 5.0).")
    # 3 · THE SAVE, ONCE, before anything is written.
    try:
        with user_error_as_prep():
            from .group import named
            kept = save_before(
                base, "prep task " + ", ".join(
                    named(request, n)
                    + (f", copied in from {reuse.name}"
                       if reuse is not None and reuse.copied(n) else "")
                    for n in numbers),
                engine=str(task.engine))
    except (CheckpointError, PrepError) as exc:
        # EITHER WAY WRITTEN DOWN (R-P15): the save's own refusal, or the
        # configuration it reads, said as prep's.
        raise _refuse(f"the folder's state could not be saved, so nothing "
                      f"was prepared: {exc}") from None
    _asked()
    ledger(base, "prep", "saved", groups=numbers, state=kept.state.short,
           note=kept.state.note, new=kept.new)
    # 4 · THE PLAN, WRITTEN as it was made -- every member, every header --
    #     deciding nothing and refusing nothing (§ 5.0, rule 3).
    the_plan.carry_out()
    # 5 · THE RECORD: the description's notes, each member's, then each
    #     group's.
    _record_preflight(base, description.findings, groups=numbers)
    for a in answers:
        a.saved = kept.said()
        _record_prepared(base, a)
    for n in numbers:
        if len(members[n]) > 1:
            ledger(base, "prep", "grouped", group=n, stages=members[n],
                   header=(str(headers[n].relative_to(base))
                           if headers[n] else None))
    if sequence:
        ledger(base, "prep", "sequenced", groups=sequence,
               header=sequence_said["header"])
    for d in (reuse.groups if reuse is not None else ()):
        for c in d.copies:
            ledger(base, "prep", "copied in", group=d.number, stage=c.stage,
                   run=str(c.run.relative_to(base)), source=c.shown,
                   copied_from=c.copied_from, ended=c.ended, size=c.size,
                   left=list(c.left))
    return answers


__all__ = ["checked", "plan_stage", "Described",
           "DECIDE", "prep_bench", "prep_task", "group_header", "PrepAnswer",
           "prepared_already", "prepared_stages"]
