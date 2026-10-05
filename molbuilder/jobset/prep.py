"""Prep — THE CONDUCTOR of `docs/execution/script-preparation.md`.

It walks floors 1 -> 4 in order and owns no decision of its own: the five
steps, and the eleven sub-steps inside step 3, are that document's (§ 4);
what each engine supplies at each step is its § 5.  **It may call, but it
may never decide** -- a value settled here is a value no floor owns, which
is the shape of the "stomp" bugs (§ 3.3).

:func:`prep_calculation` is the five entire, on the described route: a
description plus its template in, one rendered deck and wrapper **per
element** of the resolved :class:`~molbuilder.resolve.ParameterSet` out.
:func:`prep_jobset` is steps 4–5 alone, over an existing ``job-set.json`` —
the shared tail of the described route, and the library surface a
hand-built set uses directly.  It stopped being a CLI route on 2026-08-12
(U2/U4): `prep` is described-only, because the pre-made-bundle arm had no
producer left.

The framework here was first built inside the benchmark and the general part
lifted out (§ 2.3.1a: *benchmarking is `prep` whose parameters are a set
rather than a point* — the five steps are general, the grid is the
specialisation).

Wrappers render **once per distinct ``job.script``, in the JOB'S OWN
DIRECTORY** (L2, roadmap 7.10 -- until 2026-08-24 they rendered in the
bundle root and were symlinked down, which is how a ten-trial sweep came to
keep 50 rendered files at its root).  A set that genuinely shares one script
gets a real copy per directory rather than a second render.  On the described
route every element renders its own
deck, so per-script is per-element and each wrapper carries its own
element's resources.  *(A "legacy sweep whose jobs share one script"
paragraph stood here promising its own fold "with bench (plan step 6)" —
the fold landed 2026-08-12 and no producer emits shared-script sets; for a
HAND-BUILT set that shares one script, the first job's resources still
become the wrapper defaults and ``launch`` passes each job's own as flags,
which is now a property of the fallback rather than a design of its own;
R8.)*
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

from .. import script_emit as _sc
from .materialize import (job_dir_names, shape_of, materialize, stage_home,
                          ladder_homes)
from ..runrecord import write_gathered_from
from ..warmfiles import warm_list
from ..issues import calling as _calling
from .model import FILENAME as JOBSET_FILENAME, Job, JobSet, Resources
from .plan import FILENAME as _PLAN_FILE
from ..runfiles import compose as _rf, stem as _rf_stem
from ..pseudos import PSEUDO_DIRNAME
from .errors import PrepError
from .machine import machine_record, require_activation, set_machine
from .engines import EngineSeam, engine_seam, _pseudo_dir, _screen_pseudos


from contextlib import contextmanager

if TYPE_CHECKING:                      # annotations only
    from ..resolve import ParameterSet
    from ..scheduler.record import Environment
    from ..task import Task


@contextmanager
def _user_error_as_prep():
    """Translate the USER-FIXABLE refusal classes the steps below raise into
    :class:`PrepError`, so every caller of the two prep entries has ONE class
    to catch.  Without this the described route's two most likely
    first-contact failures -- missing pseudos (``ValidationError`` out of the
    deck render) and an unset activation
    (``RuntimeConfigError`` out of the wrapper render) -- escaped the CLI as
    raw tracebacks (2026-08-12 plan A8).  Only the NAMED classes translate: a
    ``TypeError`` here is a bug and should look like one.

    **The hook boundary's attribution is carried across**
    (`issues.calling`).  ``str(exc)`` does not include an exception's notes,
    so a refusal that came out of a named engine hook would arrive at the CLI
    saying what went wrong and not whose it was -- and the note would reach a
    traceback nobody sees."""
    from ..issues import ValidationError, notes_of
    from ..runtime_config import RuntimeConfigError
    from ..runwrap import WrapperError
    try:
        yield
    except (ValidationError, RuntimeConfigError, WrapperError) as exc:
        note = notes_of(exc)
        raise PrepError(str(exc) + (f"\n  ({note})" if note else "")) from exc


#: What a prep says when no machine record answers.
#:
#: ONE SENTENCE OF WHY, then the command.  The rule is `running-a-job.md`
#: § 3.1's -- a machine's facts are read from a record and nowhere else, the
#: local box included -- and the reason it is worth a refusal rather than a
#: probe is that a probed-on-the-fly number is indistinguishable from a
#: recorded one once it is in a wrapper.
def _flat_continued_from(base: Path, task, stage: str, continuation,
                         plan) -> None:
    """A flat stage's own ``<basename>.continued-from``, for its launch
    record (`project-layout.md` § 1.6.3) -- the flat layout records what a
    stage continues from too (user, 2026-10-01).  It names the run by what
    every file of it carries (`runfiles.run_name`): flat keeps no directory
    per run.  A re-prep that no longer continues -- the run card now says
    `clean` -- takes the old one away, or the launch would record a source
    it did not read.  Into ``plan``, with the rest of the prep."""
    from ..runfiles import latest_run, run_name, stem as rf_stem
    from ..runrecord import continued_from_marker, write_continued_from
    stem = rf_stem(task.label, stage_home(base, task, stage).token)
    marker = continued_from_marker(base, stem)
    if continuation is None:
        if plan.is_file(marker):
            plan.remove(marker)
        return
    token = stage_home(base, task, continuation.stage).token
    n = latest_run(base, task.label, stage=token)
    if n is None:
        return                     # concluded with no run file: name none
    write_continued_from(base, run_name(task.label, token, n),
                         basename=stem, plan=plan)


def prep_jobset(jobset: JobSet, base_dir, *, env: str = None,
                emit_sbatch: bool = True, record_dir=None,
                log=None, machine_record=None,
                render=None, points=None, plan=None,
                shape=None, this_prep=None) -> List[Path]:
    """Render launchers + lay out the per-job tree under ``base_dir``.

    ``plan`` (`jobset.planned.Plan`) receives what this writes, and is read
    for the decks it already holds; with none it is made here and carried
    out at the end (`job-system.md` § 5.0).

    ``shape`` is the layout as the caller read it from the description --
    a layer below takes it (`materialize.shape_of`); with none it is asked
    here.  ``this_prep`` is what this prep's hand-over takes, which
    `STAGE-PLAN.md` says under its table (`plan.render_plan`).

    ``points`` names, per job, the folders that hold a copy of its deck
    written at another point -- a scan's (`Rung.points`) -- each of which
    gets the job's wrapper beside it, rendered by the one call the job's own
    folder gets.

    ``render`` names the jobs whose run scripts this prep writes -- the
    stage it preps.  The plan it is handed is the calculation's, merged
    (`_merge_run_jobset`), and every other stage in it was prepped already:
    its run script and monitor stay as that stage's own prep wrote them,
    the copy its attempt ran with (`project-layout.md` § 1.6.2, § 2.6; plan
    D21: every prep rendered every stage's again until 2026-10-04 -- in the
    flat shape, the launched stage's own, in its run folder).  ``None``
    renders every job: a benchmark's plan is its own, every trial this
    prep's.

    Steps, in order:
      1. render each **distinct** ``job.script``'s ``.run.sh`` (and
         ``.sbatch`` when ``emit_sbatch`` and a scheduler is configured)
         **in that job's own directory**, beside the deck it launches —
         reusing ``runwrap.write_run_wrapper`` (no reinvention).  The header
         carries the first-seen job's resources as defaults; ``launch``
         overrides per job via CLI flags, so the defaults never decide the
         answer.  A job that SHARES another's script gets a real copy, not a
         reference: a run directory holds real files.
      2. ``materialize`` — the shared package, as real copies into each job
         directory (what a stage continues from is copied when its attempt
         is opened: ``materialize.prepare_attempt``).
      3. emit ``STAGE-PLAN.md`` beside the job-set it describes.

    Returns the per-job directories.  Raises :class:`PrepError` on an
    invalid JobSet or a script that is not in its job's directory.

    **This list said something else until 2026-09-16** — wrappers rendered at
    the bundle root and symlinked into a ``point-<name>/`` dir — which is the
    design the body deleted on 2026-08-24 (step 1's own header says *IN THE
    JOB DIR* and step 3's says *(gone)*), and ``point-<name>`` is a directory
    name ``materialize.job_dir_name`` (now `paths.trial_name`) records retiring before that.  A public
    entry point whose docstring describes a deleted design misleads at the
    lines a caller actually reads.
    """
    from ..paths import trial_label
    from ..runwrap import write_run_wrapper

    # The allocation is NOT a parameter here (U2, 2026-08-12; it was, and
    # re-applying it over per-element resources was the review's "stomp").
    # `project-layout.md` M4 still holds -- an allocation is an input to
    # *prep* -- but it enters ONCE, at resolve, where each element folds it
    # into its own resources (generator.md § 5); by this floor every job
    # already carries the answer.
    errs = jobset.validate()
    if errs:
        raise PrepError(
            "cannot prep an invalid JobSet:\n  - " + "\n  - ".join(errs))
    base = Path(base_dir).resolve()
    if not base.is_dir():
        raise PrepError(f"bundle root not found: {base}")
    from .planned import Plan
    own = plan is None
    plan = Plan() if own else plan

    # ---- 0. resolve the machine (§ 2.3.1 step ONE) ---------------------- #
    # Idempotent by contract (set_machine early-returns on an existing
    # environment.json).  Every production caller is the described route,
    # whose `prep_calculation` already ran step 1; this call serves a direct
    # caller of `prep_jobset` (its tests) -- not a re-decision.
    set_machine(base, plan=plan)

    # ---- 1. render wrappers once per distinct script, IN THE JOB DIR --- #
    # Nothing rendered lives at the bundle root (user, 2026-08-24;
    # `project-layout.md` § 1.0).  The deck was born in its directory by
    # `prep_calculation`, and the wrapper is written beside the deck it
    # launches.
    if log is not None:
        log.phase("STEP 4 · WRAPPERS — how each deck is launched")
    _sh = shape if shape is not None else shape_of(jobset, base_dir)
    _dir_of = job_dir_names(jobset, _sh)
    rendered: dict = {}
    for job in jobset.jobs:
        if render is not None and job.name not in render:
            continue
        # THE SAME QUESTION `prep_calculation` AND `materialize` ASK.  A
        # trial's files live in its attempt when the shape keeps them
        # (`project-layout.md` § 1.5a); this loop looked in the container
        # and reported the deck missing, because it was the one writer
        # that had not been told.
        _jd = base / _dir_of[job.name]
        if jobset.kind == "sweep":
            from .materialize import trial_work_dir
            _jd = trial_work_dir(_jd, _sh)
        plan.folder(_jd)
        if jobset.kind == "sweep":
            # A TRIAL'S FOLDERS SAY WHAT THEY ARE, as a stage's attempt
            # does (`project-layout.md` § 1.4a).
            from .materialize import mark_run
            mark_run(base, _jd, plan)
        if job.script in rendered:
            # A SHARED script (several trials, one deck): each directory
            # still holds its own real copy (L2) -- under the symlink
            # model one root render served every dir by reference, and a
            # directory that references is a directory that does not hold.
            # `_copy2`, not `shutil as _sh`: `_sh` is this function's SHAPE
            # (line above), and `import shutil as _sh` here rebound that
            # function-local for the whole body -- so the second job through
            # this branch handed the shutil MODULE to `trial_work_dir` as a
            # shape.  One name, two meanings, in one function.
            _src_dir = rendered[job.script]
            _stem0 = Path(job.script).stem
            for _fn in (job.script, f"{_stem0}.run.sh", f"{_stem0}.sbatch"):
                if (plan.is_file(_src_dir / _fn)
                        and not plan.is_file(_jd / _fn)):
                    plan.copy(_src_dir / _fn, _jd / _fn)
            if log is not None:
                log.note(f"{job.name}: shares {job.script}'s wrapper, "
                         f"copied from {_src_dir.name}/")
            continue
        script_path = _jd / job.script
        if not plan.is_file(script_path):
            raise PrepError(
                f"job {job.name!r}: script {job.script!r} not in "
                f"{_jd} (render the inputs before prep).")
        def _wrap(script_path):
            # The ALLOCATION, whole (architecture.md § 3.1, rule A8).  This
            # call listed nine of the wrapper's eleven keyword arguments until
            # 2026-08-17 and omitted `omp_threads`, so every deck here shipped
            # a `.sbatch` asking for `-c N` beside a `.run.sh` that baked an
            # OMP default of 1 -- invisible under sbatch, where
            # SLURM_CPUS_PER_TASK outranks the default, and silently flat on a
            # workstation, which is where a benchmark's cores-per-rank axis
            # stopped measuring anything.
            #
            # It is the second time this door lost a field to a hand-copied
            # argument list: `max_memory_mb` went the same way on 2026-08-11,
            # and that fix moved the field onto `Resources` without changing
            # how it is passed.  Passing the object is what buys the sentence
            # that fix wrote -- *carried on the allocation, it cannot be
            # forgotten by one of them.*
            write_run_wrapper(
                script_path,
                # THE LABEL TRAVELS (`gpu.md` G7).  `jobset.name` is
                # `task.label` -- the SystemLabel / JOB literal and the stem
                # of every file -- so the wrapper is TOLD the name its sweep
                # keys on.  It opened the deck and read it back until
                # 2026-09-17, which is re-deriving a value we are holding.
                # A benchmark trial's is the trial's own, the label its deck
                # carries (`paths.trial_label`; the calculation's stood here
                # until 2026-10-05, W38 M5).
                label=(trial_label(jobset.name, job.name)
                       if jobset.kind == "sweep" else jobset.name),
                resources=job.resources,
                env=env,
                emit_sbatch=emit_sbatch,
                # The BUNDLE'S scope, explicitly: the script is born in its
                # job directory now, and the renderer's parent-derived
                # fallback would read the record one level below the
                # bundle's environment.json (roadmap 7.10 M1).
                project_dir=base,
                # WHICH MACHINE THIS IS FOR, carried rather than re-derived
                # (2026-08-24).  The record always travels, so the wrapper
                # reads that machine's own activation off it instead of this
                # machine's config.  It is the
                # same lesson as the paragraph above: a fact the conductor
                # already resolved, handed over whole, cannot be forgotten
                # or answered a second way further down.
                machine_record=machine_record,
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
                # (`running-a-job.md` § 3.5).
                resumes=job.resumes,
                plan=plan,
            )
        with _user_error_as_prep():
            _wrap(script_path)
            for _point in (points or {}).get(job.name, ()):
                _wrap(Path(_point) / job.script)
        rendered[job.script] = _jd
        if log is not None:
            _stem = Path(job.script).stem
            log.received(job.script, _flat_resources(job.resources)
                         + (f", env={env}" if env else ""))
            for _w in (f"{_stem}.run.sh", f"{_stem}.sbatch"):
                if plan.is_file(_jd / _w):
                    log.produced(_w, f"{len(plan.read_text(_jd / _w).splitlines())}"
                                     f" lines")
                elif _w.endswith(".sbatch"):
                    log.note(f"{_w}: not written "
                             + ("(emit_sbatch off)" if not emit_sbatch
                                else "(no scheduler configured)"))

    # ---- 2. the shared package, copied --------------------------------- #
    dirs = materialize(jobset, base, plan, shape=_sh)
    if log is not None:
        log.phase("STEP 5 · RUN DIRECTORY — where each job will be launched")
        log.received("shape", str(_sh))
        log.received("shared package",
                     ", ".join(jobset.shared) or "nothing (W5)")
        for _d in dirs:
            log.produced(_d.name if _d.resolve() != base.resolve() else ".",
                         "the bundle root — flat runs here, nothing is linked"
                         if _d.resolve() == base.resolve() else str(_d))

    # ---- 3. (gone) -- wrappers are BORN in the job dir (step 1) -------- #
    # A whole pass of symlink-laying stood here: wrappers rendered at the
    # root, then pointed at from each directory.  With the render moved
    # into the directory there is nothing left to link, and the monitor
    # travels as a real copy with the rest of the shared package
    # (`materialize`), because a run directory holds real files
    # (`project-layout.md` § 1.0; user, 2026-08-24).

    # ---- 4. emit STAGE-PLAN.md (§ 5 D3; mirrors bench's BENCH-PLAN.md) --- #
    # The reviewable plan lands in the bundle at prep -- the table `jobset
    # plan` printed until it folded into `status <stage>` (2026-10-01), which
    # reads the same columns per stage.  It carries the CONFIG PROVENANCE --
    # which files supplied the effective execution settings -- so a
    # behaviour difference between two machines is explained by the bundle
    # itself (user request 2026-08-12; secrets excluded by construction).
    from ..runtime_config import config_provenance, format_provenance
    from .plan import render_plan
    # The plan lands BESIDE the job-set it describes: the run's at the
    # root, a bench's inside its stage's bench/ container -- so a bench
    # prep can never overwrite the run's reviewable plan (U1, 2026-08-12).
    plan_dir = Path(record_dir) if record_dir is not None else base
    # WHICH file supplied the warm-file vocabulary (U6a provenance): the
    # engine's own, or this calculation's fine-tuned copy -- a surprising
    # carry must be debuggable from the plan alone (§ 4.2a).
    try:
        _in_effect = warm_list(jobset.engine, None, base)
        _vocab = (f"warm-files: {_in_effect.path}"
                  + (" (this calculation's own)" if _in_effect.own
                     else "") + "\n")
    except Exception:
        _vocab = ""    # an engine without a rules file has no line to print
    plan.text(plan_dir / _PLAN_FILE,
              render_plan(jobset, this_prep) + "\n\n" + _vocab
              + format_provenance(config_provenance(project_dir=base)) + "\n")
    if log is not None:
        log.produced(_PLAN_FILE, str(plan_dir / _PLAN_FILE))
    if own:
        plan.carry_out()
    return dirs


# --------------------------------------------------------------------- #
#  The five steps, entire — `project-layout.md` § 2.3.1                  #
# --------------------------------------------------------------------- #

def _vibration_block(stage: str, cfg, relaxed_by, *, criterion) -> dict:
    """A SIESTA force-constant deck's `vibration` block: the facts its job's
    finish reads and no SIESTA keyword states (`engines/vibration.md`
    § 5.3), built here because only `prep` holds all of them -- the stage,
    its resolved config, the relax run it read the coordinates from
    (:func:`_vibration_stage_geometry`), the molbuilder rendering the deck.
    ``criterion`` is the relaxation rung's own ``relax_force_tol`` -- the
    force the reference geometry was relaxed to (plan § 5w K4)."""
    from .. import __version__ as _mb_version
    from ..pyscf.stages import VIBRATION_RELAX_STAGE
    from ..spectra.siesta_vibration import vibration_record
    # The record is of the ladder's relaxation rung: `_vibration_stage_geometry`
    # read it from that stage's output, found by its role (plan § 5w K12) --
    # and the block names it by the kind's name for it, `relax`, which every
    # verb resolves in any case, so the finish's remedy can
    # (`engines/vibration.md` § 5.3, § 5.5).
    return vibration_record(
        stage=stage,
        force_criterion_ev_ang=criterion,
        already_relaxed=bool(getattr(cfg, "already_relaxed", False)),
        relaxation=relaxed_by,
        relaxation_stage=VIBRATION_RELAX_STAGE,
        temperature_K=float(cfg.temperature_K),
        molbuilder_version=str(_mb_version))


def _vibration_stage_geometry(base, task, pset, struct, *, builds_on=None,
                              log=None):
    """``(structure, cell, relaxation)`` a force-constant stage of a SIESTA
    vibration is written with -- `freq`, or any stage after `relax` in a
    displacement sweep, asked of `vibration_render_kind` and never of a name
    (`engines/vibration.md` § 5.2a, § 5.9; until 2026-09-28 only a stage
    named `freq` came here, so a sweep's second stage measured the unrelaxed
    input): the sorted
    copy as given, and no cell or record of its own, when the ladder holds no
    `relax` stage; the coordinates that stage relaxed to, in the cell it ran
    in, when it does -- read from the run the stage builds on,
    ``builds_on``, the `Continuation` prep decided at its checkpoint 4a
    (`relax`'s newest attempt that is one to build on, or the `relax` run
    named with ``--from``, `engines/vibration.md` § 5.2a's table), through
    the one SIESTA output parser -- with that run's
    relaxation record (`parse.contract.relaxation_of_output`), of the same
    output and the same parse, which the deck's `vibration` block carries to
    the finish (§ 5.3) so nothing re-picks the attempt later -- and the
    structure leaves the input's own relaxation record behind, since it is
    about other coordinates (§ 5.2a).  The
    cell travels because the deck otherwise re-derives one around the new
    bounding box and shifts the atoms into it, and a relaxed geometry moved
    against the real-space grid is not stationary on that grid any more.
    Every other rung comes back unchanged.

    A force-constant stage before `relax` has ended on its own with exit
    code 0 is refused where that is decided, naming what to do first
    (`continuation.continuation_answer`, the one door -- asked here too when
    a caller hands no answer).  Refused here: a stage with no `relax` while
    the structure is not stated relaxed -- the box says *relax first* and
    the ladder holds nothing that would, so the description contradicts
    itself and is refused with the two ways out rather than measured at a
    geometry nobody chose (§ 2.2) -- and a run whose output holds no
    geometry this calculation's.
    """
    from ..pyscf.stages import VIBRATION_RELAX_STAGE, vibration_render_kind
    from .continuation import relax_stage_of
    if vibration_render_kind(pset.stage) != "vibration":
        return struct, None, None
    # THE LADDER'S RELAXATION, by the one role rule -- the name in any case
    # (`engines/stages.md` § 2), never an exact string (plan § 5w K12).
    relax = relax_stage_of(task)
    if relax is None:
        if not bool(getattr(pset[0].render_config(), "already_relaxed", False)):
            raise PrepError(
                f"the structure is not stated to be relaxed (already_relaxed "
                f"is false in the template) and this ladder has no enabled "
                f"`{VIBRATION_RELAX_STAGE}` stage to relax it -- a harmonic "
                f"analysis off a stationary point reports the wrong "
                f"frequencies (engines/vibration.md 2.2).  Either add the "
                f"`{VIBRATION_RELAX_STAGE}` stage before "
                f"`{pset.stage}` (Task setup, or task.json) and run "
                f"it first, or state already_relaxed = true in the template; "
                f"the finish then measures the forces at this geometry "
                f"and says whether the statement held.")
        return struct, None, None
    from ..runs import run_of
    if builds_on is None:
        # ASKED WITH NO ANSWER from the entry: the same door its checkpoint
        # 4a asks -- the default, `relax`'s newest attempt that is one to
        # build on, or the refusal naming what to do first.
        from .continuation import continuation_answer
        builds_on, refused = continuation_answer(base, task, pset.stage)
        if refused:
            raise PrepError(refused)
    token = stage_home(base, task, relax).token
    # THE COMMANDS, from the one composer: the calculation named, the mode
    # stated where its config sets none (`commands.run_first`).
    from .commands import block
    from .commands import run_first as _run_first
    run_first = block(_run_first(relax, base=base))
    # THE RUN IT BUILDS ON -- its attempt, or on the flat layout the folder
    # its files lie in.
    attempt = (base / builds_on.source if builds_on.source is not None
               else base)
    _run = run_of(attempt, stage=token)
    out = _run.stdout if _run is not None else None
    if out is None:
        raise PrepError(
            f"{builds_on.where()} holds no engine output of `{relax}`; "
            f"there is no geometry to read.  "
            f"Re-run it --\n{run_first}")
    from ..parse.engines.siesta import SiestaParser
    from ..parse.errors import ParseError
    try:
        traj = SiestaParser.parse(str(out))
    except ParseError as e:
        raise PrepError(
            f"{out.relative_to(base)} could not be read as a SIESTA run: {e}.  "
            f"Re-run the `{relax}` stage --\n{run_first}") from e
    frames = [fr for fr in traj.frames if fr.structure is not None]
    if not frames:
        raise PrepError(
            f"{out.name} holds no coordinate block: the `{relax}` "
            f"run never reached its first geometry.  Re-run it --\n"
            f"{run_first}")
    last = frames[-1]
    if list(last.structure.elements) != list(struct.elements):
        raise PrepError(
            f"{out.name} describes {last.structure.formula} in an order "
            f"that is not this calculation's sorted copy ({struct.formula}): "
            f"the `{relax}` stage ran a different structure.  Re-run "
            f"it --\n{run_first}")
    cell = last.lattice if last.lattice is not None else traj.lattice
    if log is not None:
        log.step(f"the geometry the `{pset.stage}` stage measures at")
        log.received(str(out.relative_to(base)),
                     f"{len(frames)} geometry step(s); the last is written as "
                     f"the deck's coordinates, in that run's cell")
    # THE ENGINE'S COORDINATES STATE THE ENGINE'S ORIGIN -- 0, set together
    # with them (`model/structure-periodicity.md` § 6.0) -- so this deck
    # applies nothing: the rule would re-centre them, moving the relaxed
    # geometry by the change in its span (-0.0168 Å on the H2 e2e).
    #
    # AND THE INPUT'S RELAXATION RECORD STAYS BEHIND (§ 5.2a): it is about
    # the input's coordinates, not these; this stage's own record rides the
    # `vibration` block.  The calculation record stays -- every stage reads
    # the one electronic state from it.
    from ..parse.contract import relaxation_of_output
    return (struct.replace(positions=np.asarray(last.structure.positions,
                                                dtype=float),
                           engine_offset=np.zeros(3),
                           info={k: v for k, v in (struct.info or {}).items()
                                 if k != "relaxation"}),
            (np.asarray(cell, dtype=float) if cell is not None else None),
            relaxation_of_output(out, traj, engine=str(task.engine)))


def _structure_for(task, base: Path):
    """The structure this calculation is *of*, from the reference in the
    description (`stages.md` § 6.3 — a reference plus a witness, never a copy).

    Looked for beside the calculation first and at the recorded path second, so
    a folder carried to a cluster with its structure alongside resolves without
    the original tree existing there.

    **The witness is checked and the mismatch is loud**: a description opened
    against a structure that has since changed would otherwise build a
    different calculation under the same id — § 1's second failure mode.
    """
    from ..workingcopy_structure import StructureCodec
    codec = StructureCodec()
    src = Path(task.structure.source)
    # The codec, not the bare loader: describe writes the structure as the
    # pair when it carries metadata (--vacuum lives NOWHERE a bare .xyz can
    # put it), and the codec applies the .molstruct.json when present and
    # changes nothing when absent.  The suffix-corrected candidate is the
    # codec's own naming rule (a .pdb source travels as <stem>.xyz).
    for candidate in (base / src.name,
                      (base / src.name).with_suffix(codec.GEOMETRY_SUFFIX),
                      src):
        if candidate.is_file():
            struct = codec.load(candidate)
            break
    else:
        raise PrepError(
            f"the structure this calculation describes is not here: "
            f"{task.structure.source!r}. `task.json` records a REFERENCE to it "
            f"(engines/stages.md § 6.3), so `prep` needs the file to be either "
            f"beside the calculation or still at that path.")
    if struct.formula != task.structure.formula or \
            struct.n_atoms != task.structure.atoms:
        raise PrepError(
            f"the structure has changed since this calculation was described: "
            f"the description witnesses {task.structure.formula} with "
            f"{task.structure.atoms} atoms, and {candidate} now holds "
            f"{struct.formula} with {struct.n_atoms}.\n"
            f"  Describing again is the honest fix -- rendering this deck would "
            f"build a different calculation under the same id.")
    return struct


# --------------------------------------------------------------------- #
#  The rung -- what a kind adds to the one table of steps               #
#  (`script-preparation.md` § 3.0)                                       #
# --------------------------------------------------------------------- #

def _at_no_point(cfg, volts):
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
    #: calculation's own, or the rung's (a SIESTA vibration's relax rung is
    #: an optimisation, `_rung_kind`).
    render_kind: str
    #: ``cfg -> cfg``: the config an element renders from -- a lead's own
    #: label, the junction's electronic state folded in.
    configure: Callable = lambda cfg: cfg
    #: ``(element, cfg) -> {keyword: value}``: what else `spec_for` is told
    #: -- a vibration's block, the relaxation it builds on, the cell; the
    #: junction's electronic state.
    spec_extra: Callable = lambda element, cfg: {}
    #: ``((folder, volts), ...)``: the folders each holding a deck written at
    #: that point -- a scan's (`engines/transport.md` § 2a.10-11) -- each
    #: with its own wrapper and attempt ladder; the element's own deck is
    #: written at the first.  Empty: one deck, in the element's folder.
    points: tuple = ()
    #: ``(cfg, volts) -> cfg``: a deck's config at its point -- the rung
    #: answers the bias, the one value the catalogue does not hold
    #: (`engines/template.md` § 6.4).
    at_point: Callable = _at_no_point
    #: ``(plan) -> None``: the data files, when the kind's come from
    #: somewhere other than the engine's library (transport: the citation);
    #: ``None`` is the engine's own step (`EngineSeam.provide_data`).
    provide_data: Optional[Callable] = None
    #: ``job -> job``: what the rung's job carries forward and runs, where
    #: the kind answers it per rung -- a transport rung's restart files, the
    #: transmission's program.
    job_facts: Callable = lambda job: job


def _described_rung(base, task, pset, *, seam, template_text, sweep,
                    log, plan, continuation) -> Rung:
    """A calculation of its described structure, one deck per element --
    every kind with no steps of its own."""
    return Rung(struct=_structure_for(task, base),
                render_kind=task.calculation)


def _siesta_vibration_rung(base, task, pset, *, seam, template_text, sweep,
                           log, plan, continuation) -> Rung:
    """A SIESTA vibration's rung (`engines/vibration.md` § 5.2a, § 5.9).

    The atoms in the order the force-constant run needs -- the held ones
    first, so it nudges one contiguous range -- on a COPY, whose permutation
    is recorded beside the calculation and inverted by the job's finish
    (`model/overview.md` § 2.2); the input order never reaches the engine
    and the sorted order never reaches a person.  The `relax` rung is the
    relaxation deck; every other rung is written at the coordinates the
    relax rung reached, in that run's own cell, so the force constants are
    taken on the grid the geometry was relaxed on, with the relaxation
    record its `vibration` block carries to the finish (§ 5.3)."""
    # A DISPLACEMENT SWEEP NEEDS A DIRECTORY PER STAGE (§ 5.9): SIESTA names
    # its force constants and the finish its spectrum by the label alone, so
    # two force-constant stages sharing the flat layout's one directory
    # would overwrite the first one's result -- counted over the stages the
    # description runs: a disabled one is never prepped (plan W38 F5).
    from ..spectra.displacement_sweep import stages_share_a_directory
    if stages_share_a_directory(task, include_disabled=False):
        from ..pyscf.stages import force_constant_stages
        _fc = force_constant_stages(task, include_disabled=False)
        raise PrepError(
            f"this flat calculation describes "
            f"{len(_fc)} force-constant stages "
            f"({', '.join(_fc)}), and in the flat "
            f"layout every stage writes the same <label>.FC and "
            f"<label>.spectra.json -- each would overwrite the last one's "
            f"result.  A displacement sweep needs the hierarchical layout "
            f"(engines/vibration.md 5.9): describe it with --shape "
            f"hierarchical, or keep one force-constant stage here.")
    from ..transport.sort import sort_by, write_permutation
    _sorted = sort_by(_structure_for(task, base), "held-first")
    _perm_path = write_permutation(base, _sorted, plan=plan)
    struct = _sorted.structure
    log.step("the atom order the engine needs")
    log.produced(_perm_path.name,
                 f"key held-first, {struct.n_atoms} atoms -> {_perm_path.name}")
    render_kind = _rung_kind(task, pset.stage)
    struct, cell, relaxed_by = _vibration_stage_geometry(
        base, task, pset, struct, builds_on=continuation, log=log)
    finishes = render_kind == "vibration"
    # THE FINISH'S FORCE CRITERION IS THE RELAXATION'S OWN (plan § 5w K4,
    # M11 SS-C6): `relax_force_tol` is read by the relaxation rung alone, so
    # it is that rung's value -- the criterion the reference geometry was
    # relaxed to -- and the template's when the structure is stated relaxed.
    # The force-constant stage's own copy was read until 2026-09-30, so a
    # preset that relaxed at 0.05 judged at 0.01.
    from ..resolve import resolved_ladder
    from ..template import stage_role
    criterion = next(
        (getattr(c, "relax_force_tol", None) for n, c in resolved_ladder(
            template_text, task, seam.config_cls)
         if stage_role(str(task.engine), "vibration", n) == "relaxation"),
        getattr(pset[0].render_config(), "relax_force_tol", None))

    def spec_extra(element, cfg):
        return {**({"cell": cell} if cell is not None else {}),
                # The finish's facts ride a force-constant deck -- never a
                # benchmark trial's: a spectrum of capped SCFs is a spectrum
                # of nothing (§ 5.5).
                **({"vibration": _vibration_block(pset.stage, cfg,
                                                   relaxed_by,
                                                   criterion=criterion)}
                   if finishes and not element.is_trial else {}),
                # THE RELAX STAGE'S RECORD reaches every deck at its
                # geometry -- a trial's too, which carries no finish (V1.36).
                **({"relaxed_by": relaxed_by}
                   if relaxed_by is not None else {})}
    return Rung(struct=struct, render_kind=render_kind, spec_extra=spec_extra)


def _transport_rung_of(base, task, pset, *, seam, template_text, sweep,
                       log, plan, continuation) -> Rung:
    """A transport rung (`engines/transport.md`): its four steps of its own.

    **compose** -- the junction and its leads, composed from the cited
    relaxation, or read from the record that travelled beside ``task.json``
    (:func:`_composed_for_prep`); the rung's structure is the junction or
    the lead taken out of it (:func:`_transport_parts`).  **electronic
    state** -- decided once, on the junction, folded into every rung.
    **points** -- a scan's rung writes one deck per bias point, each in its
    own folder (`transport.stages.rung_containers`).  Its data files come
    from the citation, and its job carries the rung's own restart files and,
    for the transmission, TBtrans's program.  The fifth, **gather** -- each
    rung's inputs from the runs upstream -- is decided with what every stage
    continues from, at the entry's checkpoint 4a (:func:`gather_sources`),
    and copied into the attempts the steps open."""
    stage = pset.stage
    if not stage:
        from .commands import name_a_stage
        from .materialize import described_refs
        raise PrepError(
            "a transport prep names its rung: the composite's stages render "
            "separately, in dependency order (engines/transport.md); "
            + name_a_stage("prep", "run", described_refs(base, task),
                           base=base))
    if sweep is not None:
        raise PrepError(
            "a transport calculation takes no parameter sweep or "
            "translation.  Its parameters come from its own template and "
            "each rung's run card, and its one axis is the bias -- a list "
            "in task.json, rendered as one deck per point "
            "(engines/transport.md 2a.10: single bias is the degenerate "
            "case of that axis, one point at zero).")
    composed = _composed_for_prep(base, task, plan)
    struct, configure, state = _transport_parts(task, stage, composed,
                                                pset[0].render_config())
    from ..transport.stages import rung_containers, warm_declaration
    points = tuple((d, v) for d, v in rung_containers(base, task, stage)
                   if v is not None)
    citation = task.slots["junction"]

    def provide_data(plan):
        # The pseudopotentials travel with the citation, screened against
        # the config the rung's decks render from: what it checks is each
        # file's XC family against the functional the run will ask for.
        _transport_provide_pseudos(composed.sorted.structure,
                                   configure(pset[0].render_config()), base,
                                   citation, plan=plan)

    def job_facts(job):
        # The rung's own restart files (`transport.stages.warm_declaration`:
        # the seed's and the device's; the leads and the transmission carry
        # none) -- and TBtrans post-processes the device run from its own
        # deck, its binary riding the allocation road into the wrapper
        # (`model.Resources.program`, `engines/transport.md` § 6.1b).
        return dataclasses.replace(
            job, warm=warm_declaration(stage, task.label, base),
            resources=(dataclasses.replace(job.resources, program="tbtrans")
                       if stage == "transmission" else job.resources))
    return Rung(struct=struct, render_kind="transport", configure=configure,
                spec_extra=lambda element, cfg: {"state": state},
                points=points, at_point=_at_bias, provide_data=provide_data,
                job_facts=job_facts)


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
    and none of the three files (W55 B1: until 2026-10-05 the entry and the
    five steps each read the description, the template and the record, and
    resolved the stage twice)."""
    task: "Task"
    template: Path
    template_text: str
    #: the machine's record, its activation checked
    environment: "Environment"
    allocation: Resources
    pset: "ParameterSet"

    @property
    def shape(self):
        """The layout, as the description states it -- handed to every layer
        that asks (`materialize.shape_of`'s rule)."""
        from ..paths import Shape
        return Shape.named(self.task.shape)


def _resolve_stage(stage, *, task, template, template_text, environment,
                   allocation=None, chosen=None, sweep=None, pins=None,
                   translation=None) -> Resolved:
    """Step 2, once (`script-preparation.md` § 3.0): the allocation folded --
    the caller's ask over the description's, and its reporting policy -- and
    the stage's elements resolved from the template, the stage, the sweep,
    the pins and the machine's record.

    Asked by the prep entry at its checkpoint 4, where the job it makes is
    placed (`job-system.md` § 5.0), and by :func:`prep_calculation` for a
    caller that holds no answer (:func:`_read_and_resolve`).  ``template``
    ``None`` is refused: the folder is a template PLUS a description."""
    from ..resolve import ResolveError, resolve
    from ..task import FILENAME as TASK_FILENAME
    from ..template import template_filename
    if template is None:
        raise PrepError(
            f"no {template_filename(task.label)} beside {TASK_FILENAME}. The portable "
            f"folder is a template PLUS a description (project-layout.md § 2.1) "
            f"and `prep` rebuilds the config from the template.")
    # THE DESCRIPTION'S OWN ALLOCATION, under the caller's (2026-08-24).
    # `task.json` carries the queue, the wall and the memory a person chose
    # for this calculation (`task.Allocation`), so a prepped bundle needs
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
    allocation = _under_description(allocation, task.allocation, chosen)
    # AND THE DESCRIPTION'S REPORTING POLICY, at the same seam and for the
    # same reason: the file says when this calculation should speak up, so a
    # prepped bundle needs no flag to know it.  A separate line rather than a
    # third field on the helper above -- that one is about the ALLOCATION's
    # precedence against a CLI flag, and notify has no flag to lose to.
    allocation = _with_notify(allocation, task.notify)
    try:
        pset = resolve(template_text, task, engine_seam(task.engine).config_cls,
                       allocation=(allocation or Resources()), stage=stage,
                       sweep=sweep, pins=pins, translation=translation,
                       environment=environment)
    except ResolveError as exc:
        raise PrepError(str(exc)) from exc
    return Resolved(task=task, template=Path(template),
                    template_text=template_text, environment=environment,
                    allocation=allocation, pset=pset)


def _read_and_resolve(base: Path, stage, *, target, allocation, chosen, sweep,
                      pins, translation) -> Resolved:
    """Steps 1 and 2 for a caller of :func:`prep_calculation` that holds no
    answer -- each file read once, here: the machine's record (its
    activation checked), the description and its template.  The prep entry
    reads them at its own checkpoints and hands its :class:`Resolved` over."""
    from ..task import FILENAME as TASK_FILENAME
    from ..task import read_task
    from ..template import find_template
    if not base.is_dir():
        raise PrepError(f"calculation folder not found: {base}")
    desc = base / TASK_FILENAME
    if not desc.is_file():
        raise PrepError(
            f"no {TASK_FILENAME} in {base}. `prep` turns a DESCRIPTION into a "
            f"runnable directory; write one first with `jobset init`.")
    # READ, CHECK, THEN WRITE.  A record that does not state how to enter the
    # named machine's environment is refused before anything is on disk --
    # the snapshot included, or the remedy's re-copied record would then
    # contradict it (W52, and its fix's review).
    environment = machine_record(base, target)
    require_activation(target, environment, base=base)
    task = read_task(desc)
    # THE one template door (`template.find_template`, architecture.md
    # § 3.2): the folder's one template, named for the label.
    try:
        template = find_template(base, task.label)
    except ValueError as exc:
        raise PrepError(str(exc)) from exc
    return _resolve_stage(
        stage, task=task, template=template,
        template_text=(template.read_text(encoding="utf-8")
                       if template is not None else None),
        environment=environment, allocation=allocation, chosen=chosen,
        sweep=sweep, pins=pins, translation=translation)


def prep_calculation(base_dir, stage: Optional[str] = None, *,
                     allocation=None, env: str = None,
                     emit_sbatch: bool = True,
                     sweep=None, pins=None, translation=None,
                     target: Optional[str] = None,
                     chosen=None,
                     opened: Optional[list] = None,
                     findings: Optional[list] = None,
                     continuation=None,
                     gather=None,
                     plan=None,
                     resolved: Optional["Resolved"] = None) -> List[Path]:
    """**`prep`, entire** — the five steps of `project-layout.md` § 2.3.1, in
    the order it calls *forced rather than chosen*.

    ``continuation`` is what the stage builds on, as :func:`prep_stage`
    decided it at checkpoint 4a (`continuation.Continuation`, `job-system.md`
    § 5.4) -- the run a stage continues from, or the `relax` run a
    force-constant stage takes its geometry from -- or ``None``: it starts
    from the calculation's structure.  The files it carries are counted
    where the plan's row is (:func:`_carrying`), and the attempt is opened
    ONCE, with them (`materialize.prepare_attempt`).  ``gather`` is a
    transport rung's inputs from the runs upstream, decided at 4a too
    (:func:`gather_sources`), copied into the attempts opened.

    1. **resolve the machine** — READ its record, persist the snapshot as
       ``environment.json``; refuse when no record answers (§ 3.1);
    2. **resolve the parameters** — the description ⊕ this stage ⊕ the sweep ⊕
       the pins, into a :class:`~molbuilder.resolve.ParameterSet`;

    -- each file read once, and the stage resolved once (:class:`Resolved`):
    ``resolved`` is the entry's, answered at its own checkpoints and handed
    over (`prep_stage`, `job-system.md` § 5.0, checkpoint 4); with none,
    steps 1 and 2 are answered here (:func:`_read_and_resolve`), from
    ``allocation``, ``chosen``, ``sweep``, ``pins`` and ``translation`` --
    which a caller handing ``resolved`` has already folded into it;
    3. **render the deck(s)** — one per element of that set;
    4. **render the wrapper**;
    5. **build the run directory**.

    **Steps 2 and 3 did not exist here until 2026-08-11**, and their absence was
    stated by the code as a refusal: `prep` demanded that the decks already be
    in the bundle root, because they were finished at ``molbuilder fdf`` time on
    a machine that could not know the rank count. That is the *one real
    migration* — the producer ran at *produce* and belonged at *prep* — and
    steps 1 and 3 are now on the same side of the split, which is what
    § 2.3.1's *"step 3 cannot precede step 1"* was always about.

    Every prep writes the PIPELINE LOG -- what each step received, decided
    and produced -- beside this prep's ``STAGE-PLAN.md``: an observer of the
    pipeline, never a step in it (`script-preparation.md` § 4.5; until
    2026-10-05 only ``--pipeline-log`` wrote it, and the Task setup tab never
    did).

    ``opened``, when given, receives the :class:`~molbuilder.jobset.materialize.Attempt`
    reports of the attempts this prep opened.  A caller reporting on the
    attempt reads its freshness from here: opening it a second time finds it
    already there, unlaunched, and calls it reused.

    ``findings``, when given, receives what each deck's checks said
    (`script_emit.prepare_deck`) -- the terminal also reads them on stderr as
    each deck renders; the Task setup tab reads them from here, through the
    one prep entry's answer.

    ``plan`` (`jobset.planned.Plan`) receives everything this prep writes,
    and is carried out by the caller -- the entry saves the folder's state
    between the two (`prep_stage`, `job-system.md` § 5.0); with none, the
    plan is made here and carried out once it stands.  Either way nothing
    is written until every step has passed: a refusal writes nothing.
    While the steps plan, every reader of the calculation's pseudopotentials
    reads the folder as the plan will leave it (`pseudos.PLANNED`).

    Returns the per-job directories. Raises :class:`PrepError`.
    """
    from ..pseudos import PLANNED
    from .planned import Plan
    base = Path(base_dir).resolve()
    if resolved is None:
        resolved = _read_and_resolve(base, stage, target=target,
                                     allocation=allocation, chosen=chosen,
                                     sweep=sweep, pins=pins,
                                     translation=translation)
    own = plan is None
    plan = Plan() if own else plan
    scope = PLANNED.set(plan)
    try:
        dirs = _plan_calculation(
            base, stage, resolved, env=env, emit_sbatch=emit_sbatch,
            sweep=sweep, target=target, opened=opened, findings=findings,
            continuation=continuation, gather=gather, plan=plan)
    finally:
        PLANNED.reset(scope)
    if own:
        plan.carry_out()
    return dirs


def _plan_calculation(base: Path, stage: Optional[str], resolved: "Resolved",
                      *, env: str = None,
                      emit_sbatch: bool = True,
                      sweep=None,
                      target: Optional[str] = None,
                      opened: Optional[list] = None,
                      findings: Optional[list] = None,
                      continuation=None,
                      gather=None,
                      plan) -> List[Path]:
    """The steps of :func:`prep_calculation`, planned into ``plan``: nothing
    is written here.  Steps 1 and 2 arrive answered (``resolved``): this
    snapshots the machine's record it was handed and plans the rest -- it
    reads none of the description, the template or the record again."""
    from ..pipeline_log import PipelineLog, config_rows
    from ..task import FILENAME as TASK_FILENAME
    # The container spelling is materialize's (the naming authority): ONE
    # function places the sweep's record and the trials' directories, so the
    # two can never disagree (A-1/A-2).  The log's own home is the same
    # container, spelled by the same function.
    from ..paths import bench_container
    from ..paths import Shape

    task, environment = resolved.task, resolved.environment
    template_path, pset = resolved.template, resolved.pset
    allocation = resolved.allocation

    # ---- 1. the machine: its record, read and checked before this -------- #
    # (`_read_and_resolve`, or the entry's checkpoint 4) -- snapshotted here,
    # the calculation's copy of the record it was handed.
    set_machine(base, target, environment=environment, plan=plan)

    # THE LOG OPENS HERE, and not before: its NAME carries the stage token, and
    # the token needs the description.  Everything step 1 did is still in hand,
    # so nothing is lost by writing that phase a moment later than it ran.
    # It is HELD, and written with the rest of the plan once every step has
    # passed (`script-preparation.md` § 4.5): a refused prep writes no log.
    _token = stage_home(base, task, stage).token
    _record = (base / bench_container(Shape.named(task.shape), _token)
               if sweep is not None else base)
    log = PipelineLog.open(_record, label=task.label, token=_token,
                           engine=task.engine, shape=task.shape)
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
    from ..runtime_config import config_provenance, format_provenance
    log.produced("config", "which file supplied each setting")
    log.text(format_provenance(config_provenance(project_dir=base)))
    seam = engine_seam(task.engine)
    log.phase("STEP 2 · RESOLVE — the values for this rung")
    log.received(template_path.name, f"{len(pset[0].provenance)} fields")
    log.received("stage", pset.stage or "(no ladder)")
    log.received("allocation", _flat_resources(allocation or Resources()))
    for _el in pset:
        _rows = config_rows(_el.values, _el.provenance,
                            _el.render_config())
        _decided = [r for r in _rows if r[2] != "template"]
        log.step(f"{_el.label} — what this rung decided")
        for _n, _v, _s in _decided:
            log.chose(_n, _v, _s)
        log.step(f"{_el.label} — the rest, as the template declares")
        for _n, _v, _s in _rows[len(_decided):]:
            log.chose(_n, _v, _s)
    log.produced("ParameterSet", f"{len(pset)} element(s) -> spec_for")

    # ---- 3. the structure: the kind's rung ----------------------------- #
    # WHAT THE DECKS DESCRIBE, and what the kind adds to the steps after
    # this one, from its one record (`script-preparation.md` § 3.0): the
    # structure as described, unless the kind composes or reorders it -- a
    # transport junction and its leads; a SIESTA vibration's held-first copy
    # and the geometry its relax rung reached.
    rung = RUNGS.get((str(task.engine), str(task.calculation)),
                     _described_rung)(base, task, pset, seam=seam,
                                      template_text=resolved.template_text,
                                      sweep=sweep, log=log, plan=plan,
                                      continuation=continuation)
    # The DATA FILES the engine will open, before any deck is written: a
    # missing pseudopotential is a run that cannot start, and finding that out
    # here costs a second (project-layout.md § 2.6).  Idempotent -- what is
    # already in the folder is left alone.  The elements come from the
    # structure, which has just been checked against the description's witness.
    if rung.provide_data is not None:
        with _user_error_as_prep(), _calling(
                "provide_data", engine=task.engine):
            rung.provide_data(plan)
    elif seam.provide_data is not None:
        with _user_error_as_prep(), _calling(
                "provide_data", engine=task.engine):
            seam.provide_data(rung.struct, pset[0].render_config(), base,
                              plan=plan)
    token = stage_home(base, task, pset.stage).token
    jobs: List[Job] = []
    from ..paths import bench_container
    from .materialize import (trial_dir,
                              trial_work_dir)
    from ..paths import Shape as _Shape
    _shape = _Shape.named(task.shape)
    # ONE line per unique finding, however many trials repeat it (user,
    # 2026-08-28, O5).  The gate still FIRES per deck -- every trial is
    # validated and its own <deck>.validation.txt carries its findings --
    # but sixteen identical thin-vacuum warnings on one terminal is
    # noise wearing a safety vest.  Scoped to THIS loop (not a module
    # global) so a long-lived server process cannot quietly swallow a
    # later prep's warnings -- and to this THREAD: the gate's report reads
    # its stream from `validation.REPORT_STREAM`, a context variable.  It
    # swapped the process's `sys.stderr` until W55 D8, and two preps at once
    # left one's filter as the server's stderr.
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
            script = _rf(element.label, seam.suffix, token or None)
            # WHERE THIS ELEMENT'S FILES GO -- its own directory, never the
            # bundle root (user, 2026-08-24; `project-layout.md` § 1.0 always
            # said it: "only rendered files and copies go down to where the
            # engine runs").  What stood here rendered every deck, wrapper,
            # validation report and molwatch log AT THE ROOT and symlinked
            # them down -- 50 files at the root of a ten-trial sweep, the
            # hierarchy inverted into a veneer of links.  The directory is the
            # same one `job_dir_names` will answer for this job, computed from
            # the same two facts (token + trial-ness), so the deck is born
            # where the launch will look for it.
            if element.is_trial:
                # ONE RULE, asked -- not composed again from the same two
                # facts.  `materialize.trial_dir` is what `job_dir_names`
                # itself uses, so the deck is born where the launch looks by
                # CONSTRUCTION rather than by a comment promising it.
                #
                # The dir is named by the JOB's name -- the sweep coordinate,
                # `_job_for`'s own first rule -- never by the element's
                # RELABELLED SystemLabel (`<calc-label>-<coord>`): the two
                # differ by the calculation prefix, and using the label here
                # wrote decks into `bench-JOB-G1K1C1/` while every reader
                # asked `job_dir_names` and looked in `bench-G1K1C1/`.
                from ..resolve import point_token as _pt
                _trial = _pt(element.point)
                _jdir = trial_work_dir(
                    base / trial_dir(_shape, token, _trial), _shape)
            else:
                _trial = None
                _sd = _shape.stage_dir(token) if token else "."
                _jdir = base if _sd == "." else base / _sd
            plan.folder(_jdir)
            if element.is_trial:
                # WHERE A TRIAL'S DECK IS BORN, ITS FOLDERS SAY WHAT THEY ARE
                # (`project-layout.md` § 1.4a), as a stage's attempt does.
                from .materialize import mark_run
                mark_run(base, _jdir, plan)
            # The deck is rendered from values ⊕ THIS element's allocation, so it
            # records the rank count it actually assumed.  Rendering from the
            # values alone emits `mpi_np auto` and the launch check then refuses a
            # deck that `prep` itself just made -- which is how this was found.
            # The stage's artifact TOKEN reaches the emitter here, as a RENDER
            # ARGUMENT (C7).  It feeds three names -- the deck, the engine's own
            # log, and the molwatch log -- and leaving it unset made two rungs of
            # one calculation write to a single `<label>.molwatch.log`.  `prep`
            # holds the StageRef, so `prep` says the word; no config field carries
            # it, for either engine (`stages.md` § 1.1).
            # THE CONFIG THIS ELEMENT RENDERS FROM, as the kind's rung says
            # (a lead's own label, the junction's state) -- identity for
            # every kind with no steps of its own.
            cfg = rung.configure(element.render_config())
            if element.is_trial:
                # The deck's OWN identity line carries the trial label -- this,
                # not the filename, is what keys SIESTA's warm files away from
                # the real run's (project-layout.md § 2.3.2).
                with _calling("relabel", engine=task.engine,
                              where=element.label):
                    cfg = seam.relabel(cfg, element.label)
                # The forced cold has ONE setter -- the measurement pin
                # (`_MEASUREMENT_PINS`, resolved with provenance "pin") --
                # and ONE verifier, the submission door
                # (`agreement.check_trial_starts_cold`; user-settled
                # 2026-08-21: prep bakes the intent, submission determines
                # the actual state).  A hard replace stood here as a second,
                # provenance-invisible setter until then (Q6a).
            # The stage's artifact token is a RENDER ARGUMENT (C7, 2026-08-12):
            # `prep` holds the StageRef, so `prep` says it, per call -- the
            # config field that used to carry it is gone, and the emitter never
            # learns the word (engines/stages.md § 1.1).
            # The render's refusals (missing pseudos above all) are user-fixable
            # and translate to PrepError -- see _user_error_as_prep
            # (2026-08-12 plan A8).
            # STEP 3, WHOLE, IN ONE CALL: validate the settings, render the deck,
            # write it through the one writer (which keeps the reader's USER-CUSTOM
            # block), then read the file back and refuse one that does not say what
            # it was meant to say.  The conductor says WHEN; the framework owns the
            # order (`script-preparation.md` § 4.3).
            log.phase(f"STEP 3 · DECK — {script}")
            log.received("config", f"{type(cfg).__name__}"
                                   + (f", stage_token={token}" if token else "")
                                   + (f", trial {element.label}"
                                      if element.is_trial else ""))
            # EVERY DECK THIS ELEMENT WRITES: the one in its own folder, and
            # one in each folder of the rung's points, each written at its
            # point -- the element's own at the first (`Rung.points`).
            spec = None
            for _out, _volts in ([(_jdir, rung.points[0][1] if rung.points
                                   else None)] + list(rung.points)):
                plan.folder(_out)
                _at = rung.at_point(cfg, _volts)
                with _user_error_as_prep():
                    with _calling("spec_for", engine=task.engine,
                                  where=script):
                        _spec = seam.spec_for(
                            rung.struct, _at, stage_token=(token or None),
                            calculation=rung.render_kind,
                            **rung.spec_extra(element, cfg),
                            # A TRIAL'S DECK NAMES ITS OWN LAUNCH (plan § 5w
                            # K12) -- the bench lane is SIESTA's alone.
                            **({"trial": _trial} if _trial else {}))
                    _sc.prepare_deck(_spec, rung.struct, _at, _out / script,
                                     log=log, dest_dir=base,
                                     findings=findings, plan=plan)
                spec = spec or _spec
            if seam.sibling_artifacts is not None:
                with _calling("sibling_artifacts", engine=task.engine,
                              where=script):
                    seam.sibling_artifacts(rung.struct, cfg, _jdir / script,
                                           kind=rung.render_kind, plan=plan)
            with _calling("label_of", engine=task.engine):
                _label = seam.label_of(cfg)
            # THE PROGRESS LOG IS SEEDED WHERE ITS RUN WILL WRITE IT: beside
            # each deck an attempt ladder runs -- the element's own, or for
            # a scan each point's, never the stage folder above them, where
            # nothing would ever write to it again.
            for _seed_dir in ([d for d, _ in rung.points] or [_jdir]):
                _seed_trajectory_log(rung.struct, cfg, _seed_dir,
                                     engine=task.engine, label=_label,
                                     token=(token or None),
                                     frame=spec.engine_frame,
                                     relaxes=(rung.render_kind
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
                finish=(None if element.is_trial else spec.finish))))
    finally:
        REPORT_STREAM.reset(_scope)
    if _once.dropped:
        print(f"  (each warning shown once; {_once.dropped} repeat(s) "
              f"across this prep's other decks suppressed -- every deck's "
              f"own .validation.txt carries its full findings)",
              file=_sys.stderr)

    # ---- 4 + 5, and the record floor 3 leaves behind -------------------- #
    # ``kind`` is INTENT, not length (review 2026-08-12: a one-point grid is
    # still a benchmark).  A sweep's whole record — its job-set, its plan,
    # later its verdict — lives in the stage's ``bench/`` container
    # (job-contracts.md § 6.3's Directories row, the cross-layer authority),
    # so two stages' benchmarks can never collide.  The ROOT job-set.json
    # is the RUN's plan and MERGES per stage, so prepping `tight` no longer
    # erases `coarse` — status, and the cross-stage ``--from`` carry whose
    # pair rule needs the source job on file, read the whole ladder.
    kind = "sweep" if sweep is not None else "ladder"
    js = JobSet(name=task.label, engine=task.engine, kind=kind,
                shared=_shared_for(base, seam, engine=task.engine,
                                   plan=plan),
                jobs=jobs)
    if kind == "sweep":
        record_dir = base / bench_container(Shape.named(task.shape), token)
        plan.folder(record_dir)
    else:
        record_dir = base
        js = _merge_run_jobset(
            base / JOBSET_FILENAME, js,
            # The CURRENT ladder bounds what the merge keeps.
            # (`read_task` refuses a description without stages and
            # `Task` refuses an empty tuple, so the stage-less fallback
            # arms that stood here were unreachable -- U6 close.)
            ladder=frozenset(s.name for s in task.stages))
    # WHAT THE HAND-OVER CARRIES, counted where the plan's row is: the run
    # was decided at 4a, and its files follow from the job this prep writes
    # against the one that run belongs to (`job-system.md` § 5.4).
    this_prep = None
    if kind == "ladder" and stage:
        if continuation is not None:
            continuation = _carrying(js, base, stage, continuation,
                                     resolved.shape)
        this_prep = _what_this_prep_takes(stage, continuation, gather)
    log.phase("FLOOR 3 · THE JOB-SET — what was declared to the runner")
    log.received("kind", kind)
    for _j in js.jobs:
        log.produced(_j.name, f"{_j.script}  "
                              f"{_flat_resources(_j.resources)}")
    log.produced(JOBSET_FILENAME, str(record_dir / JOBSET_FILENAME))
    # The allocation is NOT passed on: every job already carries its own
    # resolved resources, per element (generator.md § 5).  Passing it made
    # prep_jobset re-apply the BASE allocation over every job — the review's
    # "stomp": each trial's wrapper rendered with the base rank count
    # instead of its own translated G·K.
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
    _points = [d for d, _ in rung.points]
    dirs = prep_jobset(js, base, env=env, emit_sbatch=emit_sbatch,
                       record_dir=record_dir, log=log,
                       machine_record=environment,
                       render=(None if kind == "sweep"
                               else {j.name for j in jobs}),
                       points={j.name: _points for j in jobs}, plan=plan,
                       shape=resolved.shape, this_prep=this_prep)

    # ---- THE ATTEMPT, because PREP is what sets a stage up to run ------- #
    #
    # `prepare_attempt` says the rule in its own words: *"Set ONE stage up to
    # run ... preparing is still design and the split from starting is what
    # gives you somewhere to look before committing cluster time."*  Resolve
    # the next `run-<n>`, create it, link the deck in -- all of it design,
    # none of it spending a queue slot.  `launch` starts what prep set up and
    # records that it ran; `_launch_dir` REFUSES a hierarchical stage with no
    # attempt open (C5, 2026-08-12) precisely because opening one is not its
    # job.
    #
    # It ran in ONE CALLER -- the CLI, which called this and then opened the
    # attempt itself, under a comment calling that "the CLI's OWN addition".
    # So `prep_calculation` returned a folder that `launch` refuses, and every
    # other caller got exactly that: the browser's Prep button rendered the
    # decks, reported success, and told the person to launch something the
    # launcher would not take.  A verb that is only finished by one of its
    # callers is not a verb.
    #
    # `project-layout.md` said the opposite in two places -- *"launch adds
    # attempts"* -- which is why the browser's half was not obviously wrong:
    # it implemented the document.  The document is corrected with this.
    #
    # TRIALS ALREADY DID THIS.  A bench point's deck is rendered straight into
    # its attempt (`trial_work_dir` above), so a sweep has been complete since
    # § 1.5a gave trials attempts.  Only the ladder rung was left half-done --
    # the asymmetry was inside this function, not between two surfaces.
    if kind == "ladder" and stage:
        # ONE ATTEMPT LADDER PER POINT for a scan (`04_device/v0.2/run-<n>`,
        # layout ruled 2026-08-29), because the transmission at v reads the
        # device at v; the stage's own folder otherwise.
        reports = _open_attempts(js, base, stage,
                                 containers=_points or (None,),
                                 continuation=continuation, plan=plan,
                                 shape=resolved.shape)
        if opened is not None:
            opened.extend(reports)
        if not resolved.shape.keeps_attempts_as_directories:
            # THE FLAT LAYOUT keeps no attempt: the stage's own record of
            # what it continues from, beside its files (§ 1.6.3).
            _flat_continued_from(base, task, stage, continuation, plan)
        # A TRANSPORT RUNG'S GATHER, into the attempt opened in each
        # container -- one per bias point for a scan.
        for container, _volts, inputs in (gather or ()):
            att = next((Path(a.dir) for a in reports
                        if Path(a.dir).parent.resolve()
                        == Path(container).resolve()), None)
            if att is not None:
                _gather_into(base, att, inputs, plan)

    # THE LOG, then THE PLAN'S ROW LAST -- the moment the stage is prepped
    # (`job-system.md` § 5.0, checkpoint 6): a prep that stops before this
    # line has not prepped the stage.
    log.close()
    plan.text(log.path, (plan.read_text(log.path) if plan.is_file(log.path)
                         else "") + log.held_text())
    js.write(record_dir / JOBSET_FILENAME, plan=plan)
    return dirs


def _open_attempts(js: JobSet, base: Path, stage: str,
                   containers: Sequence[Optional[Path]] = (None,), *,
                   continuation, plan, shape) -> List:
    """Open this rung's attempt(s) — **step 6, and both arms take it.**

    Returns the :class:`~molbuilder.jobset.materialize.Attempt` reports, in
    ``containers`` order, or ``[]`` when the shape keeps no attempts.

    ``containers`` is where each attempt ladder lives. The default — one
    ``None`` — means the stage's own directory, which is every kind but a
    transport bias scan; that scan keeps one ladder PER POINT
    (``04_device/v0.2/run-<n>``; layout ruled 2026-08-29) because the
    transmission at *v* reads the device at *v*, never another point's
    converged state.  It is a LIST rather than a flag for that reason: the
    number of ladders is data the caller already holds, not a shape this
    function should re-derive.

    Flat keeps no attempt directories at all (§ 1.5a): its container IS the
    run, and ``prepare_attempt`` refuses it by name — so ``shape``, the
    description's, is read here and the refusal never has to fire.

    Idempotent by ``resolve_attempt``'s rule — reuse the last attempt until it
    has been launched, then open the next.  ``continuation`` -- REQUIRED, it
    decides which run the attempt starts from and what it carries
    (`code-audit.md` D1); ``None`` starts it clean -- is prep's decision, so
    the attempt is opened once, with its carry: it was opened with none and
    then again with it until 2026-10-01, and a refusal between the two left
    an earlier carry undone (W52).
    """
    from .materialize import prepare_attempt

    if shape is None or not shape.keeps_attempts_as_directories:
        return []
    took = continuation
    reports = [prepare_attempt(js, base, stage, container=c,
                               continue_from=(took.source if took is not None
                                              else None),
                               cold=took is None,
                               named=not (took is not None and took.by_default),
                               carry=(list(took.carries) if took is not None
                                      and took.carries is not None else None),
                               plan=plan, shape=shape)
               for c in containers]
    for rep in reports:
        _move_progress_channel_into(rep.dir, plan)
    return reports


def _carrying(js: JobSet, base: Path, stage: str, continuation, shape):
    """``continuation`` with the files it carries -- by the pair's rule, the
    job this prep writes against the one its run belongs to
    (`materialize.continuation_files`, the one check), of what that run
    holds; refused as that check refuses, nothing declared or nothing there
    (`job-system.md` § 5.4).  On the flat layout nothing is copied: the
    files lie in the folder."""
    if continuation.source is None:
        return dataclasses.replace(continuation, carries=())
    from .materialize import continuation_files
    try:
        names = continuation_files(js, base, stage, continuation.source,
                                   named=not continuation.by_default,
                                   shape=shape)
    except ValueError as exc:
        raise PrepError(str(exc)) from None
    held = Path(base) / continuation.source
    return dataclasses.replace(
        continuation,
        carries=tuple(n for n in names if (held / n).is_file()))


def _what_this_prep_takes(stage: str, continuation, gather) -> List[str]:
    """What this prep's hand-over takes, as `STAGE-PLAN.md` says it under
    its table (`job-system.md` § 5.4; D28: the table alone, each stage's
    declared restart files, read as what was copied): the line both doors
    print, with the files carried; a transport rung's gather, point by
    point; or nothing taken from another run."""
    out: List[str] = []
    if continuation is not None:
        out.append(f"`{stage}` "
                   + continuation.line(copied=continuation.carries or ()))
    for _container, volts, inputs in (gather or ()):
        if inputs:
            at = "" if volts is None else f" at {volts:g} V"
            out.append(f"`{stage}`{at} gathers "
                       + ", ".join(f"{fn} <- {src}" for src, fn in inputs))
    return out or [f"`{stage}` takes nothing from another run"]


def _launch_as_written(plan, run_dir, job) -> dict:
    """What the job will be launched with, as its header and its run script
    carry it -- A13's end point (`execution/architecture.md` § 5.2): the
    header's ``#SBATCH`` lines and the run script's stated counts, read from
    the text the plan holds by each writer's own reader, never worked out
    again."""
    from ..runwrap import stated_counts
    from ..scheduler.emit import Directives
    stem = Path(job.script).stem
    header = Path(run_dir) / f"{stem}.sbatch"
    script = Path(run_dir) / f"{stem}.run.sh"
    return {"header": (Directives.lines_of(plan.read_text(header))
                       if plan.is_file(header) else []),
            "run_script": (stated_counts(plan.read_text(script))
                           if plan.is_file(script) else [])}


def _move_progress_channel_into(attempt: Path, plan) -> None:
    """Put the seeded progress log where the run will write it.

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
    from ..runfiles import WRITTEN, role_of

    stage_dir = attempt.parent
    if stage_dir == attempt:
        return
    for role in [a.role for a in WRITTEN if a.output == "progress"]:
        # THE FOLDER AS THE PLAN LEAVES IT, read as `runfiles.find_by_role`
        # reads one on disk: each name's own role.
        for seeded in plan.glob(stage_dir, "*"):
            if role_of(seeded.name) == role:
                plan.move(seeded, attempt / seeded.name)


def _transport_provide_pseudos(struct, cfg, base: Path,
                               citation: str, *, plan=None) -> None:
    """The pseudopotentials arrive FROM THE CITATION
    (archive/2026-09-01-transport-design.md § 4.1: structure, pseudos and electronic
    template all come with the cited junction — one template governs).

    Idempotent, and the folder wins, exactly like the SIESTA arm: what
    ``pseudos/`` already holds (an earlier prep, or the travelled
    folder) is left alone; only missing species are fetched, from the
    cited calculation's own ``pseudos/`` — the files the junction
    actually relaxed with.  Then the science screening runs on what is
    actually here, same protocol, same blocking statuses.
    """
    from ..projects import find_projects_root
    from ..pseudos import psml_sources
    from ..siesta.input import copy_pseudopotentials
    from ..chemistry import species_order

    species = species_order(struct.elements)
    if not species:
        return
    pdir = _pseudo_dir(base, plan)
    # THE FOLDER WINS, by the one rule the settings gate asks too.
    want = [s for s, d in psml_sources(species, dest_dir=base).items()
            if d is None]
    if want:
        root = find_projects_root(base)
        lib = None
        if root is not None:
            cited = Path(root) / citation
            if (cited / PSEUDO_DIRNAME).is_dir():
                lib = cited / PSEUDO_DIRNAME
            elif any(cited.glob("*.psml")):
                lib = cited
        missing = (copy_pseudopotentials(want, lib, pdir, plan=plan)
                   if lib is not None else list(want))
        if missing:
            raise PrepError(
                f"this transport calculation needs "
                f"{', '.join(f'{m}.psml' for m in missing)} and there is "
                f"none in {base.name}/pseudos/ or in the cited "
                f"directory ({citation}).  The pseudopotentials travel "
                f"with the citation (archive/2026-09-01-transport-design.md 4.1) -- the "
                f"junction ran with them, so its calculation folder "
                f"should hold them; prep the junction there, or put the "
                f"files in {base.name}/pseudos/ yourself.")
    _screen_pseudos(species, cfg, pdir)


def _resolve_transport(base, task, stage: str, allocation,
                       *, pins=None, log=None):
    """Floor 3's step 2 for a transport rung — the template ⊕ its overrides
    ⊕ its run card's settings (``pins``, `stages.md` § 6.8d).

    **The same `resolve` every other kind uses.** It was unreachable for
    transport until TR1, for a plain reason: `resolve` reads a template and a
    transport calculation did not have one. Now it does, so the arm that
    `engines/transport.md` § 3.2 measured as *"a second conductor that
    decides"* can hand its deciding back to the one that already exists.

    Returns the rung's :class:`~molbuilder.resolve.ResolvedConfig` — **the
    whole element, not just its values**. What is gained over assembling it by
    hand is **provenance**: every value says whether the template, the stage or
    a pin set it, which is the whole of what the pipeline log had nothing to
    print.

    THE RESOURCES RIDE ON THE ELEMENT, and returning ``element.values`` alone
    dropped them (found 2026-09-16). `resolve` folds two riders onto each
    element's resources — ``continue_retries`` and ``use_gpu``, both config
    answers that the WRAPPER reads — and this arm rebuilt the allocation by
    hand afterwards, so neither ever arrived. Measured: ``SiestaConfig``
    defaults them to ``1`` and ``False`` while ``Resources`` defaults both to
    ``None``, so every transport wrapper rendered with no warm-retry loop at
    all, and ``use_gpu`` fell back to GREPPING the deck for ``Diag.ELPA.GPU``
    — the SIESTA-keyword re-derivation `execution/gpu.md` G7 deleted. That is
    `resolve.py`'s own A-5 finding (*"travels the whole way was true of one
    road out of two"*) opening a third road.
    """
    from ..config.siesta import SiestaConfig
    from ..resolve import ResolveError, resolve
    from ..template import find_template

    # A SHARED VALUE IS NOT A PER-STAGE OVERRIDE, NOR ONE THE RUNG FIXES --
    # refused by `resolve`, the one door every kind's prep goes through
    # (`template.shared_by_every_stage` + `why_shared`; `fixed_by_role` +
    # `why_role`), which also lays each rung's own answers on its config.
    # This step refused transport's own copy until 2026-09-28, and no other
    # kind refused at all.
    # AND THE RUNG THAT READS A VALUE (`stages`, the same § 6.4) is asked at
    # that one door too, for every kind (`template.unread_overrides`, plan
    # § 5w K4): a transmission window on the seed's rung is refused there
    # by name.  This step asked transport's own copy of it until
    # 2026-09-30.

    try:
        tmpl = find_template(base, task.label)
    except ValueError as exc:
        raise PrepError(str(exc)) from exc
    if tmpl is None:
        raise PrepError(
            f"this transport calculation has no template, so there is "
            f"nothing to resolve.  `jobset init` has written one since "
            f"2026-09-16 (TR1) -- a description made before that carries "
            f"only task.json.  Re-describe it, or write "
            f"{task.label}.template.toml beside task.json with the "
            f"electronic values the cited run used.")
    if log is not None:
        log.phase("STEP 2 · RESOLVE — the description becomes a ParameterSet")
        log.received("template", tmpl.name)
        log.received("stage", stage)
    try:
        ps = resolve(tmpl.read_text(encoding="utf-8"), task, SiestaConfig,
                     allocation=allocation, stage=stage, pins=pins)
    except ResolveError as exc:
        # `resolve` translates the template's and the overrides' refusals
        # (ValueError) into its own since 2026-09-28 -- this caller caught
        # ValueError too, and the generic caller did not, so a refused
        # template was a named refusal here and a traceback everywhere else.
        raise PrepError(str(exc)) from exc
    # ONE ELEMENT.  A transport rung is a production run; its one axis is the
    # bias, and that is the device's own directory level rather than a sweep
    # (`archive/2026-09-01-transport-design.md` § 4.3).  `resolve` still returns a list, because
    # the length is the whole of the difference between a run and a sweep and
    # no reader below floor 7 may branch on which.
    element = ps.elements[0]
    if log is not None:
        log.produced("elements", f"{len(ps)} (a run, not a sweep)")
        for _name, _src in sorted(element.provenance.items()):
            log.chose(_name, getattr(element.values, _name, None), _src)
    return element


def _transport_parts(task, stage: str, composed, config):
    """``(struct, configure, state)`` of one transport rung: the structure its
    decks describe, the config they render from, and the junction's
    electronic state.

    ONE DOOR, asked by the conductor through the rung
    (:func:`_transport_rung_of`) and by the gather when it renders an
    upstream rung NOW (:func:`_transport_rung`, plan § 5w K11, T-F30): the
    gather compared an upstream attempt with that stage folder's LAST
    render, which a change since did not touch, so a stale result was
    carried forward and recorded as consistent.
    """
    from ..transport.transiesta import electrode_hs_stem
    # (i) THE STRUCTURE.  Two of them, out of the one cited file: the
    # junction, and the lead taken out of it by its region label -- same
    # atoms, same relaxation, a subset rather than a geometry derived from
    # anywhere else.  That is what lets the seam stay `spec_for(struct,
    # cfg, ...)`: a deck describes a structure, and these are two.
    if stage in ("electrode_L", "electrode_R"):
        model = (composed.electrode_left if stage == "electrode_L"
                 else composed.electrode_right)
        struct = model.as_structure()
        # The lead's identity IS the .TSHS stem the device deck names --
        # one spelling, `electrode_hs_stem`, read by both writers.
        label = electrode_hs_stem(task.label, model.label)
    else:
        # The junction -- and for the device it carries the region
        # partition the NEGF block is built from.
        struct, label = composed.sorted.structure, task.label

    def labelled(cfg):
        return (cfg if label == task.label
                else dataclasses.replace(cfg, system_label=label))
    # THE ELECTRONIC STATE BELONGS TO THE CALCULATION (ES1,
    # `science/chemistry-correctness.md` § 2a) -- and on a transport ladder
    # that is physics, not bookkeeping: TranSIESTA joins the leads'
    # self-energies to the device, so every rung must solve the same spin
    # channels.  A blank spin is decided ONCE, on the JUNCTION, and every
    # rung -- a lead included -- is handed that answer.  Decided per rung, a
    # molecule with an open-d centre would polarize the device beside
    # non-polarized leads.
    # The VALUES are folded into the config, so every reader of the rung's
    # config -- the gate, the record, the pseudopotential screening -- reads
    # the junction's answer; the STATE itself, with where each value came
    # from, is handed to the deck writer, which says so in every rung
    # (§ 2a.5).  Folded alone, the rungs read it as *stated*.
    from ..electronic_state import electronic_state
    state = electronic_state(composed.sorted.structure, labelled(config),
                             kind="transport")
    # A STATE TRANSIESTA CANNOT RUN is refused HERE, on the junction, where
    # each value still says where it came from (`template.md` § 6.3a, ES4):
    # once folded into the rungs' configs, every rung's gate would call a
    # recorded or detected value *stated*.
    from ..validation.chemistry import check_electronic_state
    refused = [i for i in check_electronic_state(
        composed.sorted.structure, labelled(config), calculation="transport")
        if i.severity == "error"]
    if refused:
        raise PrepError(refused[0].message)

    def configure(cfg):
        return dataclasses.replace(
            labelled(cfg),
            spin_treatment=state.spin_treatment.value,
            unpaired_electrons=state.unpaired_electrons.value)
    return struct, configure, state


def _transport_rung(base, task, stage: str, composed, allocation, *,
                    pins=None, log=None):
    """``(struct, config, state, element)`` of one transport rung, resolved
    NOW from the current template, junction and run card -- what the gather
    renders an upstream rung from (:func:`_rung_deck_now`): the template ⊕
    the rung's overrides ⊕ its run card (:func:`_resolve_transport`), then
    the parts the conductor's rung reads too (:func:`_transport_parts`)."""
    element = _resolve_transport(base, task, stage, allocation, pins=pins,
                                 log=log)
    struct, configure, state = _transport_parts(task, stage, composed,
                                                element.render_config())
    return struct, configure(element.render_config()), state, element


def _at_bias(cfg, volts):
    """A transport deck's config at its bias point: the point is the rung's
    answer, the one `role` value the catalogue does not hold
    (`engines/template.md` § 6.4) -- the list is the bias's only home
    (`engines/transport.md` § 2a.10), and a deck with no point keeps the
    0 V `resolve` laid on."""
    return (cfg if volts is None
            else dataclasses.replace(cfg, bias_voltage_v=float(volts)))


def _composed_for_prep(base, task, plan):
    """The junction a transport calculation's rungs describe: the record
    beside ``task.json`` when it is for THIS citation (the folder travels,
    `project-layout.md` § 2.1), else composed afresh from the projects tree,
    its record written -- and READ BACK from it.

    Written into ``plan`` (`jobset.planned.Plan`), with everything else this
    prep writes: the fresh record is written to a temporary folder outside
    the calculation and read back from there -- the same round trip a later
    prep makes from the calculation's own copy -- and its bytes go into the
    plan."""
    from ..atom_permutation import PermutationError
    from ..projects import find_projects_root
    from ..transport.compose import (ComposeError, compose_junction,
                                     load_compose_record,
                                     write_compose_record)
    from ..transport.sort import SortError
    citation = task.slots["junction"]
    try:
        # The tree root is resolved BEFORE the record is loaded: the
        # reload re-runs the § 3 lead gates, and their principal-layer
        # half reads the CITED directory's own .ion files, which the
        # travelled folder does not carry.  Absent (the folder moved
        # out of its tree), that half degrades to UNVERIFIED honestly.
        root = find_projects_root(base)
        why: list = []
        composed = load_compose_record(base, citation=citation,
                                       tree_root=root, why=why)
        if composed is not None:
            return composed
        if root is None:
            # NAME WHICH OF THE THREE: no record, one composed from another
            # citation (a slot re-pointed after a re-relaxation), or one
            # that does not read -- with the folder outside any tree, so
            # nothing can be composed afresh.
            raise PrepError(
                f"this transport calculation cannot be composed here: "
                f"{why[0] if why else 'there is no usable record'}.  "
                f"And {base} is not inside a projects tree, so the "
                f"citation {citation!r} cannot be resolved to compose "
                f"afresh.  Prep once inside the tree that holds the "
                f"cited junction -- the record then travels with the "
                f"folder (archive/2026-09-01-transport-design.md 4.1).")
        # RENDER FROM THE RECORD, on this prep as on every later one.  The
        # fresh composition is the cited `.XV` at full float precision; the
        # record is the codec's text of it, and every later prep -- the
        # other rungs, a re-prep -- reads the record.  Rendering this one
        # from the fresh copy gave two preps of the SAME calculation two
        # sets of numbers ~1e-10 apart (measured 2026-09-25 on the
        # ENGINE-OFFSET record), and the gather's `same_calculation`
        # compares them as text: a value near a rounding boundary then
        # flips, and a good seed is refused.  One source, not rounding luck.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="molbuilder-compose-") as tmp:
            names = write_compose_record(
                tmp, compose_junction(citation, tree_root=root))
            composed = load_compose_record(tmp, citation=citation,
                                           tree_root=root, why=why)
            if composed is None:
                raise PrepError(
                    f"the composed junction was written and could not be "
                    f"read back: {why[-1] if why else 'no reason given'}")
            for name in names:
                plan.bytes(Path(base) / name, (Path(tmp) / name).read_bytes())
        return composed
    except (ComposeError, SortError, PermutationError) as exc:
        raise PrepError(str(exc)) from exc


def _transport_spec(task, stage: str, struct, config, state, volts=None, *,
                    base=None):
    """One transport deck's ``(spec, cfg)`` -- the bias point's when
    ``volts`` is given (the point is the rung's answer, `engines/template.md`
    § 6.4; a single-bias rung keeps the 0 V `resolve` laid on)."""
    from ..siesta.input import spec_for as _siesta_spec_for
    cfg = _at_bias(config, volts)
    with _user_error_as_prep():
        try:
            spec = _siesta_spec_for(struct, cfg,
                                    stage_token=(stage_home(base, task, stage).token
                                                 or None),
                                    calculation="transport", state=state)
        except ValueError as exc:
            # `transport_spec` refuses an unknown rung with a message
            # written FOR a person, and `_user_error_as_prep` translates
            # only ValidationError / RuntimeConfigError / WrapperError --
            # deliberately, so a TypeError still looks like the bug it is.
            raise PrepError(str(exc)) from exc
    return spec, cfg


def transport_inputs(base_dir, task, stage: str, *,
                     bias: Optional[float] = None) -> List[tuple]:
    """What one attempt of ``stage`` takes from the runs upstream -- the
    § 4.2 DAG's inputs, the composite's other half of *"warm files are
    COPIED in at prep"*: ``[(source attempt, filename)]``, decided at
    prep's checkpoint 4a with what every stage continues from
    (`job-system.md` § 5.0) and copied in where the attempt is planned
    (:func:`_gather_into`).

    ``--from`` carries within ONE stage (an attempt continuing an
    earlier attempt of itself); this carries BETWEEN stages, and the
    sources are structural — fixed by the design's DAG, not named by a
    person — so what keeps it honest is not a name but three gates, per
    input, each a refusal naming what to do first (strict composition,
    ruling Q2 — transport never runs its pieces for you):

    * the upstream stage must have been PREPPED (its deck rendered);
    * it must hold a CONCLUDED attempt **whose deck matches the deck that
      rung renders NOW** -- from the current template, junction and run
      card, through the rung's own door (`_transport_rung`), never the
      stage folder's last render, which a change since leaves as it was
      (plan § 5w K11).  A mismatch is a mistake: refused by name;
    * the concluded, matching attempt must actually hold the file.

    The newest qualifying attempt wins (identical decks → identical
    single-point results).  What was taken from where lands in
    ``.gathered-from`` beside the copies, so a result can always say
    which electrode run fed it.
    """
    from ..transport.stages import (per_point_rungs, rung_container,
                                     stage_inputs)
    from .continuation import usable

    base = Path(base_dir)
    enabled = {s.name for s in task.stages if s.enabled}
    inputs = stage_inputs(stage, task.label,
                          seed_enabled=("seed" in enabled))
    gathered: List[tuple] = []
    composed = None              # the junction, read once, when first needed
    for upstream, filename in inputs:
        token = stage_home(base, task, upstream).token
        # A bias scan keeps a per-point rung's products PER POINT -- the
        # transmission at v reads the device at v, never another point's
        # converged state (archive/2026-09-01-transport-design.md 4.3); a lead is every
        # point's.  The one door says which folder (`rung_container`).
        up_dir = rung_container(base, task, upstream, bias)
        stem = _rf_stem(task.label, token)
        current_deck = up_dir / _rf(task.label, ".fdf", token)
        from .commands import block, command, run_first as _run_first
        run_first = ("run it first --\n"
                     + block(_run_first(upstream, base=base))
                     + "\n  (strict composition, ruling Q2: transport never "
                       "runs its pieces for you.)")
        if not current_deck.is_file():
            raise PrepError(
                f"the {stage} stage consumes {filename} from {upstream}, "
                f"and {upstream} has not been prepped -- {run_first}")
        # Newest first, and NUMERICALLY -- `run-10` is ten, not a tenth
        # (§ 4.3: the index is not padded).  This reached across for
        # `materialize.ATTEMPT_RE` and applied it twice, once to filter and
        # once to sort; `paths` owns both halves of the name now.
        from ..paths import attempt_dir as _adir
        from ..paths import attempts_in as _ain
        attempts = [_adir(up_dir, n) for n in reversed(_ain(up_dir))]
        # ONE TO BUILD ON -- ended on its own with exit code 0
        # (`continuation.usable`); one that concluded with an error was
        # gathered until 2026-10-03.
        concluded = [d for d in attempts if usable(d, stem)]
        if not concluded:
            raise PrepError(
                f"the {stage} stage consumes {filename} from {upstream}, "
                f"and {upstream} has no attempt that CONCLUDED with exit "
                f"code 0 -- it was never launched, is still running, was "
                f"force-stopped, or failed -- "
                f"`{command('status', upstream, base=base)}` says which "
                f"(project-layout.md 1.6).  Let it finish, or {run_first}")
        # THE SAME CALCULATION, not the same bytes.  A deck that renders
        # through the framework carries a generated-at timestamp and the
        # generator's git sha, and neither says anything about what the
        # engine computes -- so a byte comparison here refused a perfectly
        # good upstream result because the seed had been re-prepped, or
        # merely because a commit landed between the two preps.  It said
        # "the junction citation or its contract changed", which was false
        # and pointed the reader at the science.
        #
        # `same_calculation` masks exactly those fields and keeps every
        # other byte, the region partition included (`script_emit`).
        # THE DECK THE UPSTREAM RUNG RENDERS NOW (plan § 5w K11, T-F30).
        # This read the stage folder's LAST render, which a changed template
        # value or a re-pointed junction leaves as it was until that rung
        # is prepped again -- so a stale result was carried forward, and
        # `.gathered-from` said it was consistent.
        if composed is None:
            composed = _composed_junction(base, task)
        now = _rung_deck_now(base, task, upstream, composed,
                             volts=(bias if upstream in per_point_rungs()
                                    else None))
        matching = [d for d in concluded
                    if (d / current_deck.name).is_file()
                    and _sc.same_calculation(
                        (d / current_deck.name).read_text(), now)]
        if not matching:
            raise PrepError(
                f"{upstream} has {len(concluded)} concluded attempt(s), "
                f"but none ran the deck {upstream} renders now -- its "
                f"template, its junction or its run card changed since "
                f"they ran, so their {filename} answers a different "
                f"calculation.  Re-{run_first}")
        src = matching[0] / filename
        if not src.is_file():
            raise PrepError(
                f"{upstream}'s concluded attempt "
                f"{matching[0].relative_to(base)} did not write "
                f"{filename} -- the run concluded without producing what "
                f"the {stage} stage consumes.  Re-{run_first}")
        gathered.append((str(matching[0].relative_to(base)), filename))
    return gathered


def _gather_into(base: Path, attempt_dir, inputs, plan) -> None:
    """``inputs`` -- :func:`transport_inputs`' decision -- copied into
    ``attempt_dir`` with the record of what came from where
    (`.gathered-from`), into ``plan``."""
    for src, filename in inputs:
        plan.copy(Path(base) / src / filename, Path(attempt_dir) / filename)
    if inputs:
        write_gathered_from(attempt_dir, inputs, plan=plan)


def _composed_junction(base, task):
    """The junction this calculation is composed from -- its record beside
    ``task.json``, for the cited junction (`_composed_for_prep` writes
    it before any rung's deck)."""
    from ..projects import find_projects_root
    from ..transport.compose import load_compose_record
    why: list = []
    composed = load_compose_record(base, citation=task.slots["junction"],
                                   tree_root=find_projects_root(base),
                                   why=why)
    if composed is None:
        raise PrepError(
            f"this calculation's junction cannot be read for the gather: "
            f"{why[0] if why else 'there is no composition record'}.  Prep "
            f"the rung again, which composes it.")
    return composed


def _rung_deck_now(base, task, stage: str, composed, *, volts=None) -> str:
    """The deck ``stage`` renders NOW -- its text exactly as `prep run`
    writes it, from the current template, junction and the rung's run card
    (`_transport_rung`, `_transport_spec`, `script_emit.render_deck`).
    A transport deck records no machine sizing, so no allocation is
    needed to render it."""
    from .model import Resources
    from .prep_inputs import run_inputs
    # THE RUNG'S PINS, as `prep run` takes them from its card (`run_inputs`).
    _shape, pins = run_inputs(base, task, stage)
    struct, config, state, _element = _transport_rung(
        base, task, stage, composed, Resources(), pins=pins or None)
    spec, cfg = _transport_spec(task, stage, struct, config, state, volts,
                                base=base)
    with _user_error_as_prep():
        return _sc.render_deck(spec, struct, cfg, verbose=True,
                               dest_dir=base).text


def gather_sources(base_dir, task, stage: str
                   ) -> List[Tuple[Path, Optional[float], List[tuple]]]:
    """What every attempt of this rung takes from the runs upstream --
    ``[(container, volts, [(source attempt, filename), ...]), ...]``, one
    entry per attempt container: the stage's own folder, or each point of a
    bias scan, gathered against **its own** voltage (the transmission at *v*
    reads the device at *v*, never another point's converged state).
    ``volts`` is ``None`` for a rung with no bias axis, CARRIED rather than
    parsed back out of a folder's name, which `bias_token` owns.

    **Decided at prep's checkpoint 4a, with what every stage continues
    from** (`job-system.md` § 5.0; the hand-over is one `Continuation`, and a
    transport rung's is this, recorded as `.gathered-from`): refused before
    anything is written when an upstream run has not concluded, or ran
    another deck (:func:`transport_inputs`'s gates; strict composition,
    ruling Q2).  The steps copy each entry into the attempt they open in its
    container (:func:`_gather_into`).  *(It was a step of its own, run after
    the five, until 2026-10-05; before 2026-09-16 it ran on the CLI road
    alone, so the browser's Prep opened an attempt and carried nothing into
    it, and a device job could reach the node without an electrode's
    `.TSHS`.)*  `prep_calculation` asked directly renders the decks without
    it: a deck is the reviewable artifact, whether or not the rungs before it
    have run.
    """
    from ..transport.stages import rung_containers
    base = Path(base_dir)
    return [(Path(container), volts,
             transport_inputs(base, task, stage, bias=volts))
            for container, volts in rung_containers(base, task, stage)]

def _merge_run_jobset(path: Path, new: JobSet,
                      ladder: Optional[frozenset] = None) -> JobSet:
    """The root ``job-set.json`` is the RUN's whole plan: each stage's prep
    updates its OWN row and leaves the others standing.

    Until 2026-08-12 every prep wrote only its own elements, so `prep run
    tight` erased `coarse` from floor 3 — breaking the status rollup and,
    worse, the ``--from`` pair rule: with the source job gone, `warm_carry`
    read the pair as unverified and silently withheld ``.CG``
    (`project-layout.md` § 2.3.4 row 3).

    ``ladder`` is the CURRENT task's stage-name set, and it bounds what is
    kept (2026-08-12): a row is standing only while its stage is still
    on the ladder — a stage removed from ``task.json`` used to stay in the
    plan forever, its deck gone.  *(Three arms replaced the plan outright
    until 2026-10-03 -- one that did not read, a pre-container sweep's, one
    under another name: the prep entry reads the plan first, and molbuilder
    writes neither of the others.)*
    """
    if not path.is_file():
        return new
    old = JobSet.load(path)
    # A STAGE IS PREPPED ONCE (`job-system.md` § 5.0), so its row is new;
    # one already here is replaced rather than doubled, should two preps of
    # one stage ever race past the entry's gate together.
    # ONE KEY for a stage's name, as every reader of the plan matches it
    # (`identity.stage_key`): a stage renamed in case only kept its row on
    # every read and lost it here, so the next prep of any stage made it
    # preppable twice (the framework inventory, 2026-10-03).
    from ..identity import stage_key
    fresh = {stage_key(j.name) for j in new.jobs}
    standing = (None if ladder is None
                else {stage_key(n) for n in ladder})
    kept = [j for j in old.jobs if stage_key(j.name) not in fresh
            and (standing is None or stage_key(j.name) in standing)]
    merged = dataclasses.replace(
        new, jobs=kept + list(new.jobs),
        shared=sorted(set(old.shared) | set(new.shared)))
    # The plan's order is the LADDER's, not the order stages were prepped
    # in: `medium` prepped before `coarse` (`--cold`) is still listed after
    # it.  The seq token is zero-padded (§ 6.3) so it sorts as it reads.
    from .materialize import stage_refs
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
        # A SEQUENCE FIELD PRINTS AS ITS VALUE, not as its repr.  Every field
        # here was a scalar until `notify_channels` (2026-08-31), and the
        # default rendering put `('slack', 'lab')` -- and, worse, a bare `()`
        # -- into the one file whose whole value is that a person can read it.
        # `()` is a real answer meaning *nowhere*, so it gets a word.
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
                         label: str, token=None, frame=None,
                         relaxes: bool = True, plan=None) -> None:
    """Write the one-block preview the Watch tab discovers before a run starts.

    The deck NAMES its trajectory log; something has to CREATE it, or the tab
    has nothing to find until the engine writes its first step. That seeding
    lived inside ``convert`` — which writes a deck to disk — and `prep` renders
    the text and writes it itself, so the preview was silently skipped.

    **Found by the trajectory-log tests when `molbuilder fdf` was deleted.**
    They named a real property of the product, not of the verb, which is why
    they were repointed rather than retired.

    ``engine`` and ``label`` come through the caller from the
    :class:`EngineSeam` — this function hardcoded ``"siesta"`` and read
    ``cfg.system_label`` until 2026-08-12, which was the seam leaking.
    """
    if not getattr(cfg, "write_molwatch_log", False):
        return
    from ..trajectory_log import molwatch_log_basename, write_initial_preview
    # ``token`` is the caller's, same as the render argument (C7): the
    # config no longer carries a stage, and nothing here re-derives one.
    # The stage's own convergence targets travel with its log, so the Watch
    # tab's threshold line is THIS stage's and not the ladder's first.  They
    # come from the RESOLVED config, which is the whole point of resolving
    # before rendering: `coarse` and `tight` disagree about both of these.
    # THE KEY NAMES ARE THE READER'S, not this writer's invention.
    # `max_force_ev_per_ang` / `max_steps` stood here until 2026-09-05 and
    # nothing read either: the trajectory card asks for
    # `max_force_tol_eV_per_A` and `max_geom_iter` (trajectory/core.js), which
    # are what the OTHER two producers of this same header emit --
    # `trajectory_log/emitter.py`'s `_LEAF_KEYS` and the `.out` parser's
    # `_set_conv_target`.  So a staged SIESTA run drew a convergence card with
    # zero rows and no threshold line, while the `.out` sitting beside it
    # parsed the same two numbers correctly: the same directory answering the
    # same question two ways depending on which file was opened.
    # ONLY A DECK THAT RELAXES HAS TARGETS -- the viewer draws "the targets
    # the run was chasing" (`web/trajectory.md` § 3), and a force-constant run
    # chases none: a threshold line over its steps called 115 nudges a
    # relaxation that never settles.  ``relaxes`` is the rung's render kind.
    targets = {}
    for key, attr in (() if not relaxes else
                      (("max_force_tol_eV_per_A", "relax_force_tol"),
                       ("max_geom_iter", "relax_steps"))):
        value = getattr(cfg, attr, None)
        if value is not None:
            targets[key] = value
    write_initial_preview(
        struct,
        base / molwatch_log_basename(label, token),
        job=label, engine=engine,
        convergence_targets=(targets or None),
        frame=frame,
        frozen_atoms=list(getattr(struct, "frozen_atoms", []) or []),
        plan=plan)


def _job_for(element, script: str, task, stage_name: Optional[str],
             seam: EngineSeam, base_dir=None,
             finish: Optional[str] = None) -> Job:
    """One element of the parameter set as one :class:`Job`.

    ``resources`` is **copied from the element**, never re-derived: the element
    resolved it once, from the allocation, and a second derivation here is the
    habit `generator.md` § 5 exists to end.

    The **name** answers *which job is this*, and there are exactly three
    answers because there are three things an element can be: a trial (named by
    its sweep coordinate), a rung of a ladder (named by the stage), or the whole
    calculation (named by its label).
    """
    from ..resolve import point_token

    if element.point:
        name = point_token(element.point)
    elif stage_name:
        name = stage_name
    else:
        name = task.label

    # THE RUNG'S OWN KIND answers what it carries and whether a re-run of it
    # resumes (`job-contracts.md` § 4.2a): a vibration's `relax` rung is an
    # optimisation, its force-constant rungs the vibration.  The
    # calculation's kind stood here until 2026-09-29, and a vibration's
    # `relax` rung lost its `.CG` (the M11 review, SS-C14).
    kind = _rung_kind(task, stage_name)
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
               finish=finish, resumes=resumes)


def _rung_kind(task, stage_name: Optional[str]) -> str:
    """The kind of run a rung IS -- ONE answer, read by the deck it renders
    and by the warm-files section it reads (`job-contracts.md` § 4.2a): the
    calculation's own, except a SIESTA vibration's rungs, which are two
    programs (`vibration_render_kind`): the `relax` rung an optimisation,
    every other rung the vibration.  A PySCF vibration relaxes inside its
    one deck, so its rungs are all the vibration.  Two answers stood until
    2026-09-29, one engine-scoped and one not (the K6 review, R9); since
    2026-09-30 it is read off the rung's ROLE, the one rule every door asks
    which items a rung reads by (`template.stage_role`, plan § 5w K4)."""
    from ..template import stage_role
    if stage_role(str(task.engine), str(task.calculation),
                  stage_name) == "relaxation":
        return "optimization"
    return str(task.calculation)


def _under_description(flags, declared, chosen=None) -> "Resources":
    """The caller's allocation over the description's -- FIELD by field.

    Two things come out of `task.json`, and they are two because they answer
    two questions:

      * ``allocation`` -- the queue, the wall, the memory and the GPU
        binding this calculation asks the SCHEDULER for (`stages.md`
        § 6.8a).
      * the run card's machine items -- the launch SHAPE the person chose:
        *"run it at eight"* (``chosen``, `stages.md` § 6.8d).  A ``bench``
        entry is a question to measure at any length, never an ask.

    A flag is what the person is asking for right now, so a stated flag wins
    and an unstated one leaves the file's answer standing.  Whole-object
    precedence would make `--np 8` erase a memory ask nobody mentioned, which
    is the class of silent loss this whole round has been about.

    **A PURE FOLD**: the two pieces arrive as arguments, so this function
    reads no file and no enumerator and can be exercised with two objects.
    The SHAPE's producer is `prep_inputs.declared_run_shape`: a direct map
    of the condition's machine items onto `Resources` fields, which leaves
    every field the condition does not name to the chain that already
    answers it (`running-a-job.md` § 3.1).
    """
    out = flags or Resources()
    import dataclasses as _dc
    patch = {}
    if declared:
        for name, val in (("domain", declared.domain),
                          ("time", declared.time),
                          ("mem", declared.mem)):
            if val and getattr(out, name, None) in (None, ""):
                patch[name] = val
        # THE BINDING SWITCH, when said: `False` is the value that matters,
        # so it is not tested for truth (`execution/gpu.md` G9).
        if declared.gpu_binding is not None and out.gpu_binding is None:
            patch["gpu_binding"] = declared.gpu_binding
    # ALREADY IN `Resources`' OWN WORDS -- `to_resources` speaks them, so
    # there is no name map here and no second place for one to drift.
    known = {f.name for f in _dc.fields(Resources)}
    for name, val in sorted((chosen or {}).items()):
        if name in known and getattr(out, name, None) in (None, ""):
            patch[name] = val
    return _dc.replace(out, **patch) if patch else out


def _with_notify(flags, declared) -> "Resources":
    """The description's `notify` block, onto the allocation that reaches
    the wrapper.

    Not an allocation and not a scheduler flag -- it rides ``Resources``
    because that is the road from a job to its wrapper, the one
    ``continue_retries`` already rides (`jobset/model.Resources`).  There is
    no ``--notify-*`` flag today, so in practice the description is the only
    voice.

    **FIELD BY FIELD ANYWAY**, for the reason the function above it states:
    a whole-object copy makes a description that sets only the period erase
    a caller's SCF trigger, which is the same silent loss as `--np 8`
    erasing a memory ask.  Written whole first, and it did exactly that --
    `Resources(notify_on_scf=True)` under `Notify(every_hours=6)` came back
    with the trigger gone.  Nothing sets these from outside yet; the point
    is that the shape cannot start losing values the day something does.

    No block leaves the fields ``None``, which the wrapper renders as no
    flags at all -- and no flag is nothing sent, the start and the end
    included: reports stay off for everyone who has not asked for them.
    """
    out = flags or Resources()
    if not declared:
        return out
    import dataclasses as _dc
    patch = {}
    if declared.on_scf_converged and out.notify_on_scf is None:
        patch["notify_on_scf"] = True
    if declared.every_hours and out.notify_every_hours is None:
        patch["notify_every_hours"] = declared.every_hours
    # `is not None`, NOT truthiness -- the other two fields are off when
    # falsy and this one is not.  An empty tuple says "send this calculation
    # nowhere", and a truthiness guard here would drop it and hand the job
    # every channel on the machine instead (`run-reports.md` 3.0).
    if out.notify_channels is None:
        # A BLOCK THAT NAMES NO CHANNELS asks for every channel of the machine
        # that runs the job, which only that machine knows -- so it travels as
        # the marker, resolved there (`run-reports.md` § 3.0).  No block at
        # all is not here: `not declared` returned above, and the wrapper
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
# Prep buttons -- and until 2026-09-29 each did its own part of the act: the
# command line ran the preflight, asked the *already under way* question,
# checked the launch agreement and wrote their ledger lines; the tab called
# the five steps alone and showed the folders.  One act, two answers.  This
# is the act, once; it prints nothing and asks nothing, and returns what it
# found and decided as data for each door to show in its own way.


@dataclass
class PrepAnswer:
    """What one prep found and decided -- the whole of what either door shows
    (`job-system.md` § 5.3's table)."""
    kind: str
    stage: Optional[str]
    #: The description's preflight notes (an error refuses instead).
    findings: list = dataclasses.field(default_factory=list)
    #: What the inputs said: a run card's value the run does not use, a
    #: bench's grid -- enumerated, crossed out, kept (`prep_inputs`).
    notes: List[str] = dataclasses.field(default_factory=list)
    #: The folder's state saved before the five steps wrote, or the one it
    #: stood at -- `checkpoint.Kept.said` (`checkpointing.md` § 9).
    saved: Optional[str] = None
    dirs: List[Path] = dataclasses.field(default_factory=list)
    provenance: Optional[dict] = None
    #: A flat run: its wrappers are rendered and there is no attempt to open.
    flat: bool = False
    #: The attempt opened or reused (`materialize.Attempt`).
    attempt: Optional[object] = None
    #: A transport bias scan: ``(attempt, volts, [(source, file), ...])``
    #: per point (`gather_sources`).
    points: List[tuple] = dataclasses.field(default_factory=list)
    #: A transport rung's carry into its one attempt: ``(source, file)``.
    gathered: List[tuple] = dataclasses.field(default_factory=list)
    #: What each deck's checks said (`script_emit.prepare_deck`), one of each
    #: -- the terminal read them on stderr as the decks rendered.
    deck_findings: list = dataclasses.field(default_factory=list)
    resources: Optional[dict] = None
    #: The stage's deck, by file name -- what the agreement is about.
    deck: Optional[str] = None
    #: `agreement.LaunchAgreement`, unless the deck makes no claim.
    agreement: Optional[object] = None
    pipeline_log: Optional[Path] = None
    #: Which run this stage continues from, and what it was
    #: (`continuation.Continuation`).
    continuation: Optional[object] = None
    #: A LINKED stage -- its kind gives its rungs roles (`template.KIND_ROLES`)
    #: -- whose input is prep's own, taken from the stages before it: what
    #: both doors say in place of "nothing carried in" (W52: a `freq` built
    #: at `relax`'s geometry was said to be like a first stage).
    linked: bool = False
    #: The person said ``--cold`` (the attempt's own ``cold`` says only that
    #: it started clean, which prep now states whenever nothing continues).
    cold: bool = False
    #: What the description asks the scheduler for -- queue, wall, memory
    #: -- as the allocation folded at step 2 holds it.
    allocation: dict = dataclasses.field(default_factory=dict)
    #: A run: the launch shape its card states (`prep_inputs`); a bench:
    #: the axes it declares.
    chosen: dict = dataclasses.field(default_factory=dict)
    bench_axes: dict = dataclasses.field(default_factory=dict)
    #: A13 -- what the job will be launched with, as its header and its run
    #: script carry it (:func:`_launch_as_written`).
    launch: Optional[dict] = None
    #: A PREVIEW: the plan, stopped before the save, named by its identity
    #: for the Prep that follows (`job-system.md` § 5.0), with what it would
    #: write.
    preview: bool = False
    plan_id: Optional[str] = None
    writes: List[str] = dataclasses.field(default_factory=list)

    def as_dict(self, base) -> dict:
        """The answer as JSON, paths relative to the calculation folder --
        what the Task setup tab's prep route returns (`web/web-api.md`)."""
        from .agreement import disagreement_note
        base = Path(base)

        def rel(p):
            try:
                return str(Path(p).resolve().relative_to(base.resolve()))
            except ValueError:
                return str(p)

        def carried(pairs):
            return [{"file": fn, "from": src} for src, fn in pairs]

        a, g = self.attempt, self.agreement
        return {
            "kind": self.kind, "stage": self.stage,
            # THE ONE WIRE FORM of a finding (`Issue.to_json`).
            "findings": [i.to_json() for i in self.findings],
            "deck_findings": [i.to_json() for i in self.deck_findings],
            "notes": list(self.notes),
            "saved": self.saved,
            "dirs": [rel(d) for d in self.dirs],
            "provenance": self.provenance,
            "flat": self.flat,
            "attempt": ({"dir": rel(a.dir), "fresh": a.fresh,
                         "brought": list(a.brought), "copied": list(a.copied),
                         "continued_from": a.continued_from, "cold": a.cold}
                        if a is not None else None),
            "points": [{"attempt": rel(att), "bias": v,
                        "gathered": carried(got)}
                       for att, v, got in self.points],
            "gathered": carried(self.gathered),
            "resources": self.resources,
            "deck": self.deck,
            "agreement": ({"verdict": g.verdict,
                           "rendered_for": g.rendered_text,
                           "launching_at": g.launch_text,
                           "note": (disagreement_note(g)
                                    if g.verdict == "differs" else None)}
                          if g is not None else None),
            "pipeline_log": rel(self.pipeline_log) if self.pipeline_log else None,
            "continuation": (dict(self.continuation.as_dict(),
                                  line=self.continuation.line(
                                  a.copied if a is not None else ()))
                             if self.continuation is not None else None),
            "linked": self.linked,
            "cold": self.cold,
            "allocation": self.allocation,
            "chosen": self.chosen,
            "bench_axes": self.bench_axes,
            "launch": self.launch,
            "preview": self.preview,
            "plan_id": self.plan_id,
            "writes": list(self.writes),
        }


def prepped_already(base, task, kind: str, stage: str) -> Optional[str]:
    """Why ``stage`` is not prepped -- it already is -- or ``None``
    (`job-system.md` § 5.0, checkpoint 2a; user, 2026-10-02: *"refuse it,
    redo via rollback"*).

    PREPPED IS WHAT `status` READS: the stage's job in the calculation's
    plan, ``job-set.json``, matched as `status` matches it
    (`identity.stage_key`) -- for a bench, the sweep its bench folder holds.
    A refused prep writes nothing but its ledger line (:func:`prep_stage`),
    so a stage is counted prepped only by a prep that
    finished."""
    from ..identity import stage_key
    from ..paths import Shape
    from .commands import rollback
    from ..paths import bench_container
    from .materialize import job_dir_names
    base = Path(base)
    if kind == "bench":
        home = base / bench_container(Shape.named(task.shape),
                                      stage_home(base, task, stage).token)
        if not (home / JOBSET_FILENAME).is_file():
            return None
        what, where = (f"the benchmark of stage {stage!r}",
                       f"{home.relative_to(base)}/")
    else:
        plan = base / JOBSET_FILENAME
        if not plan.is_file():
            return None
        js = JobSet.load(plan)
        job = next((j for j in js.jobs
                    if stage_key(j.name) == stage_key(stage)), None)
        if job is None:
            return None
        home = job_dir_names(js, Shape.named(task.shape)).get(job.name, "")
        what = f"stage {stage!r}"
        where = (f"{home}/" if home not in ("", ".")
                 else "in this calculation's folder")
    return (f"{what} is already prepped ({where}) -- a prepped stage is not "
            f"prepped again (job-system.md § 5.0).  To redo it, "
            + rollback("its prep", base=base))


def prepped_stages(base, task) -> List[str]:
    """The stages of ``task`` prepped in ``base`` -- as a run or as a
    benchmark -- in the description's order: whether the calculation has
    PRODUCED, which fixes its shape (`web/task-setup.md` § 4).  Each is
    :func:`prepped_already`'s answer, so a refused prep -- which writes
    nothing -- counts for none."""
    return [st.name for st in task.stages
            if any(prepped_already(base, task, kind, st.name)
                   for kind in ("run", "bench"))]


def prep_stage(base, kind: str, stage: Optional[str] = None, *,
               target: Optional[str] = None, allocation=None,
               from_attempt: Optional[str] = None, cold: bool = False,
               env: Optional[str] = None, emit_sbatch: bool = True,
               on_found=None, preview: bool = False,
               plan_id: Optional[str] = None) -> PrepAnswer:
    """**`prep`, the verb** -- what `molbuilder jobset prep` and the Task setup
    tab's Prep buttons both call (`job-system.md` § 5.3).

    In order -- `job-system.md` § 5.0's checkpoints: the description is
    refused unless it is one (a ``task.json`` and its template -- a
    transport description from before 2026-09-16 carries none, and its own
    door says how to add one); the stage is resolved through the one grammar
    (a name, or ``#N``), and refused when it is already prepped -- a redo is
    a rollback; the description's preflight runs -- an error refuses, the
    notes come back as ``findings``; the inputs are assembled (`prep_inputs`,
    A12).  Then the steps are PLANNED -- every deck, wrapper and data file,
    the attempt and what it carries in, a transport rung's inputs -- with
    nothing written (:func:`prep_calculation` into a
    :class:`~molbuilder.jobset.planned.Plan`); then the folder's state is
    saved, always, asking nothing (`checkpoint.save_before`); then the plan
    is written, deciding nothing; then the record -- what it continues
    from, the deck's agreement with the launch it will get, *prepped*.  A
    refusal can only come before the save, and writes nothing but its line
    in ``jobset-decisions.log`` (`job-system.md` § 5.0, rule 3).  Every
    decision lands there, whichever door called.

    ``allocation`` is what the person asks for on THIS prep -- the
    command line's flags, an empty ``Resources()`` from a surface with none
    (A12: never ``None``).  Every refusal is a :class:`PrepError` in the
    reader's own words, carrying what the entry had found by then --
    ``findings`` and ``notes``.

    ``on_found``, when given, is called with ``(findings, notes)`` as soon
    as the inputs are assembled -- before the save, and before anything is
    rendered -- so a terminal prints them ahead of what the decks say
    while they are written; the answer carries them either way.

    ``preview`` stops before the save and answers the plan -- what it would
    write, the launch the header and the run script would carry, what the
    stage builds on -- or the refusal prep would give, with nothing saved,
    written or recorded (`job-system.md` § 5.0: *a preview is the same
    entry*).  ``plan_id`` is a preview's plan, named (`Plan.identity`):
    a prep that makes a different one -- the folder changed between -- is
    refused, saying to preview again.
    """
    from ..scheduler import AmbiguousTarget, UnknownTarget
    from ..task import FILENAME as TASK_FILENAME, read_task
    from ..template import find_template
    from ..validation.task import preflight
    from .ledger import prepped as ledger_prepped
    from .ledger import record as ledger
    from .prep_inputs import bench_inputs, bench_refusal, prep_run_inputs
    base = Path(base).resolve()
    desc = base / TASK_FILENAME
    findings: list = []
    notes: List[str] = []
    deck_findings: list = []
    recorded: List[bool] = []

    def _finish(answer: PrepAnswer) -> PrepAnswer:
        # THE PREP IS RECORDED WHEN IT HAS FINISHED -- written, the last
        # line of its record (`job-system.md` § 5.0, checkpoint 7).
        answer.provenance = ledger_prepped(base, kind=kind, stage=stage,
                                           dirs=answer.dirs)
        return answer

    def _record_preflight():
        # The preflight's notes land in the ledger on the pass that ACTS or
        # REFUSES -- never on one that only asks, so the answering pass does
        # not write them twice -- and ahead of what follows them, the order
        # the terminal prints them in.  A preview only asks.
        if findings and not recorded and not preview:
            ledger(base, "prep", "preflight-report", stage=stage,
                   notes=[i.message for i in findings])
            recorded.append(True)

    def _refused(exc: PrepError) -> PrepError:
        # WHAT WAS FOUND RIDES WITH THE REFUSAL: a sentence that points at
        # "the crossed-out list above" is honest only if the list is shown.
        _record_preflight()
        # AND THE REFUSAL IS A DECISION TOO (`job-system.md` § 5.3: every
        # decision the entry makes lands in the ledger) -- in a described
        # calculation only: a folder that is not one gets no ledger of ours
        # (W52: after a refusal the stage's last line read `prepped`).
        if desc.is_file() and not preview:
            ledger(base, "prep", "refused", kind=kind, stage=stage,
                   reason=str(exc))
        exc.findings, exc.notes = tuple(findings), tuple(notes)
        return exc

    try:
        # 1 · A DESCRIBED CALCULATION is "a template PLUS task.json"
        #     (project-layout.md § 2.1), and prep builds everything else from
        #     the two.  A TRANSPORT description passes on task.json alone,
        #     so that one written before its template existed (2026-09-16,
        #     TR1) is refused by the transport door in its own words -- how
        #     to add the template -- rather than told to `init` again.  It has
        #     carried a template since (`engines/transport.md` § 2a).
        # EACH READ ONCE (W55 B1): the description here, its template beside
        # it -- every step after this reads these, never the files again.
        task = unread = None
        if desc.is_file():
            try:
                task = read_task(desc)
            except Exception as exc:                          # noqa: BLE001
                unread = exc   # its reader's own words, raised below
        is_transport = task is not None and task.calculation == "transport"
        try:
            template = (find_template(base, task.label)
                        if task is not None else None)
        except ValueError as exc:
            raise PrepError(str(exc)) from exc
        # A description that does not read passes, and its reader refuses it
        # below in its own words (`read_task`).
        if not (desc.is_file() and (task is None or is_transport
                                    or template is not None)):
            # INSIDE A CALCULATION -- one of its stage or attempt folders --
            # the folder says which one it belongs to (`calcdirs.root_of`);
            # `init` there would describe a new calculation inside an
            # attempt (W52).
            from .. import calcdirs
            root = calcdirs.root_of(base)
            if root is not None and Path(root).resolve() != base:
                raise PrepError(
                    f"{base} is a folder of the calculation at {root}; "
                    f"`prep` works on the calculation -- name that folder "
                    f"(the command line's `--bundle`).")
            raise PrepError(
                f"{base} is not a described calculation -- no task.json + "
                "template pair.  `prep` derives everything from those two "
                "(project-layout.md § 2.1); run `molbuilder jobset init` "
                "first.  (Hand-built job-sets remain launchable: `launch` "
                "and `status` read job-set.json directly.)")
        if (from_attempt or cold) and kind == "bench":
            raise PrepError(
                "--from / --cold choose what a RUN starts from; a bench "
                "trial measures its point from the structure, always "
                "(job-system.md § 7).")
        if unread is not None:
            raise unread
        template_text = (template.read_text(encoding="utf-8")
                         if template is not None else None)
        if kind == "bench":
            # A CALCULATION THAT HAS NO BENCHMARK says so before it is asked
            # which stage's -- or the stage offered is refused next (W52).
            why = bench_refusal(task)
            if why:
                raise PrepError(why)
        if stage is None:
            # ONE STAGE, named -- before anything is read of the machine or
            # written (W52: a bare `prep run` was refused by `resolve` after
            # the machine record had been snapshotted).  THE STAGES THE VERB
            # TAKES are offered: a prepped one is not (2a).
            from .commands import name_a_stage
            from .materialize import described_refs
            takes = [r for r in described_refs(base, task)
                     if not prepped_already(base, task, kind, r.name)]
            raise PrepError(
                (f"`prep {kind}` acts on ONE stage" + (
                    " -- --from / --cold describe its attempt" if
                    (from_attempt or cold) else "") + "; ")
                + (name_a_stage("prep", kind, takes, base=base) if takes
                   else "every stage is prepped already, and a prepped stage "
                        "is not prepped again (job-system.md § 5.0)."))

        # 2 · THE STAGE GRAMMAR (user-settled 2026-08-21): a ladder stage is
        #     named by its NAME, or by `#N` -- the NN of its directory --
        #     through the ONE resolver, with refs built from the full ladder,
        #     so `prep run #2` and `status #2` cannot disagree about which
        #     stage that is.
        if stage is not None and getattr(task, "stages", None):
            from ..identity import StageRef, resolve_stage_ref
            refs = [StageRef(h.seq, h.name)
                    for h in ladder_homes(base, task)]
            stage = resolve_stage_ref(refs, stage).name

        #    ...AND ENABLED: a stage the description disables is never prepped
        #    (user, 2026-10-03, Q1: "never allow use").  Transport refused
        #    one on its own, and the ladder prepped it when named.
        if stage is not None:
            from ..task import stage_disabled
            why = stage_disabled(task, stage)
            if why:
                raise PrepError(why)

        # 2a · NOT PREPPED BEFORE (user, 2026-10-02: "refuse it, redo via
        #      rollback").  Asked before anything is read of the machine or
        #      written; the refusal names the way back.  Until that day a
        #      prepped stage was re-rendered -- its attempt reused until
        #      launched, after a launch a new one -- once asked
        #      ("already under way here", run-identity.md § 6).
        if stage is not None:
            why = prepped_already(base, task, kind, stage)
            if why:
                raise PrepError(why)

        # 3 · § 6.6's PREFLIGHT, at its live moment (R5, 2026-08-12): prep
        #     on a machine whose molbuilder differs from the description's
        #     author.  The template rides along when it is where prep will
        #     look for it, which adds § 6.4/§ 6.6a's sequence warnings.  An
        #     error refuses -- carrying the notes beside it, and writing them
        #     to the ledger, as every refusal does.
        issues = preflight(task, template_text=template_text)
        findings[:] = [i for i in issues if i.severity != "error"]
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            raise PrepError(
                "the description fails its own preflight "
                "(engines/stages.md § 6.6):\n  - "
                + "\n  - ".join(i.message for i in errors))

        # 4 · THE MACHINE, READ ONCE (`script-preparation.md` § 3.0, step 1):
        #     the calculation's copy of its record -- at its first prep, the
        #     named machine's -- and it says how a shell enters an environment
        #     there, or the prep is refused before anything is written (W52).
        #     Every step after reads this record, and the snapshot is made
        #     from it.
        environment = machine_record(base, target)
        require_activation(target, environment, base=base)

        #     THE INPUTS -- one assembly per kind (`prep_inputs`, A12).  A
        #     bench measures ONE stage's configuration, and there is always a
        #     stage to name (§ 6.5); its grid is enumerated from the record
        #     just read.
        sweep = pins = translation = None
        chosen: dict = {}
        container = None
        if kind == "bench":
            from ..paths import Shape
            from ..paths import bench_container
            sweep, pins, translation = bench_inputs(
                base, target, notes=notes, task=task,
                environment=environment, template_text=template_text)
            container = base / bench_container(Shape.named(task.shape),
                                               stage_home(base, task, stage).token)
        else:
            allocation, pins, chosen = prep_run_inputs(
                base, task, stage, allocation)

        #     STEP 2, ONCE (`script-preparation.md` § 3.0): the allocation
        #     folded and the stage resolved -- what the checks below ask of
        #     the job, and what every step after reads.
        resolved = _resolve_stage(
            stage, task=task, template=template, template_text=template_text,
            environment=environment, allocation=allocation, chosen=chosen,
            sweep=sweep, pins=pins, translation=translation)
        if kind == "run":
            #  ...AND ITS GPU REQUEST AGREES WITH ITSELF (`execution/gpu.md`
            #  G5): a run on the GPU states how many, a run on the CPU none
            #  -- asked of the job this prep will write, before anything is.
            from .model import GpuRequestError, gpu_request
            try:
                gpu_request(resolved.pset.elements[0].resources)
            except GpuRequestError as exc:
                raise PrepError(f"stage {stage!r}: {exc}") from None

        #     ...AND EVERY LAUNCH VALUE IS STATED, OR THE PREP IS REFUSED --
        #      here, with the whole assembly in hand, before the question and
        #      before anything is written (`architecture.md` § 5.2; user,
        #      2026-10-02: "explicit job config is the only way allowed").
        #      A run states its processes; a run or a benchmark that this
        #      prep writes a `.sbatch` for states its queue, wall and memory.
        #      The target's record CHECKS an ask -- its queues are shown so
        #      one can be named -- and supplies no value of it.
        from .placement import launch_refusal
        why = launch_refusal(
            resolved.allocation,
            engine=task.engine, shape=(kind == "run"), stage=stage,
            header=bool(emit_sbatch and environment.scheduler == "slurm"),
            queues=[d.name for d in (environment.domains or ())],
            base=base, target=target)
        if why:
            raise PrepError(why)

        # 4a · WHAT IT BUILDS ON (`job-system.md` § 5.4, plan W37): the run
        #      a stage continues from -- the stage before it, newest, by
        #      default -- or the `relax` run a force-constant stage takes its
        #      geometry from (`engines/vibration.md` § 5.2a); a transport
        #      rung's inputs from the runs upstream.  Read BEFORE anything is
        #      written, so a refusal leaves nothing behind; a named run is
        #      taken as said.
        from .continuation import continuation_answer, force_constant_stage
        continuation = gather = None
        if stage is not None and (kind == "run"
                                  or force_constant_stage(task, stage)):
            # A BENCHMARK of a force-constant stage writes every trial at
            # the geometry `relax` reached, as its run is (§ 5.2a's table):
            # the default, and nothing carried -- a trial measures from its
            # deck.
            continuation, refused = continuation_answer(
                base, task, stage, from_attempt=from_attempt, cold=cold)
            if refused:
                raise PrepError(refused)
        if kind == "run" and is_transport:
            gather = gather_sources(base, task, stage)
        if on_found is not None:
            on_found(findings, notes)

        # 4b · THE STEPS, PLANNED -- every deck, wrapper and data file, the
        #      plan's row, the attempt and what it receives, decided with
        #      nothing written (`job-system.md` § 5.0, rule 3;
        #      `jobset.planned`): a refusal from here writes nothing but its
        #      line in the ledger, so there is nothing to put back.
        from .planned import Plan
        plan = Plan()
        opened: list = []
        dirs = prep_calculation(base, stage, env=env,
                                emit_sbatch=emit_sbatch, sweep=sweep,
                                target=target, opened=opened,
                                findings=deck_findings,
                                continuation=continuation, gather=gather,
                                plan=plan, resolved=resolved)
        flat = not resolved.shape.keeps_attempts_as_directories
        # THE PLAN, NAMED -- and a preview's, held to: a prep that would
        # write something else than the plan the person looked at refuses
        # (`job-system.md` § 5.0).
        identity = plan.identity()
        if plan_id is not None and plan_id != identity:
            raise PrepError(
                "the folder changed since the preview -- what prep would "
                "write now is not the plan you looked at.  Preview again "
                "(job-system.md § 5.0).")

        # THE ANSWER, from the plan -- what a preview shows and a prep
        # records once it is written.
        seen: set = set()
        # ONE ENTRY PER FOLDER: on the flat layout every stage's folder is
        # the calculation's one, and the answer listed it once per stage --
        # "prepped 3 job dir(s)" for one stage (W52).
        out = PrepAnswer(
            kind, stage, findings=findings, notes=notes,
            dirs=list(dict.fromkeys(dirs)),
            # ONE OF EACH: a sweep's trials repeat one finding per deck, and
            # the terminal said each once (`prep_calculation`).
            deck_findings=[i for i in deck_findings
                           if not (repr(i.to_json()) in seen
                                   or seen.add(repr(i.to_json())))])
        # THE PIPELINE LOG, which every prep writes (`script-preparation.md`
        # § 4.5): where this one is.
        from ..pipeline_log import log_name
        out.pipeline_log = (container or base) / log_name(
            task.label, stage_home(base, task, stage).token or "", task.engine,
            task.shape)
        a = resolved.allocation
        out.allocation = {"domain": a.domain, "time": a.time, "mem": a.mem}
        out.bench_axes = ({k: list(v) for k, v in (task.bench or {}).items()}
                          if kind == "bench" else {})
        out.chosen = dict(chosen) if kind == "run" else {}
        rep_stage = run_dir = job = None
        if kind == "run":
            js = JobSet.from_dict(json.loads(plan.read_text(
                base / JOBSET_FILENAME)))
            # THE ATTEMPT -- opened once, with what it continues from
            # (`_open_attempts`; until 2026-10-01 it was opened a second time
            # here, and a refusal between the two left an earlier carry
            # undone -- W52).  A later attempt is `launch`'s.  Flat keeps no
            # attempt directories: the run is the calculation's folder.
            from ..template import KIND_ROLES
            out.linked = (getattr(task, "calculation", None)
                          or "optimization") in KIND_ROLES
            out.cold = bool(cold)
            out.continuation = continuation
            # WHAT EACH ATTEMPT GATHERED, as decided at 4a: the attempt
            # opened in each container.
            got = [(next((Path(at.dir) for at in opened
                          if Path(at.dir).parent.resolve()
                          == Path(c).resolve()), None), volts, inputs)
                   for c, volts, inputs in (gather or ())]
            from ..transport.stages import scan_points
            if flat:
                out.flat = True
                run_dir, rep_stage = base, stage
            elif is_transport and scan_points(task, stage):
                out.points = got
            else:
                rep = opened[0]
                out.attempt = rep
                out.gathered = [pair for _a, _v, g in got for pair in g]
                run_dir, rep_stage = rep.dir, rep.stage
            # WHAT IT WILL LAUNCH WITH, and whether the deck agrees: `launch`
            # refuses a deck rendered for another width, and prep is the step
            # that exists so there are no surprises there (`agreement.py`).
            # A13: the end point, as the header and the run script carry it.
            job = next((j for j in js.jobs if j.name == rep_stage), None)
            if job is not None:
                r = job.resources
                out.resources = {"mpi_np": r.mpi_np,
                                 "cpus_per_task": r.cpus_per_task,
                                 "continue_retries": r.continue_retries}
                out.deck = Path(job.script).name
                out.launch = _launch_as_written(plan, run_dir, job)
                from .agreement import launch_agreement
                agreement = launch_agreement(
                    run_dir, job,
                    text=plan.read_text(Path(run_dir) / out.deck))
                if agreement.verdict != "silent":
                    out.agreement = agreement
        if preview:
            out.preview = True
            out.plan_id = identity
            out.writes = [str(Path(w).relative_to(base))
                          for w in plan.writes()]
            return out

        # 5 · THE SAVE -- always, once the whole plan stands and before
        #     anything is written (`checkpointing.md` § 9; user, 2026-10-03:
        #     "always save through checkpoint, notify user").  A redo is a
        #     rollback (2a), and this is the state it restores, so a save
        #     that fails refuses the prep.
        _record_preflight()
        from ..checkpoint import CheckpointError, save_before
        try:
            # A molbuilder.json that does not read is the person's to fix,
            # said in its own words -- the save is the first to read it.
            with _user_error_as_prep():
                kept = save_before(base, f"prep {kind} {stage}",
                                   engine=str(task.engine))
        except CheckpointError as exc:
            raise PrepError(f"the folder's state could not be saved, so "
                            f"nothing was prepped: {exc}")
        out.saved = kept.said()
        ledger(base, "prep", "saved", stage=stage, state=kept.state.short,
               note=kept.state.note, new=kept.new)

        # 6 · THE PLAN, WRITTEN -- deciding nothing and refusing nothing.  A
        #     write that fails (a full disk) leaves the stage not prepped --
        #     its row in `job-set.json` is the last thing written -- and the
        #     state just saved is the way back.
        plan.carry_out()

        # 7 · THE RECORD: what it continues from, the deck's agreement with
        #     its launch, *prepped* -- the answer both doors show.
        if kind == "run":
            if continuation is not None:
                # THE DECISION, LOGGED (`job-system.md` § 5.4): which run,
                # by default or named, what it was, and what came across.
                ledger(base, "prep", "continues", stage=stage,
                       **continuation.ledger_facts(),
                       copied=list(out.attempt.copied)
                       if out.attempt is not None else [])
            elif cold:
                ledger(base, "prep", "starts-cold", stage=stage)
            if out.agreement is not None:
                ledger(base, "prep", "launch-agreement", stage=rep_stage,
                       verdict=out.agreement.verdict,
                       rendered_for=out.agreement.rendered_text,
                       launching_at=out.agreement.launch_text)
        return _finish(out)
    except PrepError as exc:
        raise _refused(exc)
    except (UnknownTarget, AmbiguousTarget, ValueError, KeyError) as exc:
        # WHICH MACHINE is this for -- an answer only the person has
        # (`preparing-for-another-machine.md` § 4) -- and the plain
        # `ValueError`/`KeyError` the steps raise for what is the USER'S to
        # fix (a template naming an item its schema does not declare, a
        # bundle written before a rename), said the same way on both doors;
        # the Task setup route answered them 400 while the terminal showed a
        # traceback.  A `TypeError` is not translated: it is a bug, and
        # should look like one.
        raise _refused(PrepError(str(exc))) from exc


__all__ = ["prep_calculation", "prep_jobset", "prep_stage", "PrepAnswer",
           "prepped_already", "prepped_stages"]
