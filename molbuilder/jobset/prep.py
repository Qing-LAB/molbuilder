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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from typing import Callable, List, Optional, Sequence, Tuple

from .. import script_emit as _sc
from .materialize import (job_dir_names, shape_of, materialize,
                          write_gathered_from)
from ..issues import calling as _calling
from .model import FILENAME as JOBSET_FILENAME, Job, JobSet, Resources
from .plan import FILENAME as _PLAN_FILE
from ..runfiles import compose as _rf, stem as _rf_stem
from ..pseudos import PSEUDO_DIRNAME


class PrepError(Exception):
    """A prep refused -- in the reader's own words, which each surface shows
    as they are.

    What the one prep entry had already found when it refused rides with it
    (`prep_stage`): ``findings`` (the description's preflight notes),
    ``notes`` (what its inputs said -- a bench's grid with every crossed-out
    cell, a run's sizing) and ``partial`` (the answer so far, once the five
    steps have written something).  A refusal that says *see the crossed-out
    list above* is only honest if the list is shown with it; the command line
    prints these before the error, the Task setup route returns them beside
    it.  Empty on a refusal raised anywhere else.
    """
    findings: tuple = ()
    notes: tuple = ()
    partial: "Optional[PrepAnswer]" = None


from contextlib import contextmanager


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

    **The hook boundary's attribution is carried across** (§ 4.6). ``str(exc)``
    does not include an exception's notes, so a refusal that came out of a
    named engine hook would arrive at the CLI saying what went wrong and not
    whose it was -- and the note would reach a traceback nobody sees."""
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
_NO_RECORD = (
    "no machine record for this machine, so there is nothing to prep "
    "against.\n"
    "  Record it, once:\n"
    "      {cmd}\n"
    "  A machine's cores, GPUs and queues are read from a record and never "
    "probed on the fly -- so the numbers in a wrapper can always be traced "
    "to a file you can look at (running-a-job.md 3.1)."
)


def resolve_target(base_dir, target: Optional[str] = None) -> Path:
    """**Step 1 of the five: resolve the machine** (`project-layout.md`
    § 2.3.1) — read the machine's record (its cores, GPUs, scheduler and
    environment) and snapshot it as ``environment.json`` beside the bundle.

    **This step existed only inside the benchmark until 2026-08-10.**
    `bench/prep.py` did it; `prep_jobset` did not do it at all, so a staged
    calculation went straight to rendering wrappers on a machine nobody had
    asked about. § 2.3.1a is explicit about how to read that: *"`bench prep`
    is the one place this framework is already built, and it was built inside
    the benchmark because that is where the need appeared first … the general
    part needs lifting out of it"* — and *"stating it the other way round
    would make the general case look like a special case of the special
    case."*

    So the module moved out of `bench/` and became ``molbuilder/environment``.
    Its persisted artifact was **already** registered then, as
    ``molbuilder/environment@1`` (`job-contracts.md` § 6.1; ``@2`` since
    2026-08-17), which is the schema saying it was never the benchmark's to
    own.

    Written once per bundle and **not** overwritten on a later prep: the file
    records what this machine is, and re-reading on every stage would make two
    stages of one calculation disagree about their own target for no reason a
    user asked for.  It names the machine the calculation is set to -- this
    prep's ``--target`` -- and that does not change: another record reaches
    the calculation only through a new prep, from a state saved before this
    one (`configuration.md` M-3).  A preview reads without writing
    (:func:`_environment_read`).

    **IT DOES NOT PROBE.  A machine that has no record is a REFUSAL**
    *(user, 2026-09-02: "all environments have to be explicitly probed and
    stored. no environment json, error")*, and it names the one command that
    fixes it.

    This step used to run a fresh probe and write the answer down whenever no
    scope answered — which read as helpful and is the guess
    `running-a-job.md` § 3.1 forbids: the numbers a wrapper carries would
    then come from *whichever box happened to run prep*, and for a bundle
    described at a desk and run on a cluster that is the wrong machine, with
    a number that looks exactly like a right one.  Probing is one command and
    it is the user's to run, so the record is always something they can point
    at and say where it came from.

    Returns the path to ``environment.json``.
    """
    from ..scheduler import machine_for, write_environment
    from ..scheduler.record import calculation_record
    out = calculation_record(base_dir)
    if out.is_file():
        return out
    # `machine_for()` WITHOUT a bundle: the calculation has no record yet (we
    # just early-returned if it did), so this is the MACHINE scope -- what
    # `jobset probe` wrote.  Snapshotting that answer rather than re-probing
    # is what makes one probe serve every calculation here
    # (configuration.md § 5, M-3).  ``target`` names WHICH machine this is
    # for (P2); an unknown name is `machine_for`'s own error, naming the ones
    # that exist.
    #
    # NO `probe=`.  Nothing here detects anything: a record is read or the
    # prep stops.
    env = machine_for(target=target)
    if env is None:
        raise _no_record()
    # THE MACHINE IT IS SET TO, named in the copy (M-3): a later prep's
    # `--target` is checked against it.  No target is this machine --
    # `machine_for` refuses the question when another is on file.
    from dataclasses import replace
    from ..scheduler.record import LOCAL_TARGET
    return write_environment(replace(env, machine=target or LOCAL_TARGET),
                             out)


def _no_record() -> PrepError:
    """The refusal of THIS machine with no record, naming the probe that
    writes one (`scheduler.record.probe_line`, W52).  Only this machine can
    have none: a named target's record is there, or `machine_for` refuses
    the name itself (W54 R10 -- a branch for a named target stood here, and
    could not be reached)."""
    from ..scheduler.record import probe_line
    return PrepError(_NO_RECORD.format(cmd=probe_line(None)))


def _environment_read(base: Path, target: Optional[str] = None):
    """Step 1's ANSWER, read -- the record `prep` snapshots
    (:func:`resolve_target` writes it, after this is checked), and all a
    preview asks, which writes nothing (`web/task-setup.md` § 11.1).  The
    bench card snapshotted on every edit until 2026-10-01, so looking at a
    calculation with the picker on one machine tied it to that machine
    before anything was prepped (W52).  Refuses as step 1 does when no
    record answers."""
    from ..scheduler import machine_for
    env = machine_for(base, target=target)
    if env is None:
        raise _no_record()
    return env


def _flat_continued_from(base: Path, task, stage: str, continuation) -> None:
    """A flat stage's own ``<basename>.continued-from``, for its launch
    record (`project-layout.md` § 1.6.3) -- the flat layout records what a
    stage continues from too (user, 2026-10-01).  It names the run by what
    every file of it carries (`runfiles.run_name`): flat keeps no directory
    per run.  A re-prep that no longer continues -- the run card now says
    `clean` -- takes the old one away, or the launch would record a source
    it did not read."""
    from ..runfiles import latest_run, run_name, stem as rf_stem
    from .materialize import continued_from_marker
    marker = continued_from_marker(base, rf_stem(task.label,
                                                 token_for(task, stage)))
    if continuation is None:
        if marker.is_file():
            marker.unlink()
        return
    token = token_for(task, continuation.stage)
    n = latest_run(base, task.label, stage=token)
    if n is None:
        return                     # concluded with no run file: name none
    marker.write_text(run_name(task.label, token, n) + "\n",
                      encoding="utf-8")


def prep_jobset(jobset: JobSet, base_dir, *, env: str = None,
                emit_sbatch: bool = True, record_dir=None,
                log=None, machine_record=None) -> List[Path]:
    """Render launchers + lay out the per-job tree under ``base_dir``.

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
    name ``materialize.job_dir_name`` records retiring before that.  A public
    entry point whose docstring describes a deleted design misleads at the
    lines a caller actually reads.
    """
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

    # ---- 0. resolve the machine (§ 2.3.1 step ONE) ---------------------- #
    # Idempotent by contract (resolve_target early-returns on an existing
    # environment.json).  Every production caller is the described route,
    # whose `prep_calculation` already ran step 1; this call serves a direct
    # caller of `prep_jobset` (its tests) -- not a re-decision.
    resolve_target(base)

    # ---- 1. render wrappers once per distinct script, IN THE JOB DIR --- #
    # Nothing rendered lives at the bundle root (user, 2026-08-24;
    # `project-layout.md` § 1.0).  The deck was born in its directory by
    # `prep_calculation`; a deck a caller rendered at the root (hand-built
    # JobSets, pre-2026-08-24 bundles) is ADOPTED -- moved in, once --
    # so the root ends clean either way and the wrapper is written beside
    # the deck it launches.
    if log is not None:
        log.phase("STEP 4 · WRAPPERS — how each deck is launched")
    _sh = shape_of(jobset, base_dir)
    _dir_of = job_dir_names(jobset, _sh)
    rendered: dict = {}
    for job in jobset.jobs:
        # THE SAME QUESTION `prep_calculation` AND `materialize` ASK.  A
        # trial's files live in its attempt when the shape keeps them
        # (`project-layout.md` § 1.5a); this loop looked in the container
        # and reported the deck missing, because it was the one writer
        # that had not been told.
        _jd = base / _dir_of[job.name]
        if jobset.kind == "sweep":
            from .materialize import trial_work_dir
            _jd = trial_work_dir(_jd, _sh)
        _jd.mkdir(parents=True, exist_ok=True)
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
            from shutil import copy2 as _copy2
            _src_dir = rendered[job.script]
            _stem0 = Path(job.script).stem
            for _fn in (job.script, f"{_stem0}.run.sh", f"{_stem0}.sbatch"):
                if (_src_dir / _fn).is_file() and not (_jd / _fn).is_file():
                    _copy2(_src_dir / _fn, _jd / _fn)
            if log is not None:
                log.note(f"{job.name}: shares {job.script}'s wrapper, "
                         f"copied from {_src_dir.name}/")
            continue
        script_path = _jd / job.script
        if not script_path.is_file():
            _root_copy = base / job.script
            if _root_copy.is_file() and _root_copy != script_path:
                _root_copy.replace(script_path)      # adoption, not a copy
            else:
                raise PrepError(
                    f"job {job.name!r}: script {job.script!r} not in "
                    f"{_jd} (render the inputs before prep).")
        with _user_error_as_prep():
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
                label=jobset.name,
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
                # THE JOB'S LAST STEP, when its engine leaves no result
                # (`Job.finish`, `engines/vibration.md` § 5.5): the wrapper
                # runs it, and its bundle is written beside the deck.
                finish=job.finish,
                # WHETHER A RE-RUN CONTINUES (`Job.resumes`): the wrapper
                # says a retry of a run that cannot resume repeats it
                # (`running-a-job.md` § 3.5).
                resumes=job.resumes,
            )
        rendered[job.script] = _jd
        if log is not None:
            _stem = Path(job.script).stem
            log.received(job.script, _flat_resources(job.resources)
                         + (f", env={env}" if env else ""))
            for _w in (f"{_stem}.run.sh", f"{_stem}.sbatch"):
                if (_jd / _w).is_file():
                    log.produced(_w, f"{len((_jd / _w).read_text().splitlines())}"
                                     f" lines")
                elif _w.endswith(".sbatch"):
                    log.note(f"{_w}: not written "
                             + ("(emit_sbatch off)" if not emit_sbatch
                                else "(no scheduler configured)"))

    # ---- 2. the shared package, copied --------------------------------- #
    dirs = materialize(jobset, base)
    if log is not None:
        log.phase("STEP 5 · RUN DIRECTORY — where each job will be launched")
        log.received("shape", str(shape_of(jobset, base_dir)))
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
        from ..warmfiles import load_warm_files
        _vocab = f"warm-files: {load_warm_files(jobset.engine, base).path}\n"
    except Exception:
        _vocab = ""    # an engine without a rules file has no line to print
    (plan_dir / _PLAN_FILE).write_text(
        render_plan(jobset) + "\n\n" + _vocab
        + format_provenance(config_provenance(project_dir=base)) + "\n",
        encoding="utf-8")
    if log is not None:
        log.produced(_PLAN_FILE, str(plan_dir / _PLAN_FILE))

    return dirs


# --------------------------------------------------------------------- #
#  The five steps, entire — `project-layout.md` § 2.3.1                  #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class EngineSeam:
    """What an engine supplies for `prep` to run the steps over it —
    `script-preparation.md` § 4's seam, stated as data.

    **That document indexes the questions by the STEP that asks them**, which is
    the ordering to read them in: a bag of callables cannot answer *"what does
    this engine still owe?"*, and against the steps a gap is a blank row.

    **Fifteen questions, ten members.**  One is answered by shared code
    (`validation.validate`), and four arrive together through ``spec_for`` --
    the layout, the syntax, the record's values and the check rules are all the
    engine describing its deck, so they ride on one ``DeckSpec`` rather than on
    four seam members.  A member that answers ``None`` is answering *nothing*,
    which is a real answer and a recorded one (§ 4, W5): PySCF gives it to
    ``provide_data``, ``shared_package`` and ``sibling_artifacts``, and to
    ``bench_marks`` on the spec.

    Everything engine-specific that the loop below needs lives HERE, so the
    loop itself never asks which engine it is in.  ``_job_for`` branched on
    ``task.engine == "siesta"`` until 2026-08-12, which was § 7's forbidden
    ``if`` one floor down from where it was deleted.
    """
    #: The config class the template rebuilds into.
    config_cls: type
    #: ``(structure, config, stage_token=) -> DeckSpec`` — the engine
    #: DESCRIBES its deck; the framework renders, writes and checks it
    #: (`script-preparation.md` § 4.3).  The token is a RENDER ARGUMENT (step
    #: 7, C7): the emitter never learns the word, the deck's filename carries
    #: it.
    #:
    #: **It handed back finished TEXT until 2026-08-18**, and that one fact was
    #: what kept the framework's step-3 runner unreachable: given text, the
    #: conductor had no form to pass on, so it performed the write and the
    #: check itself and the ORDER of step 3 was stated in two places.  Given a
    #: form, the framework can also re-derive what the deck was supposed to
    #: contain, so nothing has to be carried alongside the text to make the
    #: check possible.
    spec_for: Callable
    #: The deck's type suffix (``.fdf``).
    suffix: str
    #: ``config -> the engine's identity literal`` (``SystemLabel`` / ``JOB``).
    label_of: Callable
    #: ``(config, label) -> config`` — the identity WRITTEN, for a trial's
    #: relabelling.  Filename relabelling alone is not the § 2.3.2
    #: protection: the deck's own ``SystemLabel`` line is what keys the warm
    #: files, and until 2026-08-12 it kept the run's label (found by the
    #: first sweep that ever rendered a deck).
    relabel: Callable
    #: ``(label, config, calculation, base_dir) -> warm-file declaration``
    #: for the Job -- the engine reads its § 4.2a rules file for the TYPE
    #: (U2), and ``base_dir`` lets a calculation's own fine-tuned copy win
    #: (U6a; the comment said 3 args while the call passed 4 -- C-d).
    warm_for: Callable
    #: ``config -> traits`` the launcher routes on (GPU solver, …).
    traits_for: Callable
    #: ``(struct, config, deck_path) -> None`` — the sibling files this
    #: engine's deck TEXT promises (E6: a charged SIESTA deck instructs
    #: running a script; the promise must be kept on every route that
    #: renders the deck).  ``None`` for an engine whose decks promise
    #: nothing.
    sibling_artifacts: Optional[Callable] = None
    #: ``(base_dir) -> [filename]`` — which files in the calculation are the
    #: SHARED PACKAGE every job links to.  The engine that put them there is
    #: the one that can name them: this was a ``*.psml`` glob in shared code,
    #: a SIESTA fact stated a floor below where SIESTA may speak, so a second
    #: engine with data files of its own would have shipped none of them.
    #: ``None`` for an engine that puts nothing in — the package is then empty,
    #: which is the honest answer rather than an accident of a glob.
    shared_package: Optional[Callable] = None
    #: ``(struct, config, base_dir) -> None`` — the DATA FILES this engine's
    #: deck cannot run without, put into the calculation.
    #:
    #: *Named ``stage_data`` for about a minute: "stage" is this project's
    #: core noun and here it was being borrowed as a verb -- the collision
    #: `submit._staged_for_launch` was renamed for.*  Distinct from
    #: ``sibling_artifacts``, which is about what a deck's own text promises;
    #: this is about what the ENGINE will open.  ``None`` for an engine that
    #: needs none (PySCF's basis sets ship inside PySCF).
    #:
    #: It belongs to `prep` because `project-layout.md` § 2.6 puts the copy on
    #: the machine that runs the job — where the library lives is a fact about
    #: that machine — and because `prep` is already what decides the shared
    #: package.  Added 2026-08-18: the rule *"a calculation copies the
    #: pseudopotentials it needs into its own shared package"* was written and
    #: unowned, so `jobset init` performed it and the browser's hand-over
    #: did not, and a calculation described in the browser prepped, laid out
    #: its directories and reported success with no pseudopotentials in it.
    provide_data: Optional[Callable] = None


def _siesta_sibling_artifacts(struct, cfg, deck_path: Path, *,
                              kind: str) -> None:
    """The sibling files a SIESTA deck's own text PROMISES.

    A charged deck instructs ``python3 makov_payne_correction.py`` in its
    header -- a promise only ``convert`` kept until E6 (redo 2026-08-12):
    the described route rendered the same header and never wrote the
    script, so `prep` shipped an instruction to run a file that did not
    exist.  Same writer both routes, so they cannot drift.

    The charge is the electronic state's (`science/chemistry-correctness.md`
    § 2a) -- the one the deck beside it was written from.  The deck has just
    been written from that state, so a label naming no element cannot reach
    here.  ONLY FOR A FINITE SYSTEM (§ 2b): the script's formula is a
    molecule's in a vacuum box, and a charged slab or crystal -- whose deck
    says it gets no formula -- got the script too until the M6 review."""
    from ..electronic_state import electronic_state
    from ..siesta.makov_payne import emit_correction_script
    state = electronic_state(struct, cfg, kind=kind)
    q = state.net_charge.value
    if q != 0 and state.finite:
        emit_correction_script(fdf_path=deck_path,
                               system_label=cfg.system_label, q=q)


def _pseudo_dir(base: Path) -> Path:
    """THE PARENT'S DATA IS GROUPED (roadmap 7.10 M6): the calculation's
    pseudopotential copies live in ``pseudos/``, one folder, instead of
    N ``<El>.psml`` entries loose at the root.  Root strays -- put there
    by `init`, an earlier prep, or a travelled bundle -- are ADOPTED,
    the same move-in the deck adoption uses.  The run directories are
    untouched by this: each still receives ``<El>.psml`` beside the
    deck (that is SIESTA's own contract; it has no search path).

    ONE rule, two providers: the SIESTA arm below and the transport
    composite's (which fetches from the citation instead of a library).
    """
    pdir = base / PSEUDO_DIRNAME
    pdir.mkdir(exist_ok=True)
    # A container, and it says so (`project-layout.md` § 1.4a): the shared
    # package holds files, never a run.  Left unstamped it was the directory
    # that reported a calculation *running* because it had no result file in
    # it -- which is the shape of answer § 1.4a exists to stop.  It reads back
    # as *support* rather than a stage, because the naming authority maps no
    # job to it; that is derived, not stored.
    from .. import calcdirs
    calcdirs.write(pdir, role=calcdirs.CONTAINER, root=base)
    for stray in base.glob("*.psml"):
        target = pdir / stray.name
        if not target.exists():
            stray.replace(target)
        else:
            stray.unlink()
    return pdir


def _siesta_provide_pseudos(struct, cfg, base: Path) -> None:
    """Put the pseudopotentials this deck needs into the calculation.

    SIESTA opens ``<element>.psml`` in the directory it runs from and has no
    search path, so a missing file is not a preference — it is a run that
    cannot start, after a queue wait and however long MPI takes to come up.

    **Idempotent, and the folder wins.**  Anything already here was put here by
    an earlier prep, by `jobset init`, or by travelling with the folder;
    `copy_pseudopotentials` leaves it alone.  Only what is missing is fetched,
    from the library named by ``psml_lib``.

    **The species come from the STRUCTURE**, which `prep` has just loaded and
    checked against the description's witness — not from a list in the
    description.  A recorded list would be a second answer to *which elements
    is this calculation of*, and the structure is the first.

    A species in neither place stops `prep` **by name**, before a deck is
    written.

    **And then the science protocol runs on what is actually there.**
    `science/pseudopotentials.md` exists because a defective `S.psml` with a
    dead p-channel shipped into a real run on 2026-06-26: wrong sulfur bonding,
    and `propor: ERROR: IMAX=0` — but only at high rank counts, so a small run
    would have reported plausible, wrong numbers instead of crashing. The check
    that catches that class reads the pseudopotentials themselves.

    It has always run against ``psml_lib`` — the LIBRARY — and it is gated on
    that field being set, so a calculation whose files are already beside it and
    whose ``psml_lib`` is empty had **nothing checked at all**: the only thing
    said was *"psml_lib is not set … once set, this preflight will check
    coverage"*, while three real pseudopotentials sat in the folder the run
    would open them from. This step makes that state the normal one, so it runs
    the protocol here, against the **calculation** — which is where the files
    the run reads actually are — and refuses on the same ERROR statuses the
    preflight and `molbuilder pseudo check` refuse on, from the same shared
    constant.
    """
    from ..pseudos import psml_sources, resolve_psml_lib
    from ..siesta.input import copy_pseudopotentials
    from ..chemistry import species_order

    species = species_order(struct.elements)
    if not species:
        return
    pdir = _pseudo_dir(base)
    # THE FOLDER WINS, by the one rule the settings gate asks too.
    want = [s for s, d in psml_sources(species, dest_dir=base).items()
            if d is None]
    if not want:
        _screen_pseudos(species, cfg, pdir)
        return

    lib_raw = getattr(cfg, "psml_lib", None)
    if not lib_raw:
        raise PrepError(
            f"this calculation needs pseudopotentials for "
            f"{', '.join(want)} and none are in {base.name}/, but no "
            f"pseudopotential directory is set.  Set `psml_lib` in the "
            f"template to the library they live in -- the convention is the "
            f"bare name `pseudopotential`, which means the projects tree "
            f"this calculation lives in (project-layout.md § 2.6, "
            f"job-contracts.md § 2.5a).")
    from ..pseudos import PsmlLibError
    try:
        lib = resolve_psml_lib(str(lib_raw), dest_dir=base)
    except PsmlLibError as exc:
        raise PrepError(str(exc))
    if not lib.is_dir():
        # Name the anchor the SPELLING asked for, in the rule's own words.
        # This used to print only the resolved path, which under the old
        # cascade was whichever candidate was tried last -- on Sol that was
        # `<calc>/projects/pseudopotential`, a folder assembled from the
        # user's working directory that nobody had chosen (2026-08-21).
        from ..pseudos import describe_psml_anchor
        raise PrepError(
            f"this calculation needs pseudopotentials for "
            f"{', '.join(want)}, and the library they should come from is "
            f"not a directory.  "
            + describe_psml_anchor(str(lib_raw), dest_dir=base)
            + "  Put the .psml files there, or set `psml_lib` to a "
              "directory that has them.")
    missing = copy_pseudopotentials(want, lib, pdir)
    if missing:
        raise PrepError(
            f"this calculation needs {', '.join(f'{m}.psml' for m in missing)}"
            f" and there is none in {base.name}/ or in {lib}.  SIESTA opens "
            f"<element>.psml in the directory it runs from and has no search "
            f"path, so it would refuse at startup.  Put the file in the "
            f"library, or point `psml_lib` at one that has it.")
    _screen_pseudos(species, cfg, pdir)


def _screen_pseudos(species, cfg, base: Path) -> None:
    """`science/pseudopotentials.md` § 1, run on the calculation's own files.

    Same engine (`pseudos.check_coverage`), same severity set
    (`pseudos.ERROR_STATUSES`) and same XC-family table
    (`pseudos.expected_xc_family`) as the render-time preflight and the
    `molbuilder pseudo check` CLI, so the three surfaces cannot disagree about
    what blocks.  What differs is only WHICH DIRECTORY is read: this one asks
    about the files the run will open.

    The three blocking statuses are `missing`, `dead_projector` and
    `xc_family_mismatch` — a file absent, a valence channel physically absent,
    or the wrong XC family.  The rest are advisory, and the settings gate
    reports them in the deck's report: it reads these same files
    (`pseudos.psml_sources`), so printing them here too said each one twice.
    """
    from ..pseudos import ERROR_STATUSES, check_coverage, expected_xc_family
    entries = check_coverage(
        species, base,
        expected_xc_family=expected_xc_family(
            getattr(cfg, "xc_authors", "") or ""),
        expected_xc_authors=(getattr(cfg, "xc_authors", "") or "") or None,
    )
    blocking = [e for e in entries if e.status in ERROR_STATUSES]
    if blocking:
        raise PrepError(
            "the pseudopotentials in this calculation do not pass the "
            "screening (science/pseudopotentials.md § 1):\n  - "
            + "\n  - ".join(f"{e.element}: {e.message}" for e in blocking)
            + "\n  These are the checks that exist because a dead-channel "
              "S.psml once shipped into a real run -- wrong bonding, and a "
              "propor IMAX=0 crash that only appeared at high rank counts.")


def _engine_seam(engine: str) -> EngineSeam:
    if engine == "siesta":
        from ..config.siesta import SiestaConfig
        from ..siesta.input import spec_for as _siesta_spec
        from ..siesta.stages import _traits, _warm_declaration
        return EngineSeam(config_cls=SiestaConfig, spec_for=_siesta_spec,
                          suffix=".fdf",
                          label_of=lambda cfg: cfg.system_label,
                          relabel=lambda cfg, label: dataclasses.replace(
                              cfg, system_label=label),
                          warm_for=_warm_declaration, traits_for=_traits,
                          sibling_artifacts=_siesta_sibling_artifacts,
                          provide_data=_siesta_provide_pseudos,
                          shared_package=_siesta_shared_package)
    if engine == "pyscf":
        from ..config.pyscf import PySCFConfig
        from ..pyscf.input import spec_for as _pyscf_spec
        from ..pyscf.stages import _traits as _pyscf_traits
        from ..pyscf.stages import _warm_declaration as _pyscf_warm
        # NO ``provide_data`` and NO ``sibling_artifacts``, and both absences
        # are ANSWERS rather than omissions (`script-preparation.md` § 4, W5):
        # PySCF's basis sets ship inside PySCF, so there is no file to put in
        # the calculation; and its script's own text instructs nothing to be
        # run beside it, so there is no promise to keep.
        return EngineSeam(config_cls=PySCFConfig, spec_for=_pyscf_spec,
                          suffix=".py",
                          label_of=lambda cfg: cfg.job_name,
                          relabel=lambda cfg, label: dataclasses.replace(
                              cfg, job_name=label),
                          warm_for=_pyscf_warm, traits_for=_pyscf_traits)
        # NO ``shared_package``: PySCF's basis sets ship inside PySCF, so
        # there is nothing in the calculation for every job to link to.
    raise PrepError(
        f"no deck writer for engine {engine!r}. An engine supplies its "
        f"catalogue rows and an answer at each preparation step "
        f"(script-preparation.md § 4); this backend has neither for that "
        f"name.")


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


def _vibration_stage_geometry(base, task, pset, struct, *, log=None):
    """``(structure, cell, relaxation)`` a force-constant stage of a SIESTA
    vibration is written with -- `freq`, or any stage after `relax` in a
    displacement sweep, asked of `vibration_render_kind` and never of a name
    (`engines/vibration.md` § 5.2a, § 5.9; until 2026-09-28 only a stage
    named `freq` came here, so a sweep's second stage measured the unrelaxed
    input): the sorted
    copy as given, and no cell or record of its own, when the ladder holds no
    `relax` stage; the coordinates that stage relaxed to, in the cell it ran
    in, when it does -- read from its newest attempt, which must have
    concluded, through the one SIESTA output parser -- with that run's
    relaxation record (`parse.contract.relaxation_of_output`), of the same
    output and the same parse, which the deck's `vibration` block carries to
    the finish (§ 5.3) so nothing re-picks the attempt later -- and the
    structure leaves the input's own relaxation record behind, since it is
    about other coordinates (§ 5.2a).  The
    cell travels because the deck otherwise re-derives one around the new
    bounding box and shifts the atoms into it, and a relaxed geometry moved
    against the real-space grid is not stationary on that grid any more.
    Every other rung comes back unchanged.

    Two refusals, each naming what to do first.  A force-constant stage
    before `relax` has concluded: the job set's own order, not a guess at
    which geometry the force constants belong to.  And one with no `relax`
    stage while the structure is not stated relaxed: the box says *relax
    first* and the ladder holds nothing that would, so the description
    contradicts itself and is refused with the two ways out rather than
    measured at a geometry nobody chose (§ 2.2).
    """
    from ..pyscf.stages import VIBRATION_RELAX_STAGE, vibration_render_kind
    if vibration_render_kind(pset.stage) != "vibration":
        return struct, None, None
    # THE LADDER'S RELAXATION, by the one role rule -- the name in any case
    # (`engines/stages.md` § 2), never an exact string (plan § 5w K12).
    relax = next((s for s in task.stages
                  if s.enabled and vibration_render_kind(s.name) != "vibration"),
                 None)
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
    from ..paths import Shape
    from .materialize import attempt_concluded, run_dir, stage_stdout
    token = token_for(task, relax.name)
    container = base / Shape.named(task.shape).stage_dir(token)
    stem = _rf_stem(task.label, token)
    # THE COMMANDS, from the one composer: the calculation named, the mode
    # stated where its config sets none (`commands.run_first`).
    from .commands import block, command, run_first as _run_first
    run_first = block(_run_first(relax.name, base=base))
    # THE NEWEST ATTEMPT, and it must have concluded: a `relax` re-launched
    # to tighten is the geometry the person means, so an older concluded
    # attempt never stands in for one still running.
    attempt = run_dir(container)
    if attempt_concluded(attempt, stem) is None:
        raise PrepError(
            f"the `{pset.stage}` stage takes its geometry from the "
            f"`{relax.name}` stage, whose newest attempt has not concluded -- "
            f"it was never launched, is still running, or was force-stopped "
            f"-- `{command('status', relax.name, base=base)}` says which "
            f"(project-layout.md 1.6).  Let it finish, or run it first --\n"
            f"{run_first}")
    out = stage_stdout(attempt, task.label, token, str(task.engine))
    if out is None:
        raise PrepError(
            f"{attempt.relative_to(base)} concluded without the engine's "
            f"output for `{relax.name}`; there is no geometry to read.  "
            f"Re-run it --\n{run_first}")
    from ..parse.engines.siesta import SiestaParser
    from ..parse.errors import ParseError
    try:
        traj = SiestaParser.parse(str(out))
    except ParseError as e:
        raise PrepError(
            f"{out.relative_to(base)} could not be read as a SIESTA run: {e}.  "
            f"Re-run the `{relax.name}` stage --\n{run_first}") from e
    frames = [fr for fr in traj.frames if fr.structure is not None]
    if not frames:
        raise PrepError(
            f"{out.name} holds no coordinate block: the `{relax.name}` "
            f"run never reached its first geometry.  Re-run it --\n"
            f"{run_first}")
    last = frames[-1]
    if list(last.structure.elements) != list(struct.elements):
        raise PrepError(
            f"{out.name} describes {last.structure.formula} in an order "
            f"that is not this calculation's sorted copy ({struct.formula}): "
            f"the `{relax.name}` stage ran a different structure.  Re-run "
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


def prep_calculation(base_dir, stage: Optional[str] = None, *,
                     allocation=None, env: str = None,
                     emit_sbatch: bool = True,
                     sweep=None, pins=None, translation=None,
                     target: Optional[str] = None,
                     chosen=None,
                     pipeline_log: bool = False,
                     opened: Optional[list] = None,
                     findings: Optional[list] = None,
                     continue_from: Optional[str] = None,
                     cold: bool = False,
                     named: bool = True) -> List[Path]:
    """**`prep`, entire** — the five steps of `project-layout.md` § 2.3.1, in
    the order it calls *forced rather than chosen*.

    ``continue_from`` / ``cold`` / ``named`` are what the stage continues
    from, as :func:`prep_stage` decided it before anything was written: the
    attempt is opened ONCE, with its carry (`materialize.prepare_attempt`).
    Saying neither leaves a reused attempt's carry as it is.

    1. **resolve the machine** — READ its record, persist the snapshot as
       ``environment.json``; refuse when no record answers (§ 3.1);
    2. **resolve the parameters** — the description ⊕ this stage ⊕ the sweep ⊕
       the pins, into a :class:`~molbuilder.resolve.ParameterSet`;
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

    ``pipeline_log`` writes a step-by-step record of what each step received,
    decided and produced, beside this prep's ``STAGE-PLAN.md``. Off by
    default: it is an observer of the pipeline, never a step in it, and no
    generated artifact differs either way (`script-preparation.md` § 4.5).

    ``opened``, when given, receives the :class:`~molbuilder.jobset.materialize.Attempt`
    reports of the attempts this prep opened.  A caller reporting on the
    attempt reads its freshness from here: opening it a second time finds it
    already there, unlaunched, and calls it reused.

    ``findings``, when given, receives what each deck's checks said
    (`script_emit.prepare_deck`) -- the terminal also reads them on stderr as
    each deck renders; the Task setup tab reads them from here, through the
    one prep entry's answer.

    Returns the per-job directories. Raises :class:`PrepError`.
    """
    # TRANSPORT IS THE COMPOSITE (archive/2026-09-01-transport-design.md § 4.2): no
    # template, no structure reference -- its stages render from the
    # composed junction, so it takes its own arm rather than crashing
    # on the structure this calculation deliberately does not carry.
    from ..task import FILENAME as _TASK_FILENAME
    from ..task import read_task as _read_task_early
    try:
        _t_calc = _read_task_early(Path(base_dir) / _TASK_FILENAME).calculation
    except Exception:
        _t_calc = None   # no/invalid description: the named refusal below owns it
    if _t_calc == "transport":
        # `chosen` TRAVELS.  It is the run's launch shape -- what the person
        # decided on the run card, or `--np` on the command line -- and this
        # hand-off dropped it, so a transport run silently fell back to the
        # target's own width while every other calculation honoured it.  Both
        # sides have always declared the parameter; only the forwarding was
        # missing, which is why nothing named it.
        return _prep_transport(base_dir, stage, allocation=allocation,
                               env=env, emit_sbatch=emit_sbatch,
                               sweep=sweep, pins=pins,
                               translation=translation, target=target,
                               chosen=chosen,
                               pipeline_log=pipeline_log,
                               opened=opened, findings=findings,
                               continue_from=continue_from, cold=cold,
                               named=named)
    from ..pipeline_log import PipelineLog, config_rows
    from ..resolve import ResolveError, resolve
    from ..task import FILENAME as TASK_FILENAME
    from ..task import read_task
    from ..template import template_path as _template_path
    # The container spelling is materialize's (the naming authority): ONE
    # function places the sweep's record and the trials' directories, so the
    # two can never disagree (A-1/A-2).  The log's own home is the same
    # container, spelled by the same function.
    from .materialize import bench_container
    from ..paths import Shape

    base = Path(base_dir).resolve()
    if not base.is_dir():
        raise PrepError(f"calculation folder not found: {base}")
    desc = base / TASK_FILENAME
    if not desc.is_file():
        raise PrepError(
            f"no {TASK_FILENAME} in {base}. `prep` turns a DESCRIPTION into a "
            f"runnable directory; write one first with `jobset init`.")

    # ---- 1. resolve the machine ---------------------------------------- #
    # READ, CHECK, THEN WRITE.  A record that does not state how to enter the
    # named machine's environment is refused before anything is on disk --
    # the snapshot included, or the remedy's re-copied record would then
    # contradict it (W52, and its fix's review).
    environment = _environment_read(base, target)
    _require_activation(target, environment, base=base)
    resolve_target(base, target)          # step 1 proper: the snapshot

    # ---- 2. resolve the parameters ------------------------------------- #
    task = read_task(desc)
    # THE LOG OPENS HERE, and not before: its NAME carries the stage token, and
    # the token needs the description.  Everything step 1 did is still in hand,
    # so nothing is lost by writing that phase a moment later than it ran.
    #
    # It is not closed on a refusal and does not need to be: every line is
    # flushed as it is written, so a prep that dies leaves a file ending at the
    # step that refused -- which is the step a reader is looking for.
    log = None
    if pipeline_log:
        _token = token_for(task, stage)
        _record = (base / bench_container(Shape.named(task.shape), _token)
                   if sweep is not None else base)
        _record.mkdir(parents=True, exist_ok=True)
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
    # THE one place this name is formed (`template.template_path`).  Six
    # call sites spelled it in two incompatible ways until 2026-08-17.
    template_path = _template_path(base, task.label)
    if not template_path.is_file():
        raise PrepError(
            f"no {template_path.name} beside {TASK_FILENAME}. The portable "
            f"folder is a template PLUS a description (project-layout.md § 2.1) "
            f"and `prep` rebuilds the config from the template.")
    seam = _engine_seam(task.engine)
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
        pset = resolve(template_path.read_text(encoding="utf-8"), task,
                       seam.config_cls, allocation=(allocation or Resources()),
                       stage=stage, sweep=sweep, pins=pins,
                       translation=translation, environment=environment)
    except ResolveError as exc:
        raise PrepError(str(exc)) from exc
    if log is not None:
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

    # ---- 3. render the deck(s) ----------------------------------------- #
    struct = _structure_for(task, base)
    # A KIND THAT NEEDS THE ATOMS IN AN ORDER THE INPUT DOES NOT HAVE sorts a
    # COPY here, records the permutation beside the calculation, and renders
    # from the copy (`model/overview.md` § 2.2).  The vibration kind on SIESTA
    # is one: the force-constant run nudges one contiguous range, so the free
    # atoms go last under the 'held-first' key.  The record is what the
    # return leg -- the job's finish (`engines/vibration.md` § 5.5) --
    # inverts, from its copy in the attempt; the input order never
    # reaches the engine and the sorted order never reaches a person.
    # WHICH DECK THIS RUNG RENDERS, and IN WHICH FRAME.  Both are the
    # calculation's unless the kind says otherwise: the SIESTA vibration's
    # `relax` stage is the relaxation deck (`engines/vibration.md` § 5.2a),
    # and every force-constant stage -- every other stage, whatever its
    # name (`pyscf/stages.vibration_render_kind`) -- is written at the
    # coordinates the relax stage relaxed to, in that run's own cell, so the
    # force constants are taken on the grid the geometry was relaxed on.
    _render_kind = task.calculation
    _render_cell = None
    # A SIESTA force-constant deck carries a `vibration` block (below), built
    # per element from its resolved config and the relax run read here.
    _finishes, _relaxed_by, _criterion = False, None, None
    if task.calculation == "vibration" and str(task.engine) == "siesta":
        # A DISPLACEMENT SWEEP NEEDS A DIRECTORY PER STAGE (`engines/
        # vibration.md` § 5.9): SIESTA names its force constants and the
        # finish its spectrum by the label alone, so two force-constant
        # stages sharing the flat layout's one directory would overwrite the
        # first one's result.  Refused before any sort, permutation record or
        # deck is written, at whichever stage the person preps first -- and
        # counted over EVERY described stage, enabled or not, since a stage
        # named here is prepped either way (plan W38 F5).
        from ..spectra.displacement_sweep import stages_share_a_directory
        if stages_share_a_directory(task, include_disabled=True):
            from ..pyscf.stages import force_constant_stages
            _fc = force_constant_stages(task, include_disabled=True)
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
        _sorted = sort_by(struct, "held-first")
        _perm_path = write_permutation(base, _sorted)
        struct = _sorted.structure
        if log is not None:
            log.step("the atom order the engine needs")
            log.produced("atom-permutation.json",
                         f"key held-first, {struct.n_atoms} atoms -> {_perm_path.name}")
        _render_kind = _rung_kind(task, pset.stage)
        struct, _render_cell, _relaxed_by = _vibration_stage_geometry(
            base, task, pset, struct, log=log)
        _finishes = _render_kind == "vibration"
        # THE FINISH'S FORCE CRITERION IS THE RELAXATION'S OWN (plan § 5w
        # K4, M11 SS-C6): `relax_force_tol` is read by the relaxation rung
        # alone, so it is that rung's value -- the criterion the reference
        # geometry was relaxed to -- and the template's when the structure
        # is stated relaxed.  The force-constant stage's own copy was read
        # until 2026-09-30, so a preset that relaxed at 0.05 judged at 0.01.
        from ..resolve import resolved_ladder
        from ..template import stage_role
        _criterion = next(
            (getattr(c, "relax_force_tol", None) for n, c in resolved_ladder(
                template_path.read_text(encoding="utf-8"), task,
                seam.config_cls)
             if stage_role(str(task.engine), "vibration", n) == "relaxation"),
            getattr(pset[0].render_config(), "relax_force_tol", None))
    # The DATA FILES the engine will open, before any deck is written: a
    # missing pseudopotential is a run that cannot start, and finding that out
    # here costs a second (project-layout.md § 2.6).  Idempotent -- what is
    # already in the folder is left alone.  The elements come from the
    # structure, which has just been checked against the description's witness.
    if seam.provide_data is not None:
        with _user_error_as_prep(), _calling(
                "provide_data", engine=task.engine, log=log):
            seam.provide_data(struct, pset[0].render_config(), base)
    token = token_for(task, pset.stage)
    jobs: List[Job] = []
    from .materialize import (bench_container, trial_dir,
                              trial_work_dir)
    from ..paths import Shape as _Shape
    _shape = _Shape.named(task.shape)
    # ONE line per unique finding, however many trials repeat it (user,
    # 2026-08-28, O5).  The gate still FIRES per deck -- every trial is
    # validated and its own <deck>.validation.txt carries its findings --
    # but sixteen identical thin-vacuum warnings on one terminal is
    # noise wearing a safety vest.  Scoped to THIS loop (not a module
    # global) so a long-lived server process cannot quietly swallow a
    # later prep's warnings.
    import io as _io
    import sys as _sys

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
    _real_stderr, _sys.stderr = _sys.stderr, _once
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
            _jdir.mkdir(parents=True, exist_ok=True)
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
            cfg = element.render_config()
            if element.is_trial:
                # The deck's OWN identity line carries the trial label -- this,
                # not the filename, is what keys SIESTA's warm files away from
                # the real run's (project-layout.md § 2.3.2).
                with _calling("relabel", engine=task.engine,
                              where=element.label, log=log):
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
            if log is not None:
                log.phase(f"STEP 3 · DECK — {script}")
                log.received("config", f"{type(cfg).__name__}"
                                       + (f", stage_token={token}" if token else "")
                                       + (f", trial {element.label}"
                                          if element.is_trial else ""))
            with _user_error_as_prep():
                with _calling("spec_for", engine=task.engine,
                              where=script, log=log):
                    spec = seam.spec_for(struct, cfg,
                                         stage_token=(token or None),
                                         calculation=_render_kind,
                                         **({"cell": _render_cell}
                                            if _render_cell is not None else {}),
                                         **({"vibration": _vibration_block(
                                                pset.stage, cfg, _relaxed_by,
                                                criterion=_criterion)}
                                            if _finishes and not element.is_trial
                                            else {}),
                                         # THE RELAX STAGE'S RECORD reaches
                                         # every deck at its geometry -- a
                                         # trial's too, which carries no
                                         # finish (V1.36).
                                         **({"relaxed_by": _relaxed_by}
                                            if _relaxed_by is not None
                                            else {}),
                                         # A TRIAL'S DECK NAMES ITS OWN
                                         # LAUNCH (plan § 5w K12) -- the
                                         # bench lane is SIESTA's alone.
                                         **({"trial": _trial}
                                            if _trial else {}))
                _sc.prepare_deck(spec, struct, cfg, _jdir / script, log=log,
                                 dest_dir=base, findings=findings)
            if seam.sibling_artifacts is not None:
                with _calling("sibling_artifacts", engine=task.engine,
                              where=script, log=log):
                    seam.sibling_artifacts(struct, cfg, _jdir / script,
                                           kind=_render_kind)
            with _calling("label_of", engine=task.engine, log=log):
                _label = seam.label_of(cfg)
            _seed_trajectory_log(struct, cfg, _jdir, engine=task.engine,
                                 label=_label, token=(token or None),
                                 frame=spec.engine_frame,
                                 relaxes=(_render_kind == "optimization"))
            if log is not None:
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
            jobs.append(_job_for(element, script, task, pset.stage, seam,
                                 base, log=log,
                                 finish=(None if element.is_trial
                                         else spec.finish)))
    finally:
        _sys.stderr = _real_stderr
    if _once.dropped:
        print(f"  (each warning shown once; {_once.dropped} repeat(s) "
              f"across the other trials suppressed -- every trial's own "
              f".validation.txt carries its full findings)",
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
                shared=_shared_for(base, seam,
                                   engine=task.engine, log=log),
                jobs=jobs)
    if kind == "sweep":
        record_dir = base / bench_container(Shape.named(task.shape), token)
        record_dir.mkdir(parents=True, exist_ok=True)
    else:
        record_dir = base
        js = _merge_run_jobset(
            base / JOBSET_FILENAME, js,
            # The CURRENT ladder bounds what the merge keeps.
            # (`read_task` refuses a description without stages and
            # `Task` refuses an empty tuple, so the stage-less fallback
            # arms that stood here were unreachable -- U6 close.)
            ladder=frozenset(s.name for s in task.stages))
    js.write(record_dir / JOBSET_FILENAME)
    if log is not None:
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
    # was written (`_require_activation`) -- whichever machine it describes.
    dirs = prep_jobset(js, base, env=env, emit_sbatch=emit_sbatch,
                       record_dir=record_dir, log=log,
                       machine_record=environment)

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
        reports = _open_attempts(js, base, stage, continue_from=continue_from,
                                 cold=cold, named=named)
        if opened is not None:
            opened.extend(reports)

    if log is not None:
        log.close()
    return dirs


def _open_attempts(js: JobSet, base: Path, stage: str,
                   containers: Sequence[Optional[Path]] = (None,), *,
                   continue_from: Optional[str], cold: bool,
                   named: bool) -> List:
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
    run, and ``prepare_attempt`` refuses it by name — so the shape is asked
    here once and the refusal never has to fire.

    Idempotent by ``resolve_attempt``'s rule — reuse the last attempt until it
    has been launched, then open the next.  ``continue_from`` / ``cold`` /
    ``named`` -- REQUIRED, they decide which run the attempt starts from
    (`code-audit.md` D1) -- are prep's decision, so the attempt is opened
    once, with its carry: it was opened with none and then again with it
    until 2026-10-01, and a refusal between the two left an earlier carry
    undone (W52).
    """
    from .materialize import prepare_attempt, shape_of as _shape_of

    sh = _shape_of(js, base)
    if sh is None or not sh.keeps_attempts_as_directories:
        return []
    reports = [prepare_attempt(js, base, stage, container=c,
                               continue_from=continue_from, cold=cold,
                               named=named)
               for c in containers]
    for rep in reports:
        _move_progress_channel_into(rep.dir)
    return reports


def _move_progress_channel_into(attempt: Path) -> None:
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
    from ..runfiles import WRITTEN, find_by_role

    stage_dir = attempt.parent
    if stage_dir == attempt:
        return
    for role in [a.role for a in WRITTEN if a.output == "progress"]:
        for seeded in find_by_role(stage_dir, role):
            seeded.replace(attempt / seeded.name)


def _require_activation(target: Optional[str], environment,
                        base=None) -> None:
    """A record that does not state how a shell enters an environment there
    is refused HERE, for every target -- this machine included.

    **Prep reads the TARGET's record, and only it** (`configuration.md` § 4):
    each machine declares its ``env_init`` in its own ``molbuilder.json`` and
    `jobset probe` copies it into the record it writes.  No target is allowed
    a substitute: generating with THIS machine's activation for another
    machine succeeds at generate time and dies on the cluster hours later, on
    a path that exists only here (2026-08-24).
    """
    from ..scheduler.record import (LOCAL_TARGET, calculation_record,
                                    probe_command, probe_steps)
    from .commands import rollback
    if (getattr(environment, "env_init", None) or {}).get("activation"):
        return
    here = target in (None, LOCAL_TARGET)
    own = calculation_record(base) if base is not None else None
    if own is not None and own.is_file():
        # THE CALCULATION'S OWN COPY ANSWERED (`configuration.md` § 5 M-3):
        # taken at its first prep and never replaced, so no probe reaches it.
        raise PrepError(
            f"this calculation's record of its machine, {own}, does not say "
            f"how a shell enters an environment there -- it was copied at the "
            f"calculation's first prep and is never replaced, so a probe does "
            f"not reach it (a copy taken before 2026-10-02 names it "
            f"`script_generation`).  Rename that key to `env_init` in it, or "
            + rollback("the calculation's first prep", base=base)
            + ("" if here else f"  Name the machine again: --target {target}."))
    whose = "this machine's record" if here else f"the record of {target!r}"
    raise PrepError(
        f"{whose} does not say "
        f"how a shell enters an environment there, and nothing else may "
        f"(docs/configuration.md § 4).  Declare it in that machine's "
        f"molbuilder.json --\n"
        f"      \"env_init\": {{\"activation\": \"conda activate\", "
        f"\"preamble\": \"source <conda root>/etc/profile.d/conda.sh\"}}\n"
        f"  or \"source activate\" after \"module load mamba\" where a "
        f"module gives the toolchain -- then probe {probe_steps(target)}:\n"
        f"      {probe_command(target)}\n"
        f"  and prep again.  A copied record that is wrong for its machine "
        f"is edited by hand.")


# --------------------------------------------------------------------- #
#  The transport arm — the composite's prep (archive/2026-09-01-transport-design.md § 4.2)  #
# --------------------------------------------------------------------- #

def _transport_provide_pseudos(struct, cfg, base: Path,
                               citation: str) -> None:
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
    pdir = _pseudo_dir(base)
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
        missing = (copy_pseudopotentials(want, lib, pdir)
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
    a pin set it, which is the whole of what `--pipeline-log` had nothing to
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

    tmpl = find_template(base)
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


def _transport_rung(base, task, stage: str, composed, allocation, *,
                    pins=None, log=None):
    """What one transport rung's decks render from --
    ``(struct, config, state, element)``.

    ONE DOOR, asked by `_prep_transport` to write the rung's decks and by
    :func:`gather_transport_inputs` to render an upstream rung NOW, from the
    current template, junction and run card (plan § 5w K11, T-F30): the
    gather compared an upstream attempt with that stage folder's LAST render,
    which a change since did not touch, so a stale result was carried
    forward and recorded as consistent.
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

    # (iii) THE CONFIG: the template ⊕ this rung's overrides ⊕ its run card,
    # with provenance recording which source set each value.
    element = _resolve_transport(base, task, stage, allocation, pins=pins,
                                 log=log)
    # WHAT THE DECK WRITER IS HANDED is values ⊕ the allocation-marked fields
    # (`ResolvedConfig.render_config`), the same object every other kind's
    # emitter gets.  Rendering from bare `.values` left the emitter blind to
    # the rank count and the memory ceiling it is supposed to record.
    config = element.render_config()
    if label != task.label:
        config = dataclasses.replace(config, system_label=label)
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
    state = electronic_state(composed.sorted.structure, config,
                             kind="transport")
    # A STATE TRANSIESTA CANNOT RUN is refused HERE, on the junction, where
    # each value still says where it came from (`template.md` § 6.3a, ES4):
    # once folded into the rungs' configs, every rung's gate would call a
    # recorded or detected value *stated*.
    from ..validation.chemistry import check_electronic_state
    refused = [i for i in check_electronic_state(
        composed.sorted.structure, config, calculation="transport")
        if i.severity == "error"]
    if refused:
        raise PrepError(refused[0].message)
    config = dataclasses.replace(
        config,
        spin_treatment=state.spin_treatment.value,
        unpaired_electrons=state.unpaired_electrons.value)
    return struct, config, state, element


def _transport_spec(task, stage: str, struct, config, state, volts=None):
    """One transport deck's ``(spec, cfg)`` -- the bias point's when
    ``volts`` is given (the point is the rung's answer, `engines/template.md`
    § 6.4; a single-bias rung keeps the 0 V `resolve` laid on)."""
    from ..siesta.input import spec_for as _siesta_spec_for
    cfg = (config if volts is None else
           dataclasses.replace(config, bias_voltage_v=float(volts)))
    with _user_error_as_prep():
        try:
            spec = _siesta_spec_for(struct, cfg,
                                    stage_token=(token_for(task, stage)
                                                 or None),
                                    calculation="transport", state=state)
        except ValueError as exc:
            # `transport_spec` refuses an unknown rung with a message
            # written FOR a person, and `_user_error_as_prep` translates
            # only ValidationError / RuntimeConfigError / WrapperError --
            # deliberately, so a TypeError still looks like the bug it is.
            raise PrepError(str(exc)) from exc
    return spec, cfg


def _prep_transport(base_dir, stage: Optional[str] = None, *,
                    allocation=None, env: str = None,
                    emit_sbatch: bool = True,
                    sweep=None, pins=None, translation=None,
                    target: Optional[str] = None,
                    chosen=None,
                    pipeline_log: bool = False,
                    opened: Optional[list] = None,
                    findings: Optional[list] = None,
                    continue_from: Optional[str] = None,
                    cold: bool = False,
                    named: bool = True) -> List[Path]:
    """`prep` for the transport COMPOSITE — one rung of the ladder.

    **The same five steps every kind takes**, with one step of its own.
    Transport's genuinely new input is the CITATION: a finished relaxation
    whose junction this calculation is built from. Composing it — copy,
    sort, gate, extract the leads — is step 3a below and belongs to this
    arm. Everything else is the shared machinery, un-forked::

        1  the machine            `_environment_read`, then `resolve_target`
        2  the description        `read_task`, and WHICH rung
        3a the citation           compose  (transport's own)
        3b the data files         the pseudos travel with the citation
        3c the deck(s)            resolve -> spec_for -> prepare_deck
        4  the wrappers           `prep_jobset`
        5  the run directories    `prep_jobset`

    Step 3c is the framework's, not this module's, and that is the point:
    `engines/transport.md` § 3.2 measured what it cost when it was not —
    a deck of 13 keywords against a template offering 45, with no
    validation report and no check gate. Every rung renders through
    `spec_for` → `DeckSpec` → `prepare_deck` now.
    """
    from ..task import FILENAME as TASK_FILENAME
    from ..task import read_task
    from ..transport.compose import (ComposeError, compose_junction,
                                     load_compose_record,
                                     write_compose_record)
    from ..atom_permutation import PermutationError
    from ..transport.sort import SortError
    from ..transport.stages import warm_declaration
    from ..runwrap import write_run_wrapper
    from ..paths import Shape

    base = Path(base_dir).resolve()
    if not base.is_dir():
        raise PrepError(f"calculation folder not found: {base}")
    desc = base / TASK_FILENAME
    if not desc.is_file():
        raise PrepError(
            f"no {TASK_FILENAME} in {base}. `prep` turns a DESCRIPTION into a "
            f"runnable directory; write one first with `jobset init`.")
    # A TRANSPORT RUNG TAKES ITS RUN CARD as every rung does (`stages.md`
    # § 6.8d, plan § 5w K5, T-F3): the card's settings arrive as ``pins``
    # and its machine items as ``chosen``.  Every pin was refused here until
    # 2026-09-30, so a `use_gpu` the run card offered on a transport rung
    # stopped the prep.  What it does not take is a sweep.
    if sweep is not None or translation is not None:
        raise PrepError(
            "a transport calculation takes no parameter sweep or "
            "translation.  Its parameters come from its own template and "
            "each rung's run card, and its one axis is the bias -- a list "
            "in task.json, rendered as one deck per point "
            "(engines/transport.md 2a.10: single bias is the degenerate "
            "case of that axis, one point at zero).")

    # ---- 1. resolve the machine ---------------------------------------- #
    # READ, CHECK, THEN WRITE.  A record that does not state how to enter the
    # named machine's environment is refused before anything is on disk --
    # the snapshot included, or the remedy's re-copied record would then
    # contradict it (W52, and its fix's review).
    environment = _environment_read(base, target)
    _require_activation(target, environment, base=base)
    resolve_target(base, target)          # step 1 proper: the snapshot

    # ---- 2. the description, and WHICH rung ---------------------------- #
    task = read_task(desc)
    if not stage:
        from .commands import enabled_refs, name_a_stage
        raise PrepError(
            "a transport prep names its rung: the composite's stages render "
            "separately, in dependency order (engines/transport.md); "
            + name_a_stage("prep", "run", enabled_refs(task), base=base))
    token = token_for(task, stage)          # refuses an unknown stage by name

    # THE PIPELINE LOG (TR4).  This arm printed "not wired for the transport
    # arm yet" until 2026-09-16 -- honest, and a documented no-op is still a
    # no-op.  What made it possible was TR1: a transport calculation has a
    # template now, so there IS a resolve step whose inputs and outputs are
    # worth recording.  Opened here rather than at step 1 because it is named
    # for the rung, and the rung is not known until the description is read.
    _tlog = None
    if pipeline_log:
        from ..pipeline_log import PipelineLog as _PL
        _tlog = _PL.open(base, label=task.label, token=token,
                         engine=task.engine, shape=task.shape)
        _tlog.phase("STEP 1 · MACHINE — where this job will run")
        _tlog.received("calculation", str(base))
        _tlog.received(TASK_FILENAME,
                       f"{task.label} · transport · {task.shape} · "
                       f"{len(task.stages or ())} stage(s)")
        for _g, _l in _environment_rows(environment):
            _tlog.produced(_g, _l)
    stage_ref = next(s for s in task.stages if s.name == stage)
    if not stage_ref.enabled:
        raise PrepError(
            f"stage {stage!r} is disabled in this description "
            f"(enabled: false in task.json).  The seed is skippable by "
            f"design (archive/2026-09-01-transport-design.md, ruling Q4) -- re-enable it "
            f"there, or prep the next stage.")

    # ---- 3a. compose the junction (or load the travelled copy) --------- #
    # The record beside task.json answers first (the folder travels;
    # `project-layout.md` § 2.1) -- but only for THIS citation.  A
    # missing or re-pointed record composes fresh from the tree.
    citation = task.slots["junction"]
    try:
        # The tree root is resolved BEFORE the record is loaded: the
        # reload re-runs the § 3 lead gates, and their principal-layer
        # half reads the CITED directory's own .ion files, which the
        # travelled folder does not carry.  Absent (the folder moved
        # out of its tree), that half degrades to UNVERIFIED honestly.
        from ..projects import find_projects_root
        root = find_projects_root(base)
        why: list = []
        composed = load_compose_record(base, citation=citation,
                                       tree_root=root, why=why)
        if composed is None:
            if root is None:
                # NAME WHICH OF THE THREE.  This asserted the first --
                # "the record is not beside task.json" -- for all of them,
                # and the likeliest here is the second: a slot re-pointed
                # after a re-relaxation, prepped on a machine outside the
                # tree, with the record sitting right there composed from
                # the previous attempt.
                raise PrepError(
                    f"this transport calculation cannot be composed here: "
                    f"{why[0] if why else 'there is no usable record'}.  "
                    f"And {base} is not inside a projects tree, so the "
                    f"citation {citation!r} cannot be resolved to compose "
                    f"afresh.  Prep once inside the tree that holds the "
                    f"cited junction -- the record then travels with the "
                    f"folder (archive/2026-09-01-transport-design.md 4.1).")
            composed = compose_junction(citation, tree_root=root)
            write_compose_record(base, composed)
            # RENDER FROM THE RECORD, on this prep as on every later one.  The
            # fresh composition is the cited `.XV` at full float precision;
            # the record is the codec's text of it, and every later prep --
            # the other rungs, a re-prep -- reads the record.  Rendering this
            # one from the fresh copy gave two preps of the SAME calculation
            # two sets of numbers ~1e-10 apart (measured 2026-09-25 on the
            # ENGINE-OFFSET record), and the gather's `same_calculation`
            # compares them as text: a value near a rounding boundary then
            # flips, and a good seed is refused.  One source, not rounding luck.
            composed = load_compose_record(base, citation=citation,
                                           tree_root=root, why=why)
            if composed is None:
                raise PrepError(
                    f"the composed junction was written and could not be "
                    f"read back: {why[-1] if why else 'no reason given'}")
    except (ComposeError, SortError, PermutationError) as exc:
        raise PrepError(str(exc)) from exc

    # ---- 3b. render this rung's deck(s) -------------------------------- #
    #
    # ONE PATH, and every rung takes it:
    #
    #     the template ⊕ this rung's overrides  ->  resolve       -> config
    #     the structure this rung describes     ->  spec_for      -> DeckSpec
    #     the DeckSpec                          ->  prepare_deck  -> the .fdf
    #
    # Nothing below floor 3 asks which rung this is.  What differs between
    # rungs is DATA -- WHICH structure it describes (i), and which layout
    # its shape selects (`transport/deck.py::SHAPE_OF_RUNG`) -- and both are
    # looked up rather than decided here.
    shape = Shape.named(task.shape)
    stage_dir = (base / shape.stage_dir(token)) if token else base
    script = _rf(task.label, ".fdf", token or None)

    # (ii) THE ALLOCATION, folded BEFORE the resolve and not after.
    #
    # The description's own queue/wall/memory ask and its reporting policy are
    # part of the allocation `resolve` is handed -- that is the order the
    # shared arm takes, and the reason is that `resolve` FOLDS RIDERS ONTO
    # WHAT IT IS GIVEN (`_resolve_transport`'s own note).  Folding afterwards
    # meant resolve saw a bare `Resources()` and the element's answer was
    # thrown away, so the two mistakes cancelled and neither was visible.
    allocation = _with_notify(
        _under_description(allocation, task.allocation, chosen), task.notify)

    # (i) + (iii) THE RUNG -- its structure, its config and the junction's
    # electronic state, through the one door the gather asks too when it
    # renders an upstream rung now (`_transport_rung`, plan § 5w K11).
    struct, config, _junction_state, element = _transport_rung(
        base, task, stage, composed, allocation or Resources(), pins=pins,
        log=_tlog)
    res = element.resources

    # The pseudopotentials travel with the citation, and the screening runs
    # against THIS config -- the one the deck renders from -- because what
    # it checks is whether each file's XC family matches the functional the
    # run will ask for.  Screening against a different config would be
    # comparing the files to a calculation nobody is doing.
    _transport_provide_pseudos(composed.sorted.structure, config, base,
                               citation)

    # (iv) THE DECKS.  A bias scan renders one per point, a single-bias
    # calculation one -- and the stage directory always holds the FIRST
    # point's, because that is where every generic reader looks for a job's
    # script and § 2a.11 documents it as "the same deck v0/ holds".
    #
    # § 2a.10: *single bias is the degenerate case of the bias axis -- one
    # point, at zero, where every list starts.*  One mechanism, so one loop:
    # the description's list is the bias's only home, and each point's deck
    # is written at that point (the rung fixes it, `engines/template.md`
    # § 6.4); a single-bias rung renders the 0 V `resolve` laid on.
    # ONE SPELLING of where a bias point lives, and of which rungs have one:
    # the door every reader of a rung's attempts asks (`rung_containers`,
    # `engines/transport.md` § 2a.11; plan § 5w K10).  Three steps below
    # need it -- the deck, the wrapper and the attempt ladder.
    from ..transport.stages import rung_containers
    point_dirs = [(d, v) for d, v in rung_containers(base, task, stage)
                  if v is not None]
    points = tuple(v for _d, v in point_dirs)

    for out_dir, volts in ([(stage_dir, points[0] if points else None)]
                           + point_dirs):
        out_dir.mkdir(parents=True, exist_ok=True)
        # THE POINT IS THE RUNG'S ANSWER -- the one `role` answer the
        # catalogue does not hold (`engines/template.md` § 6.4): the list is
        # the bias's only home (`engines/transport.md` § 2a.10), and a
        # single-bias rung keeps the answer `resolve` laid on, 0 V.
        spec, cfg = _transport_spec(task, stage, struct, config,
                                    _junction_state, volts)
        with _user_error_as_prep():
            _sc.prepare_deck(spec, struct, cfg, out_dir / script,
                             log=_tlog, dest_dir=base, findings=findings)

    # ---- 4 + 5, the shared tail ---------------------------------------- #
    if stage == "transmission":
        # TBtrans post-processes the device run from its own deck (the
        # device's and the transmission's are two texts since 2026-09-29,
        # `engines/transport.md` § 6.1b); which binary runs it is not
        # read off the deck -- it rides the allocation road
        # (`model.Resources.program`) into the wrapper.
        res = dataclasses.replace(res, program="tbtrans")
    from ..warmfiles import resumes_for
    job = Job(name=stage, script=script, resources=res,
              warm=warm_declaration(stage, task.label, base),
              resumes=resumes_for(str(task.engine), _rung_kind(task, stage),
                                  base))

    # The activation the wrappers below carry was checked at step 1, before
    # anything was written: this check stood after the per-point wrapper
    # loop until 2026-09-16, and then here -- after the decks, the compose
    # record and the pseudos -- until 2026-10-01 (W52), while its own premise
    # (`_require_activation`) is that no such file may exist.

    # Each bias point's directory gets its own wrapper, beside its own deck
    # -- the same render `prep_jobset` gives the stage directory, through
    # the same one writer, so a point runs exactly as the stage would alone
    # (the chain walker only cd's and bashes).
    for point_dir, _v in point_dirs:
        with _user_error_as_prep():
            write_run_wrapper(point_dir / script,
                              label=task.label,     # G7: told, not read
                              n_atoms=len(struct.elements),
                              resources=res, env=env,
                              emit_sbatch=emit_sbatch, project_dir=base,
                              machine_record=environment,
                              # the job's own facts, as the ladder's
                              # wrapper is given them (`prep_jobset`)
                              finish=job.finish, resumes=job.resumes)
    js = JobSet(name=task.label, engine=task.engine, kind="ladder",
                shared=_siesta_shared_package(base), jobs=[job])
    js = _merge_run_jobset(base / JOBSET_FILENAME, js,
                           ladder=frozenset(s.name for s in task.stages))
    js.write(base / JOBSET_FILENAME)
    dirs = prep_jobset(js, base, env=env, emit_sbatch=emit_sbatch,
                       record_dir=base, log=_tlog, machine_record=environment)
    # STEP 6, through the SAME door the shared arm uses.  `prep` is what sets
    # a stage up to run: `_launch_dir` refuses a hierarchical stage with no
    # attempt open precisely because opening one is not `launch`'s job, and
    # this arm ended at `prep_jobset` until 2026-09-16 -- so a transport prep
    # from the browser (`web/blueprints/build.py` calls `prep_calculation`
    # directly) reported success and handed back a folder the launcher would
    # not take, naming the command that had just run.  The CLI compensated
    # and no other caller could.
    #
    # A scan's containers are this arm's one genuine difference -- one attempt
    # ladder per point (`04_device/v0.2/run-<n>`), because the transmission at
    # v reads the device at v -- and they are DATA on the call rather than a
    # shape the helper re-derives.
    #
    # THE DAG GATHER IS NOT HERE, deliberately -- and it is not missing
    # either.  `gather_transport_inputs` refuses an upstream that has not
    # CONCLUDED, so calling it here would make `prep run device` fail until
    # the leads had actually run, and you could no longer render the device
    # deck to READ it before spending the queue.  That is the split
    # `prepare_attempt` names in its own words ("preparing is still design and
    # the split from starting is what gives you somewhere to look before
    # committing cluster time"), and nineteen tests in `test_transport_prep`
    # read a device deck without running a lead.  They are right to.
    #
    # It is a step of its own, `prep.gather_for_stage`, taken by the one
    # entry (`prep_stage`) after this returns -- so both doors take it.  It
    # ran on the CLI road alone until 2026-09-16, which was
    # survivable only while this arm opened no attempt -- `launch` refused the
    # folder by name and that refusal was accidentally the guard.  Opening the
    # attempt (above, the same day) removed the symptom and left the gap, so a
    # device job could reach the node and die for want of an electrode `.TSHS`.
    reports = _open_attempts(js, base, stage,
                             containers=[d for d, _ in point_dirs] or (None,),
                             continue_from=continue_from, cold=cold,
                             named=named)
    if opened is not None:
        opened.extend(reports)
    if _tlog is not None:
        _tlog.close()
    return dirs


def gather_transport_inputs(base_dir, task, stage: str,
                            attempt_dir, *,
                            bias: Optional[float] = None) -> List[tuple]:
    """Copy the § 4.2 DAG's inputs into ``attempt_dir`` — the composite's
    other half of *"warm files are COPIED in at prep"*.

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
    which electrode run fed it.  Returns ``[(source_rel, filename)]``.
    """
    from ..transport.stages import (per_point_rungs, rung_container,
                                     stage_inputs)
    from .materialize import attempt_concluded

    base = Path(base_dir)
    attempt_dir = Path(attempt_dir)
    enabled = {s.name for s in task.stages if s.enabled}
    inputs = stage_inputs(stage, task.label,
                          seed_enabled=("seed" in enabled))
    gathered: List[tuple] = []
    composed = None              # the junction, read once, when first needed
    for upstream, filename in inputs:
        token = token_for(task, upstream)
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
        concluded = [d for d in attempts
                     if attempt_concluded(d, stem) is not None]
        if not concluded:
            raise PrepError(
                f"the {stage} stage consumes {filename} from {upstream}, "
                f"and {upstream} has no CONCLUDED attempt -- it was never "
                f"launched, is still running, or was force-stopped -- "
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
        import shutil as _sh
        _sh.copy2(src, attempt_dir / filename)
        gathered.append((str(matching[0].relative_to(base)), filename))
    if gathered:
        write_gathered_from(attempt_dir, gathered)
    return gathered


def _composed_junction(base, task):
    """The junction this calculation is composed from -- its record beside
    ``task.json``, for the cited junction (`_prep_transport` step 3a writes
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
    from .prep_inputs import _declared_execution_pins
    card = task.run_condition(stage)
    pins = {}
    if card:
        pins, _axes, _value_axes = _declared_execution_pins(
            base, task.engine, {k: [v] for k, v in card.items()})
    struct, config, state, _element = _transport_rung(
        base, task, stage, composed, Resources(), pins=pins or None)
    spec, cfg = _transport_spec(task, stage, struct, config, state, volts)
    with _user_error_as_prep():
        return _sc.render_deck(spec, struct, cfg, verbose=True,
                               dest_dir=base).text


def gather_for_stage(base_dir, task, stage: str) -> List[Tuple[Path, Optional[float], List[tuple]]]:
    """Carry the DAG's inputs into **every attempt this rung has open**.

    Returns ``[(attempt_dir, volts, [(source_rel, filename), ...]), ...]`` —
    one entry per attempt, so a caller can report what landed where and say
    which bias point it was. ``volts`` is ``None`` for a rung with no bias
    axis, and it is CARRIED rather than parsed back out of the directory name:
    this function already knows it, and a caller re-deriving it would be a
    second reader of a spelling `bias_token` owns.

    A bias scan has one attempt per point and each is gathered against **its
    own** voltage: the transmission at *v* reads the device at *v*, never
    another point's converged state.

    **THE STEP THAT MAKES A PREPPED ATTEMPT RUNNABLE, and it is not `prep`'s.**
    `prep_calculation` renders the decks and opens the attempt without asking
    whether the rungs before it have finished, deliberately: a deck is the
    reviewable artifact, and *"preparing is still design and the split from
    starting is what gives you somewhere to look before committing cluster
    time."* Nineteen tests read a device deck without running a lead, and they
    are right to.

    Carrying the inputs is the other half, and it cannot be folded into that
    one because :func:`gather_transport_inputs` refuses an upstream that has
    not CONCLUDED — which is correct, and would make the decks unreadable
    until the whole chain had run.

    So it is a step of its own, and **the point of this function is that there
    is now ONE of it.** It ran only on the CLI road until 2026-09-16 (twice,
    hand-written, once per layout), so `web/blueprints/build.py` — the
    browser's Prep button — opened an attempt and carried nothing into it. That
    used to be survivable: the folder had no attempt at all, so `launch`
    refused it by name, and the refusal was accidentally the guard. Opening the
    attempt removed the symptom and left the gap, so a device job could reach
    the node and die for want of an electrode `.TSHS` — after the queue wait.
    """
    from ..paths import attempt_dir as _adir
    from ..paths import attempts_in as _ain
    from ..transport.stages import rung_containers

    base = Path(base_dir)
    # A point's ladder lives under its own v-dir; a single-bias rung's lives
    # under the stage directory -- the same folders the deck and the wrapper
    # were written into, from the one door.
    containers = rung_containers(base, task, stage)
    out: List[Tuple[Path, Optional[float], List[tuple]]] = []
    for container, volts in containers:
        ns = _ain(container)
        if not ns:
            continue          # nothing open here: prep has not run for it
        att = _adir(container, ns[-1])
        out.append((att, volts,
                    gather_transport_inputs(base, task, stage, att,
                                            bias=volts)))
    return out


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
    plan forever, its deck gone.  A set whose NAME differs is a different
    calculation's plan and is replaced outright, same as the legacy cases.
    """
    if not path.is_file():
        return new
    try:
        old = JobSet.load(path)
    except ValueError:
        return new              # unreadable or legacy: replaced outright
    if old.kind != "ladder":
        return new              # a pre-container sweep leftover: replaced
    if old.name != new.name:
        return new              # a renamed calculation: the old plan is
                                # another name's plan, not rows to keep
    fresh = {j.name for j in new.jobs}
    kept = [j for j in old.jobs if j.name not in fresh
            and (ladder is None or j.name in ladder)]
    merged = dataclasses.replace(
        new, jobs=kept + list(new.jobs),
        shared=sorted(set(old.shared) | set(new.shared)))
    # The plan's order is the LADDER's, not the order stages were prepped
    # in: re-prepping `coarse` must not move it below `medium`.  The seq
    # token is zero-padded (§ 6.3) so it sorts as it reads.
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
                         relaxes: bool = True) -> None:
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
        stage_name=token, convergence_targets=(targets or None),
        frame=frame)


def token_for(task, stage_name: Optional[str]) -> str:
    """This stage's ``<NN>_<name>`` — the ONE namer (decision 27).

    ``NN`` is the stage's place in the **full** ladder, so disabling one leaves
    a gap rather than renumbering what follows: renumbering would hand an
    existing output to a stage that did not produce it.

    Public since 2026-08-13: the CLI's container/underway surfaces need the
    same answer, and reaching in for a private name was the re-derivation
    habit the final review's C-c row names.
    """
    if not stage_name:
        return ""       # asked without naming a rung; every ladder has one
    from ..identity import StageRef
    # The ordinal rule is stated ONCE (StageRef.ladder -- decision 28's
    # pre-produce arm); this function reads the ref and spells the token.
    for ref in StageRef.ladder([s.name for s in task.stages]):
        if ref.name == stage_name:
            return ref.token
    # Unreachable through prep_calculation -- resolve._stage_of already
    # refused an unknown stage -- and LOUD rather than "" if a future caller
    # reaches it another way: an empty token would silently drop the stage
    # from every artifact name (job-contracts.md § 6.3).
    raise PrepError(f"stage {stage_name!r} is not in this description's "
                    f"ladder: {', '.join(s.name for s in task.stages)}.")


def _job_for(element, script: str, task, stage_name: Optional[str],
             seam: EngineSeam, base_dir=None, log=None,
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
    with _calling("warm_for", engine=task.engine, where=name, log=log):
        warm = seam.warm_for(element.label, element.values, kind, base_dir)
    from ..warmfiles import resumes_for
    resumes = resumes_for(str(task.engine), kind, base_dir)
    with _calling("traits_for", engine=task.engine, where=name, log=log):
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


def _siesta_shared_package(base: Path) -> List[str]:
    """SIESTA's shared package: the pseudopotentials it put in the folder,
    and the atom-permutation record when its decks are written from a sorted
    copy.

    The same files ``_siesta_provide_pseudos`` stages, named by the engine
    that staged them (`script-preparation.md` § 4, the data-files step).
    Under ``pseudos/`` since the layout repair (roadmap 7.10 M6); the bare
    root glob stays as the fallback for a bundle prepped before it, so a
    travelled calculation still names its package.

    THE PERMUTATION TRAVELS WITH THE RUNS, because a run of a sorted copy
    speaks the sorted order in every file it writes and the record is the
    one way back (`atom_permutation`, I7): every attempt holds its copy, so
    a SIESTA force-constant job's finish reads it beside the run it finishes
    (`engines/vibration.md` § 5.5).
    """
    from ..atom_permutation import PERMUTATION_FILE
    grouped = sorted(f"{PSEUDO_DIRNAME}/{p.name}"
                     for p in (base / PSEUDO_DIRNAME).glob("*.psml"))
    pseudos = grouped or sorted(p.name for p in base.glob("*.psml"))
    return pseudos + ([PERMUTATION_FILE]
                      if (base / PERMUTATION_FILE).is_file() else [])


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
                log=None) -> List[str]:
    """The static package every job links (`project-layout.md` § 2.1).

    **Asked of the engine, not guessed from the folder.**  An engine that puts
    no data files in has an empty package, and that is an answer rather than an
    accident of which suffix the glob happened to name.
    """
    if seam is None or seam.shared_package is None:
        return []
    with _calling("shared_package", engine=engine, log=log):
        return list(seam.shared_package(base))


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


@dataclass(frozen=True)
class Answer:
    """A person's answer to the one question prep asks -- *save the folder's
    state first?* (`checkpointing.md` § 9) -- and the words the ledger
    records it in: ``yes``, ``no``, the command line's *no answer
    (non-interactive)*, the Task setup tab's.

    ``note`` is the saved state's note when ``save`` -- the one prep drafted,
    as the person left it; ``None`` keeps the draft.  *(Until 2026-10-02 the
    one question was "already under way here -- re-render?"; a prepped stage
    is refused now, `job-system.md` § 5.0.)*"""
    save: bool
    said: str
    note: Optional[str] = None


@dataclass(frozen=True)
class SaveOffer:
    """The save, offered before prep writes (`checkpointing.md` § 9;
    `job-system.md` § 5.0, checkpoint 5): the note prep drafts, what is not
    saved, and the state the folder stands at -- ``None`` when it has no
    saved state yet.  A redo is a rollback to a state saved before a prep,
    so this is the moment one is made."""
    note: str
    unsaved: Tuple[str, ...]
    standing_at: Optional[str]


@dataclass
class PrepAnswer:
    """What one prep found and decided -- the whole of what either door shows
    (`job-system.md` § 5.3's table).  With ``offer`` set nothing was written
    and ``dirs`` is empty: the caller asks, then calls again with the
    :class:`Answer`."""
    kind: str
    stage: Optional[str]
    #: The description's preflight notes (an error refuses instead).
    findings: list = dataclasses.field(default_factory=list)
    #: What the inputs said: the run's sizing when nothing stated it, a
    #: bench's grid -- enumerated, crossed out, kept (`prep_inputs`).
    notes: List[str] = dataclasses.field(default_factory=list)
    offer: Optional[SaveOffer] = None
    #: What the save did when it was offered and answered -- the ledger's
    #: words (``saved as 4f9ca71: before prep run tight``, ``no``, ...).
    saved: Optional[str] = None
    dirs: List[Path] = dataclasses.field(default_factory=list)
    provenance: Optional[dict] = None
    #: A flat run: its wrappers are rendered and there is no attempt to open.
    flat: bool = False
    #: The attempt opened or reused (`materialize.Attempt`).
    attempt: Optional[object] = None
    #: A transport bias scan: ``(attempt, volts, [(source, file), ...])``
    #: per point (`gather_for_stage`).
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

        o, a, g = self.offer, self.attempt, self.agreement
        return {
            "kind": self.kind, "stage": self.stage,
            # THE ONE WIRE FORM of a finding (`Issue.to_json`).
            "findings": [i.to_json() for i in self.findings],
            "deck_findings": [i.to_json() for i in self.deck_findings],
            "notes": list(self.notes),
            "offer": ({"note": o.note, "unsaved": list(o.unsaved),
                       "standing_at": o.standing_at} if o else None),
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
        }


def prepped_already(base, task, kind: str, stage: str) -> Optional[str]:
    """Why ``stage`` is not prepped -- it already is -- or ``None``
    (`job-system.md` § 5.0, checkpoint 2a; user, 2026-10-02: *"refuse it,
    redo via rollback"*).

    PREPPED IS WHAT `status` READS: the stage's job in the calculation's
    plan, ``job-set.json``, matched as `status` matches it
    (`identity.stage_key`) -- for a bench, the sweep its bench folder holds.
    A prep refused after it began writing puts the plan back
    (:func:`prep_stage`), so a stage is counted prepped only by a prep that
    finished."""
    from ..identity import stage_key
    from ..paths import Shape
    from .commands import rollback
    from .materialize import bench_container, job_dir_names, shape_of
    base = Path(base)
    if kind == "bench":
        home = base / bench_container(Shape.named(task.shape),
                                      token_for(task, stage))
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
        home = job_dir_names(js, shape_of(js, base)).get(job.name, "")
        what = f"stage {stage!r}"
        where = (f"{home}/" if home not in ("", ".")
                 else "in this calculation's folder")
    return (f"{what} is already prepped ({where}) -- a prepped stage is not "
            f"prepped again (job-system.md § 5.0).  To redo it, "
            + rollback("its prep", base=base))


def save_offer(base, kind: str, stage: str,
               notes: Optional[List[str]] = None) -> Optional[SaveOffer]:
    """The save prep offers before it writes (`checkpointing.md` § 9), or
    ``None`` when the folder's state is saved -- it stands at a saved state
    and nothing has changed since.  A folder with no saved state yet is
    offered its first.  What is not saved is the checkpoint door's own
    cheap read (`Repo.status`); when that cannot be read, nothing is offered,
    and ``notes`` says why."""
    from ..checkpoint import CheckpointError, Repo
    draft = f"before prep {kind} {stage}"
    repo = Repo(str(base))
    if not repo.initialized:
        return SaveOffer(draft, (), None)
    try:
        st = repo.status()
    except CheckpointError as exc:
        if notes is not None:
            notes.append(f"no save offered -- the folder's state could not "
                         f"be read: {exc}")
        return None
    if st.clean:
        return None
    at = st.standing_at
    return SaveOffer(draft, st.unsaved(),
                     f"{at.short} ({at.note})" if at is not None else None)


def _take_the_offer(base, task, offer: SaveOffer, answer: Answer) -> str:
    """Save the folder as the person answered, before anything is written,
    and return what the ledger records.  A save asked for and not made
    refuses the prep: that state is the one a redo restores."""
    if not answer.save:
        return answer.said
    from ..checkpoint import CheckpointError, Repo
    note = (answer.note or "").strip() or offer.note
    repo = Repo(str(base))
    try:
        state = (repo.save(note) if repo.initialized
                 else repo.init(engine=task.engine, note=note))
    except CheckpointError as exc:
        raise PrepError(f"the folder's state could not be saved, so nothing "
                        f"was prepped: {exc}")
    return (f"saved as {state.short}: {note}" if state is not None
            else f"nothing to save: {note}")


def _plans_as_they_are(base: Path, *homes) -> dict:
    """The calculation's plan -- ``job-set.json``, the run's at the root and
    a bench's in its folder -- as it is before the five steps write, so a
    refusal after them can put it back (:func:`prep_stage`)."""
    out = {}
    for home in (base, *homes):
        if home is None:
            continue
        f = Path(home) / JOBSET_FILENAME
        out[f] = f.read_bytes() if f.is_file() else None
    return out


def _put_back(plans: dict) -> None:
    """Each plan as it was: a stage a refused prep had added is not counted
    prepped (`job-system.md` § 5.0)."""
    for f, was in plans.items():
        if was is None:
            f.unlink(missing_ok=True)
        else:
            f.write_bytes(was)


def prep_stage(base, kind: str, stage: Optional[str] = None, *,
               target: Optional[str] = None, allocation=None,
               from_attempt: Optional[str] = None, cold: bool = False,
               env: Optional[str] = None, emit_sbatch: bool = True,
               pipeline_log: bool = False,
               answer: Optional[Answer] = None,
               on_found=None) -> PrepAnswer:
    """**`prep`, the verb** -- what `molbuilder jobset prep` and the Task setup
    tab's Prep buttons both call (`job-system.md` § 5.3).

    In order -- `job-system.md` § 5.0's checkpoints: the description is
    refused unless it is one (a ``task.json`` and its template -- a
    transport description from before 2026-09-16 carries none, and its own
    door says how to add one); the stage is resolved through the one grammar
    (a name, or ``#N``), and refused when it is already prepped -- a redo is
    a rollback; the description's preflight runs -- an error refuses, the
    notes come back as ``findings``; the inputs are assembled (`prep_inputs`,
    A12).  Then, when the folder's state is not saved and no ``answer`` was
    given, the answer comes back with ``offer`` set and NOTHING WRITTEN.
    Answered -- or with nothing to offer -- it saves as answered, runs the
    five steps (:func:`prep_calculation`), opens the attempt, carries a
    transport rung's inputs, and compares the rendered deck with the launch
    it will get.  A refusal after the five steps began puts the plan
    (``job-set.json``) back, so the stage is not counted prepped.  Every
    decision lands in ``jobset-decisions.log``, whichever door called.

    ``allocation`` is what the person asks for on THIS prep -- the
    command line's flags, an empty ``Resources()`` from a surface with none
    (A12: never ``None``).  Every refusal is a :class:`PrepError` in the
    reader's own words, carrying what the entry had found by then --
    ``findings``, ``notes`` and the ``partial`` answer.

    ``on_found``, when given, is called with ``(findings, notes)`` as soon
    as the inputs are assembled -- before the question, and before anything
    is rendered -- so a terminal prints them ahead of what the decks say
    while they are written; the answer carries them either way.
    """
    from ..scheduler import AmbiguousTarget, UnknownTarget
    from ..task import FILENAME as TASK_FILENAME, read_task
    from ..template import find_template, template_path
    from ..validation.task import preflight
    from .ledger import prepped as ledger_prepped
    from .ledger import record as ledger
    from .prep_inputs import bench_inputs, bench_refusal, prep_run_inputs
    base = Path(base).resolve()
    desc = base / TASK_FILENAME
    findings: list = []
    notes: List[str] = []
    deck_findings: list = []
    out: Optional[PrepAnswer] = None
    recorded: List[bool] = []
    # THE PLAN AS IT WAS, taken before the five steps write and put back
    # unless the prep finishes: a stage is counted prepped only by a prep that
    # finished (`job-system.md` § 5.0).
    plans: dict = {}
    finished: List[bool] = []

    def _record_preflight():
        # The preflight's notes land in the ledger on the pass that ACTS or
        # REFUSES -- never on one that only asks, so the answering pass does
        # not write them twice -- and ahead of what follows them, the order
        # the terminal prints them in.
        if findings and not recorded:
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
        if desc.is_file():
            ledger(base, "prep", "refused", kind=kind, stage=stage,
                   reason=str(exc))
        exc.findings, exc.notes, exc.partial = (tuple(findings), tuple(notes),
                                                out)
        return exc

    try:
        # 1 · A DESCRIBED CALCULATION is "a template PLUS task.json"
        #     (project-layout.md § 2.1), and prep builds everything else from
        #     the two.  A TRANSPORT description passes on task.json alone,
        #     so that one written before its template existed (2026-09-16,
        #     TR1) is refused by the transport door in its own words -- how
        #     to add the template -- rather than told to `init` again.  It has
        #     carried a template since (`engines/transport.md` § 2a).
        is_transport = False
        if desc.is_file():
            try:
                is_transport = read_task(desc).calculation == "transport"
            except Exception:                                 # noqa: BLE001
                pass      # an unreadable description: the gate below owns it
        if not (desc.is_file() and (is_transport
                                    or find_template(base) is not None)):
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
        task = read_task(desc)
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
            from .commands import enabled_refs, name_a_stage
            takes = [r for r in enabled_refs(task)
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
            refs = StageRef.ladder([s.name for s in task.stages])
            stage = resolve_stage_ref(refs, stage).name

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
        tpl = template_path(base, task.label)
        issues = preflight(task, template_text=(
            tpl.read_text(encoding="utf-8") if tpl.is_file() else None))
        findings[:] = [i for i in issues if i.severity != "error"]
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            raise PrepError(
                "the description fails its own preflight "
                "(engines/stages.md § 6.6):\n  - "
                + "\n  - ".join(i.message for i in errors))

        # 4 · THE INPUTS -- one assembly per kind (`prep_inputs`, A12).  A
        #     bench measures ONE stage's configuration, and there is always a
        #     stage to name (§ 6.5).  Assembled before the question, and so
        #     is what they read of the target -- a bench's grid, and the type
        #     of the devices a run's condition counts -- so a refusal about
        #     THAT machine comes first; the rest of a run's machine is
        #     resolved by the five steps, after the answer.
        sweep = pins = translation = None
        chosen: dict = {}
        container = None
        if kind == "bench":
            from ..paths import Shape
            from .materialize import bench_container
            sweep, pins, translation = bench_inputs(base, target, notes=notes)
            container = base / bench_container(Shape.named(task.shape),
                                               token_for(task, stage))
        else:
            allocation, pins, chosen = prep_run_inputs(
                base, task, stage, allocation, notes=notes)

        #     ...AND EVERY LAUNCH VALUE IS STATED, OR THE PREP IS REFUSED --
        #      here, with the whole assembly in hand, before the question and
        #      before anything is written (`architecture.md` § 5.2; user,
        #      2026-10-02: "explicit job config is the only way allowed").
        #      A run states its processes; a run or a benchmark that this
        #      prep writes a `.sbatch` for states its queue, wall and memory.
        #      The target's record CHECKS an ask -- its queues are shown so
        #      one can be named -- and supplies no value of it.
        from .prep_inputs import launch_refusal
        _rec = _environment_read(base, target)
        why = launch_refusal(
            (allocation if kind == "run" else _under_description(
                allocation or Resources(), task.allocation)),
            engine=task.engine, shape=(kind == "run"), stage=stage,
            header=bool(emit_sbatch and _rec.scheduler == "slurm"),
            queues=[d.name for d in (_rec.domains or ())],
            base=base, target=target)
        if why:
            raise PrepError(why)

        # 4a · WHAT IT CONTINUES FROM (`job-system.md` § 5.4, plan W37): which run an
        #      independent stage continues from -- the stage before it,
        #      newest, by default -- read BEFORE anything is written, so a
        #      refusal leaves nothing behind.  A named run is taken as said.
        continuation = None
        if kind == "run" and stage is not None:
            from .continuation import continuation_answer
            continuation, refused = continuation_answer(
                base, task, stage, from_attempt=from_attempt, cold=cold)
            if refused:
                raise PrepError(refused)
        # WHAT THE ATTEMPT IS OPENED WITH, decided here and handed to the
        # five steps, which open it once: the run it continues from, or --
        # when it continues from nothing -- clean, so a carry an earlier
        # prep left is taken away rather than read by an engine that was
        # not told to (`materialize.prepare_attempt`).
        continue_from = (continuation.source if continuation is not None
                         else None)
        start_clean = continuation is None
        named = not (continuation is not None and continuation.by_default)
        if on_found is not None:
            on_found(findings, notes)

        # 5 · THE SAVE, OFFERED (`checkpointing.md` § 9; user, 2026-10-02:
        #     "yes, offer save").  Nothing is written before this point, and a
        #     redo is a rollback (2a) -- so a folder whose state is not saved
        #     is offered a save first, its note drafted.  Asked, never
        #     assumed: the door asks and calls again with the answer.
        offer = save_offer(base, kind, stage, notes=notes)
        if offer is not None and answer is None:
            return PrepAnswer(kind, stage, findings=findings, notes=notes,
                              offer=offer)
        _record_preflight()
        saved = None
        if offer is not None:
            saved = _take_the_offer(base, task, offer, answer)
            ledger(base, "prep", "save-offer", stage=stage, answer=saved)

        # 6 · THE FIVE STEPS -- the plan as it was kept first.
        plans.update(_plans_as_they_are(base, container))
        opened: list = []
        dirs = prep_calculation(base, stage, allocation=allocation, env=env,
                                emit_sbatch=emit_sbatch, sweep=sweep,
                                pins=pins, translation=translation,
                                target=target, chosen=chosen,
                                pipeline_log=pipeline_log, opened=opened,
                                findings=deck_findings,
                                continue_from=continue_from,
                                cold=start_clean, named=named)
        seen: set = set()
        # ONE ENTRY PER FOLDER: on the flat layout every stage's folder is
        # the calculation's one, and the answer listed it once per stage --
        # "prepped 3 job dir(s)" for one stage (W52).
        dirs = list(dict.fromkeys(dirs))
        out = PrepAnswer(
            kind, stage, findings=findings, notes=notes, dirs=list(dirs),
            saved=saved,
            provenance=ledger_prepped(base, kind=kind, stage=stage, dirs=dirs),
            # ONE OF EACH: a sweep's trials repeat one finding per deck, and
            # the terminal said each once (`prep_calculation`).
            deck_findings=[i for i in deck_findings
                           if not (repr(i.to_json()) in seen
                                   or seen.add(repr(i.to_json())))])
        if pipeline_log:
            from ..pipeline_log import log_name
            out.pipeline_log = (container or base) / log_name(
                task.label, token_for(task, stage) or "", task.engine,
                task.shape)
        if kind == "bench":
            finished.append(True)
            return out

        # 7 · THE ATTEMPT -- opened by the five steps, ONCE, with what it
        #     continues from (`_open_attempts`; until 2026-10-01 it was opened
        #     a second time here, and a refusal between the two left an
        #     earlier carry undone -- W52).  A later attempt is `launch`'s.
        #     Flat keeps no attempt directories: the run is the calculation's
        #     folder.
        from ..template import KIND_ROLES
        out.linked = (getattr(task, "calculation", None)
                      or "optimization") in KIND_ROLES
        out.cold = bool(cold)
        out.continuation = continuation
        js = JobSet.load(base / JOBSET_FILENAME)
        sh = shape_of(js, base)
        if sh is not None and not sh.keeps_attempts_as_directories:
            out.flat = True
            _flat_continued_from(base, task, stage, continuation)
            run_dir, rep_stage, copied = base, stage, []
        else:
            # A TRANSPORT BIAS SCAN keeps one attempt per point (04_device/
            # v0.2/run-<n>, layout ruled 2026-08-29), which the five steps
            # opened; each is gathered against its own voltage.
            from ..transport.stages import scan_points
            if is_transport and scan_points(task, stage):
                out.points = gather_for_stage(base, task, stage)
                finished.append(True)
                return out
            rep = opened[0]
            out.attempt = rep
            run_dir, rep_stage, copied = rep.dir, rep.stage, list(rep.copied)
            if is_transport:
                out.gathered = [pair for _att, _v, got
                                in gather_for_stage(base, task, rep.stage)
                                for pair in got]
        if continuation is not None:
            # THE DECISION, LOGGED (`job-system.md` § 5.4): which run, by
            # default or named, what it was, and what came across.
            ledger(base, "prep", "continues", stage=stage,
                   **continuation.ledger_facts(), copied=copied)
        elif cold:
            ledger(base, "prep", "starts-cold", stage=stage)

        # 8 · WHAT IT WILL LAUNCH WITH, and whether the deck agrees: `launch`
        #     refuses a deck rendered for another width, and prep is the step
        #     that exists so there are no surprises there (`agreement.py`).
        job = next((j for j in js.jobs if j.name == rep_stage), None)
        if job is not None:
            r = job.resources
            out.resources = {"mpi_np": r.mpi_np,
                             "cpus_per_task": r.cpus_per_task,
                             "continue_retries": r.continue_retries}
            out.deck = Path(job.script).name
            from .agreement import launch_agreement
            agreement = launch_agreement(run_dir, job)
            if agreement.verdict != "silent":
                out.agreement = agreement
                ledger(base, "prep", "launch-agreement", stage=rep_stage,
                       verdict=agreement.verdict,
                       rendered_for=agreement.rendered_text,
                       launching_at=agreement.launch_text)
        finished.append(True)
        return out
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
    finally:
        # A REFUSAL -- or a bug -- AFTER THE FIVE STEPS BEGAN puts the plan
        # back; an offer returned before them took nothing (`plans` empty).
        if not finished:
            _put_back(plans)


__all__ = ["prep_calculation", "prep_jobset", "prep_stage", "PrepAnswer",
           "Answer", "SaveOffer", "PrepError", "resolve_target"]
