"""Materialize engine — turn a :class:`JobSet` into on-disk per-job
directories (docs/execution/job-system.md; naming: job-contracts § 6.3).

Filesystem ONLY: it knows nothing about schedulers or engines.  For each
job it creates the directory :func:`job_dir_names` assigns (a stage's
``<NN>_<name>/``, a trial's ``<NN>_<name>/bench/bench-<point>/``) and
copies in, as real files, the static ``shared`` package plus the job's own
``script`` (`project-layout.md` § 1.0: a run directory holds everything it
runs from).
"""

from __future__ import annotations

from .model import KIND_SWEEP

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # annotations only
    from ..runfiles import RunNames

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import calcdirs, runrecord
from ..pseudos import PSEUDO_DIRNAME
from ..identity import resolve_stage_ref
from ..runfiles import FIRST_ATTEMPT
from .model import JobSet, warm_carry
from .naming import (job_dir_names, ladder_homes, open_container, run_names,
                     stage_refs, stamp_containers)
from ..paths import (attempt_dir, attempt_index, attempts_in,
                     bench_containers_in as _bench_containers_in)

def trial_work_dir(container, shape, names: "RunNames") -> Path:
    """Where a trial's files GO, given the directory it lives in.

    `naming.trial_dir` (via `naming.job_dir_names`) answers *where does this
    trial live*; this answers *where does prep put its deck and package*,
    and the two differ the moment a trial keeps attempts
    (`project-layout.md` § 1.5a):

    * **hierarchical** — the attempt, ``bench-<point>/run-<n>``;
    * **flat** — the container itself, because flat has no directory layer
      to open.

    **It takes the container rather than recomputing it**, so it cannot
    diverge from :func:`job_dir_names`.

    `resolve_attempt` is the rule, not restated: reuse the last attempt
    until it has been launched, then open the next -- a benchmark's sweep is
    prepared once (`job-system.md` § 5.0), so its prep finds ``run-0``.

    **The read-side twin is :func:`run_dir`** — *where does this trial
    actually run*, which needs no shape because by then the directory is
    there to be found.  Both resolve to the newest attempt, and that is the
    invariant: if they ever disagree, prep writes where nothing reads.
    """
    d = Path(container)
    if shape is None or not shape.keeps_attempts_as_directories:
        return d
    # NOTHING IS MADE HERE: a container not there yet holds no attempt
    # (`paths.attempts_in`), and the plan makes the folder when it is
    # written.
    attempt, _fresh = resolve_attempt(d, names)
    return attempt


def sweep_set_dirs(bundle) -> "List[Path]":
    """Every folder a SWEEP's plan can lie in, in this bundle -- the bench
    containers the layout declares (`paths.bench_containers_in`), in order.

    The search counterpart of :func:`bench_container`, and it lives beside it
    for that reason: the namer says where a sweep's state GOES and this says
    where to look for it, so a layout change moves one file instead of two
    that must be kept in step by hand.  Whether a folder holds a plan, and
    whether it is a sweep's, is the plan's door's answer (`JobSet.of`,
    `architecture.md` A16); nothing is read here.
    """
    base = Path(bundle)
    return [base / rel for rel, _tok in _bench_containers_in(base)]


def bench_stage_of(folder) -> "Optional[str]":
    """The stage a sweep measures, by its name -- read off where the sweep
    lives (`calcdirs.where`): the benchmark ``folder`` is, or the one its
    trial is in, in either layout.  A refusal or a status names the stage's
    own verbs with it, never a ``<stage>``.  ``None`` for a sweep in no
    stage's benchmark -- a hand-built one."""
    from ..identity import command_stage
    at = calcdirs.where(folder)
    if at.stage is None or (at.what != calcdirs.BENCHMARK
                            and at.trial is None):
        return None
    return command_stage(at.stage)


def bench_owner(folder) -> "Optional[Tuple[Path, str]]":
    """``(calculation folder, stage name)`` when ``folder`` IS a stage's
    benchmark -- the flat ``bench_<NN>_<stage>`` or the hierarchy's
    ``<NN>_<stage>/bench``, by where it is (`calcdirs.where`) -- else
    ``None``.  A sweep's own job-set names its trials from the calculation,
    so a reader standing in its container reads them from there."""
    from ..identity import command_stage
    at = calcdirs.where(folder)
    if at.what != calcdirs.BENCHMARK:
        return None
    return at.root, command_stage(at.stage)


@dataclass(frozen=True)
class ScratchFile:
    """One file of an engine's scratch a calculation's runs hold -- its
    stage (``None`` on the flat layout, where the stages share one folder),
    where it lies, its size in bytes, and the run it lies in
    (`paths.run_folder_of`; ``None`` on the flat layout, where the runs
    share the folder)."""
    stage: Optional[str]
    path: Path
    size: int
    run: Optional[Path] = None


def scratch_in(calc) -> List[ScratchFile]:
    """AN ENGINE'S SCRATCH a calculation's runs hold (`job-system.md`
    § 5.3, *scratch*): every file the one answer names scratch
    (`warmfiles.is_scratch`, which the checkpoint asks too) below each
    stage's folder -- on the flat layout the calculation's own -- the
    stores and every dot-folder left out.  What `jobset scratch` lists and
    empties; it reads names and sizes, never a file, and a file gone while
    it looks is passed over."""
    from ..paths import run_folder_of
    from ..warmfiles import is_scratch
    base = Path(calc.base)

    def under(folder: Path):
        for here, dirs, files in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in sorted(files):
                path = Path(here) / name
                if is_scratch(name) and not path.is_symlink():
                    try:
                        yield path, path.stat().st_size
                    except FileNotFoundError:
                        continue
    if not calc.shape.keeps_attempts_as_directories:
        return [ScratchFile(None, p, size) for p, size in under(base)]
    found = []
    for home in ladder_homes(base, calc.task):
        if home.dir is None or not home.dir.is_dir():
            continue
        for p, size in under(home.dir):
            run = run_folder_of(p.relative_to(base))
            found.append(ScratchFile(home.name, p, size,
                                     base / run if run is not None else None))
    return found


def materialize(jobset: JobSet, base_dir, plan=None, *,
                shape, only=None) -> List[Path]:
    """Create each job's directory under ``base_dir`` with its copies --
    the jobs named in ``only`` (a set of job names), or every job.

    Returns the list of created job directories (in JobSet order).  Idempotent:
    a copy already there is kept, never written twice.  Raises
    ``ValueError`` if the JobSet is structurally invalid (so a bad carry /
    duplicate name can't produce a broken tree).

    ``plan`` (`jobset.planned.Plan`) receives the folders and copies instead
    of the disk, and is read for the files it already holds -- `prep`
    decides everything before it writes (`job-system.md` § 5.0); without one
    they are written now.  ``shape`` is the layout as the caller read it
    (`Shape.named(task.shape)`, the one door; A16).
    """
    errors = jobset.validate()
    if errors:
        raise ValueError(
            "cannot materialize an invalid JobSet:\n  - "
            + "\n  - ".join(errors))
    from .planned import Plan
    own = plan is None
    plan = Plan() if own else plan
    base = Path(base_dir)
    created: List[Path] = []
    sh = shape
    dirs = job_dir_names(jobset, sh)
    for job in jobset.jobs:
        if only is not None and job.name not in only:
            continue
        # A TRIAL KEEPS ATTEMPTS EXACTLY AS A STAGE DOES, and the shape
        # decides (`project-layout.md` § 1.5a).  `trial_work_dir` is the
        # one answer to *where do this trial's files go*, and `prep` asks
        # the same one.
        d = base / dirs[job.name]
        if jobset.kind == KIND_SWEEP:
            # A TRIAL'S FOLDER IS A RUN's, opened by the one opener.
            d = open_run(base, trial_work_dir(d, sh,
                                              run_names(jobset, job, sh)),
                         plan)
        else:
            plan.folder(d)
        created.append(d)
        if d.resolve() == base.resolve():
            # FLAT: depth 1 (`project-layout.md` § 1) -- the job runs in the
            # bundle root, where every file it needs ALREADY SITS, and flat's
            # warm files are ONE SHARED SET there (§ 1) -- except the
            # pseudopotentials: they live in `pseudos/`, and SIESTA
            # opens `<element>.psml` in the directory it runs from and has no
            # search path.
            for _ps in plan.glob(base / PSEUDO_DIRNAME, "*.psml"):
                _dst = d / _ps.name
                if not plan.is_file(_dst):
                    plan.copy(_ps, _dst)
            continue
        # The static package arrives as REAL COPIES (user, 2026-08-24;
        # `project-layout.md` § 1.0: the run directory "holds everything",
        # and a symlink holds nothing).  The deck is not in this list: it is
        # born in the directory, so there is no root copy to reach for.
        for fname in list(jobset.shared):
            src = base / fname
            dst = d / os.path.basename(fname)
            if not plan.is_file(src):
                continue          # prep's own missing-input gates report it
            if not plan.is_file(dst):
                plan.copy(src, dst)
        # NOTHING ELSE IS LINKED IN: what a stage continues from is a real
        # file COPIED by `prepare_attempt` from the attempt you name
        # (project-layout.md 1.6).
    if own:
        plan.carry_out()
    return created


# --------------------------------------------------------------------- #
#  Attempts — one directory per try at a stage (project-layout.md § 1.6)  #
# --------------------------------------------------------------------- #


def latest_attempt(stage_dir: Path) -> Optional[Path]:
    """The newest attempt under ``stage_dir``, or ``None`` if there are none.

    **Where a stage's state actually is.** `project-layout.md` § 1.5 is flat
    about it — *"Where a run happens: inside the attempt directory"* — and
    *"everything the run writes"* is *"created in place"*, because the wrapper
    is invoked there. So anything asking *what happened to this stage?* asks
    here first, and only falls back to the container for a flat run, which
    § 1.5 says is untouched and *"is a run"* in its own right.

    This is a layout question, so it is answered in the layout layer rather
    than by each observer working out where to look.
    """
    ns = attempts_in(stage_dir)
    return attempt_dir(stage_dir, ns[-1]) if ns else None


def newest_on_label(stage_dir: Path, role: str
                    ) -> Tuple[Optional[Path], Optional[Path]]:
    """THE STAGE'S NEWEST ATTEMPT AND ITS FILE IN ``role`` on the label
    (`runs.on_label_at`: a run's own folder, or a run in phases' root) --
    ``(None, None)`` before its first attempt, ``(attempt, None)`` where it
    holds none yet.  The one answer to *this stage's result*."""
    attempt = latest_attempt(stage_dir)
    if attempt is None:
        return None, None
    from ..runs import on_label_at
    return attempt, on_label_at(attempt, role)


def run_dir(container: Path) -> Path:
    """**The directory a stage or trial actually uses** — its newest attempt
    when the shape keeps them, the container itself when it does not.

    THE OTHER HALF OF :func:`latest_attempt`, which answers *is there an
    attempt* and is right to return ``None`` for flat.  Almost every caller
    wants *where do I look*, and this answers it in the layout layer.

    Keep using `latest_attempt` where ``None`` is the ANSWER (has this been
    prepared at all?); use this where a path is wanted.

    **The write-side twin is :func:`trial_work_dir`** — *where does prep PUT
    this trial's files*, which takes the shape because it runs before the
    directory exists and may have to open one.  The two agree by
    construction and must keep agreeing: prep uses (or opens) the newest
    attempt, and this finds the newest.  If either ever picks a different
    one, prep writes where nothing reads.
    """
    return latest_attempt(container) or Path(container)


def resolve_attempt(stage_dir: Path, names: "RunNames") -> Tuple[Path, bool]:
    """The attempt directory to prepare into, and whether it is a fresh one.

    The last attempt is REUSED when it has not been launched -- the one prep
    opened, which `launch` then runs -- and a new one is opened only when the
    last has: what a launch of a stage again opens (`job-system.md` § 5.4).
    That makes the numbering mean something: every ``run-<n>`` on disk but
    the newest was actually started.  ``names`` are the stage's
    (:func:`run_names`), which name its launch record.
    """
    existing = attempts_in(stage_dir)
    if existing:
        last = attempt_dir(stage_dir, existing[-1])
        try:
            launched = runrecord.launch_record(last, names) is not None
        except runrecord.LaunchRecordError as e:
            from .errors import PrepError
            raise PrepError(str(e)) from e
        if not launched:
            return last, False
        return attempt_dir(stage_dir, existing[-1] + 1), True
    return attempt_dir(stage_dir, FIRST_ATTEMPT), True


def ever_launched(calc, job) -> bool:
    """Whether ``job``'s stage has been launched -- any run of it
    (`project-layout.md` § 1.6.1): where its runs are folders, every
    ``run-<n>`` but the newest was, and the newest when its launch record
    says so (:func:`resolve_attempt`'s rule); where its runs share a
    folder, its newest launched run's record.  ``calc`` is the calculation
    as the verb read it (A16): its plan and its shape.  A record that does
    not read raises, as the one door does (`runrecord.launch_record`)."""
    jobset, base, sh = calc.jobset, Path(calc.base), calc.shape
    names = run_names(jobset, job, sh)
    container = base / job_dir_names(jobset, sh)[job.name]
    if not sh.keeps_attempts_as_directories:
        return runrecord.launch_record(container, names) is not None
    ns = attempts_in(container)
    return len(ns) > 1 or (bool(ns) and runrecord.launch_record(
        attempt_dir(container, ns[-1]), names) is not None)


def newest_launch(calc, job) -> Optional[dict]:
    """The launch record of ``job``'s newest LAUNCHED run -- where its runs
    are folders, the newest ``run-<n>`` whose record is there; where they
    share a folder, its newest launched run's -- or ``None`` before its
    first launch.  A record that does not read raises, as the one door does
    (`runrecord.launch_record`)."""
    jobset, base, sh = calc.jobset, Path(calc.base), calc.shape
    names = run_names(jobset, job, sh)
    container = base / job_dir_names(jobset, sh)[job.name]
    if not sh.keeps_attempts_as_directories:
        return runrecord.launch_record(container, names)
    for n in reversed(attempts_in(container)):
        rec = runrecord.launch_record(attempt_dir(container, n), names)
        if rec is not None:
            return rec
    return None


@dataclass(frozen=True)
class Attempt:
    """One try at a stage: the directory, and what was put in it -- the
    stage's own files (``brought``) and what it continues from, one
    hand-over (``hand``, ``None`` when it takes nothing).  ``fresh`` is
    False when an unlaunched attempt was **reused** rather than opened
    (:func:`resolve_attempt`)."""
    stage:          str
    dir:            Path
    fresh:          bool
    brought:        List[str]
    hand:           Optional["HandOver"] = None

    @property
    def copied(self) -> List[str]:
        """The files it took from the run it continues from."""
        return self.hand.names_taken if self.hand is not None else []


#: Why ``--from`` and ``--cold`` mean nothing on the flat layout -- said by
#: `continuation` before prep writes anything, and by `prepare_attempt`
#: to a caller that asks it anyway.
FLAT_HAS_NO_ATTEMPTS = (
    "this calculation's shape is 'flat', which has no attempt directories "
    "to open: each run carries the number launch gives it "
    "(<label>_<NN>_<name>-run<N>.out) and every stage reads the files the "
    "stage before it left in the one folder (project-layout.md § 1) -- so "
    "there is no run to name with --from, and none to skip with --cold.")


def open_run(base, run_dir, plan) -> Path:
    """THE ONE OPENER OF A RUN FOLDER (`project-layout.md` § 1.6.2): a
    stage's attempt (:func:`prepare_attempt`, a bias point's included) and a
    benchmark trial's folder, in either shape.  It makes the folder and every
    directory above it down from the calculation root, and says what each is
    (§ 1.4a, invariant 6b): each above a container, the run a run.  The root
    says itself, through its description, and gets no record.

    ``plan`` receives the folders and the stamps, as :func:`materialize`'s
    copies.  Returns ``run_dir``."""
    base, run_dir = Path(base), Path(run_dir)
    plan.folder(run_dir)
    if run_dir.resolve() == base.resolve():
        return run_dir
    stamp_containers(base, run_dir.parents, plan)
    plan.text(*calcdirs.record(run_dir, role=calcdirs.RUN, root=base))
    return run_dir


def open_scratch(base, folder, plan) -> Path:
    """Make a run's scratch folder -- a device run's ``tsgf/``, the engine's
    scratch of the run above it (`engines/transport.md` § 2a.11) -- and say
    what it is (`calcdirs.SCRATCH`, invariant 6b): the run door and the
    Results sidebar read the role, and the checkpoint keeps the marker, so a
    restore brings the folder back empty.  Returns ``folder``."""
    base, folder = Path(base), Path(folder)
    plan.folder(folder)
    plan.text(*calcdirs.record(folder, role=calcdirs.SCRATCH, root=base))
    return folder


def bring_files(jobset: JobSet, job, base, src_dir, run, names, plan
                ) -> List[str]:
    """Copy into ``run`` what it runs from -- the deck, its wrappers, the
    shared package and the bundles that travel beside the deck -- from
    ``src_dir`` (where prep rendered them: the stage's folder, or a swept
    point's), else the calculation's root.  COPIED, never linked
    (`project-layout.md` § 1.0: the run directory "holds everything"; a
    synced-back bundle's links would dangle).  Returns what was brought.

    ONE COPY STEP for every run folder: a stage's attempt
    (:func:`prepare_attempt`) and a swept run's point
    (:func:`open_sweep_run`)."""
    base, src_dir, run = Path(base), Path(src_dir), Path(run)
    # THE BUNDLES THAT TRAVEL BESIDE THE DECK -- the monitor's, the job's
    # finish, a PySCF script's code -- from the one list the wrapper's
    # writer reads too (`runwrap.bundles_for`): one file each, so none can
    # be half-shipped.  And makov_payne_correction.py: the post-run script a
    # CHARGED deck's own header instructs the user to run "after SIESTA
    # finishes" -- HERE, beside the .out.
    from ..runfiles import beside_the_deck
    from .planned import bring
    # THE ONE LIST of what travels beside a deck (`runfiles.beside_the_deck`:
    # the bundles, the post-run script, the run script, the header, a call's
    # coordinates file -- a folder holding none of one brings none), and the
    # calculation's shared package.
    names_ = ([job.script] + list(jobset.shared)
              + list(beside_the_deck(job.script, job.finish, names)))
    # THE ONE COPY STEP (`planned.bring`): from where prep rendered them,
    # else the calculation's root.
    return bring(plan, names_, into=run, from_dir=src_dir, also=(base,))


@dataclass(frozen=True)
class SweepRun:
    """A swept stage's run (`engines/transport.md` § 2a.11): its folder, its
    points in walk order -- ``[(run-<n>/<point>, Point)]``, each frame's
    voltages in turn (`transport.stages.rung_points`) -- and whether it is a
    fresh one (``False``: the unlaunched run prep opened, reused)."""
    stage:  str
    dir:    Path
    points: List[Tuple[Path, object]]
    fresh:  bool


def open_sweep_run(calc, stage_name: str, *, plan) -> SweepRun:
    """Open a swept stage's run -- ``run-<n>/`` -- and in it each point's
    folder, holding copies of that point's prepared files (its deck, run
    script and the shared package; :func:`bring_files`), and the run's
    scratch folder where the rung keeps one.  ``points`` is the listing
    order (frames outer); the walk runs them round by round
    (`engines/transport.md` § 2a.11).  The run is the
    stage's newest when it has not been launched -- reused, and emptied
    whole first (:func:`empty_run`) -- else the next (a launch again,
    `job-system.md` § 5.4, rule 3).  Each point
    folder is a run folder (`open_run`); the run above them is their
    container.  What the
    points take from upstream is the caller's (launch's gather), and what
    each starts from the walk's."""
    from .points import points_of
    jobset, base, task, sh = (calc.jobset, Path(calc.base), calc.task,
                              calc.shape)
    job = jobset.job(stage_name)
    stage_dir = base / job_dir_names(jobset, sh)[job.name]
    rn = run_names(jobset, job, sh)
    run, fresh = resolve_attempt(stage_dir, rn)
    # THE RUNG'S PREPARED POINTS, by its kind's declaration (`jobset.points`),
    # its records as this plan holds them -- a prep composing the junction
    # in the same plan has not written its record yet.
    declared = points_of(task)
    sources = dict((pt, stage_dir / pt.rel) for pt in declared
                   .rung_points(calc, stage_name, read=plan.read_text))
    points = [(run / pt.rel, pt) for pt in sources]
    if not fresh:
        empty_run(run, plan)
    for pdir, pt in points:
        open_run(base, pdir, plan)
        bring_files(jobset, job, base, sources[pt], pdir, rn, plan)
    # THE FOLDERS THE RUN HOLDS BESIDES ITS POINTS, by its kind's
    # declaration (`points.PointsOf.containers`): a vibration's optical
    # calls' container (`engines/vibration.md` § 5.11 § 2).
    for rel in declared.containers(calc, stage_name):
        open_container(base, run / rel, plan)
    # THE RUN'S SCRATCH FOLDER, opened with the run (`engines/transport.md`
    # § 2a.11): the rung declares it (`Job.scratch_dir`).
    if job.scratch_dir:
        open_scratch(base, run / job.scratch_dir, plan)
    return SweepRun(stage=stage_name, dir=run, points=points, fresh=fresh)


def empty_run(run, plan) -> List[str]:
    """A run never launched, emptied whole -- every file in it and every
    folder below it -- for the prep or the launch that fills it again
    (`job-system.md` § 5.0, 2a: *its first run emptied of anything put
    there before*; a run a refused send opened, § 6.0 step 4).  A link is
    removed, never followed.  Returns the paths taken out, from ``run``."""
    run = Path(run)
    out: List[str] = []
    for here, dirs, files in os.walk(run, topdown=False):
        here = Path(here)
        for name in sorted(files):
            plan.remove(here / name)
            out.append(str((here / name).relative_to(run)))
        for name in sorted(dirs):
            d = here / name
            if d.is_symlink():
                plan.remove(d)
            else:
                plan.remove_folder(d)
            out.append(str(d.relative_to(run)) + "/")
    return out


def empty_old_points(stage_dir, points, plan, *, keep=()) -> List[str]:
    """A swept rung prepared again: the folders prep rendered below its
    stage for points the description no longer holds, taken out whole --
    its files rendered again (`job-system.md` § 5.0, 2a).  ``points`` are
    the points prepared now (``Point.rel``); a run folder (``run-<n>``) and
    the folders ``keep`` names -- the stage's benchmark container, the walk
    a launch writes -- are never touched.  Returns the folders taken out,
    from ``stage_dir``."""
    stage_dir = Path(stage_dir)
    wanted = {str(Path(r)) for r in points}
    above = {str(parent) for r in wanted for parent in Path(r).parents
             if str(parent) != "."}
    out: List[str] = []

    def gone(folder: Path) -> None:
        for here, dirs, files in os.walk(folder, topdown=False):
            for name in sorted(files):
                plan.remove(Path(here) / name)
            for name in sorted(dirs):
                d = Path(here) / name
                if d.is_symlink():
                    plan.remove(d)
                else:
                    plan.remove_folder(d)
        plan.remove_folder(folder)
        out.append(str(folder.relative_to(stage_dir)) + "/")

    def walk(folder: Path) -> None:
        for d in sorted(p for p in folder.iterdir()
                        if p.is_dir() and not p.is_symlink()):
            rel = str(d.relative_to(stage_dir))
            if folder == stage_dir and (attempt_index(d.name) is not None
                                        or d.name in keep):
                continue
            if rel in wanted:
                continue
            if rel in above:
                walk(d)
            else:
                gone(d)

    if stage_dir.is_dir():
        walk(stage_dir)
    return out


#: A hand-over's two records (`job-system.md` § 5.4): the one run a run
#: continues from (`.continued-from`), and a transport rung's inputs, each
#: from its run (`.gathered-from`).
CONTINUED, GATHERED = "continued", "gathered"


@dataclass(frozen=True)
class HandOver:
    """WHAT ONE RUN TAKES FROM OTHER RUNS, AND THE RECORD THAT SAYS SO --
    every hand-over's one shape (`job-system.md` § 5.4), whoever decided it:
    the continuation door, the gather door, a launch again keeping what its
    last run was given, the points' walk.

    ``target`` is the run that takes -- an attempt, a point of a run of
    points, or a flat calculation's folder -- named by its stage's
    ``names`` as run ``run``; ``files`` each file it takes, with the run it
    comes from; ``record`` which record says so: :data:`CONTINUED` -- the
    one run it continues from, ``source`` (a taken-over point's is the run
    that COMPUTED it, one hop, whatever folder its files are copied from)
    -- or :data:`GATHERED`, each file with its run.  ``walk``: copied by the
    walk just before the run starts (a point's start, which the point
    before it may finish in the same walk), not with the plan."""
    target: Path
    names: "RunNames"
    run: int
    record: str
    files: Tuple[Tuple["runrecord.RunRef", str], ...] = ()
    source: Optional["runrecord.RunRef"] = None
    walk: bool = False

    def __post_init__(self):
        if self.record not in (CONTINUED, GATHERED):
            raise ValueError(f"a hand-over's record is {CONTINUED!r} or "
                             f"{GATHERED!r}, not {self.record!r}")
        if (self.record == CONTINUED) != (self.source is not None):
            raise ValueError("a continued hand-over names the run it "
                             "continues from; a gathered one names a run "
                             "per file")

    @property
    def names_taken(self) -> List[str]:
        """The files it takes, by name, in order."""
        return [name for _src, name in self.files]

    def written_as_ended(self, base) -> Dict[str, str]:
        """THE FILES IT TAKES THAT WERE WRITTEN AS THEIR RUN ENDED
        (:func:`written_as_ended`, `running-a-job.md` § 3.6), each from the
        folder it is taken from."""
        out: Dict[str, str] = {}
        by_folder: Dict[Path, List[str]] = {}
        for src, name in self.files:
            by_folder.setdefault(Path(src.path(base)), []).append(name)
        for folder, names in by_folder.items():
            out.update(written_as_ended(folder, self.names.label, names))
        return out

    def walk_copies(self, base, cwd) -> List[Tuple[str, str]]:
        """What the walk copies just before the run starts: ``(file, folder)``
        pairs, each from ``cwd``, the folder the walk runs in."""
        dst = os.path.relpath(self.target, cwd)
        return [(os.path.relpath(src.path(base) / name, cwd), dst)
                for src, name in self.files]


def written_as_ended(folder, label: str, files, *,
                     run: Optional[str] = None) -> Dict[str, str]:
    """THE FILES WRITTEN AS THEIR RUN ENDED (`running-a-job.md` § 3.6):
    ``{name: why}`` for each of ``files`` in ``folder`` whose write time is
    at or after the second the run's monitor closed -- the monitor log's
    closing record (`parse/instruments/monitor.py`, ``ended_at``), read
    through the registry -- a file a scheduler's kill may have caught
    mid-write.  The log is the newest of ``label``'s in the folder, or
    ``run``'s own where a stage's runs share the folder (its name,
    ``H2_01_coarse-run1``).  Empty where no monitor closed, or every file
    was written before.  A fact shown, never a veto."""
    from datetime import datetime
    from .. import runfiles as _rf
    from ..parse import detect
    folder = Path(folder)
    if run is not None:
        rec = _rf.parse_run_name(run, label)
        logs = (_rf.find(folder, label, role=".monitor.log", stage=rec.stage,
                         run=rec.run) if rec is not None else [])
    else:
        logs = sorted(_rf.find(folder, label, role=".monitor.log"),
                      key=lambda pr: (pr[1].run is None, pr[1].run or 0))
    if not logs:
        return {}
    try:
        at = detect(str(logs[-1][0])).parse(logs[-1][0]).metrics.get("ended_at")
        end_s = datetime.fromisoformat(at).timestamp() if at else None
    except Exception:                       # noqa: BLE001 -- an unreadable
        return {}                           # log says nothing
    if end_s is None:
        return {}
    out: Dict[str, str] = {}
    for name in files:
        try:
            wrote = (folder / name).stat().st_mtime
        except OSError:
            continue
        if wrote >= int(end_s):
            w = datetime.fromtimestamp(wrote).astimezone()
            out[name] = (f"written as the job ended ({w:%H:%M:%S}; the job "
                         f"ended {at[11:19]}), it may be incomplete; a start "
                         f"from it may need more iterations or fail; "
                         f"`--cold` starts without it")
    return out


def hand_over(base, h: HandOver, plan=None) -> List[str]:
    """CARRY ONE HAND-OVER OUT -- the one function every hand-over goes
    through (`job-system.md` § 5.4): its files copied into its run -- as
    real files, never linked, since the engine writes to those very names
    (`project-layout.md` § 1.6.5) -- unless the walk copies them
    (``h.walk``), and its record written beside the run
    (`runrecord.write_continued_from`, `write_gathered_from`).  Into
    ``plan`` (`jobset.planned.Plan`); with none, now -- a flat run's
    record, written with its launch record.  Returns the files copied."""
    from .planned import Plan
    own = plan is None
    plan = Plan() if own else plan
    copied: List[str] = []
    if not h.walk:
        for src, name in h.files:
            plan.copy(src.path(base) / name, Path(h.target) / name)
            copied.append(name)
    if h.record == CONTINUED:
        runrecord.write_continued_from(h.target, h.source, names=h.names,
                                       run=h.run, plan=plan)
    else:
        runrecord.write_gathered_from(h.target, list(h.files), plan=plan)
    if own:
        plan.carry_out()
    return copied


def prepare_attempt(calc, stage_name: str, *,
                    continue_from: Optional[str] = None,
                    cold: bool = False,
                    carry: Optional[List[str]] = None,
                    named: bool = True, plan=None) -> "Attempt":
    """Set ONE stage up to run, and report what was done.

    ``plan`` receives the attempt's folder and everything put in it, and is
    read for the stage's files it already holds -- `prep` opens the attempt
    as part of its plan, written after the save (`job-system.md` § 5.0),
    and `launch` fills it as part of its own, written at the send; a caller
    with none has it written now.  ``calc`` is the calculation as the verb
    read it (A16): its plan -- the stage's row and the one it continues
    from -- and its shape.

    ``named`` says who chose ``continue_from``: the person, by ``--from``, or
    the launch, by default (`job-system.md` § 5.4) -- so a refusal quotes
    what was typed, never a ``--from`` nobody typed.

    The five steps § 1.6 names: **resolve** the next ``run-<n>``, **create**
    it, **copy** the deck / monitor / shared package in, **copy** whatever this
    run continues from, and **report** — the report being the point, since
    preparing is still design and the split from starting is what gives you
    somewhere to look before committing cluster time.

    ``continue_from`` is a bundle-relative attempt directory —
    ``"01_coarse/run-0"``. **Which run is never a guess** (§ 1.6):
    continuing from ``run-0`` and from ``run-2`` are different scientific
    choices, so the callers pass one they can name -- a first launch, by
    default, the stage before it's newest attempt, which must have
    finished, or the one a person named (`continuation.continuation_answer`,
    `job-system.md` § 5.4); a launch again, the SAME stage's latest
    (user, 2026-08-21).  ``cold=True`` means start
    clean, and with a directory per attempt that is simply *skip the copy* —
    there is nothing to move aside, because a fresh attempt is empty unless
    something is put in it.  A REUSED attempt -- never launched -- is
    emptied whole first (:func:`empty_run`), whoever fills it: prep
    preparing it again, or the launch that sends it.

    ``carry`` names the files to copy; it defaults to :func:`warm_carry` for
    **this pair** — the stage being prepared and the stage that produced the
    attempt named by ``continue_from``. They are **copied, never linked** — the
    engine writes to those very filenames, and writing through a link would
    destroy the result you started from.

    ``stage_name`` goes through the ONE resolver, so it takes a name, a number
    or a token — the same three spellings every other surface takes, and the
    same refusal when it matches none of them.
    """
    jobset, base, sh = calc.jobset, Path(calc.base), calc.shape
    if not sh.keeps_attempts_as_directories:
        raise ValueError(FLAT_HAS_NO_ATTEMPTS)
    dir_of = job_dir_names(jobset, sh)
    refs = stage_refs(jobset)
    stage_name = resolve_stage_ref([refs[j.name] for j in jobset.jobs],
                                   stage_name).name
    job = jobset.job(stage_name)

    # WHAT IT CONTINUES FROM, CHECKED BEFORE ANYTHING IS WRITTEN (W52), so a
    # refusal leaves no attempt opened or stripped.
    names: List[str] = (
        continuation_files(calc, stage_name, continue_from,
                           named=named, carry=carry)
        if continue_from and not cold else [])

    from .planned import Plan
    own = plan is None
    plan = Plan() if own else plan
    stage_dir = base / dir_of[stage_name]
    plan.folder(stage_dir)
    rn = run_names(jobset, job, sh)
    attempt, is_new = resolve_attempt(stage_dir, rn)
    # AN ATTEMPT NEVER LAUNCHED, REUSED -- prepared again, or filled at the
    # launch after a refused send: what was put into it before is taken
    # out first, so it starts from what this one takes.  A changed mind
    # that left an `.XV` lying there would change nothing -- the engine
    # finds it and warm-starts anyway, the *"present but not honoured"*
    # failure wearing its other face, and silent.
    if not is_new:
        empty_run(attempt, plan)
    # THE ONE OPENER (`open_run`): the attempt and every container down to
    # it, each saying what it is -- container or run, which no name says:
    # the shape and whether a rung sweeps decide (`project-layout.md`
    # § 1.4a).
    open_run(base, attempt, plan)
    # THE RUN'S SCRATCH FOLDER, opened with the run (`engines/transport.md`
    # § 2a.11): the rung declares it (`Job.scratch_dir`).
    if job.scratch_dir:
        open_scratch(base, attempt / job.scratch_dir, plan)

    brought = bring_files(jobset, job, base, stage_dir, attempt, rn, plan)

    # WHAT IT CONTINUES FROM, ONE HAND-OVER (`job-system.md` § 5.4): the
    # files the run holds of what the stage takes, and the record of the
    # run they came from, beside the run.
    hand = None
    if continue_from and not cold:
        source = runrecord.RunRef.read(continue_from, shared=False)
        took = tuple((source, name) for name in names
                     if (source.path(base) / name).is_file())
        if took:
            hand = HandOver(attempt, rn, FIRST_ATTEMPT, CONTINUED,
                            files=took, source=source)
            hand_over(base, hand, plan)
    if own:
        plan.carry_out()

    return Attempt(
        stage=stage_name,
        dir=attempt,
        fresh=is_new,
        brought=brought,
        hand=hand,
    )


def continuation_files(calc, stage_name: str, continue_from: str, *,
                       named: bool,
                       carry: Optional[List[str]] = None) -> List[str]:
    """The files ``stage_name`` would carry from ``continue_from`` --
    CHECKED, never copied: that attempt exists, the stage declares warm
    files for this pair (:func:`warm_carry`), and the attempt holds at least
    one of them.  Raises ``ValueError`` otherwise, worded for who chose the
    run: the person, by ``--from`` (``named``), or the door that chose it --
    a first launch's default, or a launch again's own latest run
    (`job-system.md` § 5.4).

    ONE CHECK, asked by :func:`prepare_attempt` before it writes anything
    and by `launch` before it shows what it will send -- so a run that
    cannot be continued is refused before the question, not after the yes.
    ``calc`` is the calculation as the verb read it (A16): the stage's row
    and the one the run came from are both its plan's.
    """
    jobset, base, shape = calc.jobset, Path(calc.base), calc.shape
    said = (f"--from {continue_from!r}" if named else
            f"the run it continues from ({continue_from})")
    src = base / continue_from
    if not src.is_dir():
        raise ValueError(
            f"{said}: no such attempt under "
            f"{base}. Name an attempt directory that has already run, "
            f"e.g. '01_coarse/run-0'.")
    job = jobset.job(stage_name)
    if job is None:
        raise ValueError(f"{stage_name!r} is not prepared -- it has no job "
                         f"to carry its files.")
    # The pair, resolved here and nowhere else -- `--from` is what names
    # the source, so this is the first moment both stages are known.
    dir_of = job_dir_names(jobset, shape)
    names = (carry if carry is not None
             else warm_carry(job, _source_job(jobset, dir_of, continue_from)))
    if not names:
        raise ValueError(
            f"{said}: {stage_name!r} declares no "
            f"warm-restart files, so there is nothing to continue.\n"
            f"  A stage whose description says `restart: clean` carries "
            f"none of the group -- its deck omits MD.UseSaveXV / "
            f"DM.UseSaveDM / MD.UseSaveCG, so files copied in would sit "
            f"there unread (run-identity.md § 4, *present but not "
            f"honoured*).  A stage continues when its `restart` is "
            f"`continue` in task.json at its prep; or start it cold.")
    if not any((src / name).is_file() for name in names):
        raise ValueError(
            f"{said}: that attempt holds none "
            f"of the files this stage would continue from "
            f"({', '.join(names)}). Did it run?")
    return list(names)


def _source_job(jobset: JobSet, dir_of: Dict[str, str], continue_from):
    """Which job produced the attempt named by ``--from``, or ``None``.

    ``continue_from`` is bundle-relative and always ``<stage dir>/run-<n>``
    (`job-system.md` § 5.3), so the stage is the attempt's PARENT read back
    through the SAME naming authority that wrote it. Nothing is parsed out of
    the name: :func:`job_dir_names` is asked, and a parent that matches no job
    simply has no answer — which :func:`warm_carry` then treats as *unverified*
    rather than guessing.
    """
    if not continue_from:
        return None
    parent = str(Path(continue_from).parent)
    for j in jobset.jobs:
        if dir_of.get(j.name) == parent:
            return j
    return None


__all__ = [
    "Attempt", "CONTINUED", "GATHERED", "HandOver", "hand_over",
    "bench_owner", "bench_stage_of", "scratch_in", "ScratchFile",
    "trial_work_dir", "materialize", "latest_attempt", "run_dir",
    "resolve_attempt", "ever_launched", "empty_run",
    "empty_old_points", "prepare_attempt", "open_sweep_run",
    "open_run",
]
