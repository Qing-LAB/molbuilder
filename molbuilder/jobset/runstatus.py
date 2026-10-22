"""Run-status reader — the "inform" layer
(`execution/job-system.md` § 2, rule 5: molbuilder informs; the user decides).

The resume contract is: **the modeling software resumes; molbuilder informs
and the user decides** (never auto-recovers — redoing a long run unknowingly
is a heavy penalty).  This module is the *inform* half: for a calculation --
its description's every stage, prepared or not -- it answers, per stage, *did
it finish? is it running? did it fail? are the warm-restart files there?* and
*what is each group's next step*, with the command each state calls for
-- so the manual continue is a one-glance decision.

REUSE, not reinvention: per-stage run state comes from
``parse.dirs.job.run_status`` (the directory-level status verb behind the Results
tab + JobMonitor; its schema is pinned by ``execution/running-a-job.md``
§ 4, the composer pattern by ``model/parse.md`` § 5); this module only adds the cross-stage
view (warm-file inventory + each group's next step).  It is **read-only**:
it inspects the tree, changes nothing.
"""

from __future__ import annotations

import dataclasses
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..identity import StageRef, stage_key
from ..paths import attempts_in
from .materialize import latest_attempt, run_dir
from .naming import job_dir_names, run_names, stage_refs
from ..runrecord import LaunchRecordError, launch_record
from .commands import (block, command, launch_lines, launch_pending,
                       prep_group_lines)
from .model import JobSet
from .continuation import (gathered_lines, point_rows, points_state,
                           taken_over)
from .plan import resources_text

# Engine-native warm-restart files keyed by the project id (system label),
# DERIVED from the one rules file (job-contracts § 4.2a; U3/W2).  The carry rows are status's
# question -- could a stage here hand state to the next one? -- asked of the
# one door with the calculation's folder, so its own list answers.
from ..warmfiles import warm_list as _warm_list
from ..paths import attempt_name



def _warm_files(engine: str, base):
    """What carries, from the list in effect for the calculation in
    ``base`` (`warmfiles.warm_list`, every section: status reads a folder,
    not a rung's kind) -- its own copy first, else the engine's
    (`job-contracts.md` § 4.2a).  Resolved AT USE — never at import (C-b):
    one malformed TOML must not kill every entry point with an import-time
    traceback; a list that does not read is said by status, naming the file
    (`warmfiles.WarmFilesError`), as prep refuses it."""
    return _warm_list(engine, None, base).carry

# run_status states we treat as "this stage is done".
_DONE = "finished"


@dataclass(frozen=True)
class StageStatus:
    """One stage's status (read-only snapshot).

    **WHICH stage this is arrives as a whole :class:`~molbuilder.identity.
    StageRef`, not as a loose name and a loose number.** Carrying the ref means the resolver in ``naming`` is the only
    place one is ever made.
    """
    ref:        StageRef
    #: The stage's directory, or ``None`` before anything prepared it.
    dir:        Optional[str]
    #: ``described`` (not prepared: the ready door's word) or the run's:
    #: pending/queued/running/failed/finished/unknown.  ``queued`` is
    #: the contract's own word (project-layout.md § 1.6, *"queued as job
    #: 481923"*) for launched-but-no-output-yet, which is exactly the state an
    #: empty directory cannot be read for.
    state:      str
    detail:     str
    warm_files: List[str] = field(default_factory=list)  # restart files present
    #: Which attempt this status was read from (``run-0``; a bias sweep's
    #: point's, ``v0.2/run-0``), or ``None`` for a flat run, which happens in
    #: the container itself (§ 1.5).
    attempt: Optional[str] = None
    #: Every attempt present, ascending -- the stage's history.  A re-run makes
    #: a new directory and leaves the old one exactly as it was (§ 1.5), so
    #: this is the list of tries, not a counter that can be off.
    attempts: List[int] = field(default_factory=list)
    #: The run's launch record, whole, or ``None`` if it has not been
    #: launched.  Carried rather than picked apart so the per-stage view has
    #: the record without a SECOND reader of the same file -- the schema is
    #: versioned (``molbuilder/run-launch@1``) and may grow fields.
    launch: Optional[Dict[str, Any]] = None
    #: A QUEUED stage's standing as its scheduler states it here, or where
    #: to ask (`runs.standing_of`; `web/results.md` § 2.4); ``None`` for
    #: any other state.
    standing: Optional[Dict[str, Any]] = None
    #: A device rung's one line about the leads' surface Green's function
    #: files -- what its run's ``tsgf/`` holds, or the size each lead's
    #: will have (`transport.gf_files.status_line`, `engines/transport.md`
    #: § 2a.11); ``None`` for every other rung.
    tsgf: Optional[str] = None
    #: Whether this stage, launched again, continues from its own latest run
    #: -- the job's one fact (`Job.relaunch_continues`), which `launch`
    #: asks too, so the status says what launching it again will do
    #: (`job-system.md` § 5.4, *A stage launched again*).
    relaunch_continues: bool = True
    #: Its group's number in the description (`engines/stages.md` § 2), or
    #: ``None`` for a job set with no description beside it.
    group: Optional[int] = None
    #: Whether anything has prepared it -- a description's stage before its
    #: first prep is listed all the same (`job-system.md` § 5.3).
    prepared: bool = True
    #: Whether the run converged, as its output ended -- ``SCF yes`` /
    #: ``SCF NO``, ``geometry yes`` / ``geometry NO`` for a relaxation --
    #: or ``None`` before an output says (`job-system.md`, *What molbuilder
    #: does for you*, 4).  Beside the state, never folded into it: a run can
    #: finish and not converge (`parse.dirs.run_status`).
    converged: Optional[str] = None
    #: A rung's points, frames outer and voltages inner -- the point's wire
    #: form (`stages.Point.to_wire`: ``frame``, ``of``, ``bias_v``) with
    #: ``point`` (its words), ``folder``, ``state`` (its run's), ``opened``
    #: (its folder is there), ``converged`` (its run's verdict), ``done``,
    #: ``why``, ``started_from`` (``start_unread``, the reader's words, when
    #: its record does not read) and ``took`` each, read from the point's
    #: own files (`point_rows`; `engines/transport.md` § 2a.11) -- or ``[]``.
    points: List[Dict[str, Any]] = field(default_factory=list)
    #: What the run GATHERED from the rungs upstream -- ``<file> <- <run>``
    #: each, the run's own ``.gathered-from`` (`runrecord.read_gathered_from`;
    #: `engines/transport.md` § 2a.11) -- or ``[]`` for a stage that gathers
    #: nothing.  A sweep's point adds what it alone took in its own row.
    gathered: List[str] = field(default_factory=list)
    #: The ready door's answer whole (`ready.readiness`), for a stage not
    #: launched: what its first launch takes, one line each, or what it
    #: waits for -- the refusal its prep, or its launch, would print, with
    #: its commands.
    takes: List[str] = field(default_factory=list)
    why: Optional[str] = None
    #: What the stage IS, the plan's columns: its deck, the restart files it declares
    #: -- what it would take from a run it continues from -- and the
    #: resources it asks for.
    script: Optional[str] = None
    carries: List[str] = field(default_factory=list)
    resources: Optional[str] = None

    @property
    def name(self) -> str:
        """The stage's name -- from the ref, so there is exactly one holder."""
        return self.ref.name

    @property
    def seq(self) -> Optional[int]:
        """Its assigned ordinal, read back off its deck -- NOT its row.

        ``None`` for a sweep point, which has no order at all
        (`project-layout.md` § 4.1).
        """
        return self.ref.seq

    def to_dict(self) -> Dict[str, Any]:
        """The wire form -- ``name`` and ``seq`` FLAT, not a nested ref.

        A surface reading JSON wants two plain fields; ``StageRef`` is how the
        library keeps them together, not a shape to export. Flattening here
        rather than nesting keeps this dict exactly what it has always been.
        """
        d = dataclasses.asdict(self)
        d.pop("ref")
        return {"name": self.ref.name, "seq": self.ref.seq, **d}


@dataclass(frozen=True)
class GroupStep:
    """One group's place in the task's next step (`job-system.md`, *The
    task*, *The next step*): its number and stages, its state, a few words
    for where it stands, and the lines that move it -- none when it waits --
    and whether it is the next step."""
    group: int
    stages: List[str]
    state: str
    words: str
    lines: List[str] = field(default_factory=list)
    next: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Its wire form, in the ladder answer (`web/results.md` § 2.4)."""
        return {"group": self.group, "stages": list(self.stages),
                "state": self.state, "words": self.words,
                "lines": list(self.lines), "next": self.next}


@dataclass(frozen=True)
class JobSetStatus:
    """The whole job-set's status + the resume pointer."""
    name:            str
    engine:          str
    stages:          List[StageStatus]
    first_incomplete: Optional[str]   # name of the first non-finished stage (resume here)
    complete:        bool             # every described stage finished
    #: The groups `prep task` offers (`ready.prep_offer`): every one not
    #: prepared and preparable -- the next step when the first incomplete
    #: stage is not prepared.
    offer: List[int] = field(default_factory=list)
    #: The calculation's folder, so the commands the renderers print name it
    #: (`commands.command`).  Not part of the wire form.
    base: Optional[str] = None
    #: The stage a benchmark's sweep measures (`materialize.bench_stage_of`):
    #: its rows are trials, and its next step is its own verbs.  ``None``
    #: for a ladder.  Not part of the wire form.
    bench_of: Optional[str] = None
    #: THE NEXT STEP, every group's (:func:`_steps`) -- ``[]`` for a job set
    #: with no description beside it.
    steps: List[GroupStep] = field(default_factory=list)
    #: The calculation the rows were read from (A16), for the renderers'
    #: way-back lines.  Not part of the wire form.
    calc: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """THE WIRE FORM -- the Results tab's ladder is this (`web/results.md`
        § 2.4), the next prep's answer included."""
        return {
            "name": self.name, "engine": self.engine,
            "first_incomplete": self.first_incomplete,
            "complete": self.complete,
            "stages": [s.to_dict() for s in self.stages],
            "offer": self.offer,
            "steps": [g.to_dict() for g in self.steps],
        }


def _warm_present(stage_dir: Path, label: str, engine: str,
                  base) -> List[str]:
    """Which engine warm-restart files actually exist (real files, not
    dangling carry symlinks) in this stage's dir -- by the list in effect
    for the calculation in ``base``."""
    out: List[str] = []
    from ..runfiles import compose
    for ext in _warm_files(engine, base):
        f = stage_dir / compose(label, ext)
        if f.is_file():                      # follows symlinks; dangling -> False
            out.append(f.name)
    return out


#: A RUN A REFUSED SEND OPENED -- the newest folder of a stage launched
#: before, with no launch record: not a run yet (a refused launch records
#: nothing), and filled anew by the stage's next launch (`job-system.md`,
#: *The task*: a run's states).
REFUSED_OPEN = ("opened by a launch whose send was refused -- not launched; "
                "its next launch fills it anew")


def _pending_detail(attempt, detail: str) -> str:
    """A run not launched, in words: prep's first run as the run door says
    it; a later one, opened by a launch whose send was refused."""
    from ..paths import attempt_index
    return (REFUSED_OPEN if attempt is not None
            and attempt_index(Path(attempt).name) else detail)


#: A PREPARED stage whose folder is not on disk -- against the design, said
#: as such (a stage not prepared is the ready door's `described`).
MISSING = ("missing", "the job set names this run's folder; it is not on "
                      "disk", None)


def every_run(base, task, st: StageStatus) -> List[Dict[str, Any]]:
    """EVERY RUN OF A STAGE, each with its own state -- the root's bird's-eye
    (`web/results.md` § 2.4): each attempt in order and, for a swept rung,
    each attempt's points (`transport.stages.points_in`), read as `status`
    reads a run (`run_state_of`), with the file it opens, root-relative:
    the run door's answer (`runs.openable`) -- the root's `task.json` for a
    transmission run, whose result is the report at its point (§ 0.1).
    ``[]`` for a stage nothing has prepared."""
    from ..runs import openable, run_of, state_of, transmission_point
    if not st.dir:
        return []
    root = Path(base)
    stage_dir = root / st.dir
    rows: List[Dict[str, Any]] = []
    for n in attempts_in(stage_dir):
        run_dir = stage_dir / attempt_name(n)
        folders: List[tuple] = []
        if getattr(task, "calculation", None) == "transport":
            from ..transport.stages import points_in
            folders = [(p, pt) for p, pt in points_in(run_dir, task,
                                                      st.ref.name, base=root)
                       if p.is_dir()]
        if not folders:
            folders = [(run_dir, None)]
        for folder, pt in folders:
            run = run_of(folder)
            if run is None or run.stage is None:
                continue
            got = state_of(run)
            # A TRANSMISSION RUN'S POINT IS THE DOOR'S, the one it opens the
            # report at -- its one point too, when the rung runs once.
            pt = transmission_point(folder) or pt
            # THE FILE THE RUN'S FOLDER OPENS, the run door's answer -- a
            # transmission run's is its calculation's report, at the root
            # (`runs.transmission_point`, results.md § 0.1).
            opens, _trail = openable(folder)
            opens_rel: Optional[str] = (
                str(Path(opens).resolve().relative_to(root.resolve()))
                if opens else None)
            # WHAT THE RUN READ AND WHAT WAS CHECKED BEFORE IT WAS WRITTEN,
            # root-relative (`web/results.md` § 2.4): the deck in the run's
            # folder, and the `.validation.txt` prep wrote beside the deck
            # it copied -- through the run door, never by a name.
            deck, report = run.deck, run.report

            def _rel(p: Optional[Path]) -> Optional[str]:
                return (str(p.resolve().relative_to(root.resolve()))
                        if p is not None else None)
            rows.append({
                "run": attempt_name(n),
                # THE POINT, in its one wire form (`stages.Point.to_wire`):
                # its frame, how many frames the set holds, its voltage --
                # what a row is named by on a page (`web/results.md` § 2.5).
                **(pt.to_wire() if pt is not None
                   else dict.fromkeys(("frame", "of", "bias_v"))),
                "dir": str(folder.relative_to(root)),
                "state": got.state,
                "detail": got.detail,
                "converged": got.converged_words,
                # ROOT-RELATIVE: the file the row opens -- the run's own, or
                # the root's task.json for a transmission point.
                "opens": opens_rel,
                "deck": _rel(deck),
                "validation": _rel(report),
            })
    return rows


def _sweep_status(base: Path, jobset: JobSet, job, task, *, d: Path,
                  names, ref) -> Optional["StageStatus"]:
    """A SWEPT RUNG's row (`engines/transport.md` § 2a.11): its latest run's
    state -- ``pending`` before it is launched; while launched, ``running``
    while a point runs, ``finished`` once every point is done, ``queued``
    when nothing has run yet, else ``failed`` -- and its points, each done
    or not done and what it started from.  ``None`` for a rung that does
    not sweep."""
    from ..transport.stages import frames_of, products_of, rung_points
    if task is None or not rung_points(task, job.name,
                                       frames=frames_of(base)):
        return None
    run = latest_attempt(d)
    if run is None:
        state, detail = MISSING[0], MISSING[1]
        return StageStatus(ref=ref, dir=d.name, state=state, detail=detail)
    try:
        launch = launch_record(run, names)
    except LaunchRecordError as e:
        return StageStatus(ref=ref, dir=d.name, state="unreadable",
                           detail=str(e), attempt=run.name)
    products = products_of(job.name, task.label, base_dir=base)
    if launch is None:
        pts, _running = point_rows(run, names, task, job.name, base=base,
                                   launch=launch, products=products)
    else:
        # THE ONE RULE the gather asks too (`continuation.points_state`).
        state, detail, pts = points_state(run, names, task, job.name,
                                          launch=launch, products=products,
                                          base=base)
    # CONVERGED, in its own column (`job-system.md`, rule 4): yes for a rung
    # with an SCF (`STAGE_FACT`: the transmission converges nothing) when
    # every point's output says it converged; NO when one says it did not;
    # otherwise nothing yet.
    from ..transport.stages import STAGE_FACT
    scf_rung = STAGE_FACT.get(job.name, "scf") != "product"
    converged = ("SCF NO" if any(p["converged"] is False for p in pts)
                 else "SCF yes" if scf_rung and pts
                 and all(p["converged"] is True for p in pts)
                 else None)
    if launch is None:
        state, detail = "pending", _pending_detail(run, "prepared, not "
                                                        "launched")
    return StageStatus(
        ref=ref, dir=d.name, state=state, detail=detail,
        converged=converged,
        attempt=run.name, attempts=attempts_in(d), launch=launch,
        relaunch_continues=job.relaunch_continues,
        script=str(job.script), carries=[w.name for w in job.warm],
        resources=resources_text(job.resources), points=pts,
        gathered=gathered_lines(run),
        tsgf=_tsgf_line(base, task, job, d, run if launch else None,
                        attempts_in(d), going=state in ("queued", "running")))


def _job_status(base: Path, jobset: JobSet, job, task, *, dirs,
                refs, shape) -> StageStatus:
    """One prepared job's status, read from where its attempts are."""
    d = base / dirs[job.name]
    # WHICH FILES are this stage's: its stage's names (`naming.
    # run_names`) -- in either shape a flat folder holds every stage, an
    # attempt may hold several run indexes.  THE LABEL IS THIS JOB'S, NOT
    # THE JOBSET'S: a trial is relabelled with its point.
    names = run_names(jobset, job, shape)
    swept = _sweep_status(base, jobset, job, task, d=d, names=names,
                          ref=refs[job.name])
    if swept is not None:
        return swept
    # WHERE the run happened, asked of the layer that decides layout -- the
    # latest attempt where there is one, the container for a flat run
    # (project-layout.md § 1.5).
    home = d
    attempt = latest_attempt(home)      # None is the ANSWER: prepared?
    observed = run_dir(home)            # ...and this is where to look
    # THE LAUNCH RECORD lies in the folder the run happened in, named by the
    # stage's names -- the one answer the writer reads too: the attempt's
    # `run.json`, a trial's at its top, a flat stage's newest run's own in
    # the folder every stage shares.  ONE THAT DOES NOT READ is said, never
    # read as launched or not (`runrecord.launch_record`).
    if not observed.is_dir():
        launch, (state, detail, converged) = None, MISSING
    else:
        # ITS STATE, THROUGH THE ONE DOOR (`runs.concluded`, which asks the
        # scheduler once where it can, `running-a-job.md` § 4.2): the
        # stage's newest run, asked of its launch record and its files --
        # a record that does not read is the state `unreadable`, said.
        from ..runs import concluded
        st = concluded(observed, names, base=base)
        state, detail, converged = st.state, st.detail, st.converged_words
        launch = (None if state == "unreadable"
                  else launch_record(observed, names))
    if state == "pending":
        detail = _pending_detail(attempt, detail)
    where = attempt.name if attempt else None
    from ..runs import standing_of
    return StageStatus(
        ref=refs[job.name], dir=d.name, state=state, detail=detail,
        converged=converged, attempt=where,
        attempts=attempts_in(home),
        launch=launch,
        standing=(standing_of(launch, base)
                  if state == "queued" else None),
        relaunch_continues=job.relaunch_continues,
        # THE SAME LABEL THE STATE WAS READ WITH -- the stage's names'.
        warm_files=_warm_present(observed, names.label, jobset.engine,
                                 base),
        # WHAT THE STAGE IS -- the plan's own columns, read per stage by
        # `status <stage>`.
        script=str(job.script),
        carries=[w.name for w in job.warm],
        resources=resources_text(job.resources),
        # WHAT THE RUN GATHERED (a transport rung), beside what it declares.
        gathered=gathered_lines(observed) if attempt is not None else [],
        tsgf=_tsgf_line(base, task, job, home,
                        attempt if launch else None, attempts_in(home),
                        going=state in ("queued", "running")),
    )


def _tsgf_line(base, task, job, stage_dir, run_dir, attempts, *,
               going: bool) -> Optional[str]:
    """The line about the leads' files of a rung that keeps them in a
    scratch folder (`Job.scratch_dir`; `transport.gf_files.status_line`):
    the newest run's -- ``run_dir`` ``None`` for a stage not launched, which
    is answered with the estimate; ``going`` whether that run is queued or
    running -- and, where the stage has older runs, each older run's after
    it, newest first, those having ended; ``None`` for every other rung."""
    if task is None or not job.scratch_dir:
        return None
    from ..paths import attempt_dir
    from ..transport.gf_files import status_line
    first = status_line(base, task, stage_dir, run_dir, job.scratch_dir,
                        going=going)
    if run_dir is None or first is None or len(attempts) < 2:
        return first
    newest = attempt_dir(Path(stage_dir), max(attempts))
    older = [status_line(base, task, stage_dir, attempt_dir(Path(stage_dir), n),
                         job.scratch_dir)
             for n in sorted(attempts, reverse=True) if n != max(attempts)]
    return "; ".join([f"{newest.name}: {first}"]
                     + [f"{attempt_dir(Path(stage_dir), n).name}: {ln}"
                        for n, ln in zip(
                            [n for n in sorted(attempts, reverse=True)
                             if n != max(attempts)], older)])


def _not_prepared(ref: StageRef, r) -> StageStatus:
    """A stage the description names and nothing has prepared -- in the
    ready door's words: ``described``, with what its first launch will take,
    or what it waits for before it can be prepared (`ready.readiness`;
    `job-system.md`, *The task*)."""
    return StageStatus(
        ref=ref, dir=None, state=r.state, detail=r.detail, prepared=False,
        takes=list(r.takes), why=r.why, group=r.group)


def jobset_status(calc, *, sweep=None) -> JobSetStatus:
    """Read the on-disk status of every stage of ``calc`` (read-only) --
    the calculation as the verb opened it (`calculation.open`, A16): its
    description, template and plan, read once.  ``sweep`` lists a
    benchmark's set of the calculation's instead of its ladder -- its
    trials, from its bench container, laid out by the description's shape.

    **The rows are the description's ladder** when a description stands
    beside the set (`job-system.md` § 5.3, 2026-10-01): every stage
    ``task.json`` names, in its order and with its number, the ones nothing
    has prepared yet as the ready door answers them -- so a calculation lists its
    stages before its first prep (``jobset`` is then ``None``), and a ladder
    prepared one stage at a time lists them all.  The Results tab's ladder is
    this answer (`web/results.md` § 2.4).  A benchmark's sweep, with no
    description beside it, lists its own jobs.

    ``first_incomplete`` is the first stage that is not ``finished`` -- the
    stage to resume from; ``None`` (and ``complete=True``) when every stage
    finished."""
    base = Path(calc.base)
    jobset = sweep if sweep is not None else calc.jobset
    task = calc.task if sweep is None else None
    kw = {}
    if jobset is not None:
        sh = calc.shape
        kw = {"dirs": job_dir_names(jobset, sh),
              "refs": stage_refs(jobset), "shape": sh}
    stages: List[StageStatus] = []
    if task is not None:
        # ONE KEY for a stage's name, in any case (`JobSet.job`): a stage
        # renamed in case only keeps its prepared job.
        from .naming import ladder_homes
        from .ready import task_state
        # THE READY DOOR'S ANSWERS, read once -- NO VERDICT: status names the
        # run, not its relaxation, and the table is read far more often than
        # a stage is prepared or launched.
        state = task_state(calc)
        # THE NUMBERS ARE THE DOOR'S -- what the folders carry (W38 F4), so
        # a row's number is its folder's, not its place in the description.
        for st, ref, r in zip(task.stages,
                              [StageRef(h.seq, h.name)
                               for h in ladder_homes(base, task)],
                              state.stages):
            job = jobset.job(st.name) if jobset is not None else None
            if job is None:
                stages.append(_not_prepared(ref, r))
                continue
            # THE DESCRIPTION'S NAME AND THE JOB REQUEST'S NUMBER on its row:
            # the ladder is the description's, and `status <stage>` finds the
            # row by it -- and
            # a stage never launched, the ready door's words for its launch.
            row = dataclasses.replace(
                _job_status(base, jobset, job, task, **kw), ref=ref,
                group=calc.request.group_of(st.name))
            if not r.launched and row.state == "pending":
                row = dataclasses.replace(row, detail=r.detail,
                                          takes=list(r.takes), why=r.why)
            stages.append(row)
        offer = state.prep_offer
        steps = _steps(calc, state.groups, stages)
    else:
        stages = [_job_status(base, jobset, job, None, **kw)
                  for job in jobset.jobs]
    first = next((s for s in stages if s.state != _DONE), None)
    # A BENCHMARK'S SWEEP is read against the calculation it measures, as
    # its job-set names its trials (`materialize.bench_owner` re-bases a
    # reader standing in the container); its stage is read off where its
    # trials live, so its next step is its own verbs.
    bench_of = None
    if sweep is not None and jobset.jobs:
        from .materialize import bench_stage_of
        bench_of = bench_stage_of(base / kw["dirs"][jobset.jobs[0].name])
    return JobSetStatus(
        name=(jobset.name if jobset is not None else task.label),
        engine=(jobset.engine if jobset is not None else str(task.engine)),
        stages=stages,
        first_incomplete=(first.name if first is not None else None),
        complete=(first is None),
        base=str(base),
        bench_of=bench_of,
        offer=(offer if task is not None else []),
        steps=(steps if task is not None else []),
        calc=calc,
    )


#: A RUN'S STATES that leave its group needing a launch again
#: (`running-a-job.md` § 4.2) -- and the ones still on their way.
_ENDED_BADLY = ("failed", "stopped", "unreadable")
_ON_ITS_WAY = ("queued", "running")


def _said_lines(text: str) -> List[str]:
    """A composed way on (`commands`), as a step's lines: each of its
    lines, the commands among them, in order."""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _steps(calc, groups, rows: List[StageStatus]) -> List[GroupStep]:
    """THE NEXT STEP (`job-system.md`, *The task*, *The next step*): every
    group of ``task``, in its order, its state read off its members' rows
    (``rows``, the run doors' answers) and the ready door's group
    (``groups``, `ready.GroupAnswer`), the first row of the contract's table
    that applies; every line composed by `commands`.  The first group not
    finished is the next step; with every group finished, the last names
    `summarize task` where the task's result gathers its stages
    (`summarize.task_summary`)."""
    from .commands import reason_of, way_back
    from .summarize import task_summary
    from ..joblist import Asked
    from .ready import held_by_sequence
    base, task = Path(calc.base), calc.task
    by = {s.name: s for s in rows}
    out: List[GroupStep] = []
    asked = Asked()
    for g in groups:
        members = [by[m] for m in g.members]
        n = g.number
        # A GROUP A SEQUENCE STILL GOING HOLDS, not launched by it yet --
        # launched before or not (`job-system.md` § 6.0, *A sequence of
        # groups*): that job's to launch.
        held = (held_by_sequence(calc, n, asked=asked)
                if all(r.prepared for r in members) else None)
        if g.members_changed:
            # A LAUNCHED GROUP WHOSE MEMBERS CHANGED -- a member taken out, or
            # a stage put in, prepared or not: offered by no verb, its way on
            # the rule's own, whole (`job-system.md`, *The task*).
            step = ("changed", "launched, its members changed",
                    _said_lines(g.members_changed))
        elif not all(r.prepared for r in members):
            step = (("not prepared", "not prepared",
                     prep_group_lines([n], base=base)) if g.preparable else
                    ("waits", "waits before it can be prepared: "
                     + reason_of(g.why or ""), []))
        elif g.changed:
            # ITS PREP FORMS IT AGAIN with every group it needs, the groups
            # prep names (`group.formed_again`, as the ready door says it).
            from .group import formed_again
            step = ("changed", "prepared, its members changed",
                    prep_group_lines(formed_again(calc.request,
                                                  calc.jobset.jobs, [n]),
                                     base=base))
        elif held is not None:
            job, after, said = held
            step = ("in a job",
                    f"goes in job {job.id} -- "
                    + (f"after group {after}" if after is not None
                       else "its first group")
                    + f" ({said})", ["molbuilder jobs status"])
        elif not g.launched:
            # PREPARED BY ANOTHER MOLBUILDER: its launch compares it with
            # what prep writes now, and a difference is refused (R-L15) --
            # the step says so, and the prep that answers it.
            step = (("ready",
                     "ready to launch" + (f" (prepared by {g.prepared_by}: "
                                          f"its launch refuses it if prep "
                                          f"would write it differently now)"
                                          if g.prepared_by else ""),
                     launch_lines("task", base=base, groups=[n])
                     + ([f"{ln}   # if the launch refuses it, first"
                         for ln in prep_group_lines([n], base=base)]
                        if g.prepared_by else []))
                    if g.launchable else
                    ("waiting", "waits for: " + reason_of(g.why or ""), []))
        else:
            live = next((r for r in members if r.state in _ON_ITS_WAY), None)
            bad = next((r for r in members if r.state in _ENDED_BADLY), None)
            opened = next((r for r in members if r.state == "pending"), None)
            if live is not None:
                step = (live.state, f"{live.state}, let it finish",
                        [command("status", base=base)])
            elif bad is not None and bad.state == "unreadable":
                step = (bad.state, f"{bad.name}: " + reason_of(bad.detail),
                        [])
            elif bad is not None:
                # LAUNCHED AGAIN, it continues from its own latest run or
                # starts over -- its job's one fact (`Job.relaunch_continues`).
                again = ("it continues from its own latest run, or `--cold` "
                         "starts it over" if bad.relaunch_continues else
                         "it starts over, nothing handed on from a run of "
                         "its own")
                # ...OR CHANGED FIRST, its way on the one composer's, whole
                # (`commands.way_back`, layout-aware).
                step = (bad.state,
                        f"{bad.state}: {bad.name} -- launch it again -- "
                        f"{again}; or, to change it first, as below",
                        launch_lines("task", bad.name, base=base)
                        + _said_lines(way_back(bad.name, calc=calc,
                                               launched=True)))
            elif opened is not None:
                step = ("pending", f"{opened.name}: {opened.detail}",
                        launch_pending(opened.name, n, opened.attempt,
                                       base=base))
            else:
                step = (_DONE, _DONE, [])
        out.append(GroupStep(group=n, stages=list(g.members), state=step[0],
                             words=step[1], lines=list(step[2])))
    first = next((i for i, s in enumerate(out) if s.state != _DONE), None)
    if first is not None:
        out[first] = dataclasses.replace(out[first], next=True)
    elif out and task_summary(task) is not None:
        out[-1] = dataclasses.replace(
            out[-1], next=True,
            words="finished -- every group has; read what they gathered",
            lines=[command("summarize", "task", base=base)])
    return out


def render_steps(status: JobSetStatus) -> str:
    """The next step as the terminal prints it -- after `status`'s table and
    at the end of `launch task` (`job-system.md`, *The task*, *The next
    step*): each group, its stages, its words, and under the next one its
    lines."""
    if not status.steps:
        return ""
    width = max(len(", ".join(g.stages)) for g in status.steps)
    out = ["each group, and what is next:"]
    for g in status.steps:
        out.append(f"  group {g.group}  {', '.join(g.stages):<{width}}  "
                   f"{g.words}" + (" -- next:" if g.next and g.lines else
                                   " -- next" if g.next else ""))
        if g.next:
            out += [f"           {ln}" for ln in g.lines]
    return "\n".join(out)


def render_status(status: JobSetStatus) -> str:
    """Human-readable status table + each group's next step, the line that
    moves it (`job-system.md` § 5.3, *The next step*)."""
    lines: List[str] = [
        f"JOB-SET STATUS -- {status.name} ({status.engine})",
        "",
    ]
    # The stage's SEQ, never its row: a row's position is the number
    # `engines/stages.md` R5 forbids as an identifier.  A sweep point
    # has no order, so it prints `-` from the one rule the plan table uses.
    # `attempt` is which run-<n> the row was READ FROM.  Without it the table
    # says "finished" without saying finished *when* -- and after a re-run the
    # difference between run-0 and run-2 is the whole question.  `-` for a flat
    # run, which happens in the container itself (project-layout.md § 1.5).
    hdr = ("seq", "stage", "group", "attempt", "state", "converged",
           "warm files", "detail")
    rows = [(s.ref.seq_text, s.ref.name,
             str(s.group) if s.group is not None else "-",
             s.attempt or "-", s.state,
             s.converged or "-", ", ".join(s.warm_files) or "-",
             s.detail + (f" \u00b7 {s.tsgf}" if s.tsgf else ""))
            for s in status.stages]
    # Widths and the rule are both driven off `hdr`, never off a literal count.
    w = [max(len(r[k]) for r in rows + [hdr]) for k in range(len(hdr))]
    def fmt(r):
        return "  ".join(s.ljust(w[k]) for k, s in enumerate(r))
    lines.append("  " + fmt(hdr))
    lines.append("  " + "  ".join("-" * n for n in w))
    lines += ["  " + fmt(r) for r in rows]
    lines.append("")
    if status.bench_of is not None:
        lines.append(_sweep_next(status))
        return "\n".join(lines)
    # THE NEXT STEP, every group's (`_steps`): a described calculation has
    # one for each group, a finished one the summary that gathers it.
    lines.append(render_steps(status))
    return "\n".join(lines)


def _sweep_took_over(s: StageStatus) -> str:
    """What a sweep's run continued from, read at the point: the points
    taken over from an earlier run, named by that run; a run that took none
    over walked every point (the first launch, or ``--cold``)."""
    from ..runrecord import RunRef
    here = RunRef.read(f"{s.dir}/{s.attempt}", shared=False)
    taken = [(p["point"], p["started_from"]) for p in s.points
             if taken_over(p, here)]
    if not taken:
        return "nothing taken over -- every point walked"
    # THE RUN a point was taken over from, the run its point lies in.
    runs = sorted({str(RunRef.read(src, shared=False).run_folder())
                   for _w, src in taken})
    return (", ".join(w for w, _s in taken) + " taken over from "
            + ", ".join(runs) + "; the rest walked")


def _sweep_next(status: JobSetStatus) -> str:
    """A benchmark's next step: its own verbs, for the stage it measures, on
    the calculation (`job-system.md` § 7) -- the trials launch together, the
    ones already launched passed over, and what they measured is read back
    once they have run."""
    stage, base = status.bench_of, status.base
    read = block([command("summarize", "bench", stage, base=base)])
    states = {s.state for s in status.stages}
    if "pending" in states:
        return ("Trials not launched yet -- launch the sweep (the ones "
                "launched are passed over):\n"
                + block(launch_lines("bench", stage, base=base))
                + "\nthen, once they have run, read what they measured:\n"
                + read)
    if states & {"queued", "running"}:
        return ("Trials queued or running -- let them finish, then read "
                "what they measured:\n" + read)
    return "Every trial has ended -- read what they measured:\n" + read


def render_stage_status(status: JobSetStatus, stage_name: str) -> str:
    """One stage, in full — the per-stage form `job-system.md` § 5.3 reserves.

    The table answers *where is this calculation up to*; this answers *what
    this stage is* -- its deck, what it declares, its resources, the plan's
    columns -- and *what happened
    to it*, which is a different question and the one you ask before
    deciding whether to run it again. It is only answerable at all
    because of the attempt layer: the tries are directories, and the launch is
    a record rather than an inference from an empty folder (§ 1.5, § 1.6).

    Everything printed comes off the :class:`StageStatus` the reader already
    built -- for a stage nothing has prepared, the ready door's answer on
    its row. Nothing here opens a file — a second reader of the launch
    record would be a second answer to *was this launched?*
    """
    s = next(x for x in status.stages if x.name == stage_name)
    if not s.prepared:
        # WHAT YOU CAN TYPE (`job-system.md` § 5.3): one waiting is told
        # what for, with the commands; one preparable, its group's prep.
        how = ("It waits:\n" + textwrap.indent(s.why, "    ") if s.why else
               "Prep its group:\n"
               + block(prep_group_lines([s.group], base=status.base)
                       if s.group is not None else
                       [command("prep", "task", base=status.base)])
               + "".join(f"\n      # {t}" for t in s.takes))
        return "\n".join([f"STAGE {s.ref.label} -- {s.state}", "",
                          f"  {s.detail}", "", how])
    rows: List[tuple] = [
        # WHAT THE STAGE IS, before what happened to it -- the plan's columns
        # (`job-system.md` § 5.3).
        ("deck", s.script or "-"),
        # WHAT IT DECLARES, never what was copied -- the plan's column
        # (`STAGE-PLAN.md`, D28): each run's `.continued-from` and launch
        # record say what came across.
        ("declares", ", ".join(s.carries) or "-"),
        ("resources", s.resources or "-"),
    ]
    if s.gathered:
        # WHAT THE RUN TOOK from the rungs upstream -- its `.gathered-from`
        # (`engines/transport.md` § 2a.11): the inputs it started from.
        rows.append(("gathered", "; ".join(s.gathered)))

    if s.tsgf:
        # THE LEADS' FILES (`engines/transport.md` § 2a.11): the newest
        # run's scratch folder, or the size to come.
        rows.append(("tsgf", s.tsgf))
    tries = ", ".join(attempt_name(n) for n in s.attempts) or "-"
    rows.append(("attempt", f"{s.attempt or '-'}"
                            + (f"   (of {tries})" if len(s.attempts) > 1
                               else "")))
    if s.launch:
        jid = s.launch.get("job_id")
        rows.append(("launched", f"{s.launch.get('mode', '?')}"
                                 + (f" as job {jid}" if jid else "")
                                 + f" at {s.launch.get('launched_at', '?')}"))
        cmd = s.launch.get("command")
        if cmd:
            rows.append(("command", " ".join(cmd)))
        # SAID EVERY TIME (checkpointing.md S3): the run it continued from,
        # or null -- it started from the structure, or from what it gathered.
        # A sweep's run continues at the point (rule 3, `engines/transport.md`
        # § 2a.11): its row says which points were taken over from which
        # runs, never the run-level marker alone.
        src = s.launch.get("continued_from")
        rows.append(("continued from",
                     _sweep_took_over(s) if s.points
                     else src if src
                     else "nothing -- it started from what it gathered"
                     if s.gathered else
                     "nothing -- it started from the structure"))
    else:
        rows.append(("launched", "no  (no launch record -- prepared, not started)"))
    rows.append(("converged", s.converged or "-"))
    rows.append(("warm files", ", ".join(s.warm_files) or "-"))
    rows.append(("detail", s.detail or "-"))
    # A SWEEP'S POINTS, one row each (`engines/transport.md` § 2a.11): done
    # or not done and why, what it started from, what it alone took.
    for p in s.points:
        rows.append((f"  {p['point']}",
                     ("done" if p["done"] else f"not done -- {p['why']}")
                     + (f"; from {p['started_from']}" if p.get("started_from")
                        else f"; {p['start_unread']}" if p.get("start_unread")
                        else "; from what it gathered"
                        if p["opened"] else "")
                     + (f"; took {', '.join(p['took'])}" if p.get("took")
                        else "")))

    # The pad comes off the longest label, never a literal.
    w = max(len(k) for k, _ in rows) + 2
    lines = [f"STAGE {s.ref.label} -- {s.state}", ""]
    lines += [f"  {k.ljust(w)}{v}" for k, v in rows]
    lines.append("")
    lines.append(
        f"Directory: {s.dir}" + (f"/{s.attempt}" if s.attempt else ""))
    return "\n".join(lines)


def _said(state: str, detail: Optional[str]) -> str:
    """A state and its detail, as the status door says them -- the detail
    only where it adds to the state."""
    return state if not detail or detail == state else f"{state} -- {detail}"


def job_members(calculation, members: Dict[str, Dict[str, str]]
                ) -> List[tuple]:
    """Each member of a job launched from this machine (`job-system.md`
    § 6.0a): ``(stage, run, said)`` -- the stage, its run as a record names
    it, and its state in the status door's words.  The run answers through
    its own launch record, by the run door (`runs.run_of`,
    `parse.dirs.run_state_of`): that run, never its stage's newest -- where
    a stage's runs share a folder, the run its warm retries reached
    (`runrecord.retried_to`).  A run of points, a container no one run of
    ours speaks for, is answered by its stage's row of `jobset status`,
    naming the newest run it reads; a run no longer on disk, and a
    calculation that does not read, are said."""
    from ..calcdirs import CONTAINER, where
    from ..runrecord import retried_to
    from ..runs import concluded, run_of
    base = Path(calculation)
    rows: Optional[Dict[str, StageStatus]] = None
    unread: Optional[str] = None
    out = []
    for stage, member in members.items():
        run = member["run"]
        record = base / member["launch_record"]
        ran = run_of(record) if record.exists() else None
        if ran is not None and ran.stage is not None:
            reached = (retried_to(ran.folder, ran.names, ran.run)
                       if ran.run is not None else None)
            st = concluded(ran.folder, ran.names, base=base, run=reached,
                           launched_in=ran.launched_in)
            out.append((stage, run if reached == ran.run else
                        f"{run}, retried to {ran.names.run_name(reached)}",
                        _said(st.state, st.detail)))
            continue
        # WHERE ITS LAUNCH RECORD'S FOLDER IS (`calcdirs.where`): a run of
        # points' run, a container.
        if (not record.parent.is_dir()
                or where(record.parent).role != CONTAINER):
            out.append((stage, run, "its run is not on disk"
                        if not record.exists()
                        else "its launch record names no run of ours"))
            continue
        if rows is None:
            rows = {}
            try:
                from ..calculation import open as open_calculation
                rows = {stage_key(s.name): s for s in jobset_status(
                    open_calculation(base)).stages}
            except (OSError, ValueError, KeyError) as exc:
                unread = f"its calculation does not read: {exc}"
        row = rows.get(stage_key(stage))
        if row is None:
            out.append((stage, run,
                        unread or "its description no longer holds it"))
            continue
        newest = (f"{row.dir}/{row.attempt}" if row.dir and row.attempt
                  else None)
        out.append((stage, run, _said(row.state, row.detail)
                    + (f" (its newest run, {newest})" if newest else "")))
    return out


@dataclass(frozen=True)
class ScratchAnswer:
    """`jobset scratch`'s answer (`job-system.md` § 5.3): every file of an
    engine's scratch the calculation's runs hold (``found``), those a run
    still going keeps, each with why (``kept``), and those free to remove
    (``free``)."""
    found: list
    kept: list
    free: list


def scratch_answer(calc) -> ScratchAnswer:
    """WHICH OF A CALCULATION'S SCRATCH A RUN STILL GOING KEEPS -- asked of
    each RUN, never of its stage's newest: a run is going when a job
    launched from this machine that holds it is going, or this machine
    cannot tell (the jobs door, `joblist.is_going`), or when the run door
    reads it queued or running (`continuation.read_run`, the door `status`
    reads); a run whose state does not read keeps its scratch too, said.
    On the flat layout the runs share one folder: while any stage is going,
    every file is kept.  Reads names, states and the jobs' files, and
    removes nothing."""
    from ..joblist import Asked, Job, every, is_going
    from ..parse.dirs.job import LIVE_STATES
    from ..paths import run_folder_of
    from .continuation import stage_run_state
    from .materialize import scratch_in
    base = Path(calc.base)
    found = scratch_in(calc)
    if not found:
        return ScratchAnswer([], [], [])
    # THE RUNS A JOB STILL GOING HOLDS -- each by its run folder.
    held: Dict[str, str] = {}
    asked = Asked()
    for job in every():
        if (not isinstance(job, Job)
                or Path(job.calculation) != base.resolve()):
            continue
        going, said = is_going(job, asked)
        if going is False:
            continue
        for member in job.members.values():
            run = run_folder_of(member["run"])
            held[str(run if run is not None else member["run"])] = (
                f"job {job.id} -- {said}")
    kept, free = [], []
    if not calc.shape.keeps_attempts_as_directories:
        going = held and next(iter(held.values()))
        if not going:
            try:
                going = next((f"{s.ref.name} is {s.state}"
                              for s in jobset_status(calc).stages
                              if s.state in LIVE_STATES), None)
            except (OSError, ValueError, KeyError) as exc:
                going = f"which stage is going does not read: {exc}"
        return ScratchAnswer(found, [(f, going) for f in found] if going
                             else [], [] if going else list(found))
    for f in found:
        why = None
        # A FILE IN NO RUN is no run's to keep.
        if f.run is not None:
            ref = f.run.relative_to(base).as_posix()
            why = held.get(ref)
            if why is None:
                try:
                    # THE RUN'S STATE BY ITS KIND'S RULE -- a run of points
                    # by its points (`continuation.stage_run_state`).
                    state, detail = stage_run_state(base, calc.task,
                                                    f.stage, f.run)
                except (OSError, ValueError, KeyError) as exc:
                    why = f"its state does not read: {exc}"
                else:
                    if state in LIVE_STATES:
                        why = state
                    elif state == "unreadable":
                        why = f"its state does not read: {detail}"
        (kept if why else free).append((f, why) if why else f)
    return ScratchAnswer(found, kept, free)


__all__ = ["StageStatus", "JobSetStatus", "jobset_status", "job_members",
           "render_status", "render_stage_status", "ScratchAnswer",
           "scratch_answer"]
