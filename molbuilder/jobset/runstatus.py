"""Run-status reader — the "inform" layer
(`execution/job-system.md` § 2, rule 5: molbuilder informs; the user decides).

The resume contract is: **the modeling software resumes; molbuilder informs
and the user decides** (never auto-recovers — redoing a long run unknowingly
is a heavy penalty).  This module is the *inform* half: for a calculation --
its description's every stage, prepared or not -- it answers, per stage, *did
it finish? is it running? did it fail? are the warm-restart files there?* and
*which is the first incomplete stage* (the one to resume from), with the
command each state calls for -- so the manual continue is a one-glance
decision.

REUSE, not reinvention: per-stage run state comes from
``parse.dirs.job.run_status`` (the directory-level status verb behind the Results
tab + JobMonitor; its schema is pinned by ``execution/running-a-job.md``
§ 4, the composer pattern by ``model/parse.md`` § 5); this module only adds the cross-stage
view (warm-file inventory + first-incomplete pointer).  It is **read-only**:
it inspects the tree, changes nothing.
"""

from __future__ import annotations

import dataclasses
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..identity import StageRef
from ..paths import attempts_in
from .materialize import (job_dir_names, latest_attempt, run_dir, run_names, shape_of, stage_refs)
from ..runrecord import LaunchRecordError, launch_record
from .commands import (block, command, launch_lines, rollback,
                       words_for)
from .model import JobSet
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
    traceback.  An engine without a rules file simply has no carry rows to
    report."""
    try:
        return _warm_list(engine, None, base).carry
    except Exception:
        return ()

# run_status states we treat as "this stage is done".
_DONE = "finished"


@dataclass(frozen=True)
class StageStatus:
    """One stage's status (read-only snapshot).

    **WHICH stage this is arrives as a whole :class:`~molbuilder.identity.
    StageRef`, not as a loose name and a loose number.** Carrying the ref means the resolver in ``materialize`` is the only
    place one is ever made.
    """
    ref:        StageRef
    #: The stage's directory, or ``None`` before anything prepared it.
    dir:        Optional[str]
    #: ready/waiting (not prepared: the ready door's word) or the run's:
    #: pending/queued/running/failed/finished/unknown.  ``queued`` is
    #: the contract's own word (project-layout.md § 1.6, *"queued as job
    #: 481923"*) for launched-but-no-output-yet, which is exactly the state an
    #: empty directory cannot be read for.
    state:      str
    detail:     str
    warm_files: List[str] = field(default_factory=list)  # restart files present
    #: Which attempt this status was read from (``run-0``; a bias scan's
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
    #: Whether this stage, launched again, continues from its own latest run
    #: -- the job's one fact (`Job.relaunch_continues`), which `launch`
    #: asks too, so the status says what launching it again will do
    #: (`job-system.md` § 5.4, *A stage launched again*).
    relaunch_continues: bool = True
    #: The group it was prepared in, its members in the ladder's order --
    #: one job when launched together (`project-layout.md` § 1.6.6) -- or
    #: ``None``.
    group: Optional[List[str]] = None
    #: Whether anything has prepared it -- a description's stage before its
    #: first prep is listed all the same (`job-system.md` § 5.3).
    prepared: bool = True
    #: Whether the run converged, as its output ended -- ``SCF yes`` /
    #: ``SCF NO``, ``geometry yes`` / ``geometry NO`` for a relaxation --
    #: or ``None`` before an output says (`job-system.md`, *What molbuilder
    #: does for you*, 4).  Beside the state, never folded into it: a run can
    #: finish and not converge (`parse.dirs.run_status`).
    converged: Optional[str] = None
    #: A swept rung's points, in bias order -- ``{bias_v, folder, done, why,
    #: started_from}`` each, read from the point's own files
    #: (`continuation.done`; `engines/transport.md` § 2a.11) -- or ``[]``.
    points: List[Dict[str, Any]] = field(default_factory=list)
    #: What the run GATHERED from the rungs upstream -- ``<file> <- <run>``
    #: each, the run's own ``.gathered-from`` (`runrecord.read_gathered_from`;
    #: `engines/transport.md` § 2a.11) -- or ``[]`` for a stage that gathers
    #: nothing.  A sweep's point adds what it alone took in its own row.
    gathered: List[str] = field(default_factory=list)
    #: For one not prepared, the ready door's answer whole
    #: (`ready.readiness`): what its prep would take, one line each, or what
    #: it waits for -- the refusal prep would print, with its commands.
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
class JobSetStatus:
    """The whole job-set's status + the resume pointer."""
    name:            str
    engine:          str
    stages:          List[StageStatus]
    first_incomplete: Optional[str]   # name of the first non-finished stage (resume here)
    complete:        bool             # every described stage finished
    #: The ready stages `prep task` offers pre-selected (`ready.preselected`,
    #: D2) -- the next step when the first incomplete stage is not prepared.
    offer: List[str] = field(default_factory=list)
    #: The calculation's folder, so the commands the renderers print name it
    #: (`commands.command`).  Not part of the wire form.
    base: Optional[str] = None
    #: The stage a benchmark's sweep measures (`materialize.bench_stage_of`):
    #: its rows are trials, and its next step is its own verbs.  ``None``
    #: for a ladder.  Not part of the wire form.
    bench_of: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """THE WIRE FORM -- the Results tab's ladder is this (`web/results.md`
        § 2.4), the next prep's answer included."""
        return {
            "name": self.name, "engine": self.engine,
            "first_incomplete": self.first_incomplete,
            "complete": self.complete,
            "stages": [s.to_dict() for s in self.stages],
            "offer": self.offer,
        }


def _warm_present(stage_dir: Path, label: str, engine: str,
                  base) -> List[str]:
    """Which engine warm-restart files actually exist (real files, not
    dangling carry symlinks) in this stage's dir -- by the list in effect
    for the calculation in ``base``."""
    out: List[str] = []
    for ext in _warm_files(engine, base):
        f = stage_dir / f"{label}{ext}"
        if f.is_file():                      # follows symlinks; dangling -> False
            out.append(f.name)
    return out


#: A PREPARED stage whose folder is not on disk -- against the design, said
#: as such (a stage not prepared is the ready door's `ready` / `waiting`).
MISSING = ("missing", "the job set names this run's folder; it is not on "
                      "disk", None)


def every_run(base, task, st: StageStatus) -> List[Dict[str, Any]]:
    """EVERY RUN OF A STAGE, each with its own state -- the root's bird's-eye
    (`web/results.md` § 2.4): each attempt in order and, for a swept rung,
    each attempt's points (`transport.stages.points_in`), read as `status`
    reads a run (`run_state_of`), with the file the run door opens for it
    (`runs.openable`).  ``[]`` for a stage nothing has prepared."""
    from ..runs import openable, run_of
    from ..parse.dirs.rundir import run_state_of
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
            folders = [(p, v) for p, v in points_in(run_dir, task, st.ref.name)
                       if p.is_dir()]
        if not folders:
            folders = [(run_dir, None)]
        for folder, volts in folders:
            run = run_of(folder)
            if run is None or run.stage is None:
                continue
            got = run_state_of(run.folder, run.names, run.run)
            opens, _trail = openable(folder)
            rows.append({
                "run": attempt_name(n),
                "point": volts,
                "dir": str(folder.relative_to(root)),
                "state": got.state,
                "detail": got.detail,
                "converged": converged_of(got),
                "opens": Path(opens).name if opens else None,
            })
    return rows


def converged_of(st) -> Optional[str]:
    """What the run's active output says it converged -- the run door's
    one scan (`RunStatus.endings`): a relaxation's geometry, else its
    SCF -- or ``None`` when it says neither yet."""
    end = (st.endings or {}).get(st.active_source) if st.active_source \
        else None
    if end is None:
        return None
    if end.relaxed is not None:
        return "geometry " + ("yes" if end.relaxed else "NO")
    if end.scf_converged is not None:
        return "SCF " + ("yes" if end.scf_converged else "NO")
    return None


def _stage_state(observed: Path, launch: Optional[Dict[str, Any]],
                 basename: Optional[str]) -> tuple:
    """(state, detail) for the directory a stage's run actually happened in.

    ``observed`` is the latest attempt where there is one, and the stage
    container for a flat run (`project-layout.md` § 1.5) — the caller resolves
    that, because *where a run happens* is a layout question and this layer
    only reads.  ``basename`` is THIS rung's run's stem -- the flat shape
    keeps every stage in one folder, and an attempt may hold several run
    indexes.

    ``launch`` is the run's launch record. It is what separates *queued* from
    *never started*, which no amount of looking at an empty directory can do:
    § 1.6's *"a queued cluster job has produced nothing yet, so 'no output' and
    'not started' look identical"*, and its promise that status can then say
    *"queued as job 481923"* rather than guessing from an absence.  The
    directory door answers both from it (`parse.dirs.job.run_status`).
    """
    if not observed.is_dir():
        return MISSING
    try:
        # THROUGH THE PACKAGE THAT OWNS THE QUESTION, never the module
        # inside it (`model/parse.md` § 5.5, R-RO2: one import surface per
        # question).
        from ..parse.dirs import run_status
        # THE RUNG'S OWN NAME: in the flat shape every stage shares the
        # directory, and without it every row showed the newest stage's
        # state.
        st = run_status(observed, basename, launch=launch)
    except Exception as e:                    # fail-soft; stay informative
        return ("unknown", f"could not decode: {e}", None)
    # The "unknown" fallback lives in the except clause above, which is the
    # only way this can fail to have an answer.
    return (st.state, st.detail, converged_of(st))


def _sweep_status(base: Path, jobset: JobSet, job, task, *, d: Path,
                  names, ref) -> Optional["StageStatus"]:
    """A SWEPT RUNG's row (`engines/transport.md` § 2a.11): its latest run's
    state -- ``pending`` before it is launched; while launched, ``running``
    while a point runs, ``finished`` once every point is done, ``queued``
    when nothing has run yet, else ``failed`` -- and its points, each done
    or not done and what it started from.  ``None`` for a rung that does
    not sweep."""
    from ..transport.stages import points_in, products_of, sweep_points
    from ..warmfiles import warm_list
    from ..parse.dirs import run_status
    from .continuation import done
    if task is None or not sweep_points(task, job.name):
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
    own = {w.name for w in job.warm}
    hand = [f"{task.label}{suf}"
            for suf in warm_list(jobset.engine, "transport", base).along
            if f"{task.label}{suf}" in own]
    products = products_of(job.name, task.label) + hand
    pts, running = [], False
    from ..runrecord import read_continued_from, read_gathered_from
    # THE RUN'S GATHER, once; a point's own `.gathered-from` adds what it
    # alone took (the transmission's device point).
    gathered = _gathered_lines(run)
    for pdir, v in points_in(run, task, job.name):
        ok, why = done(pdir, names, launch=launch, products=products)
        if not ok and pdir.is_dir():
            running = running or run_status(pdir, names.stem,
                                             launch=launch).state == "running"
        src = (read_continued_from(pdir, names, 0) if pdir.is_dir()
               else None)
        took = [g for g in _gathered_lines(pdir) if g not in gathered]
        pts.append({"bias_v": v, "folder": f"{run.name}/{pdir.name}",
                    "done": ok, "why": why, "started_from": src,
                    "took": took})
    n_done = sum(p["done"] for p in pts)
    summary = f"{n_done} of {len(pts)} points done"
    # CONVERGED, in its own column (`job-system.md`, rule 4): every point
    # done says yes for a rung with an SCF; a point that finished without
    # converging says NO; otherwise nothing yet.
    converged = ("SCF NO" if any(p["why"].startswith("finished, not converged")
                                 for p in pts)
                 else "SCF yes" if hand and pts and n_done == len(pts)
                 else None)
    if launch is None:
        state, detail = "pending", "prepared, not launched"
    elif running:
        state, detail = "running", summary
    elif n_done == len(pts):
        state, detail = "finished", summary
    elif all(p["why"] == "not run" for p in pts):
        jid = launch.get("job_id")
        state, detail = "queued", (f"queued as job {jid}" if jid
                                   else "launched, nothing run yet")
    else:
        first = next(p for p in pts if not p["done"])
        state = "failed"
        detail = f"{summary}; {first['bias_v']:g} V: {first['why']}"
    return StageStatus(
        ref=ref, dir=d.name, state=state, detail=detail,
        converged=converged,
        attempt=run.name, attempts=attempts_in(d), launch=launch,
        relaunch_continues=job.relaunch_continues,
        group=(list(job.group) if job.group else None),
        script=str(job.script), carries=[w.name for w in job.warm],
        resources=resources_text(job.resources), points=pts,
        gathered=gathered)


def _gathered_lines(folder) -> List[str]:
    """``<file> <- <run>`` for each entry of ``folder``'s ``.gathered-from``
    (`runrecord.read_gathered_from`), in the order taken; ``[]`` when there
    is none."""
    from ..runrecord import read_gathered_from
    return [f"{g['file']} <- {g['from']}" for g in read_gathered_from(folder)]


def _job_status(base: Path, jobset: JobSet, job, task, *, dirs,
                refs, shape) -> StageStatus:
    """One prepared job's status, read from where its attempts are."""
    d = base / dirs[job.name]
    # WHICH FILES are this stage's: its stage's names (`materialize.
    # run_names`) -- in either shape a flat folder holds every stage, an
    # attempt may hold several run indexes.  THE LABEL IS THIS JOB'S, NOT
    # THE JOBSET'S: a trial is relabelled with its point.
    names = run_names(jobset, job, shape)
    basename = names.stem
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
    try:
        launch = launch_record(observed, names)
        state, detail, converged = _stage_state(observed, launch, basename)
    except LaunchRecordError as e:
        launch, state, detail, converged = None, "unreadable", str(e), None
    where = attempt.name if attempt else None
    return StageStatus(
        ref=refs[job.name], dir=d.name, state=state, detail=detail,
        converged=converged, attempt=where,
        attempts=attempts_in(home),
        launch=launch,
        relaunch_continues=job.relaunch_continues,
        group=(list(job.group) if job.group else None),
        # THE SAME LABEL THE STATE WAS READ WITH -- the stage's names'.
        warm_files=_warm_present(observed, names.label, jobset.engine,
                                 base),
        # WHAT THE STAGE IS -- the plan's own columns, read per stage by
        # `status <stage>`.
        script=str(job.script),
        carries=[w.name for w in job.warm],
        resources=resources_text(job.resources),
        # WHAT THE RUN GATHERED (a transport rung), beside what it declares.
        gathered=_gathered_lines(observed) if attempt is not None else [],
    )


def _not_prepared(ref: StageRef, r) -> StageStatus:
    """A stage the description names and nothing has prepared -- in the
    ready door's words: ``ready`` with what it would take, or ``waiting``
    with what for (`ready.readiness`; `job-system.md`, *The task*)."""
    return StageStatus(
        ref=ref, dir=None, state=r.state, detail=r.detail, prepared=False,
        takes=list(r.takes), why=r.why)


def jobset_status(jobset: Optional[JobSet], base_dir) -> JobSetStatus:
    """Read the on-disk status of every stage under ``base_dir`` (read-only).

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
    from ..task import FILENAME as TASK_FILENAME, read_task
    base = Path(base_dir)
    sweep = jobset is not None and jobset.kind == "sweep"
    task = (read_task(base / TASK_FILENAME)
            if not sweep and (base / TASK_FILENAME).is_file() else None)
    if jobset is None and task is None:
        raise ValueError(f"nothing to report in {base}: no job set and no "
                         f"description")
    kw = {}
    if jobset is not None:
        sh = shape_of(jobset, base_dir)
        kw = {"dirs": job_dir_names(jobset, sh),
              "refs": stage_refs(jobset), "shape": sh}
    stages: List[StageStatus] = []
    if task is not None:
        # ONE KEY for a stage's name, in any case (`identity.stage_key`):
        # a stage renamed in case only keeps its prepared job.
        from ..identity import stage_key
        held = {stage_key(j.name): j for j in (jobset.jobs if jobset is not None
                                               else ())}
        from .materialize import ladder_homes
        from .ready import readiness
        from ..template import find_template
        tpl = find_template(base, task.label)
        text = tpl.read_text(encoding="utf-8") if tpl else None
        # THE NUMBERS ARE THE DOOR'S -- what the folders carry (W38 F4), so
        # a row's number is its folder's, not its place in the description.
        for st, ref in zip(task.stages,
                           [StageRef(h.seq, h.name)
                            for h in ladder_homes(base, task)]):
            job = held.get(stage_key(st.name))
            if job is None:
                # NO VERDICT: status names the run, not its relaxation --
                # and the table is read far more often than prepared.
                stages.append(_not_prepared(ref, readiness(
                    base, task, st.name, template_text=text,
                    verdict=False)))
                continue
            # THE DESCRIPTION'S NAME AND NUMBER on its row: the ladder is the
            # description's, and `status <stage>` finds the row by it.
            stages.append(dataclasses.replace(
                _job_status(base, jobset, job, task, **kw), ref=ref))
    else:
        stages = [_job_status(base, jobset, job, None, **kw)
                  for job in jobset.jobs]
    first = next((s for s in stages if s.state != _DONE), None)
    # A BENCHMARK'S SWEEP is read against the calculation it measures, as
    # its job-set names its trials (`materialize.bench_owner` re-bases a
    # reader standing in the container); its stage is read off where its
    # trials live, so its next step is its own verbs.
    bench_of = None
    if sweep and jobset.jobs:
        from .materialize import bench_stage_of
        bench_of = bench_stage_of(base, base / kw["dirs"][jobset.jobs[0].name])
    return JobSetStatus(
        name=(jobset.name if jobset is not None else task.label),
        engine=(jobset.engine if jobset is not None else str(task.engine)),
        stages=stages,
        first_incomplete=(first.name if first is not None else None),
        complete=(first is None),
        base=str(base),
        bench_of=bench_of,
        offer=_offer(base, task, stages),
    )


def _offer(base: Path, task, stages: List[StageStatus]) -> List[str]:
    """The ready stages `prep task` would offer pre-selected
    (`ready.preselected`, D2), read off the rows already answered."""
    if task is None:
        return []
    from .ready import Readiness, preselected
    return preselected(base, task, [
        Readiness(s.name, prepared=s.prepared, ready=s.state == "ready")
        for s in stages])


def render_status(status: JobSetStatus) -> str:
    """Human-readable status table + the next step, worded by the state of
    the first incomplete stage (`job-system.md` § 5.3)."""
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
    hdr = ("seq", "stage", "attempt", "state", "converged", "warm files",
           "detail")
    rows = [(s.ref.seq_text, s.ref.name, s.attempt or "-", s.state,
             s.converged or "-", ", ".join(s.warm_files) or "-", s.detail)
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
    if status.complete:
        lines.append("All stages finished. Nothing to resume.")
    else:
        first = next((s for s in status.stages
                      if s.name == status.first_incomplete), None)
        if first is not None and not first.prepared:
            # NOTHING TO RE-SUBMIT: the stage has no folder yet, so the next
            # step is to prepare what is ready -- the stages `prep task`
            # offers pre-selected, each line saying what it takes
            # (`job-system.md`, *The task*).
            if status.offer:
                rows = {s.name: s for s in status.stages}
                lines.append(
                    f"First incomplete stage: {first.name}, not prepared "
                    f"yet.  Ready to prepare:\n    "
                    + command("prep", *words_for("task", *status.offer),
                              base=status.base)
                    + "".join(f"\n      # {n}: {rows[n].detail}"
                              for n in status.offer))
                return "\n".join(lines)
            lines.append(
                f"First incomplete stage: {first.name}, waiting:\n"
                + textwrap.indent(first.why or first.detail, "    "))
            return "\n".join(lines)
        lines.append(next_step(first, status.first_incomplete,
                               base=status.base,
                               pending={s.name for s in status.stages
                                        if s.state == "pending"}))
    return "\n".join(lines)


def _sweep_took_over(s: StageStatus) -> str:
    """What a sweep's run continued from, read at the point: the points
    taken over from an earlier run, named by that run; a run that took none
    over walked every point (the first launch, or ``--cold``)."""
    here = f"{s.dir}/{s.attempt}/"
    taken = [(p["bias_v"], p["started_from"]) for p in s.points
             if p.get("started_from") and not p["started_from"].startswith(here)]
    if not taken:
        return "nothing taken over -- every point walked"
    runs = sorted({src.rsplit("/", 1)[0] for _v, src in taken})
    return (", ".join(f"{v:g} V" for v, _s in taken) + " taken over from "
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


def next_step(s: Optional[StageStatus], name: str, *, base,
              pending=frozenset()) -> str:
    """What to do about a PREPARED stage that has not finished, by its state
    -- each a command that works.  molbuilder does NOT auto-resume; the
    person decides (`engines/stages.md`)."""
    state = s.state if s is not None else "stopped"
    # A GROUP PREPARED AND NOT LAUNCHED goes as its one job -- every member
    # still ``pending`` -- as `launch task` takes it (`_cli._launch_unit`).
    # A member launched, or launched again, goes alone (D4).
    words = (s.group if s is not None and s.group and state == "pending"
             and all(g in pending for g in s.group) else [name])
    if state == "pending":
        return (f"First incomplete stage: {name}, prepared and not "
                f"launched:\n" + block(launch_lines("task", *words, base=base)))
    if state in ("queued", "running"):
        return (f"First incomplete stage: {name}, {state} -- let it "
                f"finish; `{command('status', name, base=base)}` shows its "
                f"run.")
    # A PREPARED STAGE IS NOT PREPARED AGAIN (user, 2026-10-02); it is
    # launched again, however it ended, warm or cold -- the person's choice
    # (`job-system.md` § 5.4, *A stage launched again*).
    warm = ("it continues from its own latest run"
            if s is None or s.relaunch_continues
            else "it starts over, nothing handed on from a run of its own")
    how = (f"launch it again -- {warm}:\n"
           + block(launch_lines("task", *words, base=base))
           + "\n  or start it over with `--cold`; or, to change it first, "
           + rollback("its prep", base=base))
    return (f"First incomplete stage: {name}, {state}.  molbuilder does NOT "
            f"auto-resume -- you decide: {how}")


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
        # what for, with the commands.
        how = ("It waits:\n" + textwrap.indent(s.why or s.detail, "    ")
               if s.state == "waiting" else
               "Prep it:\n    "
               + command("prep", *words_for("task", s.name), base=status.base)
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
        # § 2a.11): the points taken over name the run they came from.
        src = s.launch.get("continued_from")
        rows.append(("continued from",
                     src if src else _sweep_took_over(s) if s.points
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
        rows.append((f"  {p['bias_v']:g} V",
                     ("done" if p["done"] else f"not done -- {p['why']}")
                     + (f"; from {p['started_from']}" if p.get("started_from")
                        else "; from what it gathered"
                        if p["why"] != "not opened" else "")
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


__all__ = ["StageStatus", "JobSetStatus", "jobset_status", "render_status",
           "render_stage_status"]
