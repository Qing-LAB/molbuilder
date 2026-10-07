"""Run-status reader — the "inform" layer
(`execution/job-system.md` § 2, rule 5: molbuilder informs; the user decides).

The resume contract is: **the modeling software resumes; molbuilder informs
and the user decides** (never auto-recovers — redoing a long run unknowingly
is a heavy penalty).  This module is the *inform* half: for a calculation --
its description's every stage, prepped or not -- it answers, per stage, *did
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
from .commands import block, command, launch_lines, rollback
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
    #: The stage's directory, or ``None`` before anything prepped it.
    dir:        Optional[str]
    #: not-started/pending/queued/running/failed/finished/unknown.  ``queued`` is
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
    #: Whether anything has prepped it -- a description's stage before its
    #: first prep is listed all the same (`job-system.md` § 5.3).
    prepped: bool = True
    #: Whether the description runs it -- a disabled stage is listed, and is
    #: never the stage to resume from (§ 5.3).
    enabled: bool = True
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
    complete:        bool             # every enabled stage finished
    #: The run the first incomplete stage's prep will continue from, when
    #: nothing has prepped it -- `continuation.continuation_answer`'s, the answer
    #: prep itself asks (`job-system.md` § 5.4) -- or ``None``: a linked
    #: stage, the first, one that starts clean.
    resume_from: Optional[str] = None
    #: Why that prep would refuse instead -- the refusal whole, with the
    #: commands it names (`job-system.md` § 5.3: what molbuilder prints, you
    #: can type).
    resume_refused: Optional[str] = None
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
            "resume_from": self.resume_from,
            "resume_refused": self.resume_refused,
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


#: The state a stage has before `prep` has made it a directory -- the ONE
#: spelling, read by `_stage_state` for a planned stage whose directory is
#: missing and by :func:`jobset_status` for a described stage nothing has
#: prepped yet (`job-system.md` § 5.3, `web/results.md` § 2.4).
NOT_PREPPED = ("not-started", "no directory yet (not prepped)")


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
        return NOT_PREPPED
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
        return ("unknown", f"could not decode: {e}")
    # The "unknown" fallback lives in the except clause above, which is the
    # only way this can fail to have an answer.
    return (st.state, st.detail)


def _rung_homes(base: Path, task, job_name: str, d: Path) -> list:
    """WHERE THIS JOB'S ATTEMPTS ARE -- ``[(folder, volts)]``.  For a stage of
    the description, the one door's answer (`transport.stages.rung_containers`,
    plan § 5w K10): a bias scan's point folders for a rung the scan runs at
    each point, the stage folder otherwise.  For anything else -- a bench
    trial, a job set no description stands beside -- the job's own
    directory ``d``."""
    if task is None or job_name not in {s.name for s in task.stages}:
        return [(d, None)]
    from ..transport.stages import rung_containers
    return rung_containers(base, task, job_name)


def _job_status(base: Path, jobset: JobSet, job, task, *, dirs,
                refs, shape) -> StageStatus:
    """One prepped job's status, read from where its attempts are."""
    d = base / dirs[job.name]
    # WHICH FILES are this stage's: its stage's names (`materialize.
    # run_names`) -- in either shape a flat folder holds every stage, an
    # attempt may hold several run indexes.  THE LABEL IS THIS JOB'S, NOT
    # THE JOBSET'S: a trial is relabelled with its point.
    names = run_names(jobset, job, shape)
    basename = names.stem
    read = []
    for home, volts in _rung_homes(base, task, job.name, d):
        # WHERE the run happened, asked of the layer that decides layout
        # -- the latest attempt where there is one, the container for a
        # flat run (project-layout.md § 1.5).
        attempt = latest_attempt(home)  # None is the ANSWER: prepared?
        observed = run_dir(home)        # ...and this is where to look
        # THE LAUNCH RECORD lies in the folder the run happened in, named
        # by the stage's names -- the one answer the writer reads too: the
        # attempt's `run.json`, a trial's at its top, a flat stage's newest
        # run's own in the folder every stage shares.  ONE THAT DOES NOT
        # READ is said, never read as launched or not
        # (`runrecord.launch_record`).
        try:
            launch = launch_record(observed, names)
        except LaunchRecordError as e:
            read.append((home, volts, attempt, observed, None, "unreadable",
                         str(e)))
            continue
        read.append((home, volts, attempt, observed, launch)
                    + _stage_state(observed, launch, basename))
    # A SCAN'S RUNG SPEAKS FROM ITS FIRST POINT NOT FINISHED, in the
    # scan's order -- the order its chain walks -- and from its last once
    # every point has: a rung with a point outstanding is the stage to
    # resume from, and the row names the point (`web/results.md` § 2.4;
    # `engines/transport.md` § 2a.12, *which of five runs is the one
    # still outstanding*).  Any other rung has one folder, which speaks.
    home, volts, attempt, observed, launch, state, detail = next(
        (r for r in read if r[5] != _DONE), read[-1])
    where = attempt.name if attempt else None
    if volts is not None:
        detail = f"{volts:g} V: {detail}"
        where = f"{home.name}/{where}" if where else None
    return StageStatus(
        ref=refs[job.name], dir=d.name, state=state, detail=detail,
        attempt=where,
        attempts=attempts_in(home),
        launch=launch,
        relaunch_continues=job.relaunch_continues,
        # THE SAME LABEL THE STATE WAS READ WITH -- the stage's names'.
        warm_files=_warm_present(observed, names.label, jobset.engine,
                                 base),
        # WHAT THE STAGE IS -- the plan's own columns, read per stage by
        # `status <stage>`.
        script=str(job.script),
        carries=[w.name for w in job.warm],
        resources=resources_text(job.resources),
    )


def _not_prepped(ref: StageRef, stage) -> StageStatus:
    """A stage the description names and nothing has prepped -- `NOT_PREPPED`
    in the reader's own words; a disabled one says so (`job-system.md`
    § 5.3)."""
    on = getattr(stage, "enabled", True) is not False
    return StageStatus(
        ref=ref, dir=None, state=NOT_PREPPED[0],
        detail=(NOT_PREPPED[1] if on else
                "disabled in the description (enabled: false); not prepped"),
        prepped=False, enabled=on)


def jobset_status(jobset: Optional[JobSet], base_dir) -> JobSetStatus:
    """Read the on-disk status of every stage under ``base_dir`` (read-only).

    **The rows are the description's ladder** when a description stands
    beside the set (`job-system.md` § 5.3, 2026-10-01): every stage
    ``task.json`` names, in its order and with its number, the ones nothing
    has prepped yet as :data:`NOT_PREPPED` -- so a calculation lists its
    stages before its first prep (``jobset`` is then ``None``), and a ladder
    prepped one stage at a time lists them all.  The Results tab's ladder is
    this answer (`web/results.md` § 2.4).  A benchmark's sweep, with no
    description beside it, lists its own jobs.

    ``first_incomplete`` is the first stage that is not ``finished`` -- the
    stage to resume from, never a disabled one; ``None`` (and
    ``complete=True``) when every enabled stage finished."""
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
        # a stage renamed in case only keeps its prepped job.
        from ..identity import stage_key
        held = {stage_key(j.name): j for j in (jobset.jobs if jobset is not None
                                               else ())}
        from .materialize import ladder_homes
        # THE NUMBERS ARE THE DOOR'S -- what the folders carry (W38 F4), so
        # a row's number is its folder's, not its place in the description.
        for st, ref in zip(task.stages,
                           [StageRef(h.seq, h.name)
                            for h in ladder_homes(base, task)]):
            job = held.get(stage_key(st.name))
            if job is None:
                stages.append(_not_prepped(ref, st))
                continue
            # THE DESCRIPTION'S NAME AND NUMBER on its row: the ladder is the
            # description's, and `status <stage>` finds the row by it.
            row = dataclasses.replace(
                _job_status(base, jobset, job, task, **kw), ref=ref)
            if getattr(st, "enabled", True) is False:
                # PREPPED, THEN TURNED OFF: its folder is read as ever, and
                # the row says it is never used (`task.stage_disabled`).
                row = dataclasses.replace(
                    row, enabled=False,
                    detail=f"disabled in the description -- its folder is "
                           f"kept as it is and never used; {row.detail}")
            stages.append(row)
    else:
        stages = [_job_status(base, jobset, job, None, **kw)
                  for job in jobset.jobs]
    first = next((s for s in stages if s.state != _DONE and s.enabled),
                 None)
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
        **_next_continuation(base, task, first),
    )


def _next_continuation(base: Path, task, first: Optional[StageStatus]) -> dict:
    """What the next prep of the first incomplete stage will continue from,
    or why it would refuse (:func:`stage_continuation`) -- ``{}`` once it
    is prepped."""
    if task is None or first is None or first.prepped:
        return {}
    return stage_continuation(base, task, first.name)


def stage_continuation(base: Path, task, name: str) -> dict:
    """``{"resume_from": ...}`` or ``{"resume_refused": ...}`` -- what a
    prep of ``name`` would continue from, or why it would refuse -- asked of
    `continuation.continuation_answer`, the one door `prep` asks too, so the
    two never disagree about the same run.  For the stage `status <stage>`
    names as much as for the first incomplete one."""
    from .continuation import continuation_answer
    # NO VERDICT: status names the run, not its relaxation -- and the table
    # (the Results tab's ladder too) is read far more often than prepped.
    got, refused = continuation_answer(base, task, name, verdict=False)
    if got is not None:
        return {"resume_from": got.where()}
    if refused:
        return {"resume_refused": refused}
    return {}


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
    hdr = ("seq", "stage", "attempt", "state", "warm files", "detail")
    rows = [(s.ref.seq_text, s.ref.name, s.attempt or "-", s.state,
             ", ".join(s.warm_files) or "-", s.detail)
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
        lines.append("All stages finished. Nothing to resume."
                     if all(s.enabled for s in status.stages) else
                     "Every enabled stage finished. Nothing to resume.")
    else:
        first = next((s for s in status.stages
                      if s.name == status.first_incomplete), None)
        if first is not None and not first.prepped:
            # NOTHING TO RE-SUBMIT: the stage has no folder yet, so the next
            # step is to prepare it -- and an independent stage's prep takes
            # the run before it, which the line names (§ 5.4).
            if status.resume_refused:
                lines.append(
                    f"First incomplete stage: {first.name}, not prepped yet "
                    f"-- and its prep refuses for now:\n"
                    + textwrap.indent(status.resume_refused, "    "))
                return "\n".join(lines)
            lines.append(
                f"First incomplete stage: {first.name}, not prepped yet:\n    "
                + command("prep", "run", first.name, base=status.base)
                + (f"   # continues from {status.resume_from}"
                   if status.resume_from else ""))
            return "\n".join(lines)
        lines.append(next_step(first, status.first_incomplete,
                               base=status.base))
    return "\n".join(lines)


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


def next_step(s: Optional[StageStatus], name: str, *, base) -> str:
    """What to do about a PREPPED stage that has not finished, by its state
    -- each a command that works.  molbuilder does NOT auto-resume; the
    person decides (`engines/stages.md`)."""
    state = s.state if s is not None else "stopped"
    if state == "pending":
        return (f"First incomplete stage: {name}, prepped and not "
                f"launched:\n" + block(launch_lines("run", name, base=base)))
    if state in ("queued", "running"):
        return (f"First incomplete stage: {name}, {state} -- let it "
                f"finish; `{command('status', name, base=base)}` shows its "
                f"run.")
    # A PREPPED STAGE IS NOT PREPPED AGAIN (user, 2026-10-02): changing it
    # first is a rollback.
    if s is not None and s.relaunch_continues:
        how = ("launch it again -- it continues from its own latest run:\n"
               + block(launch_lines("run", name, base=base))
               + "\n  or, to change it first, " + rollback("its prep",
                                                          base=base))
    else:
        # THE SAME SENTENCE `launch` refuses a second launch with -- one
        # fact (`Job.relaunch_continues`), one answer (C10).
        from .continuation import no_relaunch
        how = no_relaunch(s is None or not s.carries, base=base)
    return (f"First incomplete stage: {name}, {state}.  molbuilder does NOT "
            f"auto-resume -- you decide: {how}")


def render_stage_status(status: JobSetStatus, stage_name: str,
                        continuation: Optional[Dict[str, str]] = None) -> str:
    """One stage, in full — the per-stage form `job-system.md` § 5.3 reserves.

    The table answers *where is this calculation up to*; this answers *what
    this stage is* -- its deck, what it declares, its resources, the plan's
    columns -- and *what happened
    to it*, which is a different question and the one you ask before
    deciding whether to run it again. It is only answerable at all
    because of the attempt layer: the tries are directories, and the launch is
    a record rather than an inference from an empty folder (§ 1.5, § 1.6).

    Everything printed comes off the :class:`StageStatus` the reader already
    built and ``continuation`` -- :func:`stage_continuation`'s answer for a
    stage nothing has prepped, which the caller asks; without it, the
    table's own answer for the first incomplete stage. Nothing here opens a
    file — a second reader of the launch record would be a second answer to
    *was this launched?*
    """
    s = next(x for x in status.stages if x.name == stage_name)
    if not s.prepped:
        # WHAT YOU CAN TYPE (`job-system.md` § 5.3): a disabled stage's prep
        # is refused, so it is told how to enable it; one whose prep would
        # refuse is told why, with the commands.
        cont = (continuation if continuation is not None else
                {"resume_from": status.resume_from,
                 "resume_refused": status.resume_refused}
                if s.name == status.first_incomplete else {})
        how = ("Enable it in Task setup (or task.json) to run it."
               if not s.enabled else
               "Its prep refuses for now:\n"
               + textwrap.indent(cont["resume_refused"], "    ")
               if cont.get("resume_refused") else
               "Prep it:\n    "
               + command("prep", "run", s.name, base=status.base)
               + (f"   # continues from {cont['resume_from']}"
                  if cont.get("resume_from") else ""))
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
        # or null -- it started from the structure.
        src = s.launch.get("continued_from")
        rows.append(("continued from",
                     src if src else "nothing -- it started from the structure"))
    else:
        rows.append(("launched", "no  (no launch record -- prepared, not started)"))
    rows.append(("warm files", ", ".join(s.warm_files) or "-"))
    rows.append(("detail", s.detail or "-"))

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
