"""Run-status reader — the "inform" layer
(`execution/job-system.md` § 2, rule 5: molbuilder informs; the user decides).

The resume contract is: **the modeling software resumes; molbuilder informs
and the user decides** (never auto-recovers — redoing a long run unknowingly
is a heavy penalty).  This module is the *inform* half: for a prepped/running
JobSet it answers, per stage, *did it finish? is it running? did it fail? are
the warm-restart files there?* and *which is the first incomplete stage* (the
one to resume from) — so the manual continue is a one-glance decision.

REUSE, not reinvention: per-stage run state comes from
``parse.dirs.job.run_status`` (the directory-level status verb behind the Results
tab + JobMonitor; its schema is pinned by ``execution/running-a-job.md``
§ 4, the composer pattern by ``model/parse.md`` § 5); this module only adds the cross-stage
view (warm-file inventory + first-incomplete pointer).  It is **read-only**:
it inspects the tree, changes nothing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..identity import StageRef
from .materialize import (attempts, job_dir_names, read_run_launch,
                          latest_attempt, run_dir, shape_of,
                          stage_refs)
from .model import JobSet
from .plan import resources_text

# Engine-native warm-restart files keyed by the project id (system label).
# DERIVED from the one rules file (job-contracts § 4.2a; U3/W2, 2026-08-13):
# the per-engine dict that stood here was the THIRD hand-kept copy of the
# vocabulary, already citing a retired doc.  The carry rows are status's
# question -- could a stage here hand state to the next one?
from ..warmfiles import carry_inventory as _carry_inventory
from ..paths import attempt_name
from ..runfiles import is_stage_token



def _warm_files(engine: str):
    """The engine's carry rows, resolved AT USE — never at import (C-b,
    2026-08-13): the module-level dict that stood here loaded BOTH
    engines' rules files the moment anything imported
    ``molbuilder.jobset``, so one malformed TOML killed every entry
    point with an import-time traceback.  Resolved here, a broken rules
    file refuses at the status call it affects, with the loader's own
    message — the same moment `prep` already owns for its copy.  An
    engine without a rules file simply has no carry rows to report."""
    try:
        return _carry_inventory(engine)
    except Exception:
        return ()

# run_status states we treat as "this stage is done".
_DONE = "finished"


@dataclass(frozen=True)
class StageStatus:
    """One stage's status (read-only snapshot).

    **WHICH stage this is arrives as a whole :class:`~molbuilder.identity.
    StageRef`, not as a loose name and a loose number.** It used to be the
    latter, and the cost was immediate: ``render_stage_status`` wanted the
    heading ``01_coarse``, had only the two halves, and so built a SECOND
    ``StageRef`` out of them to ask for it -- a caller working out an answer a
    floor below already held, inside the very object made to stop that (§ 9.6's
    rule A4). Carrying the ref means the resolver in ``materialize`` is the only
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
    #: The attempt's ``run.json``, whole, or ``None`` if it has not been
    #: launched.  Carried rather than picked apart so the per-stage view has
    #: the record without a SECOND reader of the same file -- the schema is
    #: versioned (``molbuilder/run-launch@1``) and may grow fields.
    launch: Optional[Dict[str, Any]] = None
    #: Whether a re-run of this stage continues from what the last one left
    #: -- the job's own fact (`Job.resumes`, `job-contracts.md` § 4.2a), so
    #: the status says what re-submitting it will do.
    resumes: bool = True
    #: Whether anything has prepped it -- a description's stage before its
    #: first prep is listed all the same (`job-system.md` § 5.3).
    prepped: bool = True
    #: Whether the description runs it -- a disabled stage is listed, and is
    #: never the stage to resume from (§ 5.3).
    enabled: bool = True
    #: What the stage IS, the plan's columns (`plan` folded into
    #: `status <stage>`, 2026-10-01): its deck, what it would take from a
    #: run it continues from, and the resources it asks for.
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
    #: nothing has prepped it -- `handover.handover_answer`'s, the answer
    #: prep itself asks (`job-system.md` § 5.4) -- or ``None``: a linked
    #: stage, the first, one that starts clean.
    resume_from: Optional[str] = None
    #: Why that prep would refuse instead -- the refusal's first sentence.
    resume_refused: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "engine": self.engine,
            "first_incomplete": self.first_incomplete,
            "complete": self.complete,
            "stages": [s.to_dict() for s in self.stages],
        }


def _warm_present(stage_dir: Path, label: str, engine: str) -> List[str]:
    """Which engine warm-restart files actually exist (real files, not
    dangling carry symlinks) in this stage's dir."""
    out: List[str] = []
    for ext in _warm_files(engine):
        f = stage_dir / f"{label}{ext}"
        if f.is_file():                      # follows symlinks; dangling -> False
            out.append(f.name)
    return out


def _label_of(job: Any, fallback: str) -> str:
    """The label THIS job's files carry, read off its own deck.

    `JobSet.name` is the TASK's label; a sweep trial's deck is
    `f"{task.label}-{point_token}"` (`resolve._label_for`), so the two differ
    for exactly the case that matters.  Measured on a real sweep:
    `find(trial_dir, "siesta-AuBDTAu", role=".out")` is `[]` while
    `find(trial_dir, "siesta-AuBDTAu-G0K20C1ELPA1STAGE", role=".out")` finds
    the run -- so `has_output` was False for a trial that had finished, and a
    measured 269.8 s/iter sat beside "launched, no output yet".

    THE STAGE IS FOUND, NOT ASSUMED.  A deck is `<label>_<stage>.<role>`, the
    stage token itself contains an underscore (`01_coarse`), and
    `StageRef.token` is None on a real sweep -- so neither splitting on `_` nor
    trusting the caller's token works.  `runfiles.is_stage_token` is the
    grammar's own answer: walk the underscore boundaries from the right and
    take the first tail it recognises.

    Label-free lookup was tried first and is not available: `find_by_role`
    refuses an underscore role (`_geom.log`) because it cannot be told from a
    stage name without a label.
    """
    script = getattr(job, "script", None)
    if not script:
        return fallback
    stem = Path(str(script)).name
    for role in (".run.sh", ".sbatch", ".sh", ".py", ".fdf"):
        if stem.endswith(role):
            stem = stem[: -len(role)]
            break
    parts = stem.split("_")
    for i in range(len(parts) - 1, 0, -1):
        if is_stage_token("_".join(parts[i:])):
            return "_".join(parts[:i]) or fallback
    # NO STAGE TOKEN, NO GUESS.  A ladder names its stages freely (`demo_s1.fdf`
    # with output `demo.out`), and `s1` is not a stage token -- returning the
    # whole stem would ask for `demo_s1.out` and find nothing.  The jobset's own
    # name is right whenever the deck carries no token, which is every case this
    # function is not here to fix.
    return fallback


#: The state a stage has before `prep` has made it a directory -- the ONE
#: spelling, read by `_stage_state` for a planned stage whose directory is
#: missing and by :func:`jobset_status` for a described stage nothing has
#: prepped yet (`job-system.md` § 5.3, `web/results.md` § 2.4).
NOT_PREPPED = ("not-started", "no directory yet (not prepped)")


def _stage_state(observed: Path, launch: Optional[Dict[str, Any]],
                 out_glob: str) -> tuple:
    """(state, detail) for the directory a stage's run actually happened in.

    ``observed`` is the latest attempt where there is one, and the stage
    container for a flat run (`project-layout.md` § 1.5) — the caller resolves
    that, because *where a run happens* is a layout question and this layer
    only reads.  ``out_glob`` narrows the directory to THIS rung's files --
    the flat shape keeps every stage in one.

    ``launch`` is the attempt's ``run.json``. It is what separates *queued* from
    *never started*, which no amount of looking at an empty directory can do:
    § 1.6's *"a queued cluster job has produced nothing yet, so 'no output' and
    'not started' look identical"*, and its promise that status can then say
    *"queued as job 481923"* rather than guessing from an absence.  The
    directory door answers both from it (`parse.dirs.job.run_status`): the
    rule stood HERE, above the door, until 2026-09-26, so the Results tab's
    directory door answered the same attempt "running".
    """
    if not observed.is_dir():
        return NOT_PREPPED
    try:
        # THROUGH THE PACKAGE THAT OWNS THE QUESTION, never the module
        # inside it (`model/parse.md` § 5.5, R-RO2: one import surface per
        # question).
        from ..parse.dirs import run_status
        # THE RUNG'S OWN GLOB: in the flat shape every stage shares the
        # directory, and without it every row showed the newest stage's
        # state.
        st = run_status(observed, out_glob, launch=launch)
    except Exception as e:                    # fail-soft; stay informative
        return ("unknown", f"could not decode: {e}")
    # `run_status` returns a `RunStatus` since 2026-09-09; both fields are
    # always present, so the old `.get(..., default)` pair is gone with the
    # dict.  The "unknown" fallback lives in the except clause above, which is
    # the only way this can fail to have an answer.
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


def _job_status(base: Path, jobset: JobSet, job, task, *, sh, dirs,
                refs) -> StageStatus:
    """One prepped job's status, read from where its attempts are."""
    d = base / dirs[job.name]
    # WHICH FILES are this stage's, asked of the layout (§ 9's `Shape`).
    # In the hierarchy the directory already answered; in flat every stage
    # shares one, and the deck's token in each filename is the answer.
    token = refs[job.name].token
    # THE LABEL IS THIS JOB'S, NOT THE JOBSET'S.  A sweep's `JobSet.name`
    # is `task.label`, while each trial's deck is `f"{task.label}-{token}"`
    # (`resolve._label_for`) -- so narrowing by the jobset's name matched
    # NOTHING for a trial, and a finished trial answered § 1.6's forbidden
    # "prepped, not launched".  Read off the deck the way `summarize` does
    # (`Path(job.script).stem` minus the stage suffix), which is the name
    # the files actually carry.
    job_label = _label_of(job, jobset.name)
    out_glob = (sh.stage_glob(token, job_label)
                if (sh is not None and token) else "*")
    read = []
    for home, volts in _rung_homes(base, task, job.name, d):
        # WHERE the run happened, asked of the layer that decides layout
        # -- the latest attempt where there is one, the container for a
        # flat run (project-layout.md § 1.5).  Globbing the folder
        # regardless was blind to the whole attempt layer: a finished
        # hierarchical stage read as "prepped, not launched" because its
        # .out is one level down.
        attempt = latest_attempt(home)  # None is the ANSWER: prepared?
        observed = run_dir(home)        # ...and this is where to look
        # WHERE the launch record lives mirrors where submit WRITES it
        # (submit.py `_where_recorded`): the attempt when one exists; a
        # sweep trial's at the trial's top -- until 2026-08-20 this read
        # the attempt only, so a grouped-submitted trial answered § 1.6's
        # exact forbidden line ("prepped, not launched") while its record
        # sat one level up; and a flat stage's own record, named by its
        # deck, in the directory every stage shares (2026-09-27).
        launch = read_run_launch(
            attempt if attempt is not None else home,
            basename=(None if attempt is not None
                      or jobset.kind == "sweep"
                      else Path(job.script).stem))
        read.append((home, volts, attempt, observed, launch)
                    + _stage_state(observed, launch, out_glob))
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
        attempts=attempts(home),
        launch=launch,
        resumes=job.resumes,
        # THE SAME LABEL THE STATE WAS READ WITH.  This asked for
        # `jobset.name` while everything else in the loop had moved to
        # `job_label` -- so the fix `_label_of` exists for was applied to
        # the `.out` and not to the warm files beside it.  Measured on a
        # staged sweep trial: `siesta-AuBDTAu-G0K20C1.XV` on disk, warm
        # files reported `[]`, and `jobset status` told a person there
        # was nothing to restart from.
        warm_files=_warm_present(observed, job_label, jobset.engine),
        # WHAT THE STAGE IS -- the plan's own columns, read per stage by
        # `status <stage>` since `plan` folded into it (2026-10-01).
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
    this answer (`web/results.md` § 2.4).  A set with no description beside
    it -- a hand-built one, a benchmark's sweep -- lists its own jobs.

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
        kw = {"sh": sh, "dirs": job_dir_names(jobset, sh),
              "refs": stage_refs(jobset)}
    stages: List[StageStatus] = []
    if task is not None:
        held = {j.name: j for j in (jobset.jobs if jobset is not None
                                    else ())}
        for st, ref in zip(task.stages,
                           StageRef.ladder([s.name for s in task.stages])):
            job = held.get(st.name)
            if job is None:
                stages.append(_not_prepped(ref, st))
                continue
            row = _job_status(base, jobset, job, task, **kw)
            if getattr(st, "enabled", True) is False:
                # PREPPED, THEN DISABLED: its run is read as ever, and the
                # row says the description no longer runs it.
                row = dataclasses.replace(
                    row, enabled=False,
                    detail=f"disabled in the description; {row.detail}")
            stages.append(row)
    else:
        stages = [_job_status(base, jobset, job, None, **kw)
                  for job in jobset.jobs]
    first = next((s for s in stages if s.state != _DONE and s.enabled),
                 None)
    return JobSetStatus(
        name=(jobset.name if jobset is not None else task.label),
        engine=(jobset.engine if jobset is not None else str(task.engine)),
        stages=stages,
        first_incomplete=(first.name if first is not None else None),
        complete=(first is None),
        **_next_handover(base, task, first),
    )


def _next_handover(base: Path, task, first: Optional[StageStatus]) -> dict:
    """What the next prep of the first incomplete stage will continue from,
    or why it would refuse -- asked of `handover.handover_answer`, the one
    door `prep` asks too, so the two never disagree about the same run (the
    W37 review: a rule of status's own told a stage set to start clean that
    it would continue, and named nothing on the flat layout)."""
    if task is None or first is None or first.prepped:
        return {}
    from .handover import handover_answer
    got, refused = handover_answer(base, task, first.name)
    if got is not None:
        return {"resume_from": got.where()}
    if refused:
        return {"resume_refused": refused.split("\n")[0]}
    return {}


def render_status(status: JobSetStatus) -> str:
    """Human-readable status table + the resume pointer (the inform
    surface for the CLI / plan / UI)."""
    lines: List[str] = [
        f"JOB-SET STATUS -- {status.name} ({status.engine})",
        "",
    ]
    # The stage's SEQ, never its row.  This printed `enumerate()` until
    # 2026-08-10 -- a position where a reader reads an ordinal, which is the
    # number `engines/stages.md` R5 forbids as an identifier.  A sweep point
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
    # They were two hand-written numbers until 2026-08-10, and adding the
    # `attempt` column desynchronised them immediately: six headings over a
    # five-segment rule.
    w = [max(len(r[k]) for r in rows + [hdr]) for k in range(len(hdr))]
    def fmt(r):
        return "  ".join(s.ljust(w[k]) for k, s in enumerate(r))
    lines.append("  " + fmt(hdr))
    lines.append("  " + "  ".join("-" * n for n in w))
    lines += ["  " + fmt(r) for r in rows]
    lines.append("")
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
                    f"    {status.resume_refused}")
                return "\n".join(lines)
            lines.append(
                f"First incomplete stage: {first.name}, not prepped yet:\n"
                f"    molbuilder jobset prep run {first.name}"
                + (f"   # continues from {status.resume_from}"
                   if status.resume_from else ""))
            return "\n".join(lines)
        what = ("the engine warm-starts from its own restart files"
                if first is None or first.resumes else
                "it runs again from its first step -- this kind of run does "
                "not resume")
        lines.append(
            f"First incomplete stage: {status.first_incomplete}.  "
            f"molbuilder does NOT auto-resume -- you decide: re-submit that "
            f"stage ({what}) or switch parameters (`engines/stages.md`).")
    return "\n".join(lines)


def render_stage_status(status: JobSetStatus, stage_name: str) -> str:
    """One stage, in full — the per-stage form `job-system.md` § 5.3 reserves.

    The table answers *where is this calculation up to*; this answers *what
    this stage is* -- its deck, what it carries, its resources, the columns
    `plan` printed until it folded in here (2026-10-01) -- and *what happened
    to it*, which is a different question and the one you ask before
    deciding whether to run it again. It is only answerable at all
    because of the attempt layer: the tries are directories, and the launch is
    a record rather than an inference from an empty folder (§ 1.5, § 1.6).

    Everything printed comes off the :class:`StageStatus` the reader already
    built. Nothing here opens a file — a second reader of ``run.json`` would be
    a second answer to *was this launched?*
    """
    s = next(x for x in status.stages if x.name == stage_name)
    if not s.prepped:
        # WHAT YOU CAN TYPE (`job-system.md` § 5.3): a disabled stage's prep
        # is refused on a transport ladder, so it is told how to enable it.
        nxt = s.name == status.first_incomplete
        how = ("Enable it in Task setup (or task.json) to run it."
               if not s.enabled else
               f"Its prep refuses for now: {status.resume_refused}"
               if nxt and status.resume_refused else
               f"Prep it:  molbuilder jobset prep run {s.name}"
               + (f"   # continues from {status.resume_from}"
                  if nxt and status.resume_from else ""))
        return "\n".join([f"STAGE {s.ref.label} -- {s.state}", "",
                          f"  {s.detail}", "", how])
    rows: List[tuple] = [
        # WHAT THE STAGE IS, before what happened to it -- the plan's columns
        # (`plan` folded in here, 2026-10-01; `job-system.md` § 5.3).
        ("deck", s.script or "-"),
        ("carries", ", ".join(s.carries) or "-"),
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
        # ABSENT means started from the structure (checkpointing.md S3), which
        # is a different claim from "continued from nothing" -- so it is only
        # printed when there is something to print.
        if s.launch.get("continued_from"):
            rows.append(("continued from", s.launch["continued_from"]))
    else:
        rows.append(("launched", "no  (no run.json -- prepared, not started)"))
    rows.append(("warm files", ", ".join(s.warm_files) or "-"))
    rows.append(("detail", s.detail or "-"))

    # The pad comes off the longest label, never a literal.  Hand-written and
    # it was 14 -- exactly the width of "continued from", so the one row that
    # had something to say ran its value straight into its own name.
    w = max(len(k) for k, _ in rows) + 2
    lines = [f"STAGE {s.ref.label} -- {s.state}", ""]
    lines += [f"  {k.ljust(w)}{v}" for k, v in rows]
    lines.append("")
    lines.append(
        f"Directory: {s.dir}" + (f"/{s.attempt}" if s.attempt else ""))
    return "\n".join(lines)


__all__ = ["StageStatus", "JobSetStatus", "jobset_status", "render_status",
           "render_stage_status"]
