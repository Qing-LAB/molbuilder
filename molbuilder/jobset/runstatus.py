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

THE VIEWS A READER IS SHOWN (floor 6, `execution/architecture.md` § 2.1,
§ 3.2) are composed here over the run door (floor 2) and a run of points'
rows (floor 4), each run's points read once per answer: the views a folder
is read with (:func:`views_of` -- a calculation's ladder, :func:`ladder_view`;
a run of points, :func:`points_view`; the point a folder is), and how the
run a file belongs to is doing (:func:`run_answer`).
"""

from __future__ import annotations

import dataclasses
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..identity import StageRef, stage_key
from ..paths import attempts_in
from .materialize import latest_attempt
from .naming import job_dir_names, run_names, stage_refs
from ..runrecord import LaunchRecordError, launch_record
from .commands import (block, command, launch_lines, launch_pending,
                       prep_group_lines, rollback)
from .model import JobSet
from .continuation import (PointsReading, gathered_lines, pending_words,
                           read_points, stage_run_state, taken_over)
from .plan import resources_text
from .ready import MISSING

# Engine-native warm-restart files keyed by the project id (system label),
# DERIVED from the one rules file (job-contracts § 4.2a; U3/W2).  The carry rows are status's
# question -- could a stage here hand state to the next one? -- asked of the
# one door with the calculation's folder, so its own list answers.
from ..warmfiles import warm_list as _warm_list
from ..paths import attempt_dir, attempt_name



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
    #: Which run this status was read from (``run-0``; a run of points', its
    #: points inside it), or ``None`` when no run is opened.
    attempt: Optional[str] = None
    #: Every run present (``run-<n>``), ascending -- the stage's history.  A
    #: launch again opens a new run folder and leaves the old one exactly as
    #: it was (§ 1.5), so this is the list of runs, not a counter that can
    #: be off.
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
    #: A run of points' points, in the listing order -- each the job's row
    #: shape (`runpoints.rows_of`: ``rel``, ``words``, ``phase``,
    #: ``round``, ``needs``, ``ended``, ``ran``, ``not_run``) with the
    #: point's wire form (its kind's ``to_wire()``), ``point`` (its words),
    #: ``folder``, ``state`` (its run's), ``opened`` (its folder is there),
    #: ``converged`` (its run's verdict), ``done``, ``why``,
    #: ``started_from`` (``start_unread``, the reader's words, when its
    #: record does not read) and ``took``, read from the point's own files
    #: (`continuation.point_rows`, `execution/runs-of-points.md` § 4.7) --
    #: or ``[]``.
    points: List[Dict[str, Any]] = field(default_factory=list)
    #: What the run GATHERED from the rungs upstream -- ``<file> ← <run>``
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
    #: THE RUN IT IS A COPY OF -- the newest run's ``.copied-from``, a run of
    #: the finished calculation a transport calculation reuses, tree-relative
    #: (`runrecord.read_copied_from`; `engines/transport.md` § 2a.16), or
    #: the reader's words when it does not read; ``None`` for a run made
    #: here.
    copied_from: Optional[str] = None
    #: What the stage IS, the plan's columns: its deck, the restart files it declares
    #: -- what it would take from a run it continues from -- and the
    #: resources it asks for.
    script: Optional[str] = None
    carries: List[str] = field(default_factory=list)
    resources: Optional[str] = None
    #: CHANGED SINCE PREPARED -- each value its records give now otherwise
    #: than its prep recorded, ``name: was → now`` (`ready.Readiness.
    #: changed`; `job-system.md`, *The task*) -- and whether its prep
    #: recorded none (``unrecorded``, prepared before 2.41.0).
    changed: list = field(default_factory=list)      # `prepared_with.Difference`
    unrecorded: bool = False

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


def _warm_present(run_dir: Path, label: str, engine: str,
                  base) -> List[str]:
    """Which engine warm-restart files exist in the run folder ``run_dir``
    -- by the list in effect for the calculation in ``base``."""
    out: List[str] = []
    from ..runfiles import compose
    for ext in _warm_files(engine, base):
        f = run_dir / compose(label, ext)
        if f.is_file():
            out.append(f.name)
    return out


def run_folders(base, task, stage_dir, stage: str):
    """EVERY RUN OF A STAGE, as folders -- ``(n, folder, point)``: each
    attempt ``run-<n>`` in order and, for a rung of points, each of its
    points' folders there (`PointsOf.points_in`), ``point`` ``None``
    for a run with no point level -- the declaration's walk alone, no state
    asked: the memory a stage needs (`jobset.memory`) asks it; a reader of
    the runs' states reads each run once (:func:`every_run`)."""
    root = Path(base)
    for n in attempts_in(Path(stage_dir)):
        run_dir = attempt_dir(Path(stage_dir), n)
        folders: List[tuple] = []
        from .points import points_of
        declared = points_of(task) if task is not None else None
        if declared is not None:
            folders = [(p, pt) for p, pt in declared.points_in(
                run_dir, task, stage, root) if p.is_dir()]
        for folder, pt in folders or [(run_dir, None)]:
            yield n, folder, pt


def _at_work(reading: PointsReading, run: Path):
    """THE POINT AT WORK of a run of points read once (``reading``) --
    the point running; else the last, in the listing order, whose run holds
    an engine output (its row's ``output``); ``None`` before any has
    (`architecture.md` § 3.2) -- as ``(row, folder)``."""
    from ..calcdirs import RunPlace
    rows = reading.rows
    pick = (next((r for r in rows if r["state"] == "running"), None)
            or next((r for r in reversed(rows) if r.get("output")), None))
    return ((pick, RunPlace.of(run).point(pick["rel"]))
            if pick is not None else (None, None))


def _run_opens(run: Path, at_opens: Optional[str]) -> Optional[str]:
    """WHAT A RUN OF POINTS OPENS (`web/results.md` § 3a): its own result
    where its root holds one -- a transmission run's, its calculation's
    report (`runs.openable`) -- else ``at_opens``, the file its point at
    work's folder opens; ``None`` with neither."""
    from ..runs import openable
    own, _trail = openable(run)
    return own if own is not None else at_opens


def _report_of(calc) -> Path:
    """The calculation's description -- through which its report is opened
    (`calcdirs.description_path`, `web/results.md` § 0.1)."""
    from ..calcdirs import description_path
    return description_path(Path(calc.base))


def points_view(calc, run, reading: Optional[PointsReading] = None
                ) -> Dict[str, Any]:
    """A RUN OF POINTS, AS A READER SEES IT (`architecture.md` § 3.2,
    `web/results.md` § 3a) -- ``{state, detail, at_work, opens}``: its
    state and its count in the kind's words; **the point at work**
    (:func:`_at_work`) -- its point in its one wire form, its folder, its
    record (the run door's `record_of`) and the row of the file its folder
    opens; and **what the run opens** (:func:`_run_opens`).  Every path
    whole.  ``reading`` is the run's one reading where the caller holds it
    (`continuation.read_points`)."""
    from ..calcdirs import where
    from ..identity import parse_token
    from ..runs import file_row, openable, record_of, run_of
    run = Path(run)
    if reading is None:
        reading = read_points(calc.base, calc.task,
                              parse_token(where(run).stage)[1], run)
    row, folder = _at_work(reading, run)
    at_work, at_opens = None, None
    if row is not None:
        ran = run_of(folder)
        at_opens, _trail = openable(folder)
        at_work = {"point": row["wire"], "dir": str(folder),
                   "record": record_of(ran) if ran is not None else None,
                   "opens": file_row(at_opens) if at_opens else None}
    return {"state": reading.state, "detail": reading.detail,
            "at_work": at_work, "opens": _run_opens(run, at_opens)}


def _run_row(calc, n: int, folder: Path, point, *, state, detail,
             converged, opens: Optional[str], copied: Optional[str],
             opens_at=None, ran=None) -> Dict[str, Any]:
    """One row of the ladder's every run (`web/results.md` § 2.4): the run,
    its point in its one wire form (``None``: every field ``null``), its
    folder, state, detail and verdict, the file it opens (``opens``, the
    caller's answer), whether that is its calculation's report
    (``opens_report``) and the point it opens it at (``opens_at``, its own
    unless the caller names the point at work's), the deck its run read and
    the check written before it -- through the run door (``ran``, its
    `Run`), every path whole -- and what it is a copy of (``copied``, the
    caller's reading, :func:`_copied`)."""
    deck, report = (ran.deck, ran.report) if ran is not None else (None,
                                                                   None)
    report_opened = opens is not None and Path(opens) == _report_of(calc)
    return {"run": attempt_name(n),
            **(point if point is not None
               else dict.fromkeys(("frame", "of", "bias_v"))),
            "dir": str(folder), "state": state, "detail": detail,
            "converged": converged,
            "opens": opens,
            # A ROW THAT OPENS ITS CALCULATION'S REPORT SAYS SO, and the
            # point it opens it at (`web/results.md` § 0.3, § 2.4).
            "opens_report": report_opened,
            "opens_at": ((opens_at if opens_at is not None else point)
                         if report_opened else None),
            "deck": str(deck) if deck is not None else None,
            "validation": str(report) if report is not None else None,
            # A RUN COPIED IN from the finished calculation this one reuses,
            # named by what it is a copy of (`engines/transport.md` § 2a.16).
            "copied_from": copied}


def every_run(calc, st: StageStatus) -> List[Dict[str, Any]]:
    """EVERY RUN OF A STAGE, each with its own state -- the root's bird's-eye
    (`web/results.md` § 2.4): each attempt in order, read as `status` reads
    a run -- a run of points (`calcdirs.Where.of_points`) as the run it is,
    its own row first (its state and count, what it opens and the point it
    opens a report at, :func:`_run_opens`), then each of its points' folders
    opened, from the same reading's rows; any other run by the run door
    (`runs.state_of`), its point where it has one (`points.point_of`: a
    transmission run once).  **A run's points are read once per answer**
    (`architecture.md` § 3.2): the stage's newest run is the stage row's
    own reading (``st``), in the run's own words where it is not launched.
    ``[]`` for a stage nothing has prepared."""
    from ..calcdirs import RunPlace, where
    from ..runs import openable, run_of, state_of
    from .points import point_of
    if not st.dir:
        return []
    stage_dir = Path(calc.base) / st.dir
    rows: List[Dict[str, Any]] = []
    for n in attempts_in(stage_dir):
        run_dir = attempt_dir(stage_dir, n)
        if where(run_dir).of_points:
            # THE STAGE ROW'S OWN READING for its newest run -- its state
            # whole; its detail the run's own words where the ready door
            # worded a run not launched for its launch.
            reading = (PointsReading(
                st.state,
                pending_words(run_dir) if st.state == "pending"
                else st.detail,
                tuple(st.points), st.launch)
                if run_dir.name == st.attempt
                else read_points(calc.base, calc.task, st.ref.name, run_dir))
            place = RunPlace.of(run_dir)
            # WHAT EACH POINT'S FOLDER OPENS, asked once a point -- the
            # point at work's the run's own until its root holds a result.
            opened = {r["rel"]: openable(place.point(r["rel"]))[0]
                      for r in reading.rows if r["opened"]}
            row, _folder = _at_work(reading, run_dir)
            rows.append(_run_row(
                calc, n, run_dir, None, state=reading.state,
                detail=reading.detail, converged=None,
                opens=_run_opens(run_dir, opened.get(row["rel"])
                                 if row is not None else None),
                copied=_copied(calc.base, run_dir),
                opens_at=row["wire"] if row is not None else None))
            for r in reading.rows:
                if not r["opened"]:
                    continue
                folder = place.point(r["rel"])
                rows.append(_run_row(
                    calc, n, folder, r["wire"], state=r["state"],
                    detail=r["detail"] if r["done"] else r["why"],
                    converged=r["verdict_words"], opens=opened[r["rel"]],
                    copied=r["copied_from"], ran=run_of(folder)))
            continue
        ran = run_of(run_dir)
        if ran is None:
            continue
        got = state_of(ran)
        pt = point_of(run_dir, calc.task)
        rows.append(_run_row(calc, n, run_dir,
                             pt.to_wire() if pt is not None else None,
                             state=got.state, detail=got.detail,
                             converged=got.converged_words,
                             opens=openable(run_dir)[0],
                             copied=_copied(calc.base, run_dir), ran=ran))
    return rows


def ladder_view(calc) -> Dict[str, Any]:
    """A CALCULATION'S LADDER, as a reader sees it (`web/results.md` § 2.4)
    -- the status door's one wire form (`JobSetStatus.to_dict`), the next
    step included, and under each stage EVERY RUN of it (:func:`every_run`),
    each run's points read once; a stage whose runs do not list says why on
    its row (``runs_unread``), the other stages kept."""
    status = jobset_status(calc)
    ladder = status.to_dict()
    for row, st in zip(ladder["stages"], status.stages):
        try:
            row["runs"] = every_run(status.calc, st)
        except (OSError, ValueError) as exc:
            row["runs"], row["runs_unread"] = [], str(exc)
    return ladder


def views_of(folder) -> Dict[str, Any]:
    """THE VIEWS A FOLDER IS READ WITH (`architecture.md` § 3.2,
    `model/parse.md` § 5.2) -- ``{ladder, points, point}``, what is asked
    of a folder beyond its own answer (`runs.folder_answer`), by what it is
    (`calcdirs.where`): at a calculation's root its **ladder**
    (:func:`ladder_view`); at a run of points' root and each of its points'
    folders, the **run of points' view** (:func:`points_view`); at a folder
    of a rung with points, **the point it is** (`points.point_of`), in its
    one wire form; each ``None`` elsewhere.  The calculation is opened once
    for the views (`calculation.open`), and a run of points' points are
    read once per answer; the run door places each folder it is handed by
    itself.  What does not read is said where the view would be, in its reader's words: a
    calculation that does not open, or a ladder whose records do not --
    ``{"unread": ...}`` for the ladder, the view ``unreadable``."""
    from .. import calcdirs
    from ..calculation import open as open_calculation
    from .points import point_of
    out: Dict[str, Any] = {"ladder": None, "points": None, "point": None}
    folder = Path(folder)
    at = calcdirs.where(folder)
    if at.root is None:
        return out
    of_points = (at.run_folder is not None and at.trial is None
                 and at.what in (calcdirs.ATTEMPT, calcdirs.POINT)
                 and calcdirs.where(at.run_folder).of_points)
    try:
        calc = open_calculation(at.root)
    except (OSError, ValueError) as exc:
        if at.what == calcdirs.CALCULATION:
            out["ladder"] = {"unread": str(exc)}
        if of_points:
            out["points"] = {"state": "unreadable", "detail": str(exc),
                             "at_work": None, "opens": None}
        return out
    if at.what == calcdirs.CALCULATION:
        try:
            out["ladder"] = ladder_view(calc)
        except (OSError, ValueError) as exc:
            out["ladder"] = {"unread": str(exc)}
    if of_points:
        out["points"] = points_view(calc, at.run_folder)
    pt = point_of(folder, calc.task)
    out["point"] = pt.to_wire() if pt is not None else None
    return out


def run_answer(path) -> Optional[Dict[str, Any]]:
    """``{state, detail, live, engine_elapsed_s}`` -- HOW THE RUN THE FILE
    AT ``path`` BELONGS TO IS DOING, for a viewer that follows it
    (`web/results.md` § 4.1): a run's file by the run door (`runs.state_of`,
    asked with its launch record: one that does not read is ``unreadable``,
    its file named), a file at a run of points' root by the run's one
    reading (`continuation.read_points` -- its state, its points').
    ``live`` -- `job.LIVE_STATES` -- is what the viewer follows; once ended
    (`job.FINAL_STATES`), the run's own length, its engine's wall from
    launch to exit its record states (`runs.WALL`, the Run panel's), else
    ``None`` -- as for a run of points' root, which has no record.  ``None``
    for a file of no run of ours, as for an upload: nothing is asked of a
    folder no calculation claims (`model/parse.md` § 5)."""
    from ..identity import parse_token
    from ..parse.dirs.job import FINAL_STATES, LIVE_STATES
    from ..runs import WALL, place_of, record_of, run_of, state_of
    run = run_of(path)
    took = None
    if run is None:
        p = Path(path)
        folder = p if p.is_dir() else p.parent
        place = place_of(folder)
        if not place.ours or not place.where.of_points:
            return None
        got = read_points(place.root, place.task,
                          parse_token(place.where.stage)[1], folder)
        state, detail = got.state, got.detail
    else:
        st = state_of(run)
        state, detail = st.state, st.detail
        if state in FINAL_STATES:
            # AN ENDED RUN'S OWN LENGTH -- its record's, the field alone.
            rec = record_of(run, status=st, fields=(WALL,)) or {}
            took = ((rec.get("computation") or {}).get("time")
                    or {}).get("engine_elapsed_s")
    return {"state": state, "detail": detail,
            "live": state in LIVE_STATES, "engine_elapsed_s": took}


def _copied(base, run: Optional[Path]) -> Optional[str]:
    """What ``run`` -- a run, or a point of a run of points -- is a copy of
    (`runrecord.copied_of`), or the reader's words when its record does not
    read -- said on its row."""
    from ..runrecord import copied_of
    if run is None:
        return None
    try:
        return copied_of(base, run)
    except LaunchRecordError as exc:
        return str(exc)


def _sweep_status(calc, job, *, run: Path, columns: dict
                  ) -> Optional["StageStatus"]:
    """A SWEPT RUNG's row (`engines/transport.md` § 2a.11): its latest run's
    -- ``run`` -- one reading (`continuation.read_points`): ``pending`` before
    it is launched; while launched, ``running`` while a point runs,
    ``finished`` once every point is done, ``queued`` when nothing has run
    yet, else ``failed`` -- and its points, each done or not done and what
    it started from; ``columns``, the stage's own (:func:`_job_status`).
    ``None`` for a rung that does not sweep."""
    from .points import points_of
    base, task = calc.base, calc.task
    declared = points_of(task) if task is not None else None
    if declared is None or not declared.rung_points(calc, job.name):
        return None
    got = read_points(base, task, job.name, run)
    pts = got.rows
    # CONVERGED, in its own column (`job-system.md`, rule 4): the first
    # point whose output says it did not, in that verdict's own words (a
    # relaxation's `geometry NO`, an SCF's `SCF NO`); yes for a rung that
    # computes an SCF -- its roles' fact in the kind table (`computes`; a
    # transmission, a product of the device's, converges nothing) -- when
    # every point's output says it converged; otherwise nothing yet.
    no = next((p for p in pts if p["converged"] is False), None)
    converged = (no["verdict_words"] if no is not None
                 else "SCF yes" if pts and _computes_scf(task, job.name)
                 and all(p["converged"] is True for p in pts)
                 else None)
    return StageStatus(
        state=got.state, detail=got.detail, converged=converged,
        attempt=run.name, launch=got.launch, points=pts,
        gathered=gathered_lines(run), copied_from=_copied(base, run),
        **columns)


def _computes_scf(task, stage: str) -> bool:
    """Whether the rung ``stage`` computes an SCF -- its roles' fact in the
    kind table (`execution/architecture.md` § 3.3: a role's ``computes``),
    the stage's roles by the one rule (`template.stage_roles`)."""
    from ..kind_table import kind
    from ..template import stage_roles
    k = kind(str(task.calculation))
    return any((r := k.role(name)) is not None and r.computes == "scf"
               for name in stage_roles(str(task.engine),
                                       str(task.calculation), stage) or ())


def _job_status(calc, job, *, dirs, refs) -> StageStatus:
    """One prepared job's status, read from where its runs are -- a run of
    points' row (:func:`_sweep_status`) or one run's
    (:func:`_one_run_status`), its leads' files' line (:func:`_tsgf_line`)
    and a queued row's standing (`runs.standing_of`, `running-a-job.md`
    § 4.2) asked the same way for both; a prepared stage with no run on
    disk is :data:`MISSING`, said."""
    base, jobset, task = Path(calc.base), calc.jobset, calc.task
    d = base / dirs[job.name]
    # WHAT THE STAGE IS -- the plan's own columns, and its runs: every
    # row's, whatever its run says, read per stage by `status <stage>`
    # (`job-system.md` § 5.3).
    columns = dict(ref=refs[job.name], dir=d.name, attempts=attempts_in(d),
                   relaunch_continues=job.relaunch_continues,
                   script=str(job.script),
                   carries=[w.name for w in job.warm],
                   resources=resources_text(job.resources))
    # WHERE the run happened, asked of the layer that decides layout -- the
    # stage's latest run (project-layout.md § 1.5a).
    run = latest_attempt(d)
    if run is None:
        return StageStatus(state=MISSING[0], detail=MISSING[1], **columns)
    # WHICH FILES are this stage's: its stage's names (`naming.
    # run_names`), each named once in its run (`project-layout.md`
    # § 1.6.1).  THE LABEL IS THIS JOB'S, NOT THE JOBSET'S: a trial is
    # relabelled with its point.
    names = run_names(jobset, job)
    row = (_sweep_status(calc, job, run=run, columns=columns)
           or _one_run_status(base, jobset, run=run, names=names,
                              columns=columns))
    row = dataclasses.replace(row, tsgf=_tsgf_line(
        base, task, job, d, run if row.launch else None, row.attempts,
        going=row.state in ("queued", "running")))
    if row.state != "queued":
        return row
    from ..runs import standing_of
    return dataclasses.replace(row, standing=standing_of(row.launch, base))


def _one_run_status(base: Path, jobset: JobSet, *, run: Path, names,
                    columns: dict) -> StageStatus:
    """One run's row: the stage's newest run, ``run``, through the one
    door; ``columns``, the stage's own (:func:`_job_status`)."""
    # ITS STATE, THROUGH THE ONE DOOR (`runs.concluded`, which asks the
    # scheduler once where it can, `running-a-job.md` § 4.2): the stage's
    # newest run, asked of its launch record -- the run's `run.json` -- and
    # its files.  A record that does not read is the state `unreadable`,
    # said, never read as launched or not (`runrecord.launch_record`).
    from ..runs import concluded
    st = concluded(run, names, base=base)
    state, detail = st.state, st.detail
    launch = None if state == "unreadable" else launch_record(run)
    if state == "pending":
        detail = pending_words(run, detail)
    return StageStatus(
        state=state, detail=detail,
        converged=st.converged_words, attempt=run.name,
        launch=launch,
        # THE SAME LABEL THE STATE WAS READ WITH -- the stage's names'.
        warm_files=_warm_present(run, names.label, jobset.engine, base),
        # WHAT THE RUN GATHERED (a transport rung), beside what it declares.
        gathered=gathered_lines(run),
        copied_from=_copied(base, run),
        **columns)


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
    newest = max(attempts)
    return "; ".join(
        [f"{attempt_name(newest)}: {first}"]
        + [f"{attempt_name(n)}: "
           + str(status_line(base, task, stage_dir,
                             attempt_dir(Path(stage_dir), n),
                             job.scratch_dir))
           for n in sorted(attempts, reverse=True) if n != newest])


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
    trials, from its bench container.

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
    # THE CALCULATION AS THIS STATUS READS IT: its own plan and
    # description, or a benchmark's set alone (`_job_status`).
    view = dataclasses.replace(calc, jobset=jobset, task=task)
    kw = {}
    if jobset is not None:
        kw = {"dirs": job_dir_names(jobset), "refs": stage_refs(jobset)}
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
                _job_status(view, job, **kw), ref=ref,
                group=calc.request.group_of(st.name),
                changed=list(r.changed), unrecorded=r.unrecorded)
            if not r.launched and row.state == "pending":
                row = dataclasses.replace(row, detail=r.detail,
                                          takes=list(r.takes), why=r.why)
            stages.append(row)
        offer = state.prep_offer
        steps = _steps(calc, state.groups, stages)
    else:
        stages = [_job_status(view, job, **kw) for job in jobset.jobs]
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


def _steps(calc, groups, rows: List[StageStatus]) -> List[GroupStep]:
    """THE NEXT STEP (`job-system.md`, *The task*, *The next step*): every
    group of ``task``, in its order, its state read off its members' rows
    (``rows``, the run doors' answers) and the ready door's group
    (``groups``, `ready.GroupAnswer`), the first row of the contract's table
    that applies; every line composed by `commands`.  The first group not
    finished is the next step; with every group finished, the last names
    `summarize task` where the task's result gathers its stages
    (`summarize.task_summary`)."""
    from .commands import reason_of, redo_launched, said_lines, way_back
    from .summarize import task_summary
    base, task = Path(calc.base), calc.task
    by = {s.name: s for s in rows}
    out: List[GroupStep] = []
    for g in groups:
        members = [by[m] for m in g.members]
        n = g.number
        live = next((r for r in members if r.state in _ON_ITS_WAY), None)
        bad = next((r for r in members if r.state in _ENDED_BADLY), None)
        if g.members_changed:
            # A LAUNCHED GROUP WHOSE MEMBERS CHANGED -- a member taken out, or
            # a stage put in, prepared or not: offered by no verb, which stage
            # said, its way on the rule's own, whole (`job-system.md`, *The
            # task*).
            step = ("changed", f"{g.said}: {reason_of(g.members_changed)}",
                    said_lines(g.members_changed)[1:])
        elif g.changed:
            # ITS PREP FORMS IT AGAIN with every group it needs, the groups
            # prep names (`GroupAnswer.prep_with`) -- a member not prepared
            # beside it too.
            step = ("changed", g.said,
                    prep_group_lines(g.prep_with, base=base))
        elif not all(r.prepared for r in members):
            step = (("not prepared", g.said,
                     prep_group_lines(g.prep_with, base=base))
                    if g.preparable else
                    ("waits", "waits before it can be prepared: "
                     + reason_of(g.why or ""), []))
        elif g.held is not None:
            # A GROUP A SEQUENCE STILL GOING HOLDS, not launched by it yet --
            # launched before or not (`job-system.md` § 6.0, *A sequence of
            # groups*): that job's to launch, whatever changed
            # since its prep (R-P23 refuses its prep).
            step = ("in a job", f"{g.said} ({g.held[2]})",
                    ["molbuilder jobs status"])
        elif g.since_prepared and live is None:
            # CHANGED SINCE PREPARED (`job-system.md`, *The task*): never
            # launched, prepared again; launched and changed in its run
            # card alone, prepared again for its request, its runs kept
            # (`job-request.md` § 5); launched
            # and changed in what it computes, the group started from the
            # beginning, the three lines whole (`commands.redo_launched`).
            from .prepared_with import run_card_only
            words = ("changed since prepared: "
                     + "; ".join(d.said for d in g.since_prepared))
            if not g.launched:
                step = ("changed since prepared",
                        words + " -- prepare it again",
                        prep_group_lines([n], base=base))
            elif run_card_only(g.since_prepared):
                step = ("changed since prepared",
                        words + " -- prepare its group again for its job "
                                "request; its runs are kept",
                        prep_group_lines([n], base=base))
            else:
                step = ("changed since prepared", words,
                        said_lines(redo_launched(g.members[0],
                                                 calc=calc))[1:])
        elif g.missing is not None and live is None and bad is None:
            # A PREPARED MEMBER WHOSE RUN IS NOT ON DISK -- the ready door's
            # answer (`GroupAnswer.missing`) -- against the design, said,
            # launched group or not; its group's prep prepares it again, the
            # launched members' runs kept (`job-system.md`, *The next step*).
            step = (MISSING[0], f"{MISSING[0]}: {g.missing} -- {MISSING[1]}",
                    prep_group_lines([n], base=base))
        elif not g.launched:
            # PREPARED BY ANOTHER MOLBUILDER: said, the launch sending what
            # it wrote (`job-system.md` § 6.0).
            step = (("ready",
                     "ready to launch" + (f" (prepared by {g.prepared_by})"
                                          if g.prepared_by else ""),
                     launch_lines("task", base=base, groups=[n]))
                    if g.launchable else
                    ("waiting", "waits for: " + reason_of(g.why or ""), []))
        else:
            opened = next((r for r in members if r.state == "pending"), None)
            odd = next((r for r in members if r.state != _DONE), None)
            if live is not None:
                step = (live.state, f"{live.state}, let it finish",
                        [command("status", base=base)])
            elif bad is not None and bad.state == "unreadable":
                # A RECORD THAT DOES NOT READ, in its reader's words.
                step = (bad.state, f"{bad.name}: " + reason_of(bad.detail),
                        [])
            elif bad is not None:
                # LAUNCHED AGAIN: a run of points takes over every point
                # done in its latest run (`engines/transport.md` § 2a.11,
                # `engines/vibration.md` § 5.11 § 6); any other stage
                # continues from its own latest run or starts over -- its
                # job's one fact (`Job.relaunch_continues`).
                again = ("it takes over every point done in its own latest "
                         "run, or `--cold` runs them all again"
                         if bad.points else
                         "it continues from its own latest run, or `--cold` "
                         "starts it over" if bad.relaunch_continues else
                         "it starts over, nothing handed on from a run of "
                         "its own")
                # ...OR CHANGED FIRST, its way on the one composer's, whole
                # (`commands.way_back`).
                step = (bad.state,
                        f"{bad.state}: {bad.name} -- launch it again -- "
                        f"{again}; or, to change it first, as below",
                        launch_lines("task", bad.name, base=base)
                        + said_lines(way_back(bad.name, calc=calc,
                                               launched=True)))
            elif opened is not None:
                step = ("pending", f"{opened.name}: {opened.detail}",
                        launch_pending(opened.name, n, opened.attempt,
                                       base=base))
            elif odd is None:
                # FINISHED, AND ONLY THEN: every member's newest run.
                step = (_DONE, _DONE, [])
            else:
                # ANY OTHER STATE, in its row's words -- never *finished*.
                step = (odd.state,
                        f"{odd.name}: {_said(odd.state, odd.detail)}", [])
        out.append(GroupStep(group=n, stages=list(g.members), state=step[0],
                             words=step[1], lines=list(step[2])))
    first = next((i for i, s in enumerate(out) if s.state != _DONE), None)
    if first is not None:
        out[first] = dataclasses.replace(out[first], next=True)
    elif out and task_summary(task) is not None:
        out[-1] = dataclasses.replace(
            out[-1], next=True,
            words="finished -- every group has; gather the results",
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
    # difference between run-0 and run-2 is the whole question.  `-` for a
    # stage with no run opened.
    hdr = ("seq", "stage", "group", "attempt", "state", "converged",
           "warm files", "detail")
    rows = [(s.ref.seq_text, s.ref.name,
             str(s.group) if s.group is not None else "-",
             s.attempt or "-", s.state,
             s.converged or "-", ", ".join(s.warm_files) or "-",
             s.detail + (f" \u00b7 {s.tsgf}" if s.tsgf else "")
             + (f" \u00b7 copied from {s.copied_from}" if s.copied_from
                else ""))
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


def _sweep_took_over(s: StageStatus, base) -> str:
    """What a sweep's run continued from, read at the point: the points
    taken over from an earlier run, named by that run; a run that took none
    over walked every point (the first launch, or ``--cold``)."""
    from ..runrecord import RunRef
    # A POINT COPIED from the finished calculation this one reuses -- taken
    # over from it, or in a run copied in whole (`engines/transport.md`
    # § 2a.16).
    copied = [p["point"] for p in s.points if p.get("copied_from")]
    # TAKEN OVER, as launch decided it (`continuation.taken_over`).
    taken = [(p["point"], p["taken_from"]) for p in s.points
             if taken_over(p) and not p.get("copied_from")]
    if not taken and not copied:
        return "nothing taken over -- every point walked"
    # THE RUN a point was taken over from, the run its point lies in.
    runs = sorted({str(RunRef.read(src).run_folder())
                   for _w, src in taken})
    said = []
    if taken:
        said.append(", ".join(w for w, _s in taken) + " taken over from "
                    + ", ".join(runs))
    if copied:
        said.append(", ".join(copied) + " copied from the finished "
                    "calculation")
    walked = len(s.points) - len(taken) - len(copied)
    return "; ".join(said) + ("; the rest walked" if walked else "")


def _sweep_next(status: JobSetStatus) -> str:
    """A benchmark's next step: its own verbs, for the stage it measures, on
    the calculation (`job-system.md` § 7) -- the trials launch together, the
    ones already launched passed over, and what they measured is read back
    once they have run."""
    from ..parse.dirs.job import FINAL_STATES
    stage, base = status.bench_of, status.base
    read = block([command("summarize", "bench", stage, base=base)])
    states = {s.state for s in status.stages}
    # A TRIAL WHOSE RUN IS NOT ON DISK, OR DOES NOT READ, said by name with
    # its way on (R-L28) -- before any launch, which would refuse it; never
    # *ended*.
    odd = [s for s in status.stages
           if s.state not in FINAL_STATES + ("pending", "queued", "running")]
    if odd:
        return ("; ".join(f"{s.name}: {_said(s.state, s.detail)}"
                          for s in odd)
                + (" -- " + rollback("the benchmark's prep", base=base)
                   if any(s.state == MISSING[0] for s in odd) else ""))
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
    because each run is a folder of its own, and its launch is a record
    rather than an inference from an empty folder (§ 1.5, § 1.6).

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
    runs = ", ".join(attempt_name(n) for n in s.attempts) or "-"
    rows.append(("attempt", f"{s.attempt or '-'}"
                            + (f"   (of {runs})" if len(s.attempts) > 1
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
                     # NOTHING SAID while the run's records do not read.
                     "-" if s.state == "unreadable"
                     else _sweep_took_over(s, status.base) if s.points
                     else src if src
                     else "nothing -- it started from what it gathered"
                     if s.gathered else
                     "nothing -- it started from the structure"))
    else:
        rows.append(("launched", "no  (no launch record -- prepared, not started)"))
    if s.copied_from:
        # A RUN COPIED IN from the finished calculation this one reuses
        # (`engines/transport.md` § 2a.16): a launched run of this one.
        rows.append(("copied from", s.copied_from))
    rows.append(("converged", s.converged or "-"))
    rows.append(("warm files", ", ".join(s.warm_files) or "-"))
    rows.append(("detail", s.detail or "-"))
    # CHANGED SINCE PREPARED, a launched stage's too: its runs computed the
    # description as it was (`job-system.md`, *The task*); the way on is
    # its group's step.
    if s.changed:
        rows.append(("changed since prepared",
                     "; ".join(d.said for d in s.changed)))
    elif s.unrecorded:
        from .prepared_with import UNRECORDED_SAID
        rows.append(("prepared with", UNRECORDED_SAID))
    # A SWEEP'S POINTS, one row each (`engines/transport.md` § 2a.11): done
    # or not done and why, what it started from, what it alone took.
    for p in s.points:
        rows.append((f"  {p['point']}",
                     ("done" if p["done"] else f"not done -- {p['why']}")
                     + (f"; from {p['started_from']}" if p.get("started_from")
                        else f"; {p['start_unread']}" if p.get("start_unread")
                        else f"; copied from {p['copied_from']}"
                        if p.get("copied_from")
                        # NO RECORD OF ITS OWN: it started from what its run
                        # gathered, or what prep wrote -- the run's line's
                        # own rule (`job-system.md` § 5.3).
                        else ("; from what it gathered" if s.gathered
                              else "; from what prep wrote")
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


@dataclass(frozen=True)
class JobAnswer:
    """`molbuilder jobs`' answer for one job launched from this machine
    (`job-system.md` § 6.0a): each member ``(stage, run, said)`` -- the
    stage, its run as a record names it, its state in the status door's
    words -- and, for a sequence, the groups it has not launched yet, as the
    job names them (``{"number", "stages"}``); ``unread``, why its
    calculation does not read, when it does not -- which is then all that is
    known of a group off the job's address."""
    members: List[tuple]
    not_reached: List[dict] = field(default_factory=list)
    unread: Optional[str] = None


def job_members(job) -> JobAnswer:
    """THE STATUS DOOR'S ANSWER FOR A JOB, whole (`job-system.md` § 6.0a):
    each member on the job's address, and for a sequence each group it
    launched -- on its address, or by the runs whose launch records name it
    (`ready.sequence_sent`, the one answer) -- then each group it has not
    reached.  A run answers through its own launch record, by the run door
    (`runs.run_of`, `parse.dirs.run_state_of`): that run, never its
    stage's newest; a run of points, a container no one run of ours speaks
    for, by its points (`continuation.stage_run_state`).  The calculation
    is opened once; one that does not read, and a run no longer on disk,
    are said."""
    from ..calculation import open as open_calculation
    from ..runrecord import RunRef, launch_record_path
    from .ready import sequence_sent
    base = Path(job.calculation)
    calc = None
    unread: Optional[str] = None
    try:
        calc = open_calculation(base)
    except (OSError, ValueError, KeyError) as exc:
        unread = f"its calculation does not read: {exc}"
    members = [(stage, m["run"],
                _member_said(calc, unread, base, stage,
                             base / m["launch_record"]))
               for stage, m in job.members.items()]
    not_reached: List[dict] = []
    for g in (job.sequence or {}).get("groups", []):
        sent = sequence_sent(calc, g["stages"], job)
        if sent is None:
            not_reached.append(g)
            continue
        # SENT BY THE SEQUENCE, AND NOT ON ITS ADDRESS: listed by its runs
        # (on its address, ``{}``: listed above).
        members += [(stage, str(RunRef.at(calc.base, run)),
                     _member_said(calc, unread, base, stage,
                                  launch_record_path(run)))
                    for stage, run in sent.items()]
    return JobAnswer(members, not_reached, unread)


def _member_said(calc, unread: Optional[str], base: Path, stage: str,
                 record: Path) -> str:
    """One member's state, in the status door's words -- its run's, read
    through its launch record ``record`` (:func:`job_members`)."""
    from ..calcdirs import CONTAINER, where
    from ..runs import run_of, state_of
    ran = run_of(record) if record.exists() else None
    if ran is not None:
        st = state_of(ran)
        return _said(st.state, st.detail)
    # WHERE ITS LAUNCH RECORD'S FOLDER IS (`calcdirs.where`): a run of
    # points' run, a container.
    if not record.parent.is_dir() or where(record.parent).role != CONTAINER:
        return ("its run is not on disk" if not record.exists()
                else "its launch record names no run of ours")
    # A RUN OF POINTS: THAT RUN, by the one door for a run's state whichever
    # kind (`continuation.stage_run_state`) -- never its stage's newest.
    named = (next((s.name for s in calc.task.stages
                   if stage_key(s.name) == stage_key(stage)), None)
             if calc is not None else None)
    if named is None:
        return unread or "its description no longer holds it"
    try:
        state, detail = stage_run_state(calc, named, record.parent)
    except LaunchRecordError as exc:
        state, detail = "unreadable", str(exc)
    return _said(state, detail)


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
    Reads names, states and the jobs' files, and removes nothing."""
    from ..joblist import Asked, going_of
    from ..parse.dirs.job import LIVE_STATES
    from .continuation import stage_run_state
    from .materialize import scratch_in
    base = Path(calc.base)
    found = scratch_in(calc)
    if not found:
        return ScratchAnswer([], [], [])
    asked = Asked()
    kept, free = [], []
    for f in found:
        why = None
        # A FILE IN NO RUN is no run's to keep.
        if f.run is not None:
            # A JOB THAT HOLDS THE RUN STILL GOING, or one this machine
            # cannot ask (`joblist.going_of`): its scratch kept.
            going, job, said = going_of(base, f.run, asked)
            if going is not False:
                why = f"job {job.id} -- {said}"
            if why is None:
                try:
                    # THE RUN'S STATE BY ITS KIND'S RULE -- a run of points
                    # by its points (`continuation.stage_run_state`).
                    state, detail = stage_run_state(calc, f.stage, f.run)
                except (OSError, ValueError, KeyError) as exc:
                    why = f"its state does not read: {exc}"
                else:
                    if state in LIVE_STATES:
                        why = state
                    elif state == "unreadable":
                        why = f"its state does not read: {detail}"
        (kept if why else free).append((f, why) if why else f)
    return ScratchAnswer(found, kept, free)


__all__ = ["StageStatus", "JobSetStatus", "jobset_status", "JobAnswer",
           "job_members",
           "run_folders", "every_run", "points_view", "ladder_view",
           "views_of", "run_answer",
           "render_status", "render_stage_status", "ScratchAnswer",
           "scratch_answer"]
