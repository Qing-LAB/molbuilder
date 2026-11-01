"""``run_status`` — how a run directory is doing.

``{state, detail, last_change_at, active_source, concluded}``.

**The status is the ending readers' own answer.**  How a file ended --
``run_state``, ``model/parse.md`` § 2b -- is read for each result file
(every ``.out``, and each ``*.molwatch.log`` whose footer concludes the
run) by its role's reader, ``_run_ending.ending_of``: the reading the
parsers build from.  This module greps no output itself.  How the run's
PROCESS ended -- on its own or not, with what exit code -- is the run
record's one door, ``runrecord.ending``, and the state is built on it
(`execution/architecture.md` § 3.2).

Four things a parser cannot know are settled here, because they are not
in the file:

* **which file speaks for the run** -- its folder holds its stdout and its
  molwatch log;
* **whether it was launched**, before anything is written -- the
  attempt's launch record answers (``launch``) -- **and whether its start
  began**: the start's session log, written as the run script's start goes
  ahead (`project-layout.md` § 1.6.1);
* **whether its process went without a word** -- a forced stop, which its
  job tells (:func:`_job_ended`): the scheduler's own line in the job's
  error file, or the answer about the job written once into its launch
  record -- else the monitor's closing record (:func:`_monitor_ended`);
* **whether the job is still deriving its result after its engine ended**
  -- a job with a finish (`engines/vibration.md` § 5.5), whose run's
  session log says the finish began (:func:`_finish_started`).

A run whose files state no ending, no exit and no such record is
``running`` -- not finished -- however long it has been quiet: nothing in
them tells a slow step from a stopped one (user, 2026-09-26: *"It shows
what it is"*).
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

# IT TRAVELS BESIDE EVERY JOB (`runwrap.MONITOR_COMPANIONS`,
# `execution/run-reports.md` § 2.3): the monitor reports how a run ended with
# `run_status` itself.  So what it reads with is stdlib-only and travels too,
# imported two ways -- from the package, or from beside the job.
try:                                        # inside molbuilder
    # RELATIVE, as every shipped module's is: beside a job where `molbuilder`
    # happens to be importable, an absolute import would bind this file to
    # the package while the monitor reads with the shipped copies -- two
    # versions of one reader in one process.
    from ... import runfiles as _rf
    from ... import runrecord as _rr
    from ... import wrapper_log as _wl
    from ..engines import _run_ending as _re
except ImportError:                         # beside a job, as the monitor's
    import runfiles as _rf
    import runrecord as _rr
    import wrapper_log as _wl
    import _run_ending as _re


# How each file says its run ended is `_run_ending`'s -- one reader per
# role: an `.out` through the SIESTA reading pass the registered parser
# builds its Frames from, a PySCF stdout through its decks' end lines.
# Nothing in this module greps an output: it owns only the two questions no
# single file can answer (which file speaks, and whether it was launched).


# ---- helpers --------------------------------------------------------- #


def _iso_z(ts: float) -> str:
    """Format a POSIX timestamp as an ISO-8601 UTC string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


# ---- file enumeration ------------------------------------------------ #


def _of_run(paths, basename: str) -> List[Path]:
    """Those of ``paths`` that are the run ``basename`` names -- a name that
    reads back under it (`runfiles.parse`)."""
    return [p for p in paths if _rf.parse(p.name, basename) is not None]


def _enumerate_files(run_dir: Path, basename: str) -> Dict[str, List[Path]]:
    """The run-output files of the run ``basename`` names, keyed by ROLE --
    ``{".out": [...], ".pyscf.log": [...], ".molwatch.log": [...]}`` -- and
    sorted by name within each.  Which files are a run's output is the
    catalogue's question (`runfiles.run_output_roles`, `model/parse.md`
    § 5.5, R-RO1).

    ``basename`` is the run's stem: its files are those that read back
    under it -- one set, a run folder holding one start (`project-layout.md`
    § 1.6.1).
    """
    return {role: _of_run(_rf.find_by_role(run_dir, role), basename)
            for role in _rf.run_output_roles()}


#: HOW A RUN THAT DID NOT REACH ITS END IS SAID -- the run script's exit
#: trap's conclusion note, ``rc=<N> at <date>; stopped before its end``, for
#: a failure of the script's own before its marker (`runwrap`,
#: `project-layout.md` § 1.6.3), and the words every state detail of such a
#: run begins with.  Declared here beside the marker's other notes; the
#: wrapper imports it.
STOPPED = "stopped before its end"
#: ...and how a run is said whose job ended before its start began -- no
#: session log (:func:`_started`): a run its group's walk never reached, a
#: job cancelled in its queue.
NEVER_STARTED = "never started"
#: What a conclusion marker says when the job's FINISH failed -- a step the
#: wrapper runs with the finish bundle when the engine alone leaves no
#: result: a SIESTA vibration call's plan step before its engine and its
#: record step after (`engines/vibration.md` § 5.11 § 3).  The wrapper
#: writes ``rc=<N> at <date>; finish failed (<bundle> <verb>)`` and this
#: reader reads the job as failed although the engine's output ended: the
#: engine ended, the job did not.  Declared here, where it is read; the
#: wrapper imports it (`runwrap._finish_block`).
FINISH_FAILED = "finish failed"
#: What a conclusion marker says when the job STOPPED BEFORE ITS ENGINE because
#: its finish cannot run on the job's python -- the check the wrapper makes
#: once its arguments are read, before the engine starts
#: (`runwrap._finish_check_block`, `engines/vibration.md` § 5.5).  Written as
#: ``rc=1 at <date>; finish cannot load (<bundle>)`` with no output beside
#: it, which the marker's own rule already reads as failed; the words say
#: why.  Declared here beside its sibling; the wrapper imports it.
FINISH_CANNOT_LOAD = "finish cannot load"
#: What a conclusion marker says when the job STOPPED BEFORE ITS ENGINE because
#: the activation did not give it its environment's own python -- the one
#: check every job makes once its arguments are read (`runwrap.
#: _env_python_block`, `execution/running-a-job.md` § 2): the activation left
#: no ``CONDA_PREFIX`` naming the job's env, or that env holds no
#: ``bin/python3``.  Written as ``rc=1 at <date>; no env python (<path>)``;
#: the words say why.  Declared here beside its siblings; the wrapper
#: imports it.
NO_ENV_PYTHON = "no env python"


#: The monitor log's closing record of a run whose process has gone
#: (`execution/run-reports.md` § 2.5) -- written by the monitor, which
#: imports it from here, and read by :func:`_monitor_ended`.
MONITOR_ENDED = "[MONITOR] job ended"
#: That record as its line reads -- ``[<time>] [MONITOR] job ended (<why>);
#: ...`` -- its time, the run's end (`run-reports.md` § 2.5), and its words:
#: THE ONE READING of the line, which :func:`_monitor_ended` and the monitor
#: log's parser (`parse/instruments/monitor.py`, ``ended_at``) ask.
MONITOR_ENDED_LINE = re.compile(r"^\[([^\]]+)\]\s+" + re.escape(MONITOR_ENDED)
                                + r"\s*\(([^)]*)\)")


def _monitor_ended(run_dir: Path, basename: str) -> bool:
    """Did the run's monitor see its process go: its log's closing record,
    :data:`MONITOR_ENDED`?

    What a FORCED STOP leaves when the monitor outlives it: a walltime,
    ``scancel`` or a kill never reaches the wrapper's main line, so no
    ``.concluded`` is written, and the output simply stops.  The monitor
    gets SIGTERM from the wrapper's signal trap or the scheduler, or sees
    the watched PID go, and writes this record before it asks how the run
    ended (`run-reports.md` § 2.4).  A node that dies takes the monitor with
    it, and then no file says the run is over.
    """
    for log, _got in _rf.find(run_dir, basename, role=".monitor.log"):
        try:
            with log.open(encoding="utf-8", errors="replace") as fh:
                if any(MONITOR_ENDED_LINE.match(line) for line in fh):
                    return True
        except OSError:
            continue
    return False


def _job_ended(launched_in: Path, launch: Any,
               started: bool = True) -> Optional[str]:
    """How its job says the run ended without a word -- ``None`` where it
    says nothing; ``started``, whether the run's start began, its words
    :data:`STOPPED` or :data:`NEVER_STARTED`.  First the answer written once
    into the launch record by the first reader on the machine that holds
    the job
    (`runrecord.record_ended`, `running-a-job.md` § 4.2, *Asked once,
    written once*), there wherever the folder is copied: a sent job's
    scheduler's, a run here's process gone with no marker written
    (`runs.concluded`).  Then, for a sent job, how the scheduler says it
    ended the job -- *stopped by the scheduler: <its words> (<its file>)*
    -- read from the job's error file, found through the file catalogue
    (`runfiles.find_fixed`, `SCHEDULER_STDERR`: its kind and the job's id
    read off its name) in the folder its header put it in: the one the
    launch record names, else ``launched_in``, the folder the record lies
    in -- and read by its kind's reader (`scheduler.md` R16).

    Asked only by a caller holding the launch record -- molbuilder's
    readers of a run, never the monitor beside a job, which carries no
    scheduler (`scheduler/__init__`) and passes none.
    """
    if not isinstance(launch, dict):
        return None
    rec = launch.get("ended")
    jid = launch.get("job_id")
    if isinstance(rec, dict):
        on = rec.get("asked_on") or "its machine"
        at = rec.get("asked_at") or "?"
        lead = STOPPED if started else NEVER_STARTED
        unrecorded = "; no exit recorded" if started else ""
        if launch.get("mode") != "submit":
            return (f"{lead}: its process on {on} ended"
                    + (" without recording how" if started else "")
                    + f" (asked at {at})")
        if rec.get("said"):
            return (f"the scheduler listed job {jid} ended: {rec['said']} "
                    f"(asked on {on} at {at})")
        return (f"{lead}: the scheduler no longer listed job {jid} (asked "
                f"on {on} at {at}){unrecorded}")
    if launch.get("mode") != "submit" or not jid:
        return None
    from ...scheduler.queue import ended
    stated = (launch.get("scheduler") or {}).get("stderr")
    folder = (launched_in / stated).parent if stated else launched_in
    for path, fields in _rf.find_fixed(folder, _rf.SCHEDULER_STDERR):
        if fields.get("jobid") != str(jid):
            continue
        words = ended(fields["scheduler"], path, str(jid))
        if words:
            return (f"stopped by the scheduler: {words} "
                    f"({os.path.relpath(path, launched_in)})")
    return None


def _the_jobs_own(name: str, basename: str) -> bool:
    """Whether ``name`` is a file THE JOB writes -- its outputs, the
    engine's files, its wrapper's and monitor's logs: a file of this run's
    label (`runfiles.parse`; a run's folder holds its one stage's files)
    whose catalogue row a run writes, or a run output's (`runfiles.
    Artifact.when`, ``output``: the progress log prep seeds and the run
    writes into), or an engine's own file the catalogue has no row for.
    What prep and launch put in the run -- its deck, its run script and
    header, its report -- and molbuilder's own records of it, ``run.json``,
    ``.continued-from``, ``.gathered-from`` (fixed names on no label), are
    never the job's writes (`running-a-job.md` § 4.2, *Past its wall*)."""
    label = _rf.split_stem(basename)[0]
    if _rf.parse(name, label) is None:
        return False
    row = _rf.row_for(name, label)
    return row is None or row.when == "run" or row.output is not None


def _newest_write(run_dir: Path, basename: str) -> Optional[float]:
    """When the job last wrote -- the newest mtime over the files of
    ``run_dir`` that are the job's own (:func:`_the_jobs_own`), none below
    it; what `rsync -a` carries with the folder.  ``None`` with no such
    file."""
    newest = None
    try:
        with os.scandir(run_dir) as it:
            for entry in it:
                try:
                    if (entry.is_file(follow_symlinks=False)
                            and _the_jobs_own(entry.name, basename)):
                        t = entry.stat(follow_symlinks=False).st_mtime
                        newest = t if newest is None or t > newest else newest
                except OSError:
                    continue
    except OSError:
        return None
    return newest


def _started(run_dir: Path, basename: str) -> bool:
    """Has the run's start begun -- does its folder hold the start's
    session log (`wrapper_log.log_of_run`), which the run script opens as
    its start goes ahead (`project-layout.md` § 1.6.1)?"""
    return _wl.log_of_run(run_dir, *_rf.split_stem(basename)) is not None


def _ago(seconds: float) -> str:
    """``2 h 10 min``, ``35 min``, ``20 s`` -- how long ago, in words."""
    s = int(max(seconds, 0))
    if s < 60:
        return f"{s} s"
    h, m = divmod(s // 60, 60)
    d, h = divmod(h, 24)
    return " ".join(w for w in (f"{d} d" if d else "", f"{h} h" if h else "",
                                f"{m} min" if m or not (d or h) else "") if w)


def _past_wall(run_dir: Path, basename: str, launch: Any) -> Optional[str]:
    """A SENT RUN PAST ITS WALL, wherever it is read (`running-a-job.md`
    § 4.2, *Past its wall*): the job cannot run longer than the wall its
    launch record says it was sent with (`runrecord.sent_wall_s`), and
    writes only while it runs -- so a run whose newest own write
    (:func:`_newest_write`) is older than that wall is over, said in these
    words; ``None`` for a run here (no wall), one still within it, or one
    that has written nothing of its own yet.  Asked of a run with no output
    only once its start began (:func:`_started`): a job still waiting in its
    queue writes nothing.  A job its scheduler requeued
    is the one case the folder cannot tell: the machine it was sent to
    asks the scheduler for such a run (`runs.concluded`)."""
    if not isinstance(launch, dict) or launch.get("mode") != "submit":
        return None
    wall = _rr.sent_wall_s(launch)
    newest = _newest_write(run_dir, basename)
    if wall is None or newest is None:
        return None
    if time.time() - newest <= wall:
        return None
    return (f"{STOPPED}: nothing was written for longer than "
            f"its wall ({(launch.get('placed_on') or {}).get('time')}; last "
            f"write {_iso_z(newest)}), so the job is over; how it ended was "
            f"not recorded")


def _quiet_for(run_dir: Path, basename: str, launch: Any) -> str:
    """``· last wrote 2 h 10 min ago`` for a sent run still reading
    `running` (`running-a-job.md` § 4.2): the one fact a folder gives
    where its scheduler cannot be asked -- the job's own last write
    (:func:`_newest_write`); empty for a run here."""
    if not isinstance(launch, dict) or launch.get("mode") != "submit":
        return ""
    newest = _newest_write(run_dir, basename)
    if newest is None:
        return ""
    return f" \u00b7 last wrote {_ago(time.time() - newest)} ago"


def _finish_failed(concluded: Optional[str]) -> bool:
    """Does the marker say the job's finish failed?  Read through the one
    reader of the line (`runrecord.read_concluded`), never by a second
    match."""
    got = _rr.read_concluded(concluded) if concluded else None
    return bool(got) and str(got.get("note", "")).startswith(FINISH_FAILED)


# ---- how each result file ENDED ------------------------------------- #


def _output_endings(paths: List[Path], run_dir: Path,
                    basename: str) -> "Dict[str, _re.RunEnding]":
    """Each run-output file's ending, by filename — ONE loop, one door.

    `ending_of` dispatches on the ROLE (`model/parse.md` § 5.5), so a new
    run-output row is read here without this loop changing.

    A SIESTA output is read with its run's session log as the run's
    stderr (:func:`_stderr_of`), as the wrapper's own question reads it.

    FAIL-SOFT ON A READ, and only on a read: a file that cannot be opened
    contributes nothing rather than taking the walk down.  A file whose ROLE
    has no reader is NOT absorbed -- `ending_of` raises, and it cannot happen
    here because the roles come from the same catalogue view its `READERS`
    are checked against.
    """
    endings: "Dict[str, _re.RunEnding]" = {}
    for path in paths:
        try:
            endings[path.name] = _re.ending_of(
                path, stderr=_stderr_of(run_dir, path, basename))
        except OSError:
            continue
    return endings


def _finish_started(run_dir: Path, path: Path, basename: str) -> bool:
    """Did the run that wrote ``path`` begin its job's finish -- does its
    session log (the one :func:`_stderr_of` finds) record it
    (`wrapper_log.FINISH_STARTED`)?  Asked only of an output that ended with
    no conclusion yet, so the log is read in that one case."""
    log = _stderr_of(run_dir, path, basename)
    return log is not None and _wl.finish_started(log)


def _stderr_of(run_dir: Path, path: Path, basename: str) -> Optional[Path]:
    """Where the run that wrote ``path`` sent its stderr, when it kept it
    apart: a SIESTA-family output's session log (`wrapper_log.log_of_run`),
    the one the wrapper's own ending question reads (`_mb_ending
    --stderr`).  SIESTA's ``die`` flushes stdout on node 0 alone, so a rank
    other than 0 that dies may say why only there.  ``None`` for any other
    output -- a PySCF log takes its stderr in -- and for a run with no log.
    ``basename`` is the run's stem."""
    got = _rf.parse(path.name, basename)
    if got is None or got.role not in _rf.stdout_roles("siesta"):
        return None
    return _wl.log_of_run(run_dir, *_rf.split_stem(basename))


# ---- status + progress ---------------------------------------------- #


#: The verdicts `run_status` can reach.  A CLOSED set, and the enforcement is
#: `RunStatus.__post_init__` below -- nothing in this repo type-checks, so the
#: annotation alone would refuse nothing.
#: ``pending`` and ``queued`` are the states before its start began --
#: never launched, and launched and not started (no session log) -- which a
#: caller holding the attempt's launch record gets (`run_status`'s
#: ``launch``).
#: ``unreadable`` is a run whose launch record (`run.json`) does not read,
#: or -- the run door's -- one an older molbuilder named (`runs.older_run`):
#: its state cannot be told, so it is said, the file named in its detail,
#: and never followed.
#: ``not run`` is a point of a run of points its ended walk never started
#: (`_run_ending.NOT_RUN`): ended, and never ``failed`` -- it did not run.
RUN_STATES: "tuple[str, ...]" = ("pending", "queued", "running",
                                 "finished", "failed", _re.NOT_RUN,
                                 "unreadable")

#: The states in which more can still arrive -- launched and not over.  What a
#: Results viewer follows (`web/results.md` § 4.1): a run never launched writes
#: nothing until it is, and one finished or failed writes nothing more.
LIVE_STATES: "tuple[str, ...]" = ("queued", "running")
#: The states a run has ended in -- what a loop is said *stopped* in
#: (`model/parse.md` § 5d.5); `unreadable` is neither live nor over.
FINAL_STATES: "tuple[str, ...]" = ("finished", "failed", _re.NOT_RUN)


@dataclass(frozen=True)
class RunStatus:
    """How a run directory is doing: the parser's verdict, plus what no
    single file can answer.

    `state` is one of :data:`RUN_STATES`; `active_source` names the file that
    spoke for the directory and is `None` when no result file exists yet.
    """
    state:          str
    detail:         str
    last_change_at: "Optional[str]" = None
    active_source:  "Optional[str]" = None
    #: What the run's PROCESS said on its way out -- ``rc=0 at <date>`` --
    #: or None if it never said goodbye: the line of
    #: the one door's answer (`runrecord.ending`), which the state is built
    #: on.
    #:
    #: Read by the monitor's closing report (`run-reports.md` § 2.3) and the
    #: run record's exit (`record._concluded`).
    concluded:      "Optional[str]" = None
    #: ``failed`` by the wall alone (`running-a-job.md` § 4.2, *Past its
    #: wall*): the folder's newest own write older than the wall the job
    #: was sent with, nothing else recorded -- what `runs.concluded` asks
    #: the scheduler about where it can, since a requeued job reads so too.
    by_wall:        bool = False
    #: Every run-output file's ending (`_run_ending.RunEnding`: how it ended,
    #: whether each SCF phase converged, the error), by filename -- the ONE
    #: scan the state above was judged from.  Readers: the run record's
    #: verdict (`model/parse.md` § 5d.6), and the monitor's closing report
    #: -- so none of them scans the files again.
    endings:        "Dict[str, Any]" = field(default_factory=dict)
    #: THE RUN'S OWN OUTPUTS that may speak, by filename -- its stdout, and
    #: its progress log once its footer concludes (`runfiles.Artifact.
    #: output`): one run, its files aspects of it, whose statements
    #: :attr:`verdict` reads together.
    speakers:       "Tuple[str, ...]" = ()
    #: WHETHER ITS START BEGAN -- its session log is there (:func:`_started`,
    #: `project-layout.md` § 1.6.1) -- read once with the state: what tells
    #: a point its walk never reached from one that ran (`runs._as_a_point`,
    #: `continuation.point_rows`).
    started:        bool = False

    def __post_init__(self) -> None:
        if self.state not in RUN_STATES:
            raise ValueError(
                f"RunStatus.state must be one of {RUN_STATES}; "
                f"got {self.state!r}")

    @property
    def verdict(self) -> "Optional[Tuple[str, bool]]":
        """What the run says it converged -- ``("geometry", ok)`` for a
        relaxation, else ``("SCF", ok)`` -- or ``None`` when it says neither
        yet: the one scan's answer (:attr:`endings`), which a reader decides
        on (:attr:`converged_words` says it).  The speaking output's word
        first, then each other output of the same run's (:attr:`speakers`):
        a PySCF relaxation states its verdict in its progress log's footer,
        `# relaxed:`, beside a stdout that states none (`model/parse.md`
        § 2b, P-S1)."""
        if not self.active_source:
            return None
        ends = self.endings or {}
        return _re.verdict_of(ends[n] for n in (self.active_source,
                                                *self.speakers)
                              if n in ends)

    @property
    def converged_words(self) -> "Optional[str]":
        """The verdict in the words a row shows it -- ``geometry yes``,
        ``SCF NO`` -- or ``None`` when the output says neither yet.  For
        showing; a reader that decides asks :attr:`verdict`."""
        return _re.verdict_words(self.verdict)


#: A caller that does not hold the attempt's launch record -- the monitor,
#: beside a job that is running.  Distinct from ``None``, which is the record
#: saying the attempt was never launched.
_UNASKED = object()


def run_status(run_dir, basename: str, *,
               launch: Any = _UNASKED,
               launched_in=None) -> "RunStatus":
    """How is this run doing?  ``{state, detail, last_change_at,
    active_source}``.

    ``launch`` is the attempt's launch record, ``run.json``
    (`runrecord.launch_record`): ``None`` when it was never
    launched.  It is what tells *never launched* from *launched, not
    started yet* before its start began (`project-layout.md` § 1.6) --
    ``pending`` and ``queued`` -- and a caller that holds it passes it.
    ``launched_in`` is the folder that record lies in, where it is not
    ``run_dir``: a run of points' one record, in the run's folder, answers
    for each point in it, and the scheduler's file it names is named from
    there (`scheduler.md` R16).

    **The state is built on two doors, each asked once**: how the run's
    PROCESS ended -- `runrecord.ending`, did it end on its own and with what
    exit code (`execution/architecture.md` § 3.2) -- and how each of its
    files ended -- ``run_state``, ``model/parse.md`` § 2b, read by its
    role's reader (``_run_ending.ending_of``).  This settles what neither
    can alone:

    * **which file speaks for the run.**  Its folder holds its stdout and
      its molwatch log; a parser sees one file and cannot pick.  The stdout,
      where the run printed one (`model/parse.md` § 5.1).

      ``basename`` says WHICH RUN is being asked about -- its stem
      (`runfiles.RunNames.stem`).
    """
    run_dir = Path(run_dir)
    files = _enumerate_files(run_dir, basename)
    endings = _output_endings(
        [p for role in _rf.run_output_roles() for p in files[role]], run_dir,
        basename)
    states = {name: e.run_state or "unknown" for name, e in endings.items()}
    messages = {name: e.error_message for name, e in endings.items()}
    doings = {name: e.doing for name, e in endings.items()}
    # WHICH OF THEM MAY SPEAK is the catalogue's `output` column, not a rule
    # written here.  A "stdout" file exists because the PROCESS started, so it
    # counts whether or not it ended; a "progress" file is SEEDED at prep, so
    # it counts only once its footer concludes -- otherwise a stage's seed
    # outvotes its own result (`model/parse.md` § 5.5, § 5.1).
    speaks = set(_rf.stdout_roles())
    speakers = [p for role in _rf.run_output_roles() for p in files[role]
                if role in speaks or states.get(p.name) in _re.CONCLUDED]
    began = _started(run_dir, basename)
    return replace(_build_status(
        speakers,
        states,
        _rr.ending(run_dir, basename),
        launch=launch,
        job_ended=lambda started=True: _job_ended(
            Path(launched_in) if launched_in is not None else run_dir,
            launch, started),
        monitor_ended=lambda: _monitor_ended(run_dir, basename),
        past_wall=lambda: _past_wall(run_dir, basename, launch),
        quiet_for=lambda: _quiet_for(run_dir, basename, launch),
        started=lambda: began,
        out_messages=messages,
        out_doing=doings,
        finish_started=lambda p: _finish_started(run_dir, p, basename)),
        endings=endings, speakers=tuple(p.name for p in speakers),
        started=began)




#: The detail of a run whose process went with no ending in its output and
#: no exit recorded -- a forced stop, told by the monitor's closing record.
_STOPPED_UNRECORDED = (f"{STOPPED}: no ending in its output "
                       "and no exit recorded")


def _build_status(out_paths: List[Path],
                  out_run_states: Dict[str, str],
                  end: "Optional[_rr.Ending]" = None,
                  launch: Any = _UNASKED,
                  job_ended: Callable[..., Optional[str]] = lambda *_: None,
                  monitor_ended: Callable[[], bool] = lambda: False,
                  past_wall: Callable[[], Optional[str]] = lambda: None,
                  quiet_for: Callable[[], str] = lambda: "",
                  started: Callable[[], bool] = lambda: False,
                  out_messages: Optional[Dict[str, Optional[str]]] = None,
                  out_doing: Optional[Dict[str, Optional[str]]] = None,
                  finish_started: Callable[[Path], bool] = lambda _p: False,
                  ) -> "RunStatus":
    """Build the status envelope per § 5, over the directory's RESULT
    files — every ``"stdout"`` run output plus each ``"progress"`` one whose
    footer concludes (`runfiles.Artifact.output`, `model/parse.md` § 5.5) —
    and how the run's PROCESS ended (``end``, `runrecord.ending`).

    **Finished is the door's answer: the run ended on its own with exit
    code 0** -- and *failed* when it ended on its own with any other
    (`execution/architecture.md` § 3.2, plan W38 F3).  An output that says
    how it ended speaks beside it: the engine's end, its error, its memory
    running out -- and an output saying the engine stopped is a failure
    whatever followed.  **An output that ended is not a run that ended**:
    the job may still be on its way out -- a finish deriving its result
    after the engine (`engines/vibration.md` § 5.5), the wrapper's own last
    lines -- so it is ``running`` until it concludes, and ``failed`` once
    its job says it ended (``job_ended``: the scheduler's own line,
    `scheduler.md` R16, or the answer recorded once) or the monitor's
    closing record says the process went without concluding
    (``monitor_ended``) -- each asked only then, the job's first.  Where
    nothing says anything the run is ``running`` -- not finished --
    however long it has been quiet (`running-a-job.md` § 4.2)
    -- unless its folder's newest write is older than the wall its job was
    sent with (``past_wall``, *Past its wall*): then it is over, and
    ``failed`` says how it was not recorded.  A run still ``running`` says
    where it is in the reading pass's words (``out_doing``,
    `RunEnding.doing`),
    and a sent one how long since the job last wrote (``quiet_for``).

    **Before any output**, the same evidence in the same order -- its
    conclusion, its job, its monitor -- and then **whether its start
    began** (``started``, the session log): a start with no output yet is
    ``running`` (*started, no output yet*), or ``failed`` past its wall; a
    run whose start has not begun is ``pending`` or ``queued`` by its launch
    record (`running-a-job.md` § 4.2, *The order of evidence*).

    ``out_paths`` are the files that may SPEAK, the run's stdout speaking
    where it printed one; ``last_change_at`` is the speaker's mtime, paired
    with ``active_source`` beside it.
    """
    end = end if end is not None else _rr.Ending()
    said = end.line
    if not out_paths:
        # No output at all: the conclusion is the whole answer.
        if end.concluded:
            return RunStatus(
                state=("finished" if end.ok else "failed"),
                detail=(f"concluded ({said})" if end.ok
                        else f"concluded ({said}) before any output"),
                concluded=said)
        began = started()
        stopped = job_ended(began)
        if stopped is not None:
            return RunStatus(state="failed", detail=stopped)
        if monitor_ended():
            return RunStatus(
                state="failed",
                detail=f"{_STOPPED_UNRECORDED}, before any output")
        # ITS START BEGAN -- the session log -- AND NOTHING OF ITS OUTPUT
        # YET: running, until its wall has passed since the job last wrote
        # (`_past_wall`).
        if began:
            over = past_wall()
            if over:
                return RunStatus(state="failed", detail=over, by_wall=True)
            return RunStatus(state="running",
                             detail="started, no output yet" + quiet_for())
        # NOT STARTED, and the launch record says which nothing
        # (`project-layout.md` § 1.6): never launched is ``pending``,
        # launched and not started is ``queued``.
        if launch is not _UNASKED:
            if launch is None:
                return RunStatus(state="pending",
                                 detail="prepared, not launched (no launch record)")
            jid = launch.get("job_id")
            return RunStatus(state="queued", detail=(
                f"queued as job {jid}" if jid else
                f"launched ({launch.get('mode') or '?'}), not started yet"))
        return RunStatus(state="running", detail="no result file yet")
    # THE ACTIVE SOURCE: the run's stdout, which exists because its process
    # started; its concluded progress log where it printed none
    # (`model/parse.md` § 5.1).
    stdout = set(_rf.stdout_roles())
    active = next((p for p in out_paths if _rf.role_of(p.name) in stdout),
                  out_paths[0])
    active_state = out_run_states.get(active.name, "unknown")
    ended = active_state == "ended"
    over: Optional[str] = None

    # The engine parser reports how the OUTPUT ended from markers alone --
    # "running"|"ended"|"stopped"|"out_of_memory"|"unknown".  Note what is
    # NOT consulted: whether the SCF converged.  That is P-S2's reported
    # fact, carried beside this state for the reader to show, never folded
    # into it.
    if active_state in ("out_of_memory", "stopped"):
        # WHAT STOPPED IT, in the run's own words: the line its ending
        # reader kept -- from the output, or from the stderr a rank other
        # than 0 died on (`_stderr_of`).
        said_why = (out_messages or {}).get(active.name)
        state = "failed"
        detail = ("out of memory" if active_state == "out_of_memory"
                  else STOPPED)
        if said_why:
            detail += f": {said_why}"
    elif ended and _finish_failed(said):
        # THE ENGINE ENDED, THE JOB DID NOT: its step after the engine --
        # the kind's own step of a point (`execution/runs-of-points.md`
        # § 1.2) -- failed, and the marker says so in words no engine
        # teardown writes (`project-layout.md` § 1.6.3).
        state = "failed"
        detail = (f"the engine ended, but its step after the engine failed "
                  f"({said}); the session log says why")
    elif end.concluded:
        if end.ok:
            state, detail = "finished", ("ended, exit 0" if ended
                                         else f"concluded ({said})")
        else:
            state, detail = "failed", (
                f"the engine's output ended, but the job exited with an "
                f"error ({said})" if ended else f"concluded ({said})")
    else:
        # NOT ENDED ON ITS OWN: what its job says first -- the scheduler's
        # own line, the answer recorded once; neither depends on a signal
        # reaching anything (`scheduler.md` R16) -- then the monitor's
        # closing record.
        stopped = job_ended()
        if stopped is not None:
            state, detail = "failed", stopped
        elif ended and finish_started(active):
            # THE ENGINE ENDED AND THE JOB'S FINISH BEGAN, and the wrapper
            # has not concluded: the finish is deriving the result now -- or
            # the job was stopped inside it (a walltime, a kill), which
            # leaves no marker and whose monitor saw the process go.
            # Neither is finished: a force-constant run without its spectrum
            # has not finished.
            over = None if monitor_ended() else past_wall()
            if monitor_ended() or over:
                state = "failed"
                detail = ("the engine ended and the job's finish began, but "
                          "the job stopped before it concluded -- no result "
                          "was derived; the session log says how far it got"
                          + (f" ({over})" if over else ""))
            else:
                state = "running"
                detail = ("the engine ended; the job's finish is deriving "
                          "the result")
        elif ended:
            # THE ENGINE ENDED AND THE JOB HAS NOT CONCLUDED: on its way out
            # -- or stopped there, which its monitor saw.  Not finished
            # either way: nothing builds on a run that did not end on its
            # own.
            over = None if monitor_ended() else past_wall()
            if monitor_ended() or over:
                state = "failed"
                detail = ("the engine's output ended, but the job stopped "
                          "before it concluded -- no exit recorded"
                          + (f" ({over})" if over else ""))
            else:
                state = "running"
                detail = ("the engine's output ended; the job has not "
                          "concluded")
        elif monitor_ended():
            # Content and the conclusion are silent, and the monitor saw the
            # process go: a forced stop (`_monitor_ended`).
            state, detail = "failed", _STOPPED_UNRECORDED
        else:
            # NOTHING SAYS ANYTHING: running -- unless the job's own wall
            # has passed since the folder was last written (`_past_wall`);
            # and a sent run says how long it has been quiet (`_quiet_for`).
            over = past_wall()
            if over:
                state, detail = "failed", over
            else:
                doing = (out_doing or {}).get(active.name)
                state, detail = "running", ("running"
                                            + (f" \u00b7 {doing}" if doing
                                               else "") + quiet_for())

    return RunStatus(state=state, detail=detail,
                     last_change_at=_iso_z(active.stat().st_mtime),
                     active_source=active.name,
                     concluded=said,
                     by_wall=bool(state == "failed" and over))


