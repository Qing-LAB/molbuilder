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

* **which file speaks for the directory** -- a folder holds one ``.out``
  per run index and one molwatch log per stage;
* **whether it was launched**, before anything is written -- the
  attempt's launch record answers (``launch``);
* **whether its process went without a word** -- a forced stop, which the
  monitor's closing record tells (:func:`_monitor_ended`);
* **whether the job is still deriving its result after its engine ended**
  -- a job with a finish (`engines/vibration.md` § 5.5), whose run's
  session log says the finish began (:func:`_finish_started`).

A run whose files state no ending, no exit and no such record is
``running`` -- not finished -- however long it has been quiet: nothing in
them tells a slow step from a stopped one (user, 2026-09-26: *"It shows
what it is"*).

*(Until 2026-09-04 this module was a ``JobDirParser`` returning an
eleven-field ``JobResult``: job type, system label, geometry, plots,
progress, a source-file index, a per-stage input summary, diagnostics.
Measured across the tree, ten of the eleven had no reader anywhere, and
the eleventh -- this one -- was obtained by parsing every ``.out`` to
build plot data and then throwing the plots away.  1,414 lines produced
one field that was used.  The dead half is deleted rather than fixed;
four code-quality defects went with it, including a second
``LatticeConstant`` reader that disagreed with its sibling on units and
a second ``SystemLabel`` regex that returned a different answer.)*
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional

# IT TRAVELS BESIDE EVERY JOB (`runwrap.MONITOR_COMPANIONS`,
# `execution/run-reports.md` § 2.3): the monitor reports how a run ended with
# `run_status` itself.  So what it reads with is stdlib-only and travels too,
# imported two ways -- from the package, or from beside the job -- as
# `config_dir` always has been.
try:                                        # inside molbuilder
    # RELATIVE, as every shipped module's is: beside a job where `molbuilder`
    # happens to be importable, an absolute import would bind this file to
    # the package while the monitor reads with the shipped copies -- two
    # versions of one reader in one process.
    from ... import runfiles as _rf
    from ... import runrecord as _rr
    from ... import wrapper_log as _wl
    from ...identity import parse_stage_token
    from ..engines import _run_ending as _re
except ImportError:                         # beside a job, as the monitor's
    import runfiles as _rf
    import runrecord as _rr
    import wrapper_log as _wl
    from identity import parse_stage_token
    import _run_ending as _re


# How each file says its run ended is `_run_ending`'s -- one reader per
# role: an `.out` through the SIESTA reading pass the registered parser
# builds its Frames from, a PySCF stdout through its decks' end lines.
# Nothing in this module greps an output: it owns only the two questions no
# single file can answer (which file speaks, and whether it was launched).
#
# *(Those three lines used to end "(enforced by the engine parsers own
# it)" -- two half-sentences spliced -- and cited
# `test_no_direct_out_grep_in_decoder` as the enforcing test "until
# 2026-09-05" one line before saying it retired on 2026-09-04.  The lint's
# whole body was `assert src.count("read_text") < 8`; it went with the
# decoder, and nothing replaced it because the rule is structural now.)*

# (`cg_step_milestone` had a threshold constant here.  The constant went
# with the decoder on 2026-09-04; this comment did not, and `cg_step_milestone`
# now occurs exactly once in the tree -- in the sentence naming it.)


# ---- helpers --------------------------------------------------------- #


def _detect_stage(filename: str) -> Optional[int]:
    """A file's stage ORDINAL, or ``None`` when it carries no stage token.

    The token is ``<NN>_<name>`` (``bdt_au_01_coarse.fdf``) and this returns
    the ``NN``.  Read through :func:`molbuilder.identity.parse_stage_token`,
    which is the one place the shape is written down -- the decoder used to
    carry its own ``-stage(N)`` regex, a second spelling of the emitter's
    convention that could and did drift from it.

    **Still an int, and deliberately so.**  Decision 27 kept the ordinal in
    the filename, so ordering by stage stays possible: this function is the
    first component of the active-file sort key in :func:`run_status`
    (stage first, mtime second), which is what makes a re-run of an earlier
    stage stop hijacking the run's reported state.

    *This cited ``_anchor_sort_key`` and "the ``stage`` field of the
    engine-input envelope" as the other downstream orderers until
    2026-09-05.  Neither exists: the envelope went with the run decoder on
    2026-09-04, and ``_anchor_sort_key`` has never been defined anywhere in
    the tree -- it appeared only in this sentence.*  Had the token carried
    the name alone, the anchor rule would have lost its sort key and the
    Results tab its notion of "the active stage"; that is the trap
    ``staged-runs-implementation-plan.md`` § 8d walked into and § 8e closed.
    """
    hit = parse_stage_token(filename)
    return hit[0] if hit else None


def _iso_z(ts: float) -> str:
    """Format a POSIX timestamp as an ISO-8601 UTC string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


# ---- file enumeration ------------------------------------------------ #


def _of_run(paths, basename: Optional[str]) -> List[Path]:
    """Those of ``paths`` that are the run ``basename`` names -- a name that
    reads back under it (`runfiles.parse`) -- or all of them without one."""
    return [p for p in paths
            if basename is None or _rf.parse(p.name, basename) is not None]


def _enumerate_files(run_dir: Path,
                     basename: Optional[str] = None) -> Dict[str, List[Path]]:
    """The run-output files of the run ``basename`` names, keyed by ROLE --
    ``{".out": [...], ".pyscf.log": [...], ".molwatch.log": [...]}`` -- and
    sorted by name within each.  Which files are a run's output is the
    catalogue's question (`runfiles.run_output_roles`, `model/parse.md`
    § 5.5, R-RO1).

    ``basename`` NARROWS THE DIRECTORY TO ONE RUN, and in the flat shape that
    is the whole question: every stage of a flat calculation shares one
    directory and is told apart by FILENAME (`project-layout.md` § 1), so a
    bucket built from the whole directory answers about all of them at once.
    The caller passes `Shape.run_basename(token, label)` -- ``None`` in the
    hierarchy, where the directory has already selected the run.  *(A glob,
    ``<stem>*``, until 2026-10-03, when the run's ending came to be asked of
    one door by its name.)*
    """
    return {role: _of_run(_rf.find_by_role(run_dir, role), basename)
            for role in _rf.run_output_roles()}


#: What a conclusion marker says when the job's FINISH failed -- the step the
#: wrapper runs after the engine when the engine alone leaves no result (a
#: SIESTA force-constant run's modes, `engines/vibration.md` § 5.5).  The
#: wrapper writes ``rc=<N> at <date>; finish failed (<bundle>)`` and this
#: reader reads the job as failed although the engine's output ended: the
#: engine ended, the job did not.  Declared here, where it is read; the
#: wrapper imports it (`runwrap._finish_block`).
FINISH_FAILED = "finish failed"
#: What a conclusion marker says when the job STOPPED BEFORE ITS ENGINE because
#: its finish cannot run on the job's python -- the check the wrapper makes
#: once the run index is known, before the engine starts
#: (`runwrap._finish_check_block`, `engines/vibration.md` § 5.5).  Written as
#: ``rc=1 at <date>; finish cannot load (<bundle>)`` with no output beside
#: it, which the marker's own rule already reads as failed; the words say
#: why.  Declared here beside its sibling; the wrapper imports it.
FINISH_CANNOT_LOAD = "finish cannot load"


#: The monitor log's closing record of a run whose process has gone
#: (`execution/run-reports.md` § 2.5) -- written by the monitor, which
#: imports it from here, and read by :func:`_monitor_ended`.  A warm retry's
#: ``[MONITOR] stopped`` is not one: the job goes on in the next run.
MONITOR_ENDED = "[MONITOR] job ended"


def _monitor_ended(run_dir: Path, basename: Optional[str] = None) -> bool:
    """Did the latest run's monitor see its process go -- its log's closing
    record, :data:`MONITOR_ENDED`?

    What a FORCED STOP leaves when the monitor outlives it: a walltime,
    ``scancel`` or a kill never reaches the wrapper's main line, so no
    ``.concluded`` is written, and the output simply stops.  The monitor
    gets SIGTERM from the wrapper's signal trap or the scheduler, or sees
    the watched PID go, and writes this record before it asks how the run
    ended (`run-reports.md` § 2.4).  A node that dies takes the monitor with
    it, and then no file says the run is over.
    """
    for log in _rf.at_latest_run(
            run_dir, _of_run(_rf.find_by_role(run_dir, ".monitor.log"),
                             basename)):
        try:
            with log.open(encoding="utf-8", errors="replace") as fh:
                if any(MONITOR_ENDED in line for line in fh):
                    return True
        except OSError:
            continue
    return False


def _finish_failed(concluded: Optional[str]) -> bool:
    """Does the marker say the job's finish failed?  Read through the one
    reader of the line (`runrecord.read_concluded`), never by a second
    match."""
    got = _rr.read_concluded(concluded) if concluded else None
    return bool(got) and str(got.get("note", "")).startswith(FINISH_FAILED)


# ---- how each result file ENDED ------------------------------------- #
#
# (Headed "plots from .out files" until 2026-09-18.  No plot has been built
# here since 2026-09-04 -- building them and throwing them away to reach one
# field is what got the decoder deleted, as this module's docstring says.)


def _output_endings(paths: List[Path],
                    run_dir: Path) -> "Dict[str, _re.RunEnding]":
    """Each run-output file's ending, by filename — ONE loop, one door.

    This was two functions, `_out_conclusions` and `_molwatch_conclusions`,
    each hard-wired to one role and each knowing that role's reader.  Adding
    a third role meant adding a third function, which is why `.pyscf.log`
    never got one: `ending_of` dispatches on the ROLE (`model/parse.md`
    § 5.5), so a new run-output row is read here without this loop changing.

    **Through the cheap door.**  Both halves went through
    ``detect(path).parse(path)`` until 2026-09-18 -- a whole Trajectory built
    and discarded to reach one string -- which also opened a ``ParseLogger``
    per file, so merely LOOKING at a folder created and grew a ``.parse.log``
    inside the user's project directory (measured: 540 B after one
    ``run_status``, 1080 B after two, over 67 logs).

    A SIESTA output is read with its run's session log as the run's
    stderr (:func:`_stderr_of`), as the wrapper's own question reads it.

    FAIL-SOFT ON A READ, and only on a read: a file that cannot be opened
    contributes nothing rather than taking the walk down.  A file whose ROLE
    has no reader is NOT absorbed -- `ending_of` raises, and it cannot happen
    here because the roles come from the same catalogue view its `READERS`
    are checked against.
    """
    endings: "Dict[str, _re.RunEnding]" = {}
    logs: Dict[str, Dict[Any, Path]] = {}
    for path in paths:
        try:
            endings[path.name] = _re.ending_of(
                path, stderr=_stderr_of(run_dir, path, logs))
        except OSError:
            continue
    return endings


def _finish_started(run_dir: Path, path: Path) -> bool:
    """Did the run that wrote ``path`` begin its job's finish -- does its
    session log (the one :func:`_stderr_of` finds) record it
    (`wrapper_log.FINISH_STARTED`)?  Asked only of an output that ended with
    no conclusion yet, so the log is read in that one case."""
    log = _stderr_of(run_dir, path, {})
    return log is not None and _wl.finish_started(log)


def _stderr_of(run_dir: Path, path: Path,
               logs: Dict[str, Dict[Any, Path]]) -> Optional[Path]:
    """Where the run that wrote ``path`` sent its stderr, when it kept it
    apart: a SIESTA-family output's session log -- the log whose first
    section is that run (`wrapper_log.logs_by_run`), the one the wrapper's
    own ending question reads (`_mb_ending --stderr`).  SIESTA's ``die``
    flushes stdout on node 0 alone, so a rank other than 0 that dies may say
    why only there.  ``None`` for any other output -- a PySCF log takes its
    stderr in -- and for a run with no log.  ``logs`` holds each label's
    table for the rest of this walk."""
    if _rf.role_of(path.name) != ".out":
        return None
    label = _rf.label_of_run_file(path.name)
    got = _rf.parse(path.name, label)
    if got is None or got.run is None:
        return None
    if label not in logs:
        logs[label] = _wl.logs_by_run(run_dir, label)
    return logs[label].get((got.stage, got.run))


# ---- status + progress ---------------------------------------------- #


#: The verdicts `run_status` can reach.  A CLOSED set, and the enforcement is
#: `RunStatus.__post_init__` below -- nothing in this repo type-checks, so the
#: annotation alone would refuse nothing.  Until 2026-09-09 the function
#: returned `Dict[str, Any]` and a test asserted `s["state"] in (all four)`,
#: which passes whatever the code returns.
#: ``pending`` and ``queued`` are the states before anything is written --
#: never launched, and launched and silent -- which a caller holding the
#: attempt's launch record gets (`run_status`'s ``launch``).
RUN_STATES: "tuple[str, ...]" = ("pending", "queued", "running",
                                 "finished", "failed")

#: The states in which more can still arrive -- launched and not over.  What a
#: Results viewer follows (`web/results.md` § 4.1): a run never launched writes
#: nothing until it is, and one finished or failed writes nothing more.
LIVE_STATES: "tuple[str, ...]" = ("queued", "running")


@dataclass(frozen=True)
class RunStatus:
    """How a run directory is doing: the parser's verdict, plus what no
    single file can answer.

    `state` is one of :data:`RUN_STATES`; `active_source` names the file that
    spoke for the directory (highest stage, newest mtime) and is `None` when
    no result file exists yet.
    """
    state:          str
    detail:         str
    last_change_at: "Optional[str]" = None
    active_source:  "Optional[str]" = None
    #: What the run's PROCESS said on its way out -- ``rc=0 at <date>``,
    #: ``0_NORMAL_EXIT`` -- or None if it never said goodbye: the line of
    #: the one door's answer (`runrecord.ending`), which the state is built
    #: on.
    #:
    #: Read by the monitor's closing report (`run-reports.md` § 2.3) and the
    #: run record's exit (`record._concluded`).
    concluded:      "Optional[str]" = None
    #: Every run-output file's ending (`_run_ending.RunEnding`: how it ended,
    #: whether each SCF phase converged, the error), by filename -- the ONE
    #: scan the state above was judged from.  Readers: the run record's
    #: verdict and its earlier runs (`model/parse.md` § 5d.6), and the
    #: monitor's closing report -- so none of them scans the files again.
    endings:        "Dict[str, Any]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in RUN_STATES:
            raise ValueError(
                f"RunStatus.state must be one of {RUN_STATES}; "
                f"got {self.state!r}")


#: A caller that does not hold the attempt's launch record -- the monitor,
#: beside a job that is running.  Distinct from ``None``, which is the record
#: saying the attempt was never launched.
_UNASKED = object()


def run_status(run_dir, basename: Optional[str] = None, *,
               launch: Any = _UNASKED) -> "RunStatus":
    """How is this run doing?  ``{state, detail, last_change_at,
    active_source}``.

    ``launch`` is the attempt's launch record, ``run.json``
    (`runrecord.launch_record`): ``None`` when it was never
    launched.  It is what tells *never launched* from *launched, nothing
    written yet* before any output exists (`project-layout.md` § 1.6) --
    ``pending`` and ``queued`` -- and a caller that holds it passes it.

    **The state is built on two doors, each asked once**: how the run's
    PROCESS ended -- `runrecord.ending`, did it end on its own and with what
    exit code (`execution/architecture.md` § 3.2) -- and how each of its
    files ended -- ``run_state``, ``model/parse.md`` § 2b, read by its
    role's reader (``_run_ending.ending_of``).  This settles what neither
    can alone:

    * **which file speaks for the directory.**  A folder holds one
      ``.out`` per run index and one molwatch log per stage; a parser
      sees one file and cannot pick.  Highest stage, newest mtime.

      ``basename`` says WHICH RUN is being asked about -- the deck's stem,
      in a folder every stage of a flat calculation shares -- and without it
      this answered about whichever rung ran last: measured 2026-09-08, with
      a later stage's `.out` present a finished rung read the later rung's
      "running".

    Callers wanted exactly this and had to take it out of an
    eleven-field summary: ``decode_run_dir`` answered ``status`` plus
    ten fields with no reader anywhere, and reached the per-file
    run-states by building every PLOT and discarding them.
    """
    run_dir = Path(run_dir)
    files = _enumerate_files(run_dir, basename)
    endings = _output_endings(
        [p for role in _rf.run_output_roles() for p in files[role]], run_dir)
    states = {name: e.run_state or "unknown" for name, e in endings.items()}
    messages = {name: e.error_message for name, e in endings.items()}
    # WHICH OF THEM MAY SPEAK is the catalogue's `output` column, not a rule
    # written here.  A "stdout" file exists because the PROCESS started, so it
    # counts whether or not it ended; a "progress" file is SEEDED at prep, so
    # it counts only once its footer concludes -- otherwise a stage's seed
    # outvotes its own result (`model/parse.md` § 5.5, § 5.1).  That pair of
    # rules was two hand-written functions until 2026-09-18, and the seed half
    # was the only one that said WHY.
    speaks = set(_rf.stdout_roles())
    return replace(_build_status(
        [p for role in _rf.run_output_roles() for p in files[role]
         if role in speaks or states.get(p.name) in _re.CONCLUDED],
        states,
        _rr.ending(run_dir, basename),
        launch=launch,
        monitor_ended=lambda: _monitor_ended(run_dir, basename),
        out_messages=messages,
        finish_started=lambda p: _finish_started(run_dir, p)),
        endings=endings)




#: The detail of a run whose process went with no ending in its output and
#: no exit recorded -- a forced stop, told by the monitor's closing record.
_STOPPED_UNRECORDED = ("stopped before its end: no ending in its output "
                       "and no exit recorded")


def _build_status(out_paths: List[Path],
                  out_run_states: Dict[str, str],
                  end: "Optional[_rr.Ending]" = None,
                  launch: Any = _UNASKED,
                  monitor_ended: Callable[[], bool] = lambda: False,
                  out_messages: Optional[Dict[str, Optional[str]]] = None,
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
    the monitor's closing record says the process went without concluding
    (``monitor_ended``, asked only then).  *(Until 2026-10-03 an ended
    output read finished with no conclusion at all, so status said finished
    where every hand-over refused the run, and finished beside an exit code
    of 1.)*  Where nothing says anything the run is ``running`` -- not
    finished -- however long it has been quiet (`running-a-job.md` § 4.2).

    ``out_paths`` are the files that may SPEAK; ``last_change_at`` is the
    speaker's mtime, paired with ``active_source`` beside it.
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
        if monitor_ended():
            return RunStatus(
                state="failed",
                detail=f"{_STOPPED_UNRECORDED}, before any output")
        # NOTHING WRITTEN YET, and the launch record says which nothing
        # (`project-layout.md` § 1.6): never launched is ``pending``,
        # launched and silent is ``queued`` -- the words the jobset layer
        # used for them above this door until 2026-09-26, while this door
        # answered the same directory "running".
        if launch is not _UNASKED:
            if launch is None:
                return RunStatus(state="pending",
                                 detail="prepped, not launched (no run.json)")
            jid = launch.get("job_id")
            return RunStatus(state="queued", detail=(
                f"queued as job {jid}" if jid else
                f"launched ({launch.get('mode') or '?'}), no output yet"))
        return RunStatus(state="running", detail="no result file yet")
    # Active source = highest stage, latest mtime.
    sorted_outs = sorted(
        out_paths,
        key=lambda p: (_detect_stage(p.name) or 0, p.stat().st_mtime),
    )
    active = sorted_outs[-1]
    active_state = out_run_states.get(active.name, "unknown")
    ended = active_state == "ended"

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
                  else "stopped before its end")
        if said_why:
            detail += f": {said_why}"
    elif ended and _finish_failed(said):
        # THE ENGINE ENDED, THE JOB DID NOT: its finish -- the step that
        # derives the result from what the engine left -- failed, and the
        # marker says so in words no engine teardown writes
        # (`project-layout.md` § 1.6.3).
        state = "failed"
        detail = (f"the engine ended, but the job's finish did not derive "
                  f"the result ({said}); the session log says why")
    elif end.concluded:
        if end.ok:
            state, detail = "finished", ("job_completed" if ended
                                         else f"concluded ({said})")
        else:
            state, detail = "failed", (
                f"the engine's output ended, but the job exited with an "
                f"error ({said})" if ended else f"concluded ({said})")
    elif ended and finish_started(active):
        # THE ENGINE ENDED AND THE JOB'S FINISH BEGAN, and the wrapper has
        # not concluded: the finish is deriving the result now -- or the job
        # was stopped inside it (a walltime, a kill), which leaves no marker
        # and whose monitor saw the process go.  Neither is finished: a
        # force-constant run without its spectrum has not finished.
        if monitor_ended():
            state = "failed"
            detail = ("the engine ended and the job's finish began, but the "
                      "job stopped before it concluded -- no result was "
                      "derived; the session log says how far it got")
        else:
            state = "running"
            detail = ("the engine ended; the job's finish is deriving the "
                      "result")
    elif ended:
        # THE ENGINE ENDED AND THE JOB HAS NOT CONCLUDED: on its way out --
        # or stopped there, which its monitor saw.  Not finished either way:
        # nothing builds on a run that did not end on its own.
        if monitor_ended():
            state = "failed"
            detail = ("the engine's output ended, but the job stopped before "
                      "it concluded -- no exit recorded")
        else:
            state = "running"
            detail = "the engine's output ended; the job has not concluded"
    elif monitor_ended():
        # Content and the conclusion are silent, and the monitor saw the
        # process go: a forced stop (`_monitor_ended`).
        state, detail = "failed", _STOPPED_UNRECORDED
    else:
        state, detail = "running", "running"

    return RunStatus(state=state, detail=detail,
                     last_change_at=_iso_z(active.stat().st_mtime),
                     active_source=active.name,
                     concluded=said)


