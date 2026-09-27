"""``run_status`` — how a run directory is doing.

``{state, detail, last_change_at, active_source, concluded}``.

**The status is the ending readers' own answer.**  How a file ended --
``run_state``, ``model/parse.md`` § 2b -- is read for each result file
(every ``.out``, and each ``*.molwatch.log`` whose footer concludes the
run) by its role's reader, ``_run_ending.ending_of``: the reading the
parsers build from.  This module greps no output itself.

Three things a parser cannot know are settled here, because they are not
in the file:

* **which file speaks for the directory** -- a folder holds one ``.out``
  per run index and one molwatch log per stage;
* **whether it was launched**, before anything is written -- the
  attempt's launch record answers (``launch``);
* **whether its process went without a word** -- a forced stop, which the
  monitor's closing record tells (:func:`_monitor_ended`).

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

import re
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
    from ...identity import parse_stage_token
    from ..engines import _run_ending as _re
except ImportError:                         # beside a job, as the monitor's
    import runfiles as _rf
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


def _enumerate_files(run_dir: Path, match: str = "*") -> Dict[str, List[Path]]:
    """The run-output files of the rung ``match`` names, keyed by ROLE --
    ``{".out": [...], ".pyscf.log": [...], ".molwatch.log": [...]}`` -- and
    sorted by name within each.  Which files are a run's output is the
    catalogue's question (`runfiles.run_output_roles`, `model/parse.md`
    § 5.5, R-RO1).

    ``match`` NARROWS THE DIRECTORY TO ONE RUNG, and in the flat shape that is
    the whole question: every stage of a flat calculation shares one directory
    and is told apart by FILENAME (`project-layout.md` § 1), so a bucket built
    from the whole directory answers about all of them at once.  The caller
    passes `Shape.stage_glob(token, label)`; ``"*"`` is the hierarchical
    answer, where the directory has already selected the stage.
    """
    narrowed = {c for c in run_dir.glob(match) if c.is_file()}
    return {role: sorted(p for p in _rf.find_by_role(run_dir, role)
                         if p in narrowed)
            for role in _rf.run_output_roles()}


#: SIESTA's own end-of-run marker: a FILE whose existence is the signal, and
#: whose name is SIESTA's, not ours.  A literal is forced -- it is in no
#: `runfiles.WRITTEN` row, so `find_by_role` refuses it, the same as `.XV`.
_ENGINE_EXIT_MARKER = "0_NORMAL_EXIT"


def _rung_files(run_dir: Path, role: str, match: str = "*") -> List[Path]:
    """The ``role`` files of the rung ``match`` names (`_enumerate_files`'s
    narrowing, for one role)."""
    narrowed = {c.name for c in run_dir.glob(match)}
    return [f for f in _rf.find_by_role(run_dir, role) if f.name in narrowed]


def _at_latest_run(run_dir: Path, files: List[Path]) -> List[Path]:
    """Those of ``files`` that belong to their label's LATEST run, newest
    first.

    The rule is `attempt_concluded`'s: a per-run file counts only at the
    HIGHEST index any per-run artifact of its label reached, across every
    role.  An earlier index's file beside a newer run's is a previous
    re-run's -- its goodbye, or its monitor's -- and says nothing about the
    run that followed it.
    """
    kept = []
    for f in files:
        label = _label_of_run_file(f.name)
        got = _rf.parse(f.name, label)
        idx = getattr(got, "run", None) if got else None
        newest = _rf.latest_run(run_dir, label)
        if newest is not None and idx is not None and idx < newest:
            continue
        kept.append(((idx if idx is not None else -1, f.stat().st_mtime), f))
    return [f for _key, f in sorted(kept, key=lambda k: k[0], reverse=True)]


def _process_conclusion(run_dir: Path, match: str = "*") -> Optional[str]:
    """Did this run's PROCESS get to say goodbye, and with what?

    ``"rc=0"`` / ``"rc=1 (walltime)"`` from the wrapper's marker, the literal
    ``"0_NORMAL_EXIT"`` when only the engine's is there, ``None`` when
    nothing concluded.

    Content answers *did the science finish*; this answers *did the run end
    on its own*, which an output cannot say about itself.  The wrapper
    writes its marker on the main path, so an engine error reaches it and a
    kill never does.  An engine that dies before printing leaves a marker
    and no output at all -- the case content cannot see.
    """
    marks = _rung_files(run_dir, ".concluded", match)
    if marks:
        latest = _at_latest_run(run_dir, marks)
        if not latest:
            return None                  # a previous attempt's goodbye
        try:
            return latest[0].read_text(encoding="utf-8").strip() or "rc=?"
        except OSError:
            return None
    # SIESTA's own marker carries NO LABEL, so it cannot be attributed to a
    # rung.  In the flat shape every stage shares one directory, so consulting
    # it while narrowed would let one rung's clean exit answer for all of them.
    if match == "*" and (run_dir / _ENGINE_EXIT_MARKER).is_file():
        return _ENGINE_EXIT_MARKER
    return None


#: The monitor log's closing record of a run whose process has gone
#: (`execution/run-reports.md` § 2.5) -- written by the monitor, which
#: imports it from here, and read by :func:`_monitor_ended`.  A warm retry's
#: ``[MONITOR] stopped`` is not one: the job goes on in the next run.
MONITOR_ENDED = "[MONITOR] job ended"


def _monitor_ended(run_dir: Path, match: str = "*") -> bool:
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
    for log in _at_latest_run(run_dir,
                              _rung_files(run_dir, ".monitor.log", match)):
        try:
            with log.open(encoding="utf-8", errors="replace") as fh:
                if any(MONITOR_ENDED in line for line in fh):
                    return True
        except OSError:
            continue
    return False


def _label_of_run_file(name: str) -> str:
    """The label a per-run file, ``<label>-run<N>.<role>``, carries.

    `runfiles.parse` needs the label to read a name back, and a marker or a
    monitor log is met before anything has said whose it is.  The counter
    keyword comes from `runfiles.QUALIFIERS`.

    *(This spelled `"-run"` behind an `isinstance(QUALIFIERS, dict)` guard
    that is always False -- `QUALIFIERS` is a tuple -- so the literal was
    always used while the docstring claimed otherwise.)*
    """
    cut = name.rfind("-" + _rf.QUALIFIERS[0])
    return name[:cut] if cut > 0 else name


def read_concluded(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """The conclusion marker's first line, ``rc=<N> at <when>`` as the wrapper
    writes it (`runwrap.py`), as ``{"code": N, "at": when}`` -- ``at`` only
    when stated -- or ``None`` when the text is not one (SIESTA's own
    ``0_NORMAL_EXIT``, which carries no code).  THE one reader of that line.
    """
    head = text.splitlines()[0] if text else ""
    m = re.search(r"\brc=(-?\d+)(?:\s+at\s+(.*?))?\s*$", head)
    if m is None:
        return None
    return {"code": int(m.group(1)),
            **({"at": m.group(2)} if m.group(2) else {})}


def _rc_ok(concluded: str) -> bool:
    """Did the process end successfully, from the marker's own text?

    The wrapper writes ``rc=<N> at <date>`` (`runwrap.py`), so the rc has to
    be PARSED, not string-equalled.  Measured 2026-09-18: 19 of the 20 markers
    in the checkout carry the date, and an exact test against ``"rc=0"``
    matched only the one hand-made fixture -- reporting every real successful
    conclusion as a failure.
    """
    if concluded == _ENGINE_EXIT_MARKER:
        return True
    got = read_concluded(concluded)
    return got is not None and got["code"] == 0


# ---- how each result file ENDED ------------------------------------- #
#
# (Headed "plots from .out files" until 2026-09-18.  No plot has been built
# here since 2026-09-04 -- building them and throwing them away to reach one
# field is what got the decoder deleted, as this module's docstring says.)


def _output_endings(paths: List[Path]) -> "Dict[str, _re.RunEnding]":
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

    FAIL-SOFT ON A READ, and only on a read: a file that cannot be opened
    contributes nothing rather than taking the walk down.  A file whose ROLE
    has no reader is NOT absorbed -- `ending_of` raises, and it cannot happen
    here because the roles come from the same catalogue view its `READERS`
    are checked against.
    """
    endings: "Dict[str, _re.RunEnding]" = {}
    for path in paths:
        try:
            endings[path.name] = _re.ending_of(path)
        except OSError:
            continue
    return endings


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
    #: What the run's PROCESS said on its way out -- "rc=0",
    #: "rc=1 (walltime)", "0_NORMAL_EXIT" -- or None if it never said
    #: goodbye.  Reported BESIDE the state, not folded into it.
    #:
    #: Read by the monitor's closing report (`run-reports.md` § 2.3), which
    #: asks about this rung's own files.  *(It had no reader from the day the
    #: Transport tab was reverted off it: `classify_citation` asks about a
    #: DECK and this answers about a DIRECTORY -- measured, a neighbour rung's
    #: marker reported for a citation that concluded cleanly.)*
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


def run_status(run_dir, match: str = "*", *,
               launch: Any = _UNASKED) -> "RunStatus":
    """How is this run doing?  ``{state, detail, last_change_at,
    active_source}``.

    ``launch`` is the attempt's launch record, ``run.json``
    (`jobset.materialize.read_run_launch`): ``None`` when it was never
    launched.  It is what tells *never launched* from *launched, nothing
    written yet* before any output exists (`project-layout.md` § 1.6) --
    ``pending`` and ``queued`` -- and a caller that holds it passes it.

    **The status IS the ending readers' answer**, plus what no single file
    can know.  How each file ended -- ``run_state``, ``model/parse.md``
    § 2b -- is read by its role's reader (``_run_ending.ending_of``), and
    this settles what a single file cannot:

    * **which file speaks for the directory.**  A folder holds one
      ``.out`` per run index and one molwatch log per stage; a parser
      sees one file and cannot pick.  Highest stage, newest mtime.

      ``match`` says WHICH RUNG is being asked about, and without it this
      answered about whichever rung ran last.  The caller's own existence
      check was already shape-aware -- `runstatus._stage_state` narrows with
      `Shape.stage_glob` and its comment says why -- but the call through to
      here passed no filter, so in the flat shape (one directory, every
      stage) a finished rung reported the newest rung's state.  Measured
      2026-09-08: with a later stage's `.out` present, a finished rung
      read the later rung's "running".

    Callers wanted exactly this and had to take it out of an
    eleven-field summary: ``decode_run_dir`` answered ``status`` plus
    ten fields with no reader anywhere, and reached the per-file
    run-states by building every PLOT and discarding them.
    """
    run_dir = Path(run_dir)
    files = _enumerate_files(run_dir, match)
    endings = _output_endings(
        [p for role in _rf.run_output_roles() for p in files[role]])
    states = {name: e.run_state or "unknown" for name, e in endings.items()}
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
        _process_conclusion(run_dir, match),
        launch=launch,
        monitor_ended=lambda: _monitor_ended(run_dir, match)),
        endings=endings)




#: The detail of a run whose process went with no ending in its output and
#: no exit recorded -- a forced stop, told by the monitor's closing record.
_STOPPED_UNRECORDED = ("stopped before its end: no ending in its output "
                       "and no exit recorded")


def _build_status(out_paths: List[Path],
                  out_run_states: Dict[str, str],
                  concluded: Optional[str] = None,
                  launch: Any = _UNASKED,
                  monitor_ended: Callable[[], bool] = lambda: False,
                  ) -> "RunStatus":
    """Build the status envelope per § 5, over the directory's RESULT
    files — every ``"stdout"`` run output plus each ``"progress"`` one whose
    footer concludes (`runfiles.Artifact.output`, `model/parse.md` § 5.5) —
    and the run's PROCESS conclusion.

    **Content first, process second.**  An output that states how it ended
    is the strongest evidence and keeps the answer it always gave.  The
    marker speaks where content is silent; the monitor's closing record
    (``monitor_ended``, asked only then) where the marker is silent too --
    a forced stop.  Where nothing says anything the run is ``running`` --
    not finished -- however long it has been quiet (`running-a-job.md`
    § 4.2).

    ``out_paths`` are the files that may SPEAK; ``last_change_at`` is the
    speaker's mtime, paired with ``active_source`` beside it.
    """
    if not out_paths:
        # No output at all: the marker is the whole answer.
        if concluded is not None:
            rc_ok = _rc_ok(concluded)
            return RunStatus(
                state=("finished" if rc_ok else "failed"),
                detail=(f"concluded ({concluded}) before any output"
                        if not rc_ok else f"concluded ({concluded})"),
                concluded=concluded)
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

    state = "running"
    detail = "running"
    # The engine parser reports how the run ENDED from markers alone --
    # "running"|"ended"|"stopped"|"out_of_memory"|"unknown" -- and a file
    # with no ending marker is honestly "running": not finished, however
    # long it has been quiet (`running-a-job.md` § 4.2).
    #
    # Note what is NOT consulted: whether the SCF converged.  That is
    # P-S2's reported fact, carried beside this state for the reader to
    # show, never folded into it.
    if active_state == "ended":
        state, detail = "finished", "job_completed"
    elif active_state == "out_of_memory":
        state, detail = "failed", "out of memory"
    elif active_state == "stopped":
        state, detail = "failed", "stopped before its end -- see the .out"
    elif concluded is not None and concluded != _ENGINE_EXIT_MARKER:
        # Content is silent, the process is not: the run is over and the
        # marker says how.
        #
        # THE ENGINE'S OWN MARKER IS EXCLUDED, because it cannot be
        # ATTRIBUTED.  A `.concluded` carries a label and a `-run<N>`, so
        # `_process_conclusion` can refuse a previous attempt's goodbye;
        # `0_NORMAL_EXIT` is a bare filename carrying neither, so a leftover
        # from an earlier attempt promoted a silent output to `finished` --
        # measured on a copy of `BDT-withAuJunction`, and undone with the
        # marker removed.
        # `_process_conclusion` already refuses it on the STAGE axis when
        # narrowed, for the same reason; this is the ATTEMPT axis.
        #
        # It is still REPORTED (the `concluded` field below) and still
        # decides where there is no output at all to contradict it, which is
        # the case only a marker can see.  Mtime cannot stand in for the
        # index: measured over the 31 real marker directories, SIESTA writes
        # it up to 8 s BEFORE the output's final write.
        if _rc_ok(concluded):
            state, detail = "finished", f"concluded ({concluded})"
        else:
            state, detail = "failed", f"concluded ({concluded})"
    elif monitor_ended():
        # Content and the marker are silent, and the monitor saw the
        # process go: a forced stop (`_monitor_ended`).
        state, detail = "failed", _STOPPED_UNRECORDED

    return RunStatus(state=state, detail=detail,
                     last_change_at=_iso_z(active.stat().st_mtime),
                     active_source=active.name,
                     concluded=concluded)


