"""``run_status`` — how a run directory is doing.

``{state, detail, last_change_at, active_source, concluded}``.

**The status is the parsers' own answer.**  Every engine parser already
reports how its file ended -- ``run_state``, ``model/parse.md`` § 2b --
so this asks the registry for each result file (every ``.out``, and each
``*.molwatch.log`` whose footer concludes the run: the engine-neutral
end-of-run marker, and the only one a PySCF attempt has).  It never
opens an engine output directly.

Two things a parser cannot know are settled here, because they are not
in the file:

* **which file speaks for the directory** -- a folder holds one ``.out``
  per run index and one molwatch log per stage;
* **staleness** -- no ending marker and no growth is a dead job, not a
  slow one, and only the filesystem can tell those apart.

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
from dataclasses import dataclass
from typing import Dict, List, Optional

from molbuilder.identity import parse_stage_token


# Run-state detection is fully delegated to the engine trajectory
# parsers (detect().parse() -> traj.run_state); the end-of-run and
# failure markers live in engines/siesta.py + engines/pyscf.py, NOT
# here.  Nothing in this module greps .out content: the engine parsers
# own how a run ended, and this module owns only the two questions no
# single file can answer (which file speaks, and staleness).
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
    """Bucket relevant files in the dir by kind.

    Returns {"fdf": [...], "out": [...], "xv": [...], "struct_out": [...],
             "molstruct_json": [...], "ani": [...], "molwatch": [...]}.
    Paths sorted by name within each bucket.

    ``match`` NARROWS THE DIRECTORY TO ONE RUNG, and in the flat shape that is
    the whole question: every stage of a flat calculation shares one directory
    and is told apart by FILENAME (`project-layout.md` § 1), so a bucket built
    from the whole directory answers about all of them at once.  The caller
    passes `Shape.stage_glob(token, label)`; ``"*"`` is the hierarchical
    answer, where the directory has already selected the stage.
    """
    from molbuilder.runfiles import find_by_role

    by_kind: Dict[str, List[Path]] = {
        "fdf": [], "out": [], "xv": [], "struct_out": [],
        "molstruct_json": [], "ani": [], "molwatch": [],
    }
    # THE NARROWING FIRST, because it is `match`'s whole job and no role
    # search takes a glob: this is the set of files this rung owns.
    narrowed = {c for c in run_dir.glob(match) if c.is_file()}

    # ROLES THE CATALOGUE DECLARES come from the catalogue.  These three
    # were spelled `name.endswith(".fdf")` here until 2026-09-18 -- the role
    # vocabulary written outside the module that declares it, which is the
    # exact case `runfiles.find_by_role` says it exists to end, and which
    # every sibling in this package converted on 2026-09-08
    # (`atom_metadata.py`, `contract.py`, `rundir.py`).
    for role, bucket in ((".fdf", "fdf"), (".out", "out"),
                         (".molwatch.log", "molwatch")):
        by_kind[bucket] = sorted(p for p in find_by_role(run_dir, role)
                                 if p in narrowed)

    # ...AND THE ENGINE'S OWN OUTPUTS STAY LITERAL, because molbuilder
    # declares no vocabulary for them: `.XV`, `.STRUCT_OUT` and `.ANI` are
    # SIESTA's names for SIESTA's files and appear in no `runfiles.WRITTEN`
    # row (`projects.py` states the boundary).  Asking `find_by_role` for
    # one would be refused, rightly.
    for suffix, bucket in ((".XV", "xv"), (".STRUCT_OUT", "struct_out"),
                           (".molstruct.json", "molstruct_json"),
                           (".ANI", "ani")):
        by_kind[bucket] = sorted(p for p in narrowed
                                 if p.name.endswith(suffix))
    return by_kind


#: SIESTA's own end-of-run marker: a FILE whose existence is the signal, and
#: whose name is SIESTA's, not ours.  A literal is forced -- it is in no
#: `runfiles.WRITTEN` row, so `find_by_role` refuses it, the same as `.XV`.
_ENGINE_EXIT_MARKER = "0_NORMAL_EXIT"


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
    from molbuilder.runfiles import find_by_role, latest_run, parse as rf_parse
    narrowed = {c.name for c in run_dir.glob(match)}
    marks = [m for m in find_by_role(run_dir, ".concluded")
             if m.name in narrowed]
    if marks:
        # THE INDEX IS ASKED FOR, and the rule is `attempt_concluded`'s: the
        # marker counts only at the HIGHEST index any per-run artifact
        # reached, across every role.  An earlier index's marker beside a
        # newer unconcluded `.out` is a previous re-run's goodbye.
        best = None
        for m in marks:
            got = rf_parse(m.name, _label_of_marker(m.name))
            idx = getattr(got, "run", None) if got else None
            newest = latest_run(run_dir, _label_of_marker(m.name))
            if newest is not None and idx is not None and idx < newest:
                continue                 # a previous attempt's goodbye
            key = (idx if idx is not None else -1, m.stat().st_mtime)
            if best is None or key > best[0]:
                best = (key, m)
        if best is not None:
            try:
                return best[1].read_text(encoding="utf-8").strip() or "rc=?"
            except OSError:
                return None
        return None
    # SIESTA's own marker carries NO LABEL, so it cannot be attributed to a
    # rung.  In the flat shape every stage shares one directory, so consulting
    # it while narrowed would let one rung's clean exit answer for all of them.
    if match == "*" and (run_dir / _ENGINE_EXIT_MARKER).is_file():
        return _ENGINE_EXIT_MARKER
    return None


def _label_of_marker(name: str) -> str:
    """The label a `<label>-run<N>.concluded` carries.

    `runfiles.parse` needs the label to read a name back, and a marker is the
    one artifact we meet before anything has said whose it is.  The counter
    keyword comes from `runfiles.QUALIFIERS`.

    *(This spelled `"-run"` behind an `isinstance(QUALIFIERS, dict)` guard
    that is always False -- `QUALIFIERS` is a tuple -- so the literal was
    always used while the docstring claimed otherwise.)*
    """
    from molbuilder.runfiles import QUALIFIERS
    cut = name.rfind("-" + QUALIFIERS[0])
    return name[:cut] if cut > 0 else name


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
    head = concluded.splitlines()[0] if concluded else ""
    m = re.search(r"\brc=(-?\d+)", head)
    return m is not None and int(m.group(1)) == 0


# ---- how each result file ENDED ------------------------------------- #
#
# (Headed "plots from .out files" until 2026-09-18.  No plot has been built
# here since 2026-09-04 -- building them and throwing them away to reach one
# field is what got the decoder deleted, as this module's docstring says.)


def _molwatch_conclusions(mw_paths: List[Path]) -> Dict[str, str]:
    """The CONCLUDED molwatch logs' run-states, by filename.

    A molwatch log is the engine-neutral end-of-run channel
    (``running-a-job.md`` § 4): its writer appends a conclusion footer
    when the run ends, so a log carrying one is a result file and its
    run-state counts.  One without a footer is a live view — a prep-time
    seed, or a run still going — and is deliberately NOT in the answer:
    feeding it into the state would let a stage's seed outvote its own
    ``.out``.  Fail-soft like the ``.out`` path: a log that cannot be read
    simply contributes nothing.

    **Through the cheap door**, like ``_out_conclusions`` beside it.  This
    went through ``detect(path).parse(path)`` until 2026-09-18 -- a whole
    Trajectory built and discarded to reach one string -- which also opened
    a ``ParseLogger`` per log, so every Watch poll created and grew a
    ``.parse.log`` inside the user's project directory.  Measured: 540 B
    after one ``run_status``, 1080 B after two, over 67 logs.
    """
    from molbuilder.parse.engines._run_ending import CONCLUDED
    from molbuilder.parse.engines.molwatch import scan_conclusion
    states: Dict[str, str] = {}
    for path in mw_paths:
        try:
            state = scan_conclusion(path)
        except (OSError, ValueError):
            continue
        if state in CONCLUDED:
            states[path.name] = state
    return states


# ---- status + progress ---------------------------------------------- #


#: The verdicts `run_status` can reach.  A CLOSED set, and the enforcement is
#: `RunStatus.__post_init__` below -- nothing in this repo type-checks, so the
#: annotation alone would refuse nothing.  Until 2026-09-09 the function
#: returned `Dict[str, Any]` and a test asserted `s["state"] in (all four)`,
#: which passes whatever the code returns.
RUN_STATES: "tuple[str, ...]" = ("running", "stale", "finished", "failed")


@dataclass(frozen=True)
class RunStatus:
    """How a run directory is doing: the parser's verdict, plus the two
    things no single file can answer.

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
    #: **NO PRODUCTION READER TODAY.**  Added for the Transport tab, which
    #: was then reverted off it: `classify_citation` asks about a DECK
    #: (`attempt_concluded(dir, deck.stem)`) and this answers about a
    #: DIRECTORY -- measured, a neighbour rung's marker reported for a
    #: citation that concluded cleanly.  The state machine above still uses
    #: the evidence, so it is not dead; the FIELD is unread.
    concluded:      "Optional[str]" = None

    def __post_init__(self) -> None:
        if self.state not in RUN_STATES:
            raise ValueError(
                f"RunStatus.state must be one of {RUN_STATES}; "
                f"got {self.state!r}")


def run_status(run_dir, match: str = "*") -> "RunStatus":
    """How is this run doing?  ``{state, detail, last_change_at,
    active_source}``.

    **The status IS the parser's answer**, plus the two things no parser
    can know.  Every engine parser already reports how its file ended --
    ``run_state`` on the result, ``model/parse.md`` § 2b -- so this asks
    them and then settles the two questions a single file cannot:

    * **which file speaks for the directory.**  A folder holds one
      ``.out`` per run index and one molwatch log per stage; a parser
      sees one file and cannot pick.  Highest stage, newest mtime.

      ``match`` says WHICH RUNG is being asked about, and without it this
      answered about whichever rung ran last.  The caller's own existence
      check was already shape-aware -- `runstatus._stage_state` narrows with
      `Shape.stage_glob` and its comment says why -- but the call through to
      here passed no filter, so in the flat shape (one directory, every
      stage) a finished rung reported the newest rung's state.  Measured
      2026-09-08: with a later stage's `.out` present a stale rung read
      "running"; with that one file moved aside, "stale".
    * **staleness.**  A file with no ending marker is honestly
      "running" -- nothing IN it separates a slow DFT step from a job
      the scheduler killed.  Only the filesystem can, so the age check
      lives here (``_build_status``).

    Callers wanted exactly this and had to take it out of an
    eleven-field summary: ``decode_run_dir`` answered ``status`` plus
    ten fields with no reader anywhere, and reached the per-file
    run-states by building every PLOT and discarding them.
    """
    run_dir = Path(run_dir)
    files = _enumerate_files(run_dir, match)
    out_states = _out_conclusions(files["out"])
    mw_states = _molwatch_conclusions(files["molwatch"])
    return _build_status(
        files["out"] + [p for p in files["molwatch"] if p.name in mw_states],
        {**out_states, **mw_states},
        _process_conclusion(run_dir, match))


def _out_conclusions(out_paths: List[Path]) -> Dict[str, str]:
    """Each ``.out``'s run-state, by filename, through the CHEAP door.

    `_run_ending.scan_ending` and the full parser read the SAME marker
    table -- `siesta.py` builds its fatal rules from `FATAL_MARKERS` by
    comprehension -- so there is no second list to drift, and this is the
    door `jobset/summarize.py` already asks the same question through.

    **Why the cheap one.** The full parser builds every Frame -- positions
    and forces as numpy arrays -- and this keeps one string.  That is the
    shape this module's own docstring says got the previous decoder
    deleted: *"parsing every `.out` to build plot data and then throwing
    the plots away"*.  It was re-entered here through the registry, which
    § 5.4 asks for, and § 2b's cost table says a caller that wants the
    ending uses the scanner.

    Measured 2026-09-18 over every `.out` in `projects/` + `tests/` and
    then end to end over all 119 real run directories, clock frozen:
    **119/119 identical verdicts, 8.5x faster** (12x on the reads alone).
    It also ends this function's `.parse.log` side effect -- the full
    parser opens a `ParseLogger` that appends beside its input, which is
    what wrote 177 files into `projects/` on 2026-09-18.

    Fail-soft, exactly as the molwatch sibling is: a file that cannot be
    read contributes nothing rather than taking the walk down.
    """
    from molbuilder.parse.engines._run_ending import scan_ending
    from molbuilder.parse.errors import ParseError
    states: Dict[str, str] = {}
    for path in out_paths:
        try:
            states[path.name] = (
                scan_ending(path.read_text(encoding="utf-8", errors="replace"))
                .run_state or "unknown")
        except (ParseError, OSError, ValueError):
            continue
    return states


def _build_status(out_paths: List[Path],
                  out_run_states: Dict[str, str],
                  concluded: Optional[str] = None,
                  ) -> "RunStatus":
    """Build the status envelope per § 5, over the directory's RESULT
    files — every ``.out`` plus each concluded molwatch log
    (``running-a-job.md`` § 4) — and the run's PROCESS conclusion.

    **Content first, process second, age last.**  An output that states how
    it ended is the strongest evidence and keeps the answer it always gave.
    The marker speaks where content is silent, which is exactly where the
    age rule used to guess.
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
    age_s = max(0.0, _wall_now() - active.stat().st_mtime)
    # THIS LAYER SETTLES WHAT CONTENT CANNOT (`model/parse.md` § 2b).  The
    # engine parser reports how the run ENDED from markers alone --
    # "running"|"ended"|"stopped"|"out_of_memory"|"unknown" -- and a file
    # with no ending marker is honestly "running": nothing IN it can tell
    # a slow DFT step from a job the scheduler killed.  Only the
    # filesystem can, so the age check lives here and nowhere else.
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
        # marker says how.  The age rule below only guesses `stale`.
        #
        # THE ENGINE'S OWN MARKER IS EXCLUDED, because it cannot be
        # ATTRIBUTED.  A `.concluded` carries a label and a `-run<N>`, so
        # `_process_conclusion` can refuse a previous attempt's goodbye;
        # `0_NORMAL_EXIT` is a bare filename carrying neither, so a leftover
        # from an earlier attempt promoted a silent, un-growing output to
        # `finished` -- measured `stale` -> `finished` on a copy of
        # `BDT-withAuJunction`, and back to `stale` with the marker removed.
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
    elif age_s > 60.0:
        # No marker, no growth, no goodbye: a killed job, and only the
        # clock can say so.
        state, detail = "stale", f"no file growth in {int(age_s)}s"

    return RunStatus(state=state, detail=detail,
                     last_change_at=_iso_z(active.stat().st_mtime),
                     active_source=active.name,
                     concluded=concluded)


def _wall_now() -> float:
    """Wall-clock now() in POSIX seconds.  Indirected so tests can
    monkeypatch without touching time.time globally."""
    import time
    return time.time()
