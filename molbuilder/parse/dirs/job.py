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
* **whether it was launched**, before anything is written -- the
  attempt's launch record answers (``launch``).

A file with no ending is ``running`` -- not finished -- however long it
has been quiet: nothing in it or beside it tells a slow step from a
stopped one (user, 2026-09-26: *"It shows what it is"*).

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
from typing import Any, Dict, List, Optional

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
# role, over the SIESTA family's table (`siesta_grammar`) and the PySCF
# decks' end lines -- and it is the scan the full parsers agree with
# (`tests/test_run_ending_one_table.py`).  Nothing in this module greps an
# output: it owns only the two questions no single file can answer (which
# file speaks, and whether it was launched).
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

    Returns {"fdf": [...], "xv": [...], "struct_out": [...],
             "molstruct_json": [...], "ani": [...]} plus one bucket per
    RUN-OUTPUT ROLE, keyed by the role itself: {".out": [...],
    ".pyscf.log": [...], ".molwatch.log": [...]}.  Paths sorted by name
    within each bucket.

    **The run-output buckets are keyed by role and not by a nickname**, and
    that is the point rather than a detail: they used to be `"out"` and
    `"molwatch"`, a private two-word vocabulary for a three-row catalogue
    column, and the third row had no word so it was not looked for at all.

    ``match`` NARROWS THE DIRECTORY TO ONE RUNG, and in the flat shape that is
    the whole question: every stage of a flat calculation shares one directory
    and is told apart by FILENAME (`project-layout.md` § 1), so a bucket built
    from the whole directory answers about all of them at once.  The caller
    passes `Shape.stage_glob(token, label)`; ``"*"`` is the hierarchical
    answer, where the directory has already selected the stage.
    """
    by_kind: Dict[str, List[Path]] = {
        "fdf": [], "xv": [], "struct_out": [],
        "molstruct_json": [], "ani": [],
    }
    # THE NARROWING FIRST, because it is `match`'s whole job and no role
    # search takes a glob: this is the set of files this rung owns.
    narrowed = {c for c in run_dir.glob(match) if c.is_file()}

    # ROLES THE CATALOGUE DECLARES come from the catalogue.  These were
    # spelled `name.endswith(".fdf")` here until 2026-09-18 -- the role
    # vocabulary written outside the module that declares it, which is the
    # exact case `runfiles.find_by_role` says it exists to end, and which
    # every sibling in this package converted on 2026-09-08
    # (`atom_metadata.py`, `contract.py`, `rundir.py`).
    by_kind["fdf"] = sorted(p for p in _rf.find_by_role(run_dir, ".fdf")
                            if p in narrowed)
    # WHICH FILES ARE A RUN'S OUTPUT IS THE CATALOGUE'S QUESTION, and the
    # buckets are keyed by the ROLE because the role is what they are.  The
    # pair `("out", "molwatch")` stood here as a literal list until
    # 2026-09-18 and did not name `.pyscf.log`, so a finished PySCF run that
    # writes no molwatch log had no result file at all as far as this module
    # was concerned (`model/parse.md` § 5.5, R-RO1).
    for role in _rf.run_output_roles():
        by_kind[role] = sorted(p for p in _rf.find_by_role(run_dir, role)
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
    narrowed = {c.name for c in run_dir.glob(match)}
    marks = [m for m in _rf.find_by_role(run_dir, ".concluded")
             if m.name in narrowed]
    if marks:
        # THE INDEX IS ASKED FOR, and the rule is `attempt_concluded`'s: the
        # marker counts only at the HIGHEST index any per-run artifact
        # reached, across every role.  An earlier index's marker beside a
        # newer unconcluded `.out` is a previous re-run's goodbye.
        best = None
        for m in marks:
            got = _rf.parse(m.name, _label_of_marker(m.name))
            idx = getattr(got, "run", None) if got else None
            newest = _rf.latest_run(run_dir, _label_of_marker(m.name))
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
    cut = name.rfind("-" + _rf.QUALIFIERS[0])
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

    **The status IS the parser's answer**, plus what no parser can know.
    Every engine parser already reports how its file ended -- ``run_state``
    on the result, ``model/parse.md`` § 2b -- so this asks them and then
    settles what a single file cannot:

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
        launch=launch), endings=endings)




def _build_status(out_paths: List[Path],
                  out_run_states: Dict[str, str],
                  concluded: Optional[str] = None,
                  launch: Any = _UNASKED,
                  ) -> "RunStatus":
    """Build the status envelope per § 5, over the directory's RESULT
    files — every ``"stdout"`` run output plus each ``"progress"`` one whose
    footer concludes (`runfiles.Artifact.output`, `model/parse.md` § 5.5) —
    and the run's PROCESS conclusion.

    **Content first, process second.**  An output that states how it ended
    is the strongest evidence and keeps the answer it always gave.  The
    marker speaks where content is silent.  Where neither says anything the
    run is ``running`` -- not finished -- however long it has been quiet
    (`running-a-job.md` § 4.2).

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

    return RunStatus(state=state, detail=detail,
                     last_change_at=_iso_z(active.stat().st_mtime),
                     active_source=active.name,
                     concluded=concluded)


