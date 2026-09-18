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
    from molbuilder.runfiles import find_by_role, run_output_roles

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
    by_kind["fdf"] = sorted(p for p in find_by_role(run_dir, ".fdf")
                            if p in narrowed)
    # WHICH FILES ARE A RUN'S OUTPUT IS THE CATALOGUE'S QUESTION, and the
    # buckets are keyed by the ROLE because the role is what they are.  The
    # pair `("out", "molwatch")` stood here as a literal list until
    # 2026-09-18 and did not name `.pyscf.log`, so a finished PySCF run that
    # writes no molwatch log had no result file at all as far as this module
    # was concerned (`model/parse.md` § 5.5, R-RO1).
    for role in run_output_roles():
        by_kind[role] = sorted(p for p in find_by_role(run_dir, role)
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


def _output_endings(paths: List[Path]) -> Dict[str, str]:
    """Each run-output file's run-state, by filename — ONE loop, one door.

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
    from molbuilder.parse.engines._run_ending import ending_of
    states: Dict[str, str] = {}
    for path in paths:
        try:
            states[path.name] = ending_of(path).run_state or "unknown"
        except OSError:
            continue
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
    from molbuilder.parse.engines._run_ending import CONCLUDED
    from molbuilder.runfiles import run_output_roles, stdout_roles
    run_dir = Path(run_dir)
    files = _enumerate_files(run_dir, match)
    states = _output_endings(
        [p for role in run_output_roles() for p in files[role]])
    # WHICH OF THEM MAY SPEAK is the catalogue's `output` column, not a rule
    # written here.  A "stdout" file exists because the PROCESS started, so it
    # counts whether or not it ended; a "progress" file is SEEDED at prep, so
    # it counts only once its footer concludes -- otherwise a stage's seed
    # outvotes its own result (`model/parse.md` § 5.5, § 5.1).  That pair of
    # rules was two hand-written functions until 2026-09-18, and the seed half
    # was the only one that said WHY.
    speaks = set(stdout_roles())
    outputs = [p for role in run_output_roles() for p in files[role]]
    return _build_status(
        [p for role in run_output_roles() for p in files[role]
         if role in speaks or states.get(p.name) in CONCLUDED],
        states,
        _process_conclusion(run_dir, match),
        # LIVENESS IS NOT THE SPEAKER, so it gets its own list: every
        # run-output file, INCLUDING the progress log that may not speak.
        fresh_paths=outputs)




def _build_status(out_paths: List[Path],
                  out_run_states: Dict[str, str],
                  concluded: Optional[str] = None,
                  fresh_paths: "Optional[List[Path]]" = None,
                  ) -> "RunStatus":
    """Build the status envelope per § 5, over the directory's RESULT
    files — every ``"stdout"`` run output plus each ``"progress"`` one whose
    footer concludes (`runfiles.Artifact.output`, `model/parse.md` § 5.5) —
    and the run's PROCESS conclusion.

    **Content first, process second, age last.**  An output that states how
    it ended is the strongest evidence and keeps the answer it always gave.
    The marker speaks where content is silent, which is exactly where the
    age rule used to guess.

    ``out_paths`` are the files that may SPEAK; ``fresh_paths`` is every
    run-output file and is what STALENESS is measured on.  They are two lists
    because they answer two questions (§ 5.5, *liveness is not the speaker*):
    an unconcluded progress log must not outrank a real result, and it is
    also the only thing proving a block-buffered run alive.

    ``last_change_at`` stays the SPEAKER's mtime, paired with
    ``active_source`` beside it -- a timestamp taken from a file other than
    the one named would be a third answer nobody asked for.  The age that
    decided `stale` is reported in ``detail``.
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
    # LIVENESS IS NOT THE SPEAKER (`model/parse.md` § 5.5).  Staleness is
    # measured on the FRESHEST run-output file, which is not necessarily the
    # one that speaks: the wrapper runs the engine with no `-u`, so its stdout
    # is BLOCK-BUFFERED -- a real PySCF log grew 13 KB across 146 s, two
    # flushes in the whole run -- while the progress log flushes per step and
    # is deliberately NOT a speaker until its footer concludes.  Taking the
    # speaker's mtime therefore reported a live run as dead.
    #
    # `fresh_paths` defaulting to the speakers is the pre-2026-09-18 answer,
    # kept only so a caller that has no separate list is not silently given a
    # different rule than the one it asked for.
    _fresh = fresh_paths if fresh_paths else out_paths
    age_s = max(0.0, _wall_now() - max(p.stat().st_mtime for p in _fresh))
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
