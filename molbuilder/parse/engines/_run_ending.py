"""How a run ended -- one reader per ROLE, and a cheap way to ask.

Contract: `model/parse.md` § 2b and § 5.5.  The marker STRINGS each have one
home, and it is not here: SIESTA's are lines of a FOREIGN format and live in
the SIESTA family's one table, `siesta_grammar`, which the full parser builds
its rules from too; PySCF's are strings our own emitters print, so each
emitter declares its constant (§ 5.5).  This module imports both and owns the
dispatch -- which is P-S4's "one reader per question" made structural rather
than aspirational.  *(SIESTA's markers were declared here until 2026-09-26,
and the parser retyped four of them as literals beside the import it did
use.)*

**DISPATCH IS ON THE ROLE, never on the engine** (:data:`READERS`,
:func:`ending_of`).  A directory whose engine is unknown, or which holds two
engines' outputs, needs no special case: each file is read by the reader its
own role names, and `model/parse.md` § 5.1 picks which of them speaks.

**No arrays, and deliberately.**  Answering *"did this run end, and
how"* is a substring scan.  The full :mod:`~molbuilder.parse.engines.siesta`
parser needs numpy because it builds Frames -- positions and forces as
arrays -- but a caller that wants the ENDING does not, and until
2026-08-25 it paid for them anyway: `jobset/summarize.py` measured **45 ms
per trial** (272 ms for a six-trial sweep, on 152 KB files) building one
Frame per file to read one string field, on a summary that polls every
15 s.  A relaxation `.out` with hundreds of frames costs far more.

So the scanner is a single pass over the same table the heavy parser
builds its rules from -- there is no second list to drift.  (`jobset/summarize.py` grew a private `_DONE_MARKERS`
tuple exactly that way, and it disagreed with the parser about a capped
benchmark.)

**It travels beside every job** (`runwrap.MONITOR_COMPANIONS`,
`execution/run-reports.md` § 2.3): the monitor reports how a run ended with
this reader, through `parse/dirs/job.py`'s `run_status`.  So everything it
imports is stdlib-only and travels too -- the SIESTA family's table, the
molwatch log's grammar, the PySCF decks' end lines, `runfiles` -- and each is
imported two ways, from the package or from beside the job, as `config_dir`
always has been.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

# THE EMITTERS' OWN END LINES, imported rather than spelled -- the
# `ROLE_GEOM_TRAJ` pattern (`parse/dirs/rundir.py`) applied to a line.  Two,
# because PySCF has two decks and they print different lines: the relaxation
# deck says "Job complete in", the spectrum deck "Total wall time:".
# SIESTA's markers are the family's one table's (`siesta_grammar`), the
# molwatch log's footer its format's (`molwatch_grammar`).
try:                                        # inside molbuilder
    from ...pyscf.end_lines import (END_MARKER as PYSCF_END_MARKER,
                                    SPECTRUM_END_MARKER
                                    as PYSCF_SPECTRUM_END_MARKER)
    from ... import runfiles as _rf
    from . import molwatch_grammar as _MG
    from . import siesta_grammar as _G
except ImportError:                         # beside a job, as the monitor's
    from end_lines import (END_MARKER as PYSCF_END_MARKER,
                           SPECTRUM_END_MARKER as PYSCF_SPECTRUM_END_MARKER)
    import runfiles as _rf
    import molwatch_grammar as _MG
    import siesta_grammar as _G

#: The run is over and will produce nothing more -- § 2b P-S1's vocabulary
#: split by the only question a watcher asks.  ``unknown`` is deliberately
#: NOT here: "no evidence either way" is not evidence of ending, and a
#: watcher that reads it as one stops watching a run that is still alive.
#:
#: This tuple exists because a private restatement of it cost eleven hours.
#: `cli.py`'s ``watch tail`` carried its own ``("finished", "error")``; when
#: the § 2b rename retired both names the loop simply never terminated, and
#: the test that should have caught it polled until it was killed.  P-S4 is
#: not a style rule -- one door, or the copies drift silently.
CONCLUDED: Tuple[str, ...] = ("ended", "stopped", "out_of_memory")

#: The table's out-of-memory markers -- a cause that outranks the others.
_OOM_MARKERS = frozenset(m for m, st in _G.FATAL_MARKERS
                         if st == "out_of_memory")

@dataclass(frozen=True)
class RunEnding:
    """§ 2b's two independent facts, and the sentence for the first --
    and, for a run with SCF phases, each phase's convergence (``periodic``,
    ``negf``), which the run record reports (`model/parse.md` § 5d.6); for a
    relaxation, whether its geometry converged; and the fatal marker that
    decided a stop."""
    run_state:     str                    # P-S1
    scf_converged: Optional[bool] = None  # P-S2 -- a fact, not a verdict
    error_message: Optional[str] = None
    phases:        Dict[str, Optional[bool]] = field(default_factory=dict)
    #: ``True`` -- the relaxation converged; ``False`` -- it ran out of moves
    #: unconverged; ``None`` -- a single point, or not there yet.
    relaxed:       Optional[bool] = None
    #: WHAT STOPPED IT: the first fatal line's marker from the table
    #: (``_G.FATAL_MARKERS``) -- the lines after it are ``die``'s cascade, and
    #: an out-of-memory line outranks the rest wherever it falls -- or
    #: ``_G.SCF_NOT_CONV_MARKER`` when SIESTA stated the SCF's failure fatal.
    cause:         Optional[str] = None


def scan_ending(text: str, *more: str) -> RunEnding:
    """How this run ended, from markers alone -- one pass, no arrays.

    ``more`` is what the run said on its other channel, read after the
    output by the same rules: SIESTA's stderr, when its wrapper keeps it
    apart (:func:`_siesta_ending`).  The deck's echo is the output's alone,
    so each text is read from outside it.

    ``running`` is the honest answer for a file with no ending marker --
    not finished: nothing in it separates a slow DFT step from a job the
    scheduler killed (§ 2b P-S1).

    ``scf_converged`` is the LAST phase's, as the parser's is: a TranSIESTA
    device's periodic initialization converging does not speak for its NEGF
    loop, so a new phase clears it, and so does SIESTA taking a convergence
    back (``SCF cycle continued``).
    """
    run_state = "running"
    scf_converged: Optional[bool] = None
    scf_not_conv_line: Optional[str] = None
    error_message: Optional[str] = None
    phase: Optional[str] = None
    relaxed: Optional[bool] = None
    cause: Optional[str] = None

    phases: Dict[str, Optional[bool]] = {}
    in_echo = False
    for i, raw in ((i, raw) for i, t in enumerate((text, *more))
                   for raw in t.splitlines()):
        if i and in_echo:
            in_echo = False           # another channel: not inside the echo
        # THE DECK'S ECHO IS NOT THE RUN SPEAKING (`siesta_grammar`'s
        # INPUT_ECHO): a marker inside it is the deck's own comment.
        edge = _G.input_echo_edge(raw)
        if edge is not None:
            in_echo = edge
            continue
        if in_echo:
            continue
        row = _G.scf_row(raw)
        if row is not None:
            if row.phase != phase:
                phase, scf_converged = row.phase, None
                phases[phase] = None
            continue
        line = raw.lower()
        if _G.SCF_CONTINUED_MARKER in line:
            scf_converged = None
        elif _G.SCF_CONVERGED_MARKER in line:
            scf_converged = True
        elif _G.SCF_NOT_CONV_MARKER in line:
            scf_converged = False
            if scf_not_conv_line is None:
                scf_not_conv_line = raw.strip()[:200]
            # SIESTA STATING IT FATAL is the cause of the death that follows
            # (`siesta_grammar.SCF_NOT_CONV_REQUIRED`); the tolerated form
            # is not, and a later crash keeps its own cause.
            if _G.SCF_NOT_CONV_REQUIRED in line and cause is None:
                cause = _G.SCF_NOT_CONV_MARKER
        elif _G.SCF_NOT_CONVERGED_MARKER in line:
            scf_converged = False
        else:
            if _G.RELAXED_MARKER in line:
                relaxed = True
            elif _G.UNRELAXED_MARKER in line:
                relaxed = False
            for marker, state in _G.FATAL_MARKERS:
                if marker in line:
                    # Any fatal line proves the stop -- and an OOM outranks
                    # a generic abort wherever it falls.
                    oom = state == "out_of_memory"
                    if oom or run_state != "out_of_memory":
                        run_state = state
                    # THE FIRST FATAL LINE IS THE CAUSE and what follows it
                    # the cascade -- SIESTA's `die` prints its message and
                    # then ``Stopping Program from Node`` -- as the parser
                    # keeps its first line; the memory, once seen, is the
                    # cause.
                    if cause is None or (oom and cause not in _OOM_MARKERS):
                        cause = marker
                    if error_message is None:
                        error_message = raw.strip()[:200]
                    break
            else:
                if _G.RUN_END.match(raw) and run_state == "running":
                    run_state = "ended"
            continue
        # An SCF marker: it speaks for the phase of the row before it.
        if phase is not None:
            phases[phase] = scf_converged

    # The held SCF line is the informative cause when the run is proven
    # dead -- it outranks the cascade marker that recorded itself above.
    if run_state in ("stopped", "out_of_memory") and scf_not_conv_line:
        error_message = scf_not_conv_line
    return RunEnding(run_state, scf_converged, error_message, phases,
                     relaxed, cause)


# ---- PySCF: our own decks' end lines, and Python's own failure shape ---- #

#: The two lines a molbuilder PySCF deck prints when it reaches its own end,
#: IMPORTED from the emitters that print them.  Matched case-insensitively
#: anywhere in the line, like everything else here.
#:
#: Reaching either means the run ENDED, and that outranks anything the script
#: caught and reported on the way: a real relaxation log carries *"Frequency
#: analysis FAILED: unsupported format string ..."* three lines above its
#: "Job complete in 145.8 s", and the run did finish -- the geometry and the
#: energy are on disk.
PYSCF_END_MARKERS: Tuple[str, ...] = (PYSCF_END_MARKER,
                                      PYSCF_SPECTRUM_END_MARKER)

#: PYTHON'S traceback header, which is Python's and not ours -- so unlike the
#: end lines it IS sniffed, exactly as SIESTA's markers are.  An uncaught
#: exception is the only death shape that leaves a fingerprint in the file.
#:
#: A `SystemExit` deliberately has none: the four the decks raise
#: (`pyscf/input.py`'s missing-optimizer guard, and three in
#: `vibration_emitters.py`) print their message with NO traceback, and they
#: share no prefix worth pinning.  That run is answered where it should be --
#: the wrapper's `.concluded` carries `rc=1`, and `parse/dirs/job.py`
#: `_build_status` reads the process where content is silent (§ 2b: nothing
#: IN a file separates a slow step from a job that was killed).
PYSCF_TRACEBACK_MARKER = "traceback (most recent call last)"


def scan_pyscf_ending(text: str) -> RunEnding:
    """How a PySCF run ended, from markers alone -- one pass, no arrays.

    The PySCF sibling of :func:`scan_ending`, and deliberately a much shorter
    table: SIESTA's ``siesta_grammar.FATAL_MARKERS`` are **not** shared.  Measured
    2026-09-18 over 135 real output files, its five out-of-memory markers fire
    0 times and the three that do fire are SIESTA's own sentences.  Borrowing
    them would have this reader answer `out_of_memory` for a PySCF log that
    merely quoted one.

    ``scf_converged`` is left None: § 2b P-S2's fact is REPORTED, never a
    verdict, and nothing reads it for PySCF yet.  It is a row to add, not a
    shape to change.
    """
    run_state = "running"
    error_message: Optional[str] = None
    ended = False
    for raw in text.splitlines():
        line = raw.lower()
        # ANCHORED AT COLUMN 0.  A SUBSTRING TEST IS WRONG HERE -- unlike
        # SIESTA's markers, which `scan_ending` finds as substrings outside
        # the deck's echo -- and not subtly: PySCF's
        # `Mole.build()` calls `dump_input()`, which ECHOES THE DECK'S OWN
        # SOURCE into `mol.stdout` -- and when the deck writes no separate
        # `.log`, `mol.stdout` IS the stdout the wrapper captures.  So the
        # line `print(f'Total wall time: {t1 - t0:.1f} s')` appears in the
        # log during `=== Stage: build molecule ===`, in the first seconds.
        #
        # Measured on `projects/BDT/spectrum/BDT-only`: the echo is line 1139
        # of 11606 and the real end line is 11605.  Fed the first 2000 lines
        # -- a run 17% of the way through a 62-minute job -- the substring
        # form answered `ended`, so `run_status` said `finished`.  A running
        # job reporting finished is worse than the defect this reader was
        # added to fix, and a killed one reported it too: `ended` outranks
        # the traceback branch below.
        #
        # The anchor separates them exactly: both real end lines are printed
        # at column 0, and both echoed ones begin `print(`.
        if any(line.startswith(m.lower()) for m in PYSCF_END_MARKERS):
            ended = True
            continue
        if PYSCF_TRACEBACK_MARKER in line:
            run_state = "stopped"
            error_message = raw.strip()[:200]
    if ended:
        # The end line outranks a caught-and-reported failure above it.
        return RunEnding("ended")
    if run_state == "stopped":
        # The EXCEPTION line is the sentence a person wants, not the header.
        for raw in reversed(text.splitlines()):
            if raw.strip() and not raw[:1].isspace():
                error_message = raw.strip()[:200]
                break
    return RunEnding(run_state, None, error_message)


# ---- one reader per ROLE ------------------------------------------------ #


def _read(path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _siesta_ending(path, stderr=None) -> RunEnding:
    """The ``.out``'s ending -- and, given ``stderr``, the file SIESTA's
    stderr went to (its wrapper's session log), read after it when the
    ``.out`` states no ending.  SIESTA's ``die`` writes its message to both
    channels but flushes stdout on node 0 alone
    (``Src/siesta_handlers_m.F90``), so a rank other than 0 that dies may say
    why on stderr only."""
    if stderr is None:
        return scan_ending(_read(path))
    # AN OUTPUT SIESTA NEVER WROTE TO is not there at all -- the tee creates
    # it on the first line -- and a run that died before its first line may
    # still have said why on stderr.
    text = _read(path) if Path(path).exists() else ""
    end = scan_ending(text)
    if end.run_state != "running":
        return end
    return scan_ending(text, _read(stderr))


def _pyscf_ending(path) -> RunEnding:
    return scan_pyscf_ending(_read(path))


def _molwatch_ending(path) -> RunEnding:
    """The progress log's FOOTER, or `running` when it carries none.

    A molwatch log is SEEDED at prep (`jobset/prep.py::_seed_trajectory_log`),
    so it exists before the engine does: it speaks only once its footer
    concludes, which is what `runfiles.Artifact.output == "progress"` says and
    why an empty one must not outrank a real result.  `scan_conclusion`
    returns `"running"` for exactly that case.
    """
    return RunEnding(_MG.scan_conclusion(path))


#: ROLE -> the reader that knows how that file says it ended.
#:
#: Keyed on the role and never on the engine, which is load-bearing (§ 5.5): a
#: directory whose engine is unknown, or one holding both engines' outputs,
#: needs no special case here.  The keys must be exactly
#: `runfiles.run_output_roles()` -- the catalogue declares WHICH files are run
#: output, this declares HOW each is read, and
#: `tests/test_run_ending_one_table.py` asserts the two sets are equal.  That
#: one assertion is what keeps "adding an engine is two edits" true across the
#: layer split.
READERS: "Dict[str, Callable[[Path], RunEnding]]" = {
    ".out":          _siesta_ending,
    ".pyscf.log":    _pyscf_ending,
    ".molwatch.log": _molwatch_ending,
}


def ending_of(path, *, stderr=None) -> RunEnding:
    """How the run that wrote *path* ended -- dispatched on the file's ROLE.

    ``stderr`` names the file the engine's stderr went to when its output
    does not carry it -- SIESTA's, whose wrapper keeps it in the session log
    (a PySCF log takes its stderr in, ``2>&1``).

    The one door for *"how did this end"* over a run-output file.  The role
    comes from `runfiles.role_of`, which reads it off the name without a
    label, so a caller holding one path needs nothing else.

    Refuses a file that is not run output rather than answering `unknown`: the
    roles come from :data:`READERS`, whose keys are the catalogue's own
    `run_output_roles()`, so reaching this raise means a caller asked about
    the wrong file -- a mistake to hear about, not to absorb.  Read errors are
    NOT absorbed here either; the collecting loop is where fail-soft belongs,
    and it catches `OSError` alone so a role mistake still surfaces.
    """
    role = _rf.role_of(Path(path).name)
    reader = READERS.get(role)
    if reader is None:
        raise ValueError(
            f"{Path(path).name!r} is not a run-output file: its role is "
            f"{role!r}, and the run-output roles are "
            f"{', '.join(_rf.run_output_roles())} (`runfiles.WRITTEN`'s `output` "
            f"column, `model/parse.md` § 5.5).")
    if stderr is None:
        return reader(path)
    if reader is not _siesta_ending:
        raise ValueError(
            f"{Path(path).name!r} carries its engine's stderr itself; only a "
            f"SIESTA-family output is read beside a separate one.")
    return reader(path, stderr)


# ---- The shell's door onto this reader ------------------------------------ #
#
# THE WRAPPER ASKS, IT DOES NOT GREP (`execution/run-reports.md` § 2.3).  The
# generated `.run.sh` decided its warm retries and its failure hints by
# grepping the output for strings it typed itself -- `SCF_NOT_CONV`,
# `outcoor: Final (unrelaxed) ...`, `propor: ERROR` -- because nothing it
# could run beside the job could read the output the way molbuilder does.
# This module travels beside every job now, so the wrapper runs it with the
# job's own python and the markers keep their one home.

#: What a shell may ask, each answered by the exit status (0 yes, 1 no).
#: ``stopped-by MARKER`` asks for the table's own marker, which the wrapper
#: renders from `siesta_grammar` -- ``propor: error`` for its hint,
#: ``scf_not_conv`` for its warm retry.
QUESTIONS: Dict[str, Callable[..., bool]] = {
    "relaxation-capped": lambda e: e.relaxed is False,
    "stopped-by":        lambda e, marker: e.cause == marker,
}


def say(end: RunEnding) -> str:
    """The ending as one line -- what the wrapper prints after a failure,
    when the process is gone: an output with no ending in it says so."""
    bits = [end.run_state if end.run_state != "running"
            else "the output states no ending"]
    if end.error_message:
        bits.append(end.error_message)
    if end.scf_converged is not None:
        bits.append("SCF converged" if end.scf_converged
                    else "SCF did not converge")
    if end.relaxed is not None:
        bits.append("geometry relaxed" if end.relaxed
                    else "geometry did not converge within its moves")
    return " -- ".join(bits)


def main(argv=None) -> int:
    """``OUTPUT [--stderr FILE] [QUESTION [ARG]]`` -- beside a job,
    ``python mb_monitor.pyz ending ...`` (the bundle's entry hands ``ending``
    here; `runwrap.MONITOR_BUNDLE`): with a question, the exit status answers
    it (:data:`QUESTIONS`); without one, the ending is printed as :func:`say`
    words it.  ``--stderr`` is :func:`ending_of`'s.  2 when the output cannot
    be read or the question is not one."""
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    stderr = None
    if "--stderr" in args:
        at = args.index("--stderr")
        if at + 1 >= len(args):
            print("_run_ending: --stderr names a file", file=sys.stderr)
            return 2
        stderr = args[at + 1]
        del args[at:at + 2]
    if not args:
        print("usage: mb_monitor.pyz ending OUTPUT [--stderr FILE] "
              "[QUESTION [ARG]]", file=sys.stderr)
        return 2
    try:
        end = ending_of(args[0], stderr=stderr)
    except (OSError, ValueError) as exc:
        print(f"_run_ending: {exc}", file=sys.stderr)
        return 2
    if len(args) == 1:
        print(say(end))
        return 0
    ask = QUESTIONS.get(args[1])
    if ask is None:
        print(f"_run_ending: not a question: {args[1]!r} "
              f"({', '.join(QUESTIONS)})", file=sys.stderr)
        return 2
    return 0 if ask(end, *args[2:]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

