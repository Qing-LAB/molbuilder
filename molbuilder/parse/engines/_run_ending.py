"""How a run ended -- the markers, one reader per ROLE, and a cheap way to ask.

Contract: `model/parse.md` § 2b and § 5.5.  This module owns the marker
STRINGS for a FOREIGN format and nothing else; both readers below share this
one table, which is P-S4's "one reader per question" made structural rather
than aspirational.

**FOREIGN, which is the split that decides what is declared here.**  SIESTA
prints `>> end of run` and we read it, so the string is ours to sniff and it
lives below.  A format molbuilder GENERATES does not get a sniffed reader
(§ 5.5): PySCF's end lines are strings our own emitters print, so each
emitter declares its constant and this module IMPORTS it -- there is no
second home for a line, and respelling the print without respelling the
constant is what the tests catch.

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

So the markers live here, the scanner is a single pass, and the heavy
parser consults the same table for its own rules -- there is no second
list to drift.  (`jobset/summarize.py` grew a private `_DONE_MARKERS`
tuple exactly that way, and it disagreed with the parser about a capped
benchmark.)

The two emitter imports below are the one exception to *this module imports
nothing of ours*, and they cost nothing: importing any submodule of
`parse.engines` runs that package's ``__init__``, which already loads the
numpy parsers, and `pyscf.input` measured 0.0 ms on top.  What the rule was
protecting is intact -- nothing here builds an array.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

# THE EMITTERS' OWN END LINES, imported rather than spelled -- the
# `ROLE_GEOM_TRAJ` pattern (`parse/dirs/rundir.py`) applied to a line.  Two,
# because PySCF has two decks and they print different lines: the relaxation
# deck says "Job complete in", the spectrum deck "Total wall time:".
#
# `molbuilder.pyscf` and `molbuilder.parse` are both L2, so this is a
# same-layer import (`tests/test_layering.py`), and it costs nothing: by the
# time anything reaches this module `molbuilder.parse.engines.__init__` has
# already imported the numpy parsers, and `pyscf.input` measured 0.0 ms on
# top of that.
from ...pyscf.input import END_MARKER as PYSCF_END_MARKER
from ...pyscf.vibration_emitters import END_MARKER as PYSCF_SPECTRUM_END_MARKER

#: Markers that prove the run DID NOT reach its own end, in the order a
#: reader should prefer them.  Each entry is (substring, run_state).
#: Matched case-insensitively anywhere in the line -- SIESTA prefixes them
#: with "node 0: " under MPI.
FATAL_MARKERS: Tuple[Tuple[str, str], ...] = (
    # Out of memory is called out from the generic aborts because it is
    # the most common cause and the most actionable: "you ran out of
    # memory" is the one sentence that tells a user what to change.
    ("out of memory",                "out_of_memory"),
    ("oom-kill",                     "out_of_memory"),
    ("killed process",               "out_of_memory"),
    ("cannot allocate memory",       "out_of_memory"),
    ("insufficient virtual memory",  "out_of_memory"),
    ("siesta: error",                "stopped"),
    ("propor: error",                "stopped"),
    ("stopping program from node",   "stopped"),
    ("siesta died",                  "stopped"),
    ("abnormal_termination",         "stopped"),
)

#: The run reached its own end.  SIESTA prints this at the very bottom.
END_MARKER = ">> end of run"

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

#: SCF convergence -- REPORTED, never a verdict (§ 2b P-S2).
SCF_CONVERGED_MARKER = "scf convergence by"
#: The informative non-convergence line.  It is the best cause-of-death
#: sentence when something else proves death, and proves nothing alone:
#: a benchmark deck sets `SCF.MustConverge .false.` and SIESTA prints it
#: on the way to a clean exit.
SCF_NOT_CONV_MARKER = "scf_not_conv"
#: The softer informational form.
SCF_NOT_CONVERGED_MARKER = "scf did not converge"


@dataclass(frozen=True)
class RunEnding:
    """§ 2b's two independent facts, and the sentence for the first."""
    run_state:     str                    # P-S1
    scf_converged: Optional[bool] = None  # P-S2 -- a fact, not a verdict
    error_message: Optional[str] = None


def scan_ending(text: str) -> RunEnding:
    """How this run ended, from markers alone -- one pass, no arrays.

    ``running`` is the honest answer for a file with no ending marker:
    nothing IN it separates a slow DFT step from a job the scheduler
    killed.  Only the filesystem can, and `parse/dirs/job.py` does
    (§ 2b P-S1).
    """
    run_state = "running"
    scf_converged: Optional[bool] = None
    scf_not_conv_line: Optional[str] = None
    error_message: Optional[str] = None

    for raw in text.splitlines():
        line = raw.lower()
        if SCF_CONVERGED_MARKER in line:
            scf_converged = True
            continue
        if SCF_NOT_CONV_MARKER in line:
            scf_converged = False
            if scf_not_conv_line is None:
                scf_not_conv_line = raw.strip()[:200]
            continue
        if SCF_NOT_CONVERGED_MARKER in line:
            scf_converged = False
            continue
        for marker, state in FATAL_MARKERS:
            if marker in line:
                # An OOM outranks a generic abort: the aborts that follow
                # it are the cascade, and the memory is the cause.
                if run_state != "out_of_memory":
                    run_state = state
                if error_message is None:
                    error_message = raw.strip()[:200]
                break
        else:
            if line.startswith(END_MARKER) and run_state == "running":
                run_state = "ended"

    # The held SCF line is the informative cause when the run is proven
    # dead -- it outranks the cascade marker that recorded itself above.
    if run_state in ("stopped", "out_of_memory") and scf_not_conv_line:
        error_message = scf_not_conv_line
    return RunEnding(run_state, scf_converged, error_message)


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
    table: SIESTA's :data:`FATAL_MARKERS` are **not** shared.  Measured
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
        # ANCHORED AT COLUMN 0, exactly as `scan_ending` anchors SIESTA's
        # (line 151).  A SUBSTRING TEST IS WRONG HERE and not subtly: PySCF's
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


def _siesta_ending(path) -> RunEnding:
    return scan_ending(_read(path))


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
    from .molwatch import scan_conclusion
    return RunEnding(scan_conclusion(path))


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


def ending_of(path) -> RunEnding:
    """How the run that wrote *path* ended -- dispatched on the file's ROLE.

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
    from molbuilder.runfiles import role_of, run_output_roles
    role = role_of(Path(path).name)
    reader = READERS.get(role)
    if reader is None:
        raise ValueError(
            f"{Path(path).name!r} is not a run-output file: its role is "
            f"{role!r}, and the run-output roles are "
            f"{', '.join(run_output_roles())} (`runfiles.WRITTEN`'s `output` "
            f"column, `model/parse.md` § 5.5).")
    return reader(path)
