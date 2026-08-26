"""How a SIESTA run ended -- the markers, and a cheap way to ask.

Contract: `model/parse.md` § 2b.  This module owns the marker STRINGS and
nothing else; both readers below share this one table, which is P-S4's
"one reader per question" made structural rather than aspirational.

**Stdlib only, and deliberately.**  Answering *"did this run end, and
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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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
