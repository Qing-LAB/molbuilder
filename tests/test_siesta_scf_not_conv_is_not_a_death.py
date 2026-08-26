"""``SCF_NOT_CONV:`` is the cause of an abort, not proof that one happened.

The failure this closes (found 2026-08-25 on a real Au-BDT-Au sweep)
=====================================================================

The Results tab reported **six failed trials and "0 done"** for a benchmark
that had run perfectly — every trial dir held SIESTA's ``0_NORMAL_EXIT``,
every ``.out`` ended ``>> End of run`` / ``Job completed``, and every trial
had produced a timing the summary was happily displaying beside the word
*failed*.

**A benchmark deck breaks the assumption the parser was written on.** It
sets::

    MaxSCFIterations  3
    SCF.MustConverge  .false.

— three SCF steps, convergence explicitly *not* required, because what is
being measured is seconds per iteration, not a converged density. SIESTA
prints ``SCF_NOT_CONV:``, then carries straight on ("Using DM_out to
compute the final energy and forces"), reaches End-of-run, and exits 0.

The parser treated that line as fatal on sight, per a comment stating it is
emitted only with ``MustConverge=true`` and is *"always followed in the same
run by ABNORMAL_TERMINATION + Stopping Program"*. True for a production
run; false for every benchmark this project generates.

**What the line is actually for** (the 2026-05-30 change) is supplying the
INFORMATIVE root cause instead of the cascade ``Stopping Program from Node:
0``. It still does: it is held, and promoted to ``error_message`` by
whatever really proves the run died — a fatal marker, or the strict EOF
check. So the message survives and the verdict stops being wrong.
"""
from __future__ import annotations

import pytest

from molbuilder.parse.engines.siesta import SiestaOutFileParser

#: Three SCF steps then the non-convergence line — the shape every deck
#: this project writes for a benchmark produces.
_SCF_THEN_NOT_CONV = (
    "Siesta Version: 5.4.2\n"
    "   scf:    1 -1740000.0 -1740000.0 -1740000.0  0.9  0.5  30.0\n"
    "   scf:    2 -1740500.0 -1741300.0 -1741300.0  0.4  0.2  30.8\n"
    "   scf:    3 -1741700.0 -1741700.0 -1741700.0  0.1  0.1  46.8\n"
    "SCF_NOT_CONV: SCF did not converge  in maximum number of steps.\n"
)
_FINISHED = (">> End of run:  25-AUG-2026   4:08:52\nJob completed\n")


def _parse(tmp_path, text):
    f = tmp_path / "probe-run0.out"
    f.write_text(text, encoding="utf-8")
    return SiestaOutFileParser().parse(f)


def test_a_benchmark_that_ran_out_of_scf_steps_and_finished_is_finished(
        tmp_path):
    """THE LIVE DEFECT.  `MustConverge .false.` means SIESTA prints the
    line and keeps going; a run that reaches End-of-run finished."""
    r = _parse(tmp_path, _SCF_THEN_NOT_CONV
               + "Using DM_out to compute the final energy and forces\n"
               + _FINISHED)
    assert r.run_state == "ended", (
        f"a completed benchmark reads {r.run_state!r} -- this is the bug "
        f"that showed six healthy trials as failed")
    assert not r.error_message, (
        f"a run that ended carries an error message: {r.error_message!r}")
    # P-S2: the science is REPORTED beside the ending, not folded into it.
    assert r.scf_converged is False, (
        "the run did not converge and the reader must say so plainly -- "
        f"got {r.scf_converged!r}")


def test_a_real_abort_still_errors_and_keeps_the_informative_cause(tmp_path):
    """The half that must NOT regress.  With `MustConverge` on, SIESTA
    aborts — and the message must be the root cause, not the cascade the
    2026-05-30 change existed to skip past."""
    r = _parse(tmp_path, _SCF_THEN_NOT_CONV
               + "ABNORMAL_TERMINATION\nStopping Program from Node:    0\n")
    assert r.run_state == "stopped"
    assert "SCF_NOT_CONV" in (r.error_message or ""), (
        f"the cascade won over the root cause: {r.error_message!r}")


def test_a_torn_run_is_still_RUNNING_to_the_parser(tmp_path):
    """P-S1's layering, and it is deliberate.

    A file with no ending marker and no abort marker is honestly
    ``running``: nothing IN it distinguishes a slow DFT step from a job
    the scheduler killed thirty seconds ago.  The parser does not guess --
    ``parse/dirs/job.py`` settles it with the file's age, which is the
    only evidence that can.

    This used to read ``error``, decided by the last SCF block having not
    converged -- the science answering a question about the process.  That
    is exactly what P-S2 forbids."""
    r = _parse(tmp_path, _SCF_THEN_NOT_CONV)
    assert r.run_state == "running", (
        f"the parser guessed {r.run_state!r} from content that cannot "
        f"support it")
    assert r.scf_converged is False, "the fact itself is still reported"


def test_a_converged_run_is_untouched(tmp_path):
    """The control: nothing above may change the ordinary case."""
    r = _parse(tmp_path,
               "Siesta Version: 5.4.2\n"
               "   scf:    1 -1740000.0 -1740000.0 -1740000.0  0.9  0.5  30.0\n"
               "SCF Convergence by DM+H criterion\n" + _FINISHED)
    assert r.run_state == "ended"
    assert r.scf_converged is True
    assert not r.error_message
