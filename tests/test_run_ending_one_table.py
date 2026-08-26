"""The cheap ending scan and the full parser cannot disagree.

`model/parse.md` § 2b P-S4 -- one reader per question.  Two entry points
answer *"how did this run end"*: :func:`scan_ending` (stdlib-only, one
pass) and the full :class:`SiestaParser` (which also builds Frames).  They
share ``FATAL_MARKERS``, and this file is the guard that keeps them
sharing it.

The history is why the guard exists.  `jobset/summarize.py` carried a
private `_DONE_MARKERS` tuple whose own comment knew a capped benchmark
with ``SCF.MustConverge .false.`` exits cleanly -- while the parser beside
it did not.  The bench summary asked both and rendered the wrong one: six
healthy trials shown as failures.
"""
from __future__ import annotations

import pathlib

import pytest

from molbuilder.parse.engines._run_ending import scan_ending
from molbuilder.parse.engines.siesta import SiestaParser

_FIXTURES = sorted(
    (pathlib.Path(__file__).parent / "watch/fixtures/siesta_frozen").glob("*.out"))


def test_the_scanner_needs_no_numpy():
    """It is stdlib-only ON PURPOSE: answering how a run ended is a
    substring scan, and a caller should not pay for arrays it will not
    read."""
    import ast
    src = (pathlib.Path("molbuilder/parse/engines/_run_ending.py")
           .read_text(encoding="utf-8"))
    imported = {n.split(".")[0]
                for node in ast.walk(ast.parse(src))
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for n in ([a.name for a in node.names]
                          + ([node.module] if isinstance(node, ast.ImportFrom)
                             and node.module else []))}
    assert "numpy" not in imported, f"the cheap door grew a heavy import: {imported}"


@pytest.mark.parametrize("out", _FIXTURES, ids=lambda p: p.name)
def test_both_doors_give_the_same_ending(out):
    """Every frozen fixture, both ways."""
    cheap = scan_ending(out.read_text(errors="replace"))
    full  = SiestaParser.parse(str(out))
    assert cheap.run_state == full.run_state, (
        f"{out.name}: the scan says {cheap.run_state!r}, the parser says "
        f"{full.run_state!r} -- two answers to one question is the defect "
        f"this file exists to prevent")
    assert cheap.scf_converged == full.scf_converged, (
        f"{out.name}: convergence disagrees "
        f"({cheap.scf_converged!r} vs {full.scf_converged!r})")


def test_a_capped_benchmark_reads_ended_through_BOTH_doors(tmp_path):
    """The live case, asserted on both sides at once."""
    text = ("Siesta Version: 5.4.2\n"
            "   scf:    1 -1740000.0 -1740000.0 -1740000.0  0.9  0.5  30.0\n"
            "SCF_NOT_CONV: SCF did not converge  in maximum number of steps.\n"
            "Using DM_out to compute the final energy and forces\n"
            ">> End of run:  25-AUG-2026   4:08:52\nJob completed\n")
    f = tmp_path / "bench-run0.out"; f.write_text(text, encoding="utf-8")
    assert scan_ending(text).run_state == "ended"
    assert SiestaParser.parse(str(f)).run_state == "ended"
    assert scan_ending(text).scf_converged is False
