"""A trial's `.out` is scanned WHOLE, not from its tail.

`scan_ending` looks at every line, because the SCF-convergence markers
appear once per cycle rather than at the end, and `scf_not_conv_line`
reports the FIRST of them. A tail window would answer for the last
cycles only.

**This exists because the code invited the opposite.** A
`out_tail = _read(out, tail=16384)` sat beside the full read until
2026-09-04 — the orphan of an older tail-window design, consumed by
nothing, re-reading 16 KB per trial per summary on a view that polls
every 15 s. Removing it, the obvious next "optimisation" is to feed that
same window to the scan. Measured before writing this: making that
change breaks nothing in the suite, because every fixture `.out` is
smaller than the window.
"""
from __future__ import annotations

from pathlib import Path

from molbuilder.jobset.summarize import parse_point


def _out_with_early_marker(d: Path, basename: str) -> None:
    """A `.out` whose fatal marker is at the TOP, padded past 16 KB.

    The padding is the test: with a tail window the marker falls off the
    front and the trial reads as completed.
    """
    d.mkdir(parents=True, exist_ok=True)
    body = (
        "                           Welcome to SIESTA\n"
        "reinit: System Label: j\n"
        "siesta: ERROR: out of memory in dense solver\n"
        + "".join(f"scf: {i:6d}   -100.0   -100.0  0.0001\n"
                 for i in range(1600))          # well past the 16 KB window
        + "siesta: Final energy (eV):\nJob completed\n>> End of run:\n")
    assert len(body) > 3 * 16384, "the padding must exceed the tail window"
    (d / f"{basename}-run0.out").write_text(body, encoding="utf-8")


def test_a_fatal_marker_at_the_top_of_a_long_out_is_still_seen(tmp_path):
    """48 KB of clean output after an OOM does not make the run clean."""
    _out_with_early_marker(tmp_path, "j")
    pt = parse_point("t", tmp_path, "j", "cpu", {})
    assert pt.state != "completed", (
        "the ending scan missed a fatal marker in the first KB of a 60 KB "
        "file — it is reading a window, not the file, so a long run's early "
        "failure is reported as a clean finish")
