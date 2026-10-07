"""Runtime facts must reach the ARTIFACT, not just the process.

Law A says a parameter is an explicit field or it is reported.  P2 added
five SCF facts to ``_RUNTIME_INFO`` and its tests asserted the emitted
assignment lines -- which passed, and proved nothing, because the values
never reached ``<job>.molwatch.log``, the file /results actually renders.
Two independent causes, both closed here:

1. **The writer filtered.**  ``MolwatchEmitter.__init__`` carried a
   literal tuple of eleven key names.  It had already drifted from
   ``molbuilder.runtime_info.RUNTIME_INFO_KEYS`` -- ``max_memory_mb`` is
   IN the canonical list, was written by every script, and was dropped
   on the way to disk.  Nothing could catch it: the canonical tuple is
   imported by nobody and the reader accepts any ``runtime.<key>``, so
   the writer was the one closed door in an open pipe.

2. **The header was frozen too early.**  __init__ writes the whole
   header and a log header cannot be rewritten once data follows, so
   constructing the emitter before the SCF setup froze the dict at that
   moment.  The GPU keys survived only because the probe happens to run
   earlier.

Measured before the fix: 17 keys handed in, 11 persisted.  After: 17.

These tests assert on the FILE, not on the emitted source, because
asserting on the source is precisely what let the defect through.
"""
from __future__ import annotations

import numpy as np

from molbuilder.runtime_info import RUNTIME_INFO_KEYS
from molbuilder.trajectory_log.emitter import MolwatchEmitter


class _Mol:
    """Minimal mol stand-in -- the emitter only needs these three."""
    natm = 2

    def atom_coords(self, unit="Ang"):
        return np.zeros((2, 3))

    def atom_symbol(self, i):
        return "O"


def _header_keys(path):
    return [
        line.split(":", 1)[0].replace("# runtime.", "").strip()
        for line in path.read_text().splitlines()
        if line.startswith("# runtime.")
    ]


def test_every_canonical_key_survives_to_disk(tmp_path):
    """RUNTIME_INFO_KEYS calls itself the source-of-truth list.  If the
    writer drops one of them the list is a lie -- which it was, for
    ``max_memory_mb``."""
    log = tmp_path / "j.molwatch.log"
    ri = {k: "X" for k in RUNTIME_INFO_KEYS}
    MolwatchEmitter(str(log), "j", _Mol(), runtime_info=ri)
    assert set(_header_keys(log)) >= set(RUNTIME_INFO_KEYS)


def test_max_memory_mb_specifically(tmp_path):
    """The key the old whitelist actually lost.  Named on its own so a
    regression reads as itself rather than as a set-difference."""
    log = tmp_path / "j.molwatch.log"
    MolwatchEmitter(str(log), "j", _Mol(), runtime_info={"max_memory_mb": 4000})
    assert "# runtime.max_memory_mb: 4000" in log.read_text()


def test_a_key_the_writer_has_never_heard_of_is_still_written(tmp_path):
    """The writer must not filter at all.

    This is the guard that makes the whole class of bug impossible: any
    future runtime fact reaches disk without anyone remembering to add
    it to a second list.  A reinstated whitelist fails here even if it
    happens to contain every key that exists today.
    """
    log = tmp_path / "j.molwatch.log"
    MolwatchEmitter(str(log), "j", _Mol(),
                    runtime_info={"a_fact_invented_by_this_test": 42})
    assert "# runtime.a_fact_invented_by_this_test: 42" in log.read_text()


def test_values_with_newlines_cannot_break_the_line_parse(tmp_path):
    """Dropping the whitelist widened what can reach this loop, so the
    newline-stripping it always did now matters more."""
    log = tmp_path / "j.molwatch.log"
    MolwatchEmitter(str(log), "j", _Mol(),
                    runtime_info={"gpu_name": "line1\nline2\rline3"})
    text = log.read_text()
    assert "# runtime.gpu_name: line1 line2 line3" in text
    assert len(_header_keys(log)) == 1
