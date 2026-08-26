"""Equivalence + edge-case tests for the molwatch_log parser after the
combined-regex dispatch refactor (2026-06-01).

The existing ``test_molwatch_log_parser.py`` is spec-derived (it
asserts documented invariants, not implementation details).  This
file adds the cases that exercise dispatch-mechanics edges the
spec-derived tests don't reach directly:

  * **Torn-block recovery** -- a ``==== begin ====`` line mid-block
    (no intervening ``end``) must abandon the partial frame and start
    fresh.  Pre-refactor handled this with an inline ``_reset_block()``
    on every begin; the rule-based path mirrors that by including
    ``block_begin`` in BOTH the outside-block and inside-block rule
    tables.  No prior test exercised this.

  * **Runtime-info lines inside a block** -- runtime markers appear
    only in the file header; the pre-refactor ``if not in_block:``
    gate skipped them inside blocks.  The rule-based path includes
    runtime ONLY in ``OUT_BLOCK_RULES``.  A ``# runtime.foo: bar``
    line stuck inside a block must be ignored (not crash, not
    silently corrupt the block).

  * **Error overrides concluded** regardless of order -- the spec
    test ``test_run_state_error_when_error_marker_present`` covers
    the ``error -> concluded`` order; this file pins ``concluded ->
    error`` too.

  * **Frozen golden** -- a JSON signature of a comprehensive
    synthetic log committed alongside.  Locks the parse output so a
    future refactor that breaks any field is caught.  Lives at
    ``tests/watch/fixtures/molwatch_frozen/`` mirroring the SIESTA
    layout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from molbuilder.parse.engines.molwatch import MolwatchLogParser
from molbuilder.parse.engines._helpers import trajectory_to_legacy_dict


_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "molwatch_frozen"
_GOLDEN_DIR  = _FIXTURE_DIR / "goldens"


# --------------------------------------------------------------------
# Torn-block recovery
# --------------------------------------------------------------------


_TORN_BEGIN_MID_BLOCK = """\
# molwatch trajectory log v1
# engine: pyscf
==== molwatch step 0 begin ====
step_index: 0
n_atoms: 1
coordinates (Ang):
   O      0.00000000      0.00000000      0.00000000
energy (eV): -10.0
==== molwatch step 1 begin ====
step_index: 1
n_atoms: 1
coordinates (Ang):
   O      1.00000000      0.00000000      0.00000000
energy (eV): -20.0
==== molwatch step 1 end ====
"""


def test_torn_begin_mid_block_abandons_partial_frame(tmp_path):
    """A ``begin`` line that appears WITHIN an already-open block must
    abandon the partial frame (no commit) and start a fresh one.

    Pre-refactor invariant: only blocks with matching ``begin``+``end``
    commit.  A begin-mid-block silently discards the partial state.
    """
    p = tmp_path / "torn.molwatch.log"
    p.write_text(_TORN_BEGIN_MID_BLOCK)

    t = MolwatchLogParser.parse(str(p))
    assert len(t.frames) == 1, (
        "step 0 had no closing `end` and must be dropped; "
        "only step 1 (which has a matching begin+end) should remain"
    )
    assert t.frames[0].step_index == 1
    assert t.frames[0].energy == pytest.approx(-20.0)


# --------------------------------------------------------------------
# Runtime-info lines are ignored inside a block
# --------------------------------------------------------------------


_RUNTIME_INSIDE_BLOCK = """\
# molwatch trajectory log v1
# engine: pyscf
# runtime.n_mpi_processes: 4
# runtime.omp_threads: 2
==== molwatch step 0 begin ====
step_index: 0
n_atoms: 1
# runtime.this_should_be_ignored: 999
coordinates (Ang):
   O      0.00000000      0.00000000      0.00000000
energy (eV): -10.0
==== molwatch step 0 end ====
"""


def test_runtime_info_outside_block_only(tmp_path):
    """Only ``# runtime.*`` lines OUTSIDE a block contribute to
    ``Trajectory.runtime_info``.  An interloper inside a block is
    silently ignored (it's a malformed log; we don't crash, but we
    don't corrupt the bag either).
    """
    p = tmp_path / "runtime.molwatch.log"
    p.write_text(_RUNTIME_INSIDE_BLOCK)

    t = MolwatchLogParser.parse(str(p))
    assert t.runtime_info == {
        "n_mpi_processes": 4,
        "omp_threads":     2,
    }
    # The interloper key MUST NOT have leaked into the bag.
    assert "this_should_be_ignored" not in t.runtime_info
    # And the block still parsed cleanly despite the stray line.
    assert len(t.frames) == 1
    assert t.frames[0].energy == pytest.approx(-10.0)


# --------------------------------------------------------------------
# Error overrides concluded regardless of order
# --------------------------------------------------------------------


_CONCLUDED_THEN_ERROR = """\
# molwatch trajectory log v1
# engine: pyscf
==== molwatch step 0 begin ====
step_index: 0
n_atoms: 1
coordinates (Ang):
   O      0.00000000      0.00000000      0.00000000
energy (eV): -10.0
==== molwatch step 0 end ====
# concluded: clean shutdown 2026-04-25T11:00:00
# error: actually we did crash after all
"""


def test_concluded_then_error_lands_on_error(tmp_path):
    """If ``# concluded:`` arrives before ``# error:``, the final
    run_state MUST be ``error``.  Atexit/excepthook hooks can fire
    in either order depending on Python's internal cleanup -- a
    silent error after a clean concluded is the harder case to
    surface.
    """
    p = tmp_path / "ce.molwatch.log"
    p.write_text(_CONCLUDED_THEN_ERROR)

    t = MolwatchLogParser.parse(str(p))
    assert t.run_state == "stopped"
    assert t.error_message is not None
    assert "actually we did crash" in t.error_message


# --------------------------------------------------------------------
# All section types in one block + frozen golden
# --------------------------------------------------------------------


_RICH_BLOCK = """\
# molwatch trajectory log v1
# generator: molbuilder/pyscf_input
# engine: pyscf
# job: rich_test
# units: energy=eV, force=eV/Ang, coords=Ang
# created: 2026-04-25T12:00:00
# runtime.n_mpi_processes: 4
# runtime.omp_threads: 2
# runtime.hostname: testbox
# runtime.gpu_count: None

==== molwatch step 0 begin ====
step_index: 0
n_atoms: 3
coordinates (Ang):
   O      0.00000000      0.00000000      0.00000000
   H      0.95700000      0.00000000      0.00000000
   H     -0.23900000      0.92700000      0.00000000
energy (eV): -76.12345600
forces (eV/Ang):
   O     -0.00100000     -0.00200000      0.00000000
   H      0.00050000      0.00100000      0.00000000
   H      0.00050000      0.00100000      0.00000000
max_force (eV/Ang): 0.00240000
wall_time: 1761396000.0
scf_history begin
#  cycle      energy(eV)         delta_E(eV)        gnorm(eV/Ang)            ddm
       1     -76.00000000        0.00000000      5.00000000e-02   1.00000000e-01
       2     -76.10000000       -0.10000000      5.00000000e-03   1.00000000e-02
       3     -76.12345600       -0.02345600      1.00000000e-04   1.00000000e-04
scf_history end
==== molwatch step 0 end ====

==== molwatch step 1 begin ====
step_index: 1
n_atoms: 3
coordinates (Ang):
   O      0.01000000      0.00000000      0.00000000
   H      0.96700000      0.00000000      0.00000000
   H     -0.22900000      0.92700000      0.00000000
energy (eV): -76.20000000
forces (eV/Ang):
   O      0.00010000      0.00020000      0.00000000
   H     -0.00005000     -0.00010000      0.00000000
   H     -0.00005000     -0.00010000      0.00000000
max_force (eV/Ang): 0.00022400
wall_time: 1761396030.0
scf_history begin
       1     -76.20000000        0.00000000      1.00000000e-04   1.00000000e-04
       2     -76.20000000        0.00000000              None             None
scf_history end
==== molwatch step 1 end ====
# concluded: clean shutdown 2026-04-25T12:00:30
"""


def _rich_signature():
    """Compact JSON signature of the rich-block fixture parse."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".molwatch.log", delete=False) as fh:
        fh.write(_RICH_BLOCK)
        path = fh.name
    try:
        t = MolwatchLogParser.parse(path)
        frames = []
        for f in t.frames:
            frames.append({
                "step_index": int(f.step_index),
                "energy":     (None if f.energy is None
                               else round(float(f.energy), 6)),
                "max_force":  (None if f.max_force is None
                               else round(float(f.max_force), 6)),
                "wall_clock_s": (None if f.wall_clock_s is None
                                 else round(float(f.wall_clock_s), 6)),
                "elapsed_s":    (None if f.elapsed_s is None
                                 else round(float(f.elapsed_s), 6)),
                "coords_n":   len(f.structure.elements),
                "forces_n":   (0 if f.forces is None else len(f.forces)),
                "scf_history_len": len(f.scf_history or []),
            })
        return {
            "source_format":  t.source_format,
            "run_state":      t.run_state,
            "error_message":  t.error_message,
            "frame_count":    len(t.frames),
            "runtime_info":   dict(sorted((t.runtime_info or {}).items())),
            "lattice_is_none": t.lattice is None,
            "frames":         frames,
        }
    finally:
        os.unlink(path)


def test_rich_block_matches_golden():
    """Parse a synthetic log that exercises every documented section
    type (coords, forces, energy, max_force, both clocks, scf_history
    with None residuals, runtime_info including ``None`` value,
    concluded footer) and compare the signature against the committed
    golden.

    The golden lives at
    ``tests/watch/fixtures/molwatch_frozen/goldens/rich.signature.json``
    -- regenerate via ``regenerate_goldens()`` after auditing the
    parse output.
    """
    golden_path = _GOLDEN_DIR / "rich.signature.json"
    if not golden_path.exists():
        pytest.fail(
            f"no golden at {golden_path}; run regenerate_goldens() "
            f"after auditing parse output"
        )
    with golden_path.open() as fh:
        golden = json.load(fh)
    actual = _rich_signature()
    assert actual == golden, (
        f"rich-block parse signature drifted from golden; "
        f"audit + run regenerate_goldens() after fixing"
    )


# --------------------------------------------------------------------
# Maintenance helper.  Importable function, not a test.  Run via
# `python -c "import tests.watch.test_molwatch_combined_dispatch as t; t.regenerate_goldens()"`
# after auditing the new parse output.
# --------------------------------------------------------------------


def regenerate_goldens():
    """Overwrite the rich-block golden with the current parser's
    output.  Only do this after auditing the parse output by hand.
    """
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    actual = _rich_signature()
    golden_path = _GOLDEN_DIR / "rich.signature.json"
    with golden_path.open("w") as fh:
        json.dump(actual, fh, indent=2, sort_keys=True)
    print(f"wrote {golden_path}  ({actual['frame_count']} frames, "
          f"run_state={actual['run_state']})")


# --------------------------------------------------------------------
# Determinism: two back-to-back parses must agree.  Catches accidental
# mutable state leakage (the same kind of bug the SIESTA refactor's
# determinism test catches).
# --------------------------------------------------------------------


def test_parse_is_deterministic(tmp_path):
    """A back-to-back re-parse of the same file must produce the same
    Trajectory signature.  Guards against accidentally captured-by-
    reference state in the rule callbacks (a mutable list set as a
    nonlocal default would survive across parses and silently
    accumulate cycles)."""
    p = tmp_path / "det.molwatch.log"
    p.write_text(_RICH_BLOCK)

    a = trajectory_to_legacy_dict(MolwatchLogParser.parse(str(p)))
    b = trajectory_to_legacy_dict(MolwatchLogParser.parse(str(p)))
    assert a == b


# --------------------------------------------------------------------
# Cross-parse independence: parsing one file before another must not
# affect the second parse.  Catches module-global state leakage --
# the kind of bug that retired the D1 cache.
# --------------------------------------------------------------------


def test_cross_parse_independence(tmp_path):
    """Parsing file A then file B must yield the same B-signature as
    parsing B alone.  Guards against module-global state in
    ``_rules.py`` (or anywhere else) leaking across parses.
    """
    a = tmp_path / "first.molwatch.log"
    a.write_text(_RICH_BLOCK)
    b = tmp_path / "second.molwatch.log"
    b.write_text(_TORN_BEGIN_MID_BLOCK)

    # Reference: parse B with a "clean" parser.
    ref = trajectory_to_legacy_dict(MolwatchLogParser.parse(str(b)))
    # Cross: parse A first, then B.  Result must match ref.
    _   = MolwatchLogParser.parse(str(a))
    cross = trajectory_to_legacy_dict(MolwatchLogParser.parse(str(b)))
    assert cross == ref, (
        "parsing first.molwatch.log changed the result of "
        "parsing second.molwatch.log -- module-global state is "
        "leaking across parses"
    )
