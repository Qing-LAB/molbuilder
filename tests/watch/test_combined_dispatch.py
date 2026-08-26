"""End-to-end parse-equivalence + perf benchmark for the combined-regex
section dispatch refactor (Commit 2 of Strategy D, 2026-06-01).

Why this test exists
====================

Commit 2 changed the SIESTA parser's per-line dispatch from

    for rule in rules:
        if rule.start(line):
            ...
            break

to

    rule = compiled.find_match(line)
    if rule is not None:
        ...

The two implementations are designed to be semantically identical (see
:class:`molbuilder.parsers._rules.CompiledRules`).  But "designed to be"
is not "is".  This test parses a representative set of real SIESTA .out
files and compares the resulting :class:`Trajectory` against a frozen
JSON signature.

Verified before this commit landed: dispatch logs on the same file are
byte-for-byte identical between the OLD per-rule for-loop and the NEW
``CompiledRules.find_match``.  The golden-signature test below is the
durable check that guards against future drift.

Test corpus
===========

Frozen .out files live in ``tests/watch/fixtures/siesta_frozen/``.
COPIES of real SIESTA runs (not the originals in ``projects/``), so the
tests are immune to live SIESTA runs appending to project files mid-
test.  Each file's basename encodes its expected shape so a failure
message points at the run-state pattern, e.g.::

    hemeC-stage1-scf_not_conv-5fr.out
    BDT-stage3-propor_error-32fr.out

Coverage matrix:

  * ``hemeC-stage1-scf_not_conv-5fr.out``       error / SCF_NOT_CONV / 5 frames
    -- exercises: fatal_scf_not_conv rule, multi-frame attempt, max_force
       after each forces block, scf_data + scf_header in 5 cycles.
  * ``hemeC-stage3-scf_not_conv-1fr.out``       error / SCF_NOT_CONV / 1 frame
    -- exercises: single-step run that aborted, ensures small-file edge
       case (torn at first outcoor) handles cleanly.
  * ``hemeC-stage2-run3-finished-42fr.out``     finished / 42 frames
    -- exercises: full successful geometry optimisation, scf_converged
       rule firing every step, end_of_run marker, lattice extraction.
  * ``BDT-stage3-propor_error-32fr.out``        error / propor: ERROR / 32 frames
    -- exercises: fatal_propor_error rule path (different shape from
       fatal_scf_not_conv); tunneling project with NEGF setup.
  * ``BDT_METAL-stage5-propor_error-11fr.out``  error / propor: ERROR / 11 frames
    -- exercises: fatal_propor_error with smaller frame count + metal
       (different element set + force scale).

Together these touch every section rule in the SIESTA parser's list and
both terminal ``run_state`` values the corpus can hold -- ``ended`` and
``stopped`` (`model/parse.md` § 2b).

Goldens
=======

JSON files in ``tests/watch/fixtures/siesta_frozen/goldens/`` carry the
expected :class:`Trajectory` signature -- see ``_siesta_signature.py``
for the field schema.  Signatures are intentionally compact: scalars +
per-frame summary stats (energy, force-sum, scf-history-len), not the
raw N x 3 arrays.  This catches drift on every parser-observable field
without committing megabytes of golden data.

Regenerating goldens
====================

If a frozen .out file is intentionally swapped or a parser fix changes
the expected output, regenerate via::

    python -c "import tests.watch.test_combined_dispatch as t; t.regenerate_goldens()"

The whole point of the golden is to detect unintended drift, so audit
the new parse output by hand before regenerating.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from molbuilder.parse.engines.siesta import SiestaParser

# Local import via sys.path trick mirrored from test_section_rule.py
sys.path.insert(0, os.path.dirname(__file__))
from _siesta_signature import signature  # noqa: E402


_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "siesta_frozen"
_GOLDEN_DIR  = _FIXTURE_DIR / "goldens"


def _frozen_files():
    """Return sorted list of frozen .out fixture paths."""
    if not _FIXTURE_DIR.exists():
        return []
    return sorted(_FIXTURE_DIR.glob("*.out"))


# --------------------------------------------------------------------
# Equivalence: each frozen file's parse signature == its committed golden.
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "out_path",
    _frozen_files() or [pytest.param(None,
                          marks=pytest.mark.skip(
                              reason="no frozen SIESTA fixtures; populate "
                                     "tests/watch/fixtures/siesta_frozen/"))],
    ids=lambda v: v.name if hasattr(v, "name") else str(v),
)
def test_dispatch_equivalence_against_golden(out_path):
    """Parse a frozen .out file; compare against the JSON golden.

    Each frozen fixture name encodes its expected shape so a failure
    message instantly tells you which run-state / error path drifted.
    """
    golden_path = _GOLDEN_DIR / (out_path.name + ".signature.json")
    if not golden_path.exists():
        pytest.fail(
            f"no golden for {out_path.name} at {golden_path}; "
            f"run regenerate_goldens() after auditing parse output"
        )

    actual = signature(SiestaParser().parse(str(out_path)))
    with golden_path.open() as fh:
        golden = json.load(fh)

    # Compare top-level scalars first so the error message points at
    # the highest-level drift before diving into per-frame detail.
    for key in ("source_format", "run_state", "error_message",
                "frame_count", "lattice_sum", "parse_warning_count"):
        assert actual[key] == golden[key], (
            f"{out_path.name}: top-level field {key!r} drifted: "
            f"got {actual[key]!r}, golden {golden[key]!r}"
        )
    assert actual["runtime_info"] == golden["runtime_info"], (
        f"{out_path.name}: runtime_info drift: "
        f"got {actual['runtime_info']!r}, golden {golden['runtime_info']!r}"
    )
    # Per-frame.  Index-by-index so the first divergent frame is
    # reported, not a wall of all frames.
    assert len(actual["frames"]) == len(golden["frames"]), (
        f"{out_path.name}: frame count: got {len(actual['frames'])}, "
        f"golden {len(golden['frames'])}"
    )
    for i, (af, gf) in enumerate(zip(actual["frames"], golden["frames"])):
        for key in ("step_index", "energy", "max_force",
                    "forces_sum", "forces_n",
                    "coords_sum", "coords_n", "scf_history_len"):
            assert af[key] == gf[key], (
                f"{out_path.name} frame {i}: {key!r} drifted: "
                f"got {af[key]!r}, golden {gf[key]!r}"
            )


# --------------------------------------------------------------------
# Run-state coverage check.  Ensures the corpus actually exercises
# every terminal run_state.  If someone deletes a fixture and the
# corpus shrinks to all-finished or all-error, the equivalence test
# above still passes but the coverage promise is broken.
# --------------------------------------------------------------------


def test_corpus_covers_every_run_state():
    """The frozen fixture set must cover the ENDINGS a real run reaches.

    `model/parse.md` § 2b's vocabulary is
    ``running``/``ended``/``stopped``/``out_of_memory``/``unknown``.
    ``running`` is excluded because a torn live run is not reproducible
    as a fixture, and ``out_of_memory`` because the corpus has no OOM
    capture yet -- when one is added, assert it here."""
    pairs = _frozen_files()
    if not pairs:
        pytest.skip("no frozen SIESTA fixtures")
    states = set()
    error_messages = set()
    for p in pairs:
        golden_path = _GOLDEN_DIR / (p.name + ".signature.json")
        with golden_path.open() as fh:
            g = json.load(fh)
        states.add(g["run_state"])
        if g["error_message"]:
            # Bucket by error PREFIX so propor ERROR and SCF_NOT_CONV
            # both register even though their full messages differ
            # between SIESTA versions.
            error_messages.add(g["error_message"][:20])
    assert "ended"   in states, "corpus lacks a run that reached its end"
    assert "stopped" in states, "corpus lacks a run that stopped short"
    # Different error paths through the parser must each be represented.
    assert any(e.startswith("SCF_NOT_CONV") for e in error_messages), (
        "corpus lacks a SCF_NOT_CONV failure")
    assert any("propor" in e for e in error_messages), (
        "corpus lacks a propor ERROR failure")


# --------------------------------------------------------------------
# Determinism: re-parsing the SAME file twice must yield the SAME
# signature.  Catches accidental mutable-state-in-rule-list bugs that
# only surface across multiple parses (rule closure captures a list
# by reference instead of by value, etc).
# --------------------------------------------------------------------


def test_parse_is_deterministic():
    """Pick the smallest frozen file, parse twice, assert identical
    signatures.  Guards against accidental mutable state in either
    ``compile_rules`` (cached state across parses) or rule closures."""
    pairs = _frozen_files()
    if not pairs:
        pytest.skip("no frozen SIESTA fixtures")
    out_path = min(pairs, key=lambda p: p.stat().st_size)
    p = SiestaParser()
    sig_a = signature(p.parse(str(out_path)))
    sig_b = signature(p.parse(str(out_path)))
    assert sig_a == sig_b, (
        f"non-deterministic parse on {out_path.name}: two back-to-back "
        f"parses returned different signatures.  Likely cause: mutable "
        f"state captured by reference in a rule closure."
    )


# --------------------------------------------------------------------
# Determinism across parser instances: a fresh SiestaParser() must
# produce the same signature as one used previously.  Catches accidental
# module-global state in _rules.py (the kind of bug that retired the
# D1 cache).
# --------------------------------------------------------------------


def test_parse_independent_across_parser_instances():
    """A NEW :class:`SiestaParser` instance must produce the same
    signature as one used to parse a different file first.  Catches
    module-global state leakage."""
    pairs = _frozen_files()
    if len(pairs) < 2:
        pytest.skip("need at least 2 frozen fixtures")
    file_a, file_b = pairs[0], pairs[1]

    # Reference: parse B with a fresh parser.
    ref_b = signature(SiestaParser().parse(str(file_b)))

    # Cross: parse A, then B, with the SAME parser.  B's signature must
    # still match ref_b -- if it doesn't, A's parse left state behind.
    p = SiestaParser()
    _ = p.parse(str(file_a))
    cross_b = signature(p.parse(str(file_b)))
    assert cross_b == ref_b, (
        f"parsing {file_a.name} BEFORE {file_b.name} changed the "
        f"signature of {file_b.name}: module-global state is leaking "
        f"across parses.  Check _rules.py for shared mutable state."
    )


# --------------------------------------------------------------------
# Perf regression guard.  Not a strict <X seconds budget (machine-
# variant) -- just verifies the parse completes in a sane wall-clock
# window so a 10x slowdown gets caught in CI.
# --------------------------------------------------------------------


def test_parse_perf_envelope():
    """Parse the largest frozen file; assert wall time is under a
    generous ceiling.

    Tuned to catch order-of-magnitude regressions, not micro-drift.
    The pre-refactor parser handled the 1.6 MB stage2.out in ~1.0 s on
    a 2024 dev box; the combined-regex path is faster.  We allow 8 s
    which is ~8x the pre-refactor budget -- enough headroom for a
    cold-cache CI box without being so loose a real regression slips
    through.
    """
    pairs = _frozen_files()
    if not pairs:
        pytest.skip("no frozen SIESTA fixtures")
    out_path = max(pairs, key=lambda p: p.stat().st_size)
    p = SiestaParser()
    start = time.perf_counter()
    t = p.parse(str(out_path))
    elapsed = time.perf_counter() - start
    assert len(t.frames) > 0, f"{out_path.name} parsed to zero frames"
    assert elapsed < 8.0, (
        f"parse of {out_path.name} ({out_path.stat().st_size} bytes) "
        f"took {elapsed:.2f}s, expected < 8s.  Possible regression in "
        f"the combined-regex dispatch path."
    )


# --------------------------------------------------------------------
# Maintenance helper.  Importable function, not a test.  Run via
# `python -c "import tests.watch.test_combined_dispatch as t; t.regenerate_goldens()"`
# after auditing the new parse output.
# --------------------------------------------------------------------


def regenerate_goldens():
    """Re-parse every frozen .out file and overwrite the JSON golden.

    Only do this after auditing the new parse output by hand -- this
    defeats the regression-detection purpose if run blindly.
    """
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    p = SiestaParser()
    for out_path in _frozen_files():
        sig = signature(p.parse(str(out_path)))
        golden_path = _GOLDEN_DIR / (out_path.name + ".signature.json")
        with golden_path.open("w") as fh:
            json.dump(sig, fh, indent=2, sort_keys=True)
        print(f"wrote {golden_path}  ({sig['frame_count']} frames, "
              f"run_state={sig['run_state']})")
