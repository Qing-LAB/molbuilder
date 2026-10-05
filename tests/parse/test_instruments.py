"""The wrapper's own files, read through the registry — `parse.md` § 5c.

Three parsers, one for each thing the wrapper measures beside the deck,
plus the resolver that decides which of two sources to believe. These
lived in `bench/result.py` and read bytes directly until 2026-09-04.

The `.log` / `.csv` suffixes join a registry that RAISES on ambiguity, so
the first test here is that they claim their own files and nothing else's.
"""
from __future__ import annotations

import pytest

from molbuilder.parse.instruments import utilisation


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 7 tests here read an SCF-timing log, a monitor log or a
# .util.csv typed by hand (`process/testing.md` § 6).


def test_a_relaxations_step_boundaries_are_not_iterations():
    """The delta INTO a new SCF's first row holds the previous SCF's end, the
    forces, the move and the next step's setup -- not an iteration, for the
    reason the first delta is dropped.  A measured fixture: the timing log of
    a real 2-rank H2 CG relaxation (2026-09-26, `fixtures/scf_timing/`),
    five SCFs of 8, 6, 4, 6 and 6 iterations; its four boundaries took about
    2.4 s each against 0.58 s an iteration within an SCF, and averaged in
    they read 0.8 s.

    MUTATION THIS MUST FAIL AGAINST: keep the deltas into iteration 1."""
    from pathlib import Path
    from molbuilder.parse.instruments.scf_timing_rows import scf_timing_metrics
    text = (Path(__file__).parent / "fixtures" / "scf_timing"
            / "h2mon_01_coarse-run0.scf-timing.log").read_text()
    m = scf_timing_metrics(text)
    # 29 deltas, 4 of them boundaries, the first dropped as warm-up
    assert (m["iters_measured"], m["rows"]) == (24, 30), m
    assert m["s_per_iter"] == pytest.approx(0.58, abs=0.005), m


# ---------------------------------------------------------------------------
#  A number that was never measured
# ---------------------------------------------------------------------------
#  Both of these shipped on 2026-09-04 and both put a value where an
#  absence belonged -- forbidden pattern #9, `model/parse.md` § 7.  The
#  suite was green with both defects live, because every fixture in it
#  carries a well-formed epoch column and a monitor summary that states
#  BOTH means; nothing asked what happens when the measurement is
#  partial, which is the shape a KILLED trial leaves and the one a
#  benchmark most needs to read.


def test_a_half_stated_summary_is_not_called_the_monitors_own():
    """One `util_basis` label cannot describe two means of different origin.

    `monitor.summary()` writes its ``cpu mean=`` and ``gpuN sm mean=``
    bits independently, so a line truncated mid-write states one and not
    the other.  Stamping ``monitor-summary`` over the half that came
    from the change-gated csv is exactly what `plan` § E2 says the field
    exists to prevent.
    """
    mixed = utilisation({"stated_cpu_mean_pct": 40.0},
                        {"cpu_mean_pct": 31.5, "gpu_sm_mean_pct": 77.0})
    assert mixed["cpu_mean_pct"] == 40.0, "the exact CPU figure still wins"
    assert mixed["gpu_sm_mean_pct"] == 77.0, "the csv still supplies the GPU"
    assert mixed["util_basis"] == "mixed", (
        "the GPU figure is a reconstruction and the label says otherwise: "
        + repr(mixed))

    # A CPU-only node is NOT mixed: neither source carries a GPU mean, so
    # there is one basis and it is the monitor's.
    cpu_only = utilisation({"stated_cpu_mean_pct": 40.0},
                           {"cpu_mean_pct": 31.5, "monitored_elapsed_s": 9.0})
    assert cpu_only["util_basis"] == "monitor-summary", repr(cpu_only)


def test_a_symlink_loop_does_not_break_the_envelope(tmp_path):
    """Describing where a result came from must not fail after reading it.

    `_source_str` guarded `resolve()` with `except OSError` and named a
    symlink loop as the reason.  A loop raises `RuntimeError`, which is
    not an `OSError`, so the guard never fired on its own example --
    every builder would have propagated it out of a parse that had
    already succeeded.
    """
    import os
    from molbuilder.parse.types import _source_str

    a, b = tmp_path / "A", tmp_path / "B"
    os.symlink(b, a)
    os.symlink(a, b)
    assert _source_str(a) == str(a), "a loop must fall back to the raw string"

    # An ordinary path still resolves, or the guard is just a mute.
    real = tmp_path / "real.log"
    real.write_text("x")
    assert _source_str(real) == str(real.resolve())
