"""The wrapper's own files, read through the registry — `parse.md` § 5c.

Three parsers, one for each thing the wrapper measures beside the deck,
plus the resolver that decides which of two sources to believe.
"""
from __future__ import annotations

import pytest

from molbuilder.parse.instruments import utilisation


# ---------------------------------------------------------------------------
#  A number that was never measured
# ---------------------------------------------------------------------------
#  A partial measurement is the shape a KILLED trial leaves and the one
#  a benchmark most needs to read.


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


# ---------------------------------------------------------------------------
#  Which file times a run's SCF rows
# ---------------------------------------------------------------------------

#: A RUN'S FILES BY ROLE, and the role of the file that times it
#: (`scf_timing_rows.timing_file`, `model/parse.md` § 5c).  A SIESTA-family
#: run is timed by the tee its run script cuts from its ``.out`` alone --
#: its progress log is prep's seed, which SIESTA never writes; until 2.64.0
#: a SIESTA run with no tee yet was timed from that seed, and read zero
#: iterations.  A PySCF run is timed by the progress log its deck stamps.
TIMING_ROWS = [
    ({".out", ".molwatch.log"}, None),
    ({".out", ".scf-timing.log", ".molwatch.log"}, ".scf-timing.log"),
    ({".pyscf.log", ".molwatch.log"}, ".molwatch.log"),
]


@pytest.mark.parametrize("roles, timed_by", TIMING_ROWS,
                         ids=["siesta-before-its-tee", "siesta-with-its-tee",
                              "pyscf"])
def test_a_run_is_timed_by_the_file_its_output_says(tmp_path, roles,
                                                    timed_by):
    from molbuilder.parse.instruments.scf_timing_rows import timing_file
    files = {r: tmp_path / f"run{r}" for r in roles}
    got = timing_file(files)
    assert got == (files[timed_by] if timed_by else None), got
