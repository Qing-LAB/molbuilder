"""The wrapper's own files, read through the registry — `parse.md` § 5c.

Three parsers, one for each thing the wrapper measures beside the deck,
plus the resolver that decides which of two sources to believe.
"""
from __future__ import annotations

import pytest

from molbuilder import monitor as M
from molbuilder import monitor_log as ML
from molbuilder.monitor import UtilAccum, UtilSample
from molbuilder.parse.instruments import utilisation
from molbuilder.parse.instruments.monitor import monitor_metrics


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


# ---------------------------------------------------------------------------
#  The monitor's lines, written and read (`monitor_log`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sm, bound", [(90.0, "gpu"), (50.0, "host"),
                                       (70.0, "mixed")],
                         ids=["GPU-bound", "host-bound", "mixed"])
def test_what_the_monitor_writes_its_reader_reads(sm, bound, monkeypatch):
    """`model/parse.md` § 5c, the `monitor_metrics` row, and
    `execution/run-reports.md` § 2.5, the closing record: each fact, from the
    lines the monitor's own writers make.  THE SILENT FAILURE: a reader's
    pattern built wrong from the words `monitor_log` holds finds nothing, and
    a run states no verdict, no memory peak and no end, with no error.  This
    is the only reading of these lines short of a real run.  The machine's
    own reads are given; what is tested is the words."""
    monkeypatch.setattr(M, "machine_identity", lambda: {
        "node": "n1", "cores": "4", "mem_gb": "8", "gpu": "RTX A, RTX B"})
    monkeypatch.setattr(M, "_read_mem_peak_gb", lambda: 2.5)
    monkeypatch.setattr(M, "_read_mem_limit_gb", lambda: 8.0)
    monkeypatch.setattr(M, "_read_cpu_used_ns", lambda: (1, "cgroup-v2"))
    monkeypatch.setattr(M, "_read_mem_used_gb", lambda: (1.0, "cgroup-v2"))
    accum = UtilAccum()
    accum.add(UtilSample(epoch=1.0, cpu_pct=40.0, mem_gb=1.0,
                         gpus=[(0, sm, 1.0, 2.0)]))
    metrics = monitor_metrics("\n".join([
        f"[t] {ML.MACHINE_TAG} {M.machine_line()}",
        f"[t] {ML.UTIL_SUMMARY_TAG} {accum.summary()}",
        f"[t] {ML.UTIL_BASIS_TAG} {M.measurement_provenance(M._Basis(4, 'affinity'))}",
        "[t] " + ML.ended_record("watched pid 1 gone")]))
    assert metrics == {
        "machine": {"node": "n1", "cores": "4", "mem_gb": "8",
                    "gpu": "RTX A, RTX B"},
        "bound": bound,
        "stated_cpu_mean_pct": 40.0,
        "stated_gpu_sm_mean_pct": sm,
        "mem_basis": "cgroup-v2",
        "mem_peak_kernel_gb": 2.5,
        "mem_limit_gb": 8.0,
        "ended_at": "t",
        "ended_why": "watched pid 1 gone"}
