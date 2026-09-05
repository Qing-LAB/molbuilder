"""The wrapper's own files, read through the registry — `parse.md` § 5c.

Three parsers, one for each thing the wrapper measures beside the deck,
plus the resolver that decides which of two sources to believe. These
lived in `bench/result.py` and read bytes directly until 2026-09-04.

The `.log` / `.csv` suffixes join a registry that RAISES on ambiguity, so
the first test here is that they claim their own files and nothing else's.
"""
from __future__ import annotations

import pytest

from molbuilder.parse import detect, parse
from molbuilder.parse.instruments import utilisation

_TIMING = "1000.0 scf: 1\n1002.0 scf: 2\n1004.5 scf: 3\n1007.0 scf: 4\n"
_MONITOR = (
    "[2026-08-27T14:02:11] [MACHINE] node=sol-g042 cores=48 mem_gb=503.5 "
    "gpu=NVIDIA A100-SXM4-80GB\n"
    "[2026-08-27T15:00:00] [UTIL-SUMMARY] cpu mean=73% (10-90); "
    "gpu0 sm mean=88% (0-99) -> GPU-bound\n")
_CSV = "epoch,iso,cpu_pct,mem_gb\n100,x,50.0,1.0\n200,x,90.0,2.0\n300,x,90.0,2.0\n"


@pytest.mark.parametrize("name,body,expect", [
    ("job-run0.scf-timing.log", _TIMING, "scf-timing"),
    ("job.monitor.log",         _MONITOR, "monitor-log"),
    ("job.util.csv",            _CSV,    "util-csv"),
])
def test_each_instrument_is_claimed_by_exactly_one_parser(tmp_path, name, body,
                                                          expect):
    """`detect` is exactly-one-or-raise, so a suffix that overlaps an
    existing parser turns a readable file into an `AmbiguousFormatError`
    — which is how `<job>_optimized.xyz` became an HTTP 500."""
    f = tmp_path / name
    f.write_text(body)
    assert detect(f).name == expect


def test_the_timing_log_drops_the_warm_up_delta(tmp_path):
    """Iteration 1→2 can still carry warm-up, so it is dropped when two
    or more deltas exist: 2.0 discarded, mean of 2.5 and 2.5."""
    f = tmp_path / "job-run0.scf-timing.log"
    f.write_text(_TIMING)
    m = parse(f).metrics
    assert m == {"s_per_iter": 2.5, "iters_measured": 2}


def test_the_monitor_states_the_machine_and_the_verdict(tmp_path):
    f = tmp_path / "job.monitor.log"
    f.write_text(_MONITOR)
    m = parse(f).metrics
    assert m["machine"]["cores"] == "48"
    assert m["machine"]["gpu"] == "NVIDIA A100-SXM4-80GB", (
        "the gpu field is everything after `gpu=` — device models contain "
        "spaces, which is why the monitor puts it last")
    assert m["bound"] == "gpu"
    assert m["stated_cpu_mean_pct"] == 73.0


def test_a_killed_run_has_a_csv_and_no_summary(tmp_path):
    """The trial a benchmark most needs to read.

    The monitor writes `[UTIL-SUMMARY]` only in its terminal branch, so a
    trial the scheduler killed leaves samples and no summary. Nothing may
    raise, and the means must fall back to the csv — saying so in
    `util_basis`, or a reader cannot tell an exact figure from a
    reconstruction.
    """
    mon = tmp_path / "job.monitor.log"
    mon.write_text(_MONITOR.splitlines()[0] + "\n")     # MACHINE line only
    csv = tmp_path / "job.util.csv"
    csv.write_text(_CSV)
    merged = utilisation(parse(mon).metrics, parse(csv).metrics)
    assert merged["util_basis"] == "util-csv"
    assert merged["cpu_mean_pct"] != 73.0, (
        "with no summary the mean must come from the samples")


def test_the_monitors_own_mean_wins_when_it_stated_one(tmp_path):
    """Every tick beats a change-gated subset — `parse.md` § 5a."""
    mon = tmp_path / "job.monitor.log"
    mon.write_text(_MONITOR)
    csv = tmp_path / "job.util.csv"
    csv.write_text(_CSV)
    merged = utilisation(parse(mon).metrics, parse(csv).metrics)
    assert merged["cpu_mean_pct"] == 73.0
    assert merged["util_basis"] == "monitor-summary"


def test_the_reader_can_read_what_the_monitor_writes(monkeypatch, tmp_path):
    """The `[MACHINE]` line's WRITER and READER, joined.

    `monitor.machine_line()` formats it; `instruments/monitor.py`'s regex
    reads it back. Both are tested — separately. Nothing fed one to the
    other, so renaming a key (`cores=` -> `ncores=`) leaves the writer's
    test green (it only inspects the `gpu=` tail) while the reader
    silently answers `{}` and every trial loses its machine.

    The `gpu` field is last on purpose, because device models contain
    spaces and the reader takes everything after `gpu=`. That is a
    contract BETWEEN the two, so it is asserted between them.
    """
    from molbuilder import monitor

    monkeypatch.setattr(monitor, "machine_identity", lambda: {
        "node": "sol-g042", "cores": "48", "mem_gb": "503.5",
        "gpu": "NVIDIA A100-SXM4-80GB, NVIDIA H200"})

    log = tmp_path / "job.monitor.log"
    log.write_text(f"[2026-08-27T14:02:11] [MACHINE] {monitor.machine_line()}\n")

    machine = parse(log).metrics["machine"]
    assert machine == {"node": "sol-g042", "cores": "48", "mem_gb": "503.5",
                       "gpu": "NVIDIA A100-SXM4-80GB, NVIDIA H200"}, (
        f"the reader could not reconstruct what the writer wrote: {machine}")
