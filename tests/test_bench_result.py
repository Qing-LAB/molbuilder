"""Tests for the bench-result schema + parsers + winner logic
(molbuilder/bench/result.py)."""
from __future__ import annotations

import pytest

from molbuilder.bench.result import (
    BenchPoint, BenchResult, build_bench_result, choose_winner,
    parse_mpi_ranks, parse_sacct_mem, parse_scf_timing,
    parse_util_csv_peak_mem, parse_util_summary, recommend_resources,
)


def test_parse_mpi_ranks():
    assert parse_mpi_ranks("* Running on 20 nodes in parallel.\n") == 20
    assert parse_mpi_ranks("header\n* Running on   8 nodes in parallel.") == 8
    assert parse_mpi_ranks("no parallel line here") is None


# --------------------------------------------------------------------- #
#  parsers                                                              #
# --------------------------------------------------------------------- #

# Real gpu-k8 trace (5 iters): steady-state ~1538 s/iter; dropping the
# first delta (iter1->2) leaves iters 3-5.
_TIMING = """\
1782535754.956785509 1    scf:    1 -1731471.140364
1782537294.309523016 2    scf:    2 -1739770.552290
1782538833.546167403 3    scf:    3 -1739904.135072
1782540369.867679698 4    scf:    4 -1741157.398168
1782541907.019907646 5    scf:    5 -1741564.799727
"""


def test_parse_scf_timing_steady_state():
    r = parse_scf_timing(_TIMING)
    # deltas: 1539.35, 1539.24, 1536.32, 1537.15 -> drop first -> mean of 3.
    assert r["iters_measured"] == 3
    assert r["s_per_iter"] == pytest.approx(1537.6, abs=0.5)


def test_parse_scf_timing_too_few():
    assert parse_scf_timing("100.0 1 scf: 1\n") == \
        {"s_per_iter": None, "iters_measured": 0}
    assert parse_scf_timing("")["s_per_iter"] is None


@pytest.mark.parametrize("log,cpu,sm,bound", [
    ("[t] [UTIL-SUMMARY] cpu mean=40% (35-45); gpu0 sm mean=91% (80-97) "
     "-> GPU-bound (GPU saturated)", 40.0, 91.0, "gpu"),
    ("[t] [UTIL-SUMMARY] cpu mean=95% (90-99); gpu0 sm mean=48% (10-70) "
     "-> host/CPU-bound (GPU starved)", 95.0, 48.0, "host"),
    ("[t] [UTIL-SUMMARY] cpu mean=60% (50-70); gpu0 sm mean=70% (60-80) "
     "-> mixed (GPU not saturated)", 60.0, 70.0, "mixed"),
])
def test_parse_util_summary(log, cpu, sm, bound):
    r = parse_util_summary(log)
    assert r["cpu_mean_pct"] == cpu
    assert r["gpu_sm_mean_pct"] == sm
    assert r["bound"] == bound


def test_parse_util_summary_multi_gpu_takes_max():
    r = parse_util_summary(
        "[UTIL-SUMMARY] cpu mean=30% (20-40); gpu0 sm mean=80% (..); "
        "gpu1 sm mean=92% (..) -> GPU-bound (..)")
    assert r["gpu_sm_mean_pct"] == 92.0


def test_parse_util_summary_absent():
    assert parse_util_summary("nothing here")["bound"] is None


def test_parse_util_csv_peak_mem():
    csv = ("epoch,iso,cpu_pct,mem_gb,gpu0_sm,gpu0_memutil,gpu0_vram_gb\n"
           "1,a,3,137.2,0,0,0\n"
           "2,b,5,260.9,0,0,0\n"
           "3,c,4,180.0,0,0,0\n")
    assert parse_util_csv_peak_mem(csv) == 260.9
    assert parse_util_csv_peak_mem("epoch,mem_gb\n") is None
    assert parse_util_csv_peak_mem("no_mem_col\n1\n") is None


@pytest.mark.parametrize("text,expect", [
    ("57522630.ba+ batch CANCELLED cpu=10-14:33:36,energy=0.01G,"
     "mem=433.15G,pages=0", 433.2),
    ("MaxRSS\nmem=128.00G,foo=1", 128.0),
    ("mem=512000M", 500.0),                # 512000 MiB -> 500 GiB
    ("no memory here", None),
])
def test_parse_sacct_mem(text, expect):
    got = parse_sacct_mem(text)
    if expect is None:
        assert got is None
    else:
        assert got == pytest.approx(expect, abs=0.2)


# --------------------------------------------------------------------- #
#  winner + recommend + schema                                          #
# --------------------------------------------------------------------- #


def _pts():
    return [
        BenchPoint("gpu-k8", "gpu", {"gpus": 1, "ranks_per_gpu": 8},
                   {"s_per_iter": 1538.0, "peak_rss_gb": 25.2},
                   bound="gpu", state="completed"),
        BenchPoint("gpu-k4", "gpu", {"gpus": 1, "ranks_per_gpu": 4},
                   {"s_per_iter": 1938.0, "peak_rss_gb": 22.3},
                   bound="gpu", state="completed"),
        BenchPoint("cpu-np64", "cpu", {"ranks": 64},
                   {"s_per_iter": None, "peak_rss_gb": 433.2},
                   state="timeout"),
    ]


def test_choose_winner_fastest_completed():
    c = choose_winner(_pts())
    assert c["engine"] == "gpu"
    assert c["knobs"] == {"gpus": 1, "ranks_per_gpu": 8}
    assert "gpu-k8 fastest" in c["rationale"]
    assert "vs gpu-k4" in c["rationale"]


def test_choose_winner_ignores_non_completed_and_timeless():
    # only the timed-out CPU point -> no winner
    pts = [BenchPoint("cpu", "cpu", {}, {"s_per_iter": None},
                      state="timeout")]
    assert choose_winner(pts) == {}


def test_recommend_from_winner_peak_rss():
    pts = _pts()
    rec = recommend_resources(pts, choose_winner(pts))
    assert rec["mem_gb"] == 29          # ceil(25.2 * 1.15) = ceil(28.98)
    assert "time" in rec and rec["time"].count(":") == 2


def test_recommend_mem_uses_true_ceil():
    # rss*safety just above an integer must round UP (the old +0.999 trick
    # floored values within 0.001 of an int).
    pts = [BenchPoint("g", "gpu", {"gpus": 1, "ranks_per_gpu": 8},
                      {"s_per_iter": 100.0, "peak_rss_gb": 25.218},
                      bound="gpu", state="completed")]
    rec = recommend_resources(pts, choose_winner(pts))
    assert rec["mem_gb"] == 30           # ceil(29.0007), not 29


def test_build_and_round_trip():
    res = build_bench_result(
        _pts(),
        environment={"schema": "molbuilder/environment@1",
                     "scheduler": "slurm"},
        system={"engine": "siesta", "n_atoms": 444},
        now_iso="2026-06-27T22:00:00Z")
    d = res.to_dict()
    assert d["schema"] == "molbuilder/bench-result@1"
    assert d["choice"]["knobs"]["ranks_per_gpu"] == 8
    assert d["points"][0]["metrics"]["s_per_iter"] == 1538.0

    back = BenchResult.from_dict(d)
    assert back.choice["engine"] == "gpu"
    assert back.points[0].label == "gpu-k8"
    assert back.system["n_atoms"] == 444


def test_from_dict_rejects_major_mismatch():
    with pytest.raises(ValueError, match="schema mismatch"):
        BenchResult.from_dict({"schema": "molbuilder/bench-result@2"})


def test_choice_is_consumable_by_format_run():
    # The decoupling contract: the bench-result 'choice' feeds prep-run's
    # adapter.format_run directly (§ 5.4) -- no shared code path.
    from molbuilder.bench.adapters import SlurmAdapter
    from molbuilder.environment import Environment, Topology
    res = build_bench_result(_pts())
    env = Environment(scheduler="slurm",
                      topology=Topology(cores_per_socket=24, gpus_per_node=4,
                                        gpu_type="a100"))
    script = SlurmAdapter().format_run(
        res.choice, env, script_base="prod")["run-production.sh"]
    # winner gpu-k8 (gpus=1, K=8) -> -n 8, c re-resolved 24//8=3.
    assert "sbatch --gres=gpu:a100:1 -n 8 -c 3 prod.sbatch" in script
