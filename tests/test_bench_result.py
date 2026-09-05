"""Tests for the bench-result schema + parsers + winner logic
(molbuilder/bench/result.py)."""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.bench.result import (
    BenchPoint, BenchResult, build_bench_result, choose_winner,
    compare_asked_to_ran, parse_effective_run, parse_mpi_ranks,
    parse_sacct_mem,
)
# The wrapper's own instruments are registered parsers since 2026-09-04
# (`parse.md` § 5c); the logic is unchanged, only its address moved.
from molbuilder.parse.instruments.monitor import monitor_metrics
from molbuilder.parse.instruments.scf_timing import scf_timing_metrics
from molbuilder.parse.instruments.util_csv import util_csv_metrics


def _bound(text):
    return monitor_metrics(text)["bound"]


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
    r = scf_timing_metrics(_TIMING)
    # deltas: 1539.35, 1539.24, 1536.32, 1537.15 -> drop first -> mean of 3.
    assert r["iters_measured"] == 3
    assert r["s_per_iter"] == pytest.approx(1537.6, abs=0.5)


def test_parse_scf_timing_too_few():
    assert scf_timing_metrics("100.0 1 scf: 1\n") == \
        {"s_per_iter": None, "iters_measured": 0}
    assert scf_timing_metrics("")["s_per_iter"] is None


@pytest.mark.parametrize("log,bound", [
    ("[t] [UTIL-SUMMARY] cpu mean=20% (10-30); gpu0 sm mean=91% (88-95) "
     "-> GPU-bound (host has headroom)", "gpu"),
    ("[t] [UTIL-SUMMARY] cpu mean=95% (90-99); gpu0 sm mean=48% (40-55) "
     "-> host/CPU-bound (GPU starved)", "host"),
    ("[t] [UTIL-SUMMARY] cpu mean=60% (50-70); gpu0 sm mean=70% (60-80) "
     "-> mixed (GPU not saturated)", "mixed"),
])
def test_parse_util_bound_reads_the_verdict_only(log, bound):
    """The monitor's summary line contributes exactly its VERDICT — the
    utilisation numbers on it are a digest of util.csv's raw samples and
    are read from there (`util_csv_metrics`), one home per fact."""
    assert _bound(log) == bound


def test_parse_util_bound_absent():
    assert _bound("nothing here") is None


def test_parse_util_csv_reads_all_five_metrics():
    csv = ("epoch,iso,cpu_pct,mem_gb,gpu0_sm,gpu0_memutil,gpu0_vram_gb,"
           "gpu1_sm,gpu1_vram_gb\n"
           "100,a,30,137.2,80,10,10.0,90,11.5\n"
           "105,b,50,260.9,70,10,12.5,92,11.0\n"
           "141,c,40,180.0,90,10,11.0,94,10.0\n")
    r = util_csv_metrics(csv)
    assert r["peak_rss_gb"] == 260.9
    assert r["wall_s"] == 41.0                    # last epoch - first
    # MEANS ARE OVER TIME, NOT OVER ROWS (2026-08-25).  `util.csv` is
    # change-gated -- a row is written only when a metric moves past its
    # threshold or a 300 s keepalive fires -- so the rows are deliberately
    # not uniformly spaced and `sum/len` weights a one-second transient as
    # heavily as five minutes of steady state.  On a real 316 s CPU trial
    # that read 31.5% where the truth was 40.3%: a healthy run made to look
    # idle.  Each row is held to weigh the interval until the NEXT row; the
    # last has no successor and so contributes nothing, which is right --
    # `wall_s` ends AT it, so it spans none of the window.
    #
    #   cpu: (30x5 + 50x36) / 41 = 1950/41 = 47.56
    assert r["cpu_mean_pct"] == 47.6
    # per-GPU mean first, then the max ACROSS GPUs:
    #   gpu0: (80x5 + 70x36) / 41 = 2920/41 = 71.2
    #   gpu1: (90x5 + 92x36) / 41 = 3762/41 = 91.76   <- the max
    assert r["gpu_sm_mean_pct"] == 91.8
    assert r["gpu_vram_peak_gb"] == 12.5          # peak anywhere
    # missing pieces are absent, not zero
    assert util_csv_metrics("epoch,mem_gb\n") == {}
    assert util_csv_metrics("") == {}
    # Two rows, values 3 then 5, and the answer is 3.0 -- not the 4.0 a
    # row-average gives.  The 5 was the reading AT the closing instant of a
    # window it spans none of; 3 held for the whole two seconds.  This is
    # the smallest case where the two definitions visibly disagree, which
    # is why it is pinned rather than left to the richer CSV above.
    slim = util_csv_metrics("epoch,cpu_pct\n7,3\n9,5\n")
    assert slim == {"wall_s": 2.0, "cpu_mean_pct": 3.0}


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


def test_a_sweep_proposes_no_wall_and_no_memory():
    """Replaces `test_recommend_from_winner_peak_rss` and
    `test_recommend_mem_uses_true_ceil` (deleted 2026-08-24, user).

    Those pinned `recommend_resources`, which derived
    ``mem_gb = peak RSS x 1.15`` and
    ``time = s/iter x prod_iters(200) x 1.5``.  The safety factors and the
    production iteration count were chosen by nobody -- the last of them a
    default in the function's own signature -- and `summarize` wrote both
    into `run-config.toml`, from which `prep` folded them into an allocation
    and `sbatch` received them.  That is the mechanism the estimation purge
    was ordered to end; it survived in the one path the purge missed.

    What a sweep proposes now is what it MEASURED.  The wall and the memory
    are the person's to state (`execution/submission.md` S1, S2).
    """
    import molbuilder.bench.result as _r
    assert not hasattr(_r, "recommend_resources"), (
        "the benchmark must not size a wall or a memory")

    res = build_bench_result(
        _pts(), environment={"schema": "molbuilder/environment@1",
                             "scheduler": "slurm"}, system={})
    assert not hasattr(res, "recommend")
    assert "recommend" not in res.to_dict()


def test_run_config_proposes_no_wall_and_no_memory():
    """The other end of the same path: whatever the sweep measured, the
    report must recommend no `time` and no `mem`.

    A benchmark measures how fast a shape runs; it has no evidence about how
    long *your* job needs or how much it will hold, and those two asks stay
    the person's (2026-08-24).  The rule outlived the file it was written
    for -- it was `run-config.toml`, which `prep` folded into an allocation,
    and it is now a report nobody reads but you (`architecture.md` § 5.2)."""
    from molbuilder.jobset.summarize import recommendation_text
    res = build_bench_result(
        _pts(), environment={"schema": "molbuilder/environment@1",
                             "scheduler": "slurm"}, system={})
    text = recommendation_text(res, stage="tight") or ""
    assert '"time"' not in text, text
    assert '"mem"' not in text, text


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


def test_choice_survives_a_json_round_trip_for_the_offer():
    """RETIRED 2026-08-12: `adapter.format_run` no longer exists -- a
    verdict reaches production prep through `run-config.toml` (written
    by summarize, applied by `_apply_run_config`; pinned end-to-end in
    test_prep_bench_fold).  What
    THIS file still owns is the artifact: the `choice` written here must
    carry the knobs that offer reads back.
    """
    import json
    res = build_bench_result(_pts())
    back = json.loads(res.to_json())
    knobs = (back.get("choice") or {}).get("knobs") or {}
    assert knobs, "choice.knobs is what prep-run's offer consumes"



# --------------------------------------------------------------------- #
#  What the trial ACTUALLY ran -- the readback + the comparison          #
# --------------------------------------------------------------------- #
#
# Restores, on the current design, the check the deleted legacy bench
# module carried as `parse_point_out`'s effective_np / effective_omp /
# effective_bs / effective_diag.  Without it a benchmark records the
# settings it ASKED for as though they were the measurement, and a silent
# fallback (ELPA -> CPU solver, a launcher handing back fewer ranks)
# competes in the ranking under a label describing a run that never
# happened.

#: A REAL SIESTA run's output, not a hand-written sample -- the setup
#: lines this parser depends on are exactly as the binary printed them.
_FROZEN_OUT = (Path(__file__).parent / "watch" / "fixtures" /
               "siesta_frozen" / "hemeC-stage2-run3-finished-42fr.out")

#: The wrapper's launch record.  The thread count appears in no SIESTA
#: output, so this file is its only witness.
_WRAP_LOG = """\
[2026-08-13 09:14:02] INFO  resolved launch : mpirun -np 8 job.fdf > job-run0.out
[2026-08-13 09:14:02] INFO  launch mode     : mpirun (local)
[2026-08-13 09:14:02] INFO  ranks / omp     : 8 ranks x 2 OMP threads
"""


def test_effective_run_is_read_from_a_real_siesta_output():
    out = _FROZEN_OUT.read_text(encoding="utf-8", errors="replace")
    eff = parse_effective_run(out, _WRAP_LOG)
    assert eff["mpi_np"] == 8              # "* Running on 8 nodes in parallel."
    assert eff["omp_threads"] == 2         # wrapper log only
    assert eff["blocksize"] == 8           # "* ProcessorY, Blocksize:  4  8"
    assert eff["diag_algorithm"] == "D&C"  # "diag: Algorithm  = D&C"


def test_the_setup_lines_sit_far_past_a_16kb_head():
    """Why summarize reads a wide window: the eigensolver and the parallel
    grid are printed AFTER the basis/pseudopotential report.  In this real
    42-atom run they are ~49 KB in -- a 16 KB head (the window that finds
    the rank count) sees neither, and the check would silently never fire."""
    raw = _FROZEN_OUT.read_bytes()
    assert raw.find(b"* Running on") < 16384
    assert raw.find(b"diag: Algorithm") > 16384
    head16 = raw[:16384].decode("utf-8", "replace")
    assert parse_effective_run(head16, "").get("diag_algorithm") is None


def test_effective_run_reports_only_what_it_could_read():
    """No artifacts -> no claims.  'Could not check' must be tellable from
    'checked and matched', so absent keys are absent, not defaulted."""
    assert parse_effective_run("", "") == {}


def test_the_wrapper_log_is_not_a_rank_witness():
    """The wrapper writes `ranks / omp` BEFORE it launches, so its rank
    count is what it INTENDED -- MPI can hand back fewer.  Taking it as a
    fallback would record an intention as an observation, and it would
    agree with the request by construction, which is the one thing this
    check exists to detect.  Only SIESTA's own line witnesses ranks.

    The thread count is different and legitimately comes from here: no
    SIESTA output states it, and the wrapper exports it in the same
    script that runs the engine."""
    assert parse_effective_run("", _WRAP_LOG) == {"omp_threads": 2}
    both = parse_effective_run("* Running on 4 nodes in parallel.\n", _WRAP_LOG)
    assert both["mpi_np"] == 4, "SIESTA's count must win over the wrapper's 8"
    assert both["omp_threads"] == 2


def test_an_adapted_block_size_never_bars_a_trial():
    """SIESTA shrinking the requested block so every rank gets one
    (initparallel.F), or ELPA rounding it up to a power of two, is
    documented behaviour that depends on the RANK COUNT -- the axis a
    sweep varies.  Comparing it would mark most trials of a small system
    as "ran something else" and leave the benchmark with no winner.  It
    is recorded, never compared."""
    assert compare_asked_to_ran({"blocksize": 256}, {"blocksize": 64}) == {}


def test_elpa_gpu_key_is_captured_when_the_build_reached_elpa():
    out = ("* Running on 4 nodes in parallel.\n"
           "* ProcessorY, Blocksize:    2   64\n"
           "diag: Algorithm                                = ELPA-2stage\n"
           "diag: ELPA GPU string key                      = nvidia-gpu\n")
    eff = parse_effective_run(out, "")
    assert eff["diag_algorithm"] == "ELPA-2stage"
    assert eff["elpa_gpu"] == "nvidia-gpu"


def test_agreement_is_silence():
    asked = {"mpi_np": 8, "cpus_per_task": 2, "diag_algorithm": "ELPA-1STAGE"}
    ran = {"mpi_np": 8, "omp_threads": 2, "diag_algorithm": "ELPA-1stage"}
    # Case differs because the deck shouts and SIESTA prints mixed case.
    assert compare_asked_to_ran(asked, ran) == {}


def test_a_silent_eigensolver_fallback_is_caught():
    """The failure this whole check exists for: the deck asked for the GPU
    eigensolver, SIESTA used the CPU one and said so only in its output."""
    m = compare_asked_to_ran({"diag_algorithm": "ELPA-1STAGE"},
                             {"diag_algorithm": "D&C"})
    assert m == {"diag_algorithm": {"asked": "ELPA-1STAGE", "ran": "D&C"}}


def test_fewer_ranks_and_wrong_threads_are_caught():
    m = compare_asked_to_ran({"mpi_np": 8, "cpus_per_task": 4},
                             {"mpi_np": 4, "omp_threads": 8})
    assert m["mpi_np"] == {"asked": 8, "ran": 4}
    assert m["omp_threads"] == {"asked": 4, "ran": 8}


def test_a_knob_only_one_side_knows_is_not_a_disagreement():
    """Silence is the honest answer to an unanswered question -- claiming a
    mismatch from a missing readback would bar good trials from winning."""
    assert compare_asked_to_ran({"mpi_np": 8}, {}) == {}
    assert compare_asked_to_ran({}, {"mpi_np": 4}) == {}


def test_a_trial_that_ran_something_else_cannot_win_even_if_fastest():
    pts = [
        BenchPoint("gpu-k8", "gpu", {"gpus": 1},
                   {"s_per_iter": 10.0}, state="completed",
                   effective={"diag_algorithm": "D&C"},
                   mismatch={"diag_algorithm": {"asked": "ELPA-1STAGE",
                                                "ran": "D&C"}}),
        BenchPoint("gpu-k4", "gpu", {"gpus": 2},
                   {"s_per_iter": 99.0}, state="completed"),
    ]
    c = choose_winner(pts)
    assert c["label"] == "gpu-k4", "the fastest row measured another machine"
    assert "excluded" in c["rationale"] and "gpu-k8" in c["rationale"]
    assert "asked ELPA-1STAGE, ran D&C" in c["rationale"]


def test_no_winner_when_every_timed_trial_ran_something_else():
    """Better no recommendation than the least-wrong of a bad table."""
    pts = [BenchPoint(f"p{i}", "gpu", {}, {"s_per_iter": float(i + 1)},
                      state="completed",
                      mismatch={"mpi_np": {"asked": 8, "ran": 4}})
           for i in range(3)]
    assert choose_winner(pts) == {}


def test_the_readback_survives_the_json_round_trip():
    pts = [BenchPoint("g", "gpu", {"gpus": 1}, {"s_per_iter": 5.0},
                      state="completed",
                      effective={"blocksize": 64, "diag_algorithm": "D&C"},
                      mismatch={"mpi_np": {"asked": 8, "ran": 4}})]
    back = BenchResult.from_dict(build_bench_result(pts).to_dict())
    assert back.points[0].effective["blocksize"] == 64
    assert back.points[0].mismatch["mpi_np"]["ran"] == 4
