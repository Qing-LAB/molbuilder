"""Tests for the bench-result schema + parsers + winner logic
(molbuilder/bench/result.py)."""
from __future__ import annotations


import pytest

from molbuilder.bench.result import (
    BenchPoint,
    BenchResult,
    build_bench_result,
    choose_winner,
    compare_asked_to_ran,
    parse_effective_run,
)
# The wrapper's own instruments are registered parsers since 2026-09-04
# (`parse.md` § 5c); the logic is unchanged, only its address moved.


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 10 tests here parsed timing logs, monitor lines, utilisation
# rows, wrapper logs, SIESTA lines or sacct text typed by hand (`process/testing.md` § 6).


# --------------------------------------------------------------------- #
#  winner + recommend + schema                                          #
# --------------------------------------------------------------------- #


def _pts():
    return [
        BenchPoint("gpu-k8", "gpu", {"gpus": 1, "ranks_per_gpu": 8},
                   {"s_per_iter": 1538.0, "mem_peak_gb": 25.2},
                   bound="gpu", state="completed"),
        BenchPoint("gpu-k4", "gpu", {"gpus": 1, "ranks_per_gpu": 4},
                   {"s_per_iter": 1938.0, "mem_peak_gb": 22.3},
                   bound="gpu", state="completed"),
        BenchPoint("cpu-np64", "cpu", {"ranks": 64},
                   {"s_per_iter": None, "mem_peak_gb": 433.2},
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


def test_effective_run_reports_only_what_it_could_read():
    """No artifacts -> no claims.  'Could not check' must be tellable from
    'checked and matched', so absent keys are absent, not defaulted."""
    assert parse_effective_run("", "") == {}


def test_an_adapted_block_size_never_bars_a_trial():
    """SIESTA shrinking the requested block so every rank gets one
    (initparallel.F), or ELPA rounding it up to a power of two, is
    documented behaviour that depends on the RANK COUNT -- the axis a
    sweep varies.  Comparing it would mark most trials of a small system
    as "ran something else" and leave the benchmark with no winner.  It
    is recorded, never compared."""
    assert compare_asked_to_ran({"blocksize": 256}, {"blocksize": 64}) == {}


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
