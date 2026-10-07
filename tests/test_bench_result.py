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


def test_a_sweep_proposes_no_wall_and_no_memory():
    """What a sweep proposes is what it MEASURED (user, 2026-08-24) -- never
    a wall or a memory sized by safety factors and an iteration count chosen
    by nobody.  The wall and the memory are the person's to state
    (`execution/submission.md` S1, S2).
    """
    import molbuilder.bench.result as _r
    assert not hasattr(_r, "recommend_resources"), (
        "the benchmark must not size a wall or a memory")

    res = build_bench_result(
        _pts(), environment={"schema": "molbuilder/environment@1",
                             "scheduler": "slurm"}, system={})
    assert not hasattr(res, "recommend")
    assert "recommend" not in res.to_dict()


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
    """The artifact: the `choice` written here must carry the knobs that
    the offer reads back, through a JSON round trip.
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
# Without this check a benchmark records the settings it ASKED for as though they were the measurement, and a silent
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


def test_the_readback_survives_the_json_round_trip():
    pts = [BenchPoint("g", "gpu", {"gpus": 1}, {"s_per_iter": 5.0},
                      state="completed",
                      effective={"blocksize": 64, "diag_algorithm": "D&C"},
                      mismatch={"mpi_np": {"asked": 8, "ran": 4}})]
    back = BenchResult.from_dict(build_bench_result(pts).to_dict())
    assert back.points[0].effective["blocksize"] == 64
    assert back.points[0].mismatch["mpi_np"]["ran"] == 4
