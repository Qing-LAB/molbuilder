"""Tests for the summarize driver (molbuilder/bench/summarize.py)."""
from __future__ import annotations

import json

from molbuilder.bench import summarize
from molbuilder.bench.result import BenchResult


# Real gpu timing trace shape (5 iters); fast point vs slow point.
def _timing(per_iter: float) -> str:
    lines = []
    t = 1000.0
    for i in range(1, 6):
        lines.append(f"{t:.3f} {i}    scf:    {i} -100.{i}")
        t += per_iter
    return "\n".join(lines) + "\n"


def _make_point(bundle, g, k, per_iter, sm, cpu, bound, mem, *,
                completed=True):
    d = bundle / f"point-G{g}K{k}"
    d.mkdir()
    (d / "job-gpu-run0.scf-timing.log").write_text(_timing(per_iter))
    (d / "job-gpu.monitor.log").write_text(
        f"[t] [UTIL-SUMMARY] cpu mean={cpu}% (..); gpu0 sm mean={sm}% (..) "
        f"-> {bound}\n")
    (d / "job-gpu.util.csv").write_text(
        "epoch,iso,cpu_pct,mem_gb,gpu0_sm,gpu0_memutil,gpu0_vram_gb\n"
        f"1,a,5,{mem},0,0,0\n2,b,5,{mem - 1},0,0,0\n")
    out = "scf: 5\n>> End of run: completed\n" if completed else "scf: 3\n"
    (d / "job-gpu-run0.out").write_text(out)


def _bundle(tmp_path):
    b = tmp_path / "bundle"
    b.mkdir()
    (b / "environment.json").write_text(json.dumps(
        {"schema": "molbuilder/environment@1", "scheduler": "slurm",
         "topology": {"cores_per_socket": 24}}))
    (b / "job-gpu.fdf").write_text("NumberOfAtoms 444\nDiag.ELPA.GPU .true.\n")
    # gpu-k8 fast + GPU-bound; gpu-k4 slower
    _make_point(b, 1, 8, 1538.0, 91, 40, "GPU-bound (GPU saturated)", 25.2)
    _make_point(b, 1, 4, 1938.0, 70, 60, "mixed (..)", 22.3)
    return b


def test_summarize_bundle_builds_result_and_winner(tmp_path):
    b = _bundle(tmp_path)
    res = summarize.summarize_bundle(b, now_iso="2026-06-27T22:00:00Z")

    assert {p.label for p in res.points} == {"point-G1K8", "point-G1K4"}
    win = res.choice
    assert win["engine"] == "gpu"
    assert win["knobs"] == {"gpus": 1, "ranks_per_gpu": 8}   # the fast one
    assert "point-G1K8 fastest" in win["rationale"]

    # per-point metrics parsed from the artifacts
    p8 = next(p for p in res.points if p.label == "point-G1K8")
    assert p8.s_per_iter() == 1538.0
    assert p8.metrics["gpu_sm_mean_pct"] == 91.0
    assert p8.metrics["peak_rss_gb"] == 25.2
    assert p8.bound == "gpu"
    assert p8.state == "completed"
    # system descriptor read from the fdf
    assert res.system["n_atoms"] == 444
    assert res.environment["scheduler"] == "slurm"


def test_run_summarize_writes_valid_bench_result(tmp_path):
    b = _bundle(tmp_path)
    res, out_path = summarize.run_summarize(b, now_iso="t")
    assert out_path == b / "bench-result.json"
    doc = json.loads(out_path.read_text())
    assert doc["schema"] == "molbuilder/bench-result@1"
    # round-trips + the choice feeds prep-run
    back = BenchResult.from_dict(doc)
    assert back.choice["knobs"]["ranks_per_gpu"] == 8


def test_incomplete_point_is_not_chosen(tmp_path):
    b = tmp_path / "b"
    b.mkdir()
    _make_point(b, 1, 8, 1500.0, 90, 40, "GPU-bound", 25.0, completed=False)
    res = summarize.summarize_bundle(b)
    pt = res.points[0]
    assert pt.state == "incomplete"
    assert res.choice == {}            # no completed point -> no winner


def test_discover_ignores_non_point_dirs(tmp_path):
    b = tmp_path / "b"
    b.mkdir()
    (b / "notapoint").mkdir()
    (b / "point-bad").mkdir()
    _make_point(b, 2, 4, 1000.0, 80, 30, "GPU-bound", 30.0)
    pts = summarize.discover_points(b)
    assert [p.label for p in pts] == ["point-G2K4"]


def test_summary_text_smoke(tmp_path):
    b = _bundle(tmp_path)
    res, out_path = summarize.run_summarize(b, now_iso="t")
    txt = summarize.summary_text(res, out_path)
    assert "ranked points" in txt
    assert "point-G1K8" in txt and "winner" in txt
