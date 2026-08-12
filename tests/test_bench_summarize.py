"""The sweep reader's pure units — ``parse_point``, the winner, the text.

The bundle-walking half this file used to exercise (``discover_points``'
directory regex, ``summarize_bundle``/``run_summarize``) was DELETED
2026-08-12 (u5) with the shipped-bundle lifecycle; the LIVE, data-keyed
path — ``discover_points_from_jobset`` / ``run_summarize_jobset`` — is
covered end-to-end in tests/test_prep_bench_fold.py.  What remains here
are the parsing units both paths share.
"""
from __future__ import annotations

from pathlib import Path

from molbuilder.bench.result import build_bench_result, choose_winner
from molbuilder.bench.summarize import parse_point, summary_text


def _point_dir(tmp_path, name, basename, out_text=None):
    d = tmp_path / name
    d.mkdir()
    if out_text is not None:
        (d / f"{basename}-run0.out").write_text(out_text)
    return d


def test_parse_point_states_are_the_three_honest_answers(tmp_path):
    done = _point_dir(tmp_path, "a", "job", "x\nsiesta: Final energy = -1\n")
    part = _point_dir(tmp_path, "b", "job", "still going\n")
    none = _point_dir(tmp_path, "c", "job")
    assert parse_point("a", done, "job", "gpu", {}).state == "completed"
    assert parse_point("b", part, "job", "gpu", {}).state == "incomplete"
    assert parse_point("c", none, "job", "gpu", {}).state == "unknown"


def test_cpu_point_recovers_np_from_its_own_out(tmp_path):
    d = _point_dir(tmp_path, "cpu", "job",
                   "* Running on    8 nodes in parallel\n>> End of run:\n")
    pt = parse_point("cpu", d, "job", "cpu", {})
    # exchange vocabulary (U13): the knob is the job-set's own field name
    assert pt.knobs.get("mpi_np") == 8
    assert pt.state == "completed"


def test_cpu_ranks_survive_a_real_sized_out(tmp_path):
    """THE U11 pin.  "Running on N nodes" is a LAUNCH HEADER -- the first
    KB of the .out -- while the done-markers are in the tail.  Until
    2026-08-12 both were searched in one 16 KB tail window, so any run
    whose .out outgrew it (i.e. any real run) silently lost its rank
    count and the verdict's CPU half had no np.  The tiny fixture above
    fits both ends in one window, which is exactly why the bug never
    fired in tests."""
    body = "scf: iteration data line\n" * 4000        # ~100 KB of middle
    d = _point_dir(tmp_path, "cpu2", "job",
                   "* Running on   16 nodes in parallel\n"
                   + body + ">> End of run:\n")
    pt = parse_point("cpu2", d, "job", "cpu", {})
    assert pt.knobs.get("mpi_np") == 16, "header lost outside the tail window"
    assert pt.state == "completed"


def test_an_incomplete_point_is_never_the_winner(tmp_path):
    """U20: rebuilt twice over.  The old assert read `(choice or
    {}).get("label", "y") == "y"` -- vacuously green while choose_winner
    returned no "label" key (it gained one at U13).  De-vacuousing it then
    exposed the FIXTURE: the 'done' point had no timing, so under the
    contract ("the fastest COMPLETED point by s/iter; {} if no point
    produced a time") NOBODY could win and the old default hid that too.
    Now the incomplete point carries the FASTER time and must still
    lose -- which is the claim in this test's name."""
    dx = _point_dir(tmp_path, "x", "job", "going\n")
    (dx / "job-run0.scf-timing.log").write_text(
        "100.0 scf 1\n101.0 scf 2\n102.0 scf 3\n")      # 1 s/iter, unfinished
    fast_but_unfinished = parse_point("x", dx, "job", "gpu", {})
    dy = _point_dir(tmp_path, "y", "job", ">> End of run:\n")
    (dy / "job-run0.scf-timing.log").write_text(
        "100.0 scf 1\n104.0 scf 2\n108.0 scf 3\n")      # 4 s/iter, done
    done = parse_point("y", dy, "job", "gpu", {})
    assert fast_but_unfinished.s_per_iter() < done.s_per_iter()
    choice = choose_winner([fast_but_unfinished, done])
    assert choice, "a completed, timed point must produce a winner"
    assert choice["label"] == "y"


def test_summary_text_names_every_point_and_the_output(tmp_path):
    res = build_bench_result(
        [parse_point("p1", _point_dir(tmp_path, "p1", "job",
                                      ">> End of run:\n"), "job", "gpu", {})],
        environment={}, system={}, now_iso="2026-08-12T00:00:00Z")
    text = summary_text(res, Path("/tmp/bench-result.json"))
    assert "p1" in text and "completed" in text
    assert "bench-result.json" in text
