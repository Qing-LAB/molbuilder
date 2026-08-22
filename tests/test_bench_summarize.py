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
from molbuilder.jobset.summarize import parse_point, summary_text


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
    # the exact path the caller was told about, not a substring of the
    # test's own argument (the R2-7 class)
    assert str(Path("/tmp/bench-result.json")) in text


# --------------------------------------------------------------------- #
#  What the trial actually ran -- the WIRING, not the parsers            #
# --------------------------------------------------------------------- #
#
# The parsers themselves are pinned in test_bench_result.py.  What these
# pin is that `parse_point` actually READS a trial's artifacts and fills
# `effective` / `mismatch` -- a mutation blanking the readback passed the
# whole bench suite before these existed, which is the shape of the
# original defect: the check can be absent and everything stays green.

def _ran_trial(tmp_path, name, basename, *, ranks, omp, algorithm,
               asked_algorithm):
    """A trial directory holding what a real run leaves behind: its deck
    (what was ASKED), its .out and its wrapper log (what RAN)."""
    d = _point_dir(
        tmp_path, name, basename,
        "* Running on {} nodes in parallel.\n".format(ranks)
        + "padding\n" * 3000                       # push setup lines deep
        + "* ProcessorY, Blocksize:    2   64\n"
        + "diag: Algorithm                              = {}\n".format(algorithm)
        + "scf: 1\n>> End of run:\n")
    (d / f"{basename}.fdf").write_text(
        f"SystemLabel {basename}\nDiag.Algorithm {asked_algorithm}\n")
    (d / f"{basename}.runwrap-20260813-091402.log").write_text(
        f"[..] INFO  ranks / omp     : {ranks} ranks x {omp} OMP threads\n")
    return d


def test_parse_point_reads_back_what_the_trial_really_ran(tmp_path):
    d = _ran_trial(tmp_path, "ok", "job", ranks=8, omp=2,
                   algorithm="ELPA-1stage", asked_algorithm="ELPA-1STAGE")
    pt = parse_point("ok", d, "job", "gpu",
                     {"mpi_np": 8, "cpus_per_task": 2})
    assert pt.effective["mpi_np"] == 8
    assert pt.effective["omp_threads"] == 2
    assert pt.effective["blocksize"] == 64
    assert pt.effective["diag_algorithm"] == "ELPA-1stage"
    assert pt.mismatch == {}, "asked and ran agree; case must not matter"


def test_parse_point_catches_a_trial_that_ran_something_else(tmp_path):
    """The deck asked for the GPU eigensolver on 8 ranks; SIESTA used the
    CPU solver and the launcher gave 4.  Both must show up."""
    d = _ran_trial(tmp_path, "fell-back", "job", ranks=4, omp=2,
                   algorithm="D&C", asked_algorithm="ELPA-1STAGE")
    pt = parse_point("fell-back", d, "job", "gpu",
                     {"mpi_np": 8, "cpus_per_task": 2})
    assert pt.mismatch["mpi_np"] == {"asked": 8, "ran": 4}
    assert pt.mismatch["diag_algorithm"] == {"asked": "ELPA-1STAGE",
                                             "ran": "D&C"}
    assert pt.state == "completed", "the run finished -- it just ran elsewise"


def test_a_trial_with_no_artifacts_claims_nothing(tmp_path):
    """'Could not check' must never masquerade as 'checked and matched'."""
    d = _point_dir(tmp_path, "bare", "job")
    pt = parse_point("bare", d, "job", "gpu", {"mpi_np": 8})
    assert pt.effective == {} and pt.mismatch == {}


def test_summary_text_shows_a_mismatch_and_refuses_a_bogus_winner(tmp_path):
    d = _ran_trial(tmp_path, "only", "job", ranks=4, omp=2,
                   algorithm="D&C", asked_algorithm="ELPA-1STAGE")
    (d / "job-run0.scf-timing.log").write_text(
        "1782535754.0 1 scf: 1\n1782535854.0 2 scf: 2\n"
        "1782535954.0 3 scf: 3\n")
    pt = parse_point("only", d, "job", "gpu", {"mpi_np": 8})
    res = build_bench_result([pt])
    text = summary_text(res, Path("bench-result.json"))
    assert "ran something other than asked" in text
    assert "asked 8, ran 4" in text
    assert res.choice == {}, "the only timed trial measured another setup"
    assert "NO WINNER" in text


def test_one_deck_reader_and_it_takes_the_first_match(tmp_path):
    """libfdf's `fdf_locate` walks from the top and STOPS at the first
    matching label, so a deck naming a keyword twice is read with its
    FIRST value.  There were two readers here -- this one and
    `_winner_mechanism`'s loop, which kept the LAST -- so a duplicated
    keyword made the verdict name an algorithm SIESTA never used, and
    that verdict is what `prep run` offers to apply to production."""
    from molbuilder.jobset.summarize import deck_value
    deck = tmp_path / "job.fdf"
    deck.write_text("Diag.Algorithm ELPA-1STAGE\n"
                    "Diag.Algorithm D&C\n"
                    "BlockSize 256\n")
    assert deck_value(deck, "Diag.Algorithm") == "ELPA-1STAGE"
    assert deck_value(deck, "BlockSize") == "256"
    # `_norm` folds separators and case: one keyword, many spellings.
    assert deck_value(deck, "diag_algorithm") == "ELPA-1STAGE"
    assert deck_value(deck, "MeshCutoff") is None
    assert deck_value(tmp_path / "absent.fdf", "BlockSize") is None


def test_both_block_sizes_are_recorded_and_neither_bars_the_trial(tmp_path):
    """End-to-end: the deck asks 256, the .out reports 64.  Both land on
    the point as data; the trial stays eligible to win."""
    d = _ran_trial(tmp_path, "bs", "job", ranks=8, omp=2,
                   algorithm="ELPA-1stage", asked_algorithm="ELPA-1STAGE")
    (d / "job.fdf").write_text("SystemLabel job\n"
                               "Diag.Algorithm ELPA-1STAGE\n"
                               "BlockSize 256\n")
    pt = parse_point("bs", d, "job", "gpu", {"mpi_np": 8, "cpus_per_task": 2})
    assert pt.effective["blocksize"] == 64          # what SIESTA used
    assert pt.effective["blocksize_asked"] == 256    # what the deck asked
    assert "blocksize" not in pt.mismatch, "an adapted block size is not a fault"


def test_a_recovered_rank_count_is_never_compared_with_itself(tmp_path):
    """A CPU point whose job-set carries no rank count has it recovered
    FROM the .out.  That recovered value must not then be compared
    against the same .out: an empty `mismatch` would read as 'checked and
    matched' when nothing was checked.  The knob is still recorded --
    only the CLAIM of having verified it is withheld."""
    d = _ran_trial(tmp_path, "cpu", "job", ranks=4, omp=2,
                   algorithm="D&C", asked_algorithm="D&C")
    pt = parse_point("cpu", d, "job", "cpu", {})     # job-set knows no ranks
    assert pt.knobs["mpi_np"] == 4, "the observation is still recorded"
    assert pt.effective["mpi_np"] == 4
    assert "mpi_np" not in pt.mismatch
