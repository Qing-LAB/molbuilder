"""`--mode ask` — submit nothing, and say when it would start.

User, 2026-08-27: *there's no prediction when the cluster is used. It has to
be on the site, and the user has to decide within minutes. You come back, tune
it, and submit for a different cluster or reduce those resources and see if
they can get a better waiting time, or just say, okay, I can live with that.*

And on where it lives: *instead of submit, we can just say ask — we don't have
to reinvent something.*

**That is the design, not a convenience.** `--mode ask` walks the identical
path `--mode submit` walks and inserts one flag, so the line asked about IS
the line that would be sent. A separate verb would have re-rendered the flags,
and two renderings of one fact are two things that can disagree.

`sbatch --test-only` validates a request and predicts a start time. **It
creates no job**, which is what makes it safe to run in a loop while tuning.
"""
from __future__ import annotations

import pytest

from molbuilder.jobset.ask import Prediction, parse_test_only, prediction_table


# --------------------------------------------------------------------- #
#  reading what the scheduler said                                       #
# --------------------------------------------------------------------- #

def test_a_prediction_is_read_whole():
    got = parse_test_only(
        "sbatch: Job 62238108 to start at 2026-08-27T14:30:00 using 48 "
        "processors on nodes sg013 in partition htc")
    assert got.start == "2026-08-27T14:30:00"
    assert got.procs == 48
    assert got.nodes == "sg013"
    assert got.refused is None


def test_a_prediction_without_the_trimmings_still_reads():
    """Not every SLURM version prints the processor and node clause, and the
    time is the part that matters."""
    got = parse_test_only("sbatch: Job 5 to start at 2026-08-27T09:00:00")
    assert got.start == "2026-08-27T09:00:00"
    assert got.procs is None and got.nodes is None


@pytest.mark.parametrize("text,why", [
    ("sbatch: error: Batch job submission failed: Requested node "
     "configuration is not available", "the queue cannot take it"),
    ("sbatch: error: invalid partition specified: nosuch", "no such queue"),
    ("", "nothing at all"),
    ("could not ask the scheduler: [Errno 2] No such file", "no sbatch here"),
])
def test_no_time_means_UNKNOWN_and_the_reason_is_kept(text, why):
    """**A missing prediction is the absence of an answer, and dressing it as
    a good one is how a person waits a day for a queue that looked instant.**

    The reason is kept because it is often the whole answer — *"Requested node
    configuration is not available"* says the ask does not fit any machine
    here, which is exactly what the person needs to change.
    """
    got = parse_test_only(text)
    assert got.start is None, why
    assert got.refused, f"{why}: the reason was thrown away"


def test_a_refusal_is_never_mistaken_for_a_time():
    got = parse_test_only("sbatch: error: Job violates accounting/QOS policy")
    assert got.start is None
    assert "QOS" in got.refused


# --------------------------------------------------------------------- #
#  what a person reads                                                   #
# --------------------------------------------------------------------- #

def test_the_table_says_nothing_was_submitted():
    """The single most important line: this ran `sbatch`, and a person who
    thinks their job is now queued will not launch it."""
    out = prediction_table([Prediction(label="htc", start="2026-08-27T14:00",
                                       procs=48, nodes="sg013")])
    assert "nothing was submitted" in out


def test_an_unknown_is_shown_as_unknown_with_its_reason():
    out = prediction_table([
        Prediction(label="htc", start="2026-08-27T14:00"),
        Prediction(label="highmem",
                   refused="Requested node configuration is not available")])
    assert "no prediction" in out
    assert "Requested node configuration is not available" in out
    assert "2026-08-27T14:00" in out


def test_the_table_does_not_rank_the_queues():
    """Sorting would be a recommendation. The wait is one of the things
    being weighed; the others — what else is running, whose allocation, how
    long the job really needs — are not on this machine.

    Same rule `queue_table` already holds: show what exists, let the person
    pick.
    """
    preds = [Prediction(label="later", start="2026-08-29T00:00"),
             Prediction(label="sooner", start="2026-08-27T01:00")]
    out = prediction_table(preds)
    assert out.index("later") < out.index("sooner"), \
        "the answers were reordered, which is a recommendation"
    for word in ("recommended", "best", "fastest", "you should"):
        assert word not in out.lower()


def test_the_table_says_the_time_is_an_estimate_that_moves():
    """A queue prediction is true of the queue as it was asked, and a person
    who reads it as a promise will be surprised."""
    out = prediction_table([Prediction(label="htc", start="2026-08-27T14:00")])
    assert "ESTIMATE" in out or "estimate" in out
    assert "moves" in out


def test_the_table_names_the_next_move():
    """*Come back, tune it, ask again, or say I can live with that.* The
    loop only works if the table says how to re-enter it."""
    out = prediction_table([Prediction(label="htc", start="x")])
    assert "--domain" in out and "ask again" in out


def test_an_empty_ask_says_so_rather_than_printing_a_header():
    assert prediction_table([]) == "nothing to ask about."


# --------------------------------------------------------------------- #
#  the mode itself                                                       #
# --------------------------------------------------------------------- #

def test_ask_is_gated_like_submit_not_like_direct():
    """One job at a time. It submits nothing, but it is still N scheduler
    queries fired from one command — and the answer is only useful for a job
    you are about to hand over, which is one job."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/jobset/submit.py").read_text()
    fn = src[src.index("def _refuse_batch_submission"):
             src.index("def submit_jobset")]
    assert 'mode in ("submit", "ask")' in fn, \
        "ask escaped the one-at-a-time gate"


def test_ask_walks_the_SAME_path_as_submit():
    """The whole reason it is a mode and not a verb. If these ever became
    two code paths, the line asked about could stop being the line sent."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/jobset/submit.py").read_text()
    disp = src[src.index("    if mode in (\"submit\", \"ask\"):"):
               src.index("    if mode == \"direct\":")]
    assert "_submit_slurm" in disp
    assert disp.count("return") == 1, "ask branched away from submit"


def test_ask_adds_test_only_and_changes_nothing_else():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/jobset/submit.py").read_text()
    assert 'cmd = [cmd[0], "--test-only"] + cmd[1:]' in src, \
        "the flag must be inserted into the real command, not a rebuilt one"


def test_ask_records_no_launch():
    """A launch record says a job exists. After this one does not, so
    writing one would make `status` report a job nobody submitted."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/jobset/submit.py").read_text()
    body = src[src.index("        if ask:\n            # NOTHING WAS"):
               src.index("        if cp.returncode != 0:")]
    assert "_record_launch" not in body
    assert "continue" in body
