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

#: VERBATIM from `sbatch --test-only` on ASU Sol, 2026-08-27.  Note the
#: token between the timestamp and `using` -- that is what SLURM printed,
#: and it is why the three facts are read independently.
SOL_PREDICTION = ("sbatch: Job 62266174 to start at 2026-08-27T11:22:03 a "
                  "using 4 processors on nodes sc078 in partition htc")

#: Also verbatim.  The refusal path was written against an INVENTED
#: `sbatch: error: ...` line; the real prefix is `allocation failure:`.
SOL_REFUSAL = "allocation failure: Requested node configuration is not available"


def test_the_REAL_prediction_from_sol_is_read_whole():
    """**Against what SLURM actually printed, not what I assumed.**

    One regex chained the three facts with optional tails, so it required
    them to be adjacent — and Sol puts a token between the timestamp and
    `using`. The time still parsed while the processor count AND the node
    name were silently lost. Read separately, whatever SLURM inserts is
    ignored and one missing field cannot take the others with it.
    """
    got = parse_test_only(SOL_PREDICTION)
    assert got.start == "2026-08-27T11:22:03"
    assert got.procs == 4, "the stray token ate the processor count"
    assert got.nodes == "sc078", "the stray token ate the node name"
    assert got.refused is None


def test_the_REAL_refusal_from_sol_survives_a_wrong_guess():
    """The refusal was written against an invented `sbatch: error: …`
    prefix. Sol says `allocation failure: …`.

    **It works because the parser keeps the raw line rather than matching a
    known prefix** — a parser that recognised prefixes would have thrown
    away the one sentence worth reading.
    """
    got = parse_test_only(SOL_REFUSAL)
    assert got.start is None
    assert "Requested node configuration is not available" in got.refused


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


def test_no_scheduler_is_its_own_ANSWER_not_an_empty_table(): 
    """The workstation path, and it had no test at all until 2026-08-27 —
    the wording could be changed freely and nothing noticed.

    A missing scheduler is not "the queue could not say"; it is "there is
    no queue". Rendering the normal table would head it *asked the
    scheduler* when none was asked, and offer to change `--domain`, which
    means nothing here.
    """
    out = prediction_table([Prediction(label="relax", no_scheduler=True)])
    assert "no scheduler on this machine" in out
    assert "nothing to wait for" in out
    assert "asked the scheduler" not in out
    assert "--domain" not in out, "offered a knob that does nothing here"
    assert "would start" not in out, "rendered the table header anyway"


def test_no_scheduler_does_not_end_by_pointing_at_submit():
    """**Caught by running it, not by a test.** The table said *there is no
    scheduler here* and the closing line said *launch it with `--mode
    submit` when the answer suits you* — a contradiction in consecutive
    sentences, pointing at a mode this machine cannot run.

    Guarded at the source, because the closing line lives in the CLI and
    the table cannot see it.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/jobset/_cli.py").read_text()
    branch = src[src.index("would send: "):src.index("would send: ") + 900]
    assert "if all(p.no_scheduler for p in preds):" in branch
    assert "--mode direct" in branch, \
        "the no-scheduler case must point at the mode that DOES work here"


def test_the_fact_and_the_ACTION_are_not_said_twice():
    """The table states the fact; the CLI's closing line says what to do.
    Both saying `--mode direct` reads as a stutter, and it was."""
    out = prediction_table([Prediction(label="relax", no_scheduler=True)])
    assert out.count("--mode direct") == 0, \
        "the table took the caller's line as well as its own"


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

def test_ask_is_NOT_gated_by_the_one_at_a_time_rule():
    """**I had this backwards, and the rule's own words say so** (caught by
    the user on a 4-trial bench, 2026-08-27).

    `_refuse_batch_submission` exists because jobs queued together start
    together, contend, and make a sweep measure contention rather than
    scaling. Its docstring is explicit: *"a rule about the SCHEDULER, not
    about doing several things"* — which is why `--mode direct` is untouched.

    `--test-only` enqueues nothing, so none of that harm is reachable. And
    the sweep is exactly where asking pays: a grid's trials ask for
    different shapes, G1 schedules sooner than G4, so seeing their waits
    side by side is what tells you which to submit. Gating it made the
    feature useless precisely where it was most useful.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/jobset/submit.py").read_text()
    fn = src[src.index("def _refuse_batch_submission"):
             src.index("def submit_jobset")]
    assert 'if mode == "submit" and len(jobset.jobs) > 1:' in fn, \
        "ask was gated by the submission rule again"
    assert '"ask"' not in fn.split("if mode ==")[1], \
        "ask must not appear in the refusal condition"


def test_the_number_of_QUERIES_is_bounded_and_says_what_it_skipped():
    """Politeness, not a rule about queues. And **no silent cap**: a partial
    answer that does not say it is partial reads as a complete one."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/jobset/submit.py").read_text()
    assert "ASK_MAX_QUERIES" in src
    assert 'JobResult(job.name, [], "not asked")' in src, \
        "trials past the cap must be named, not dropped"
    cli = (Path(__file__).resolve().parents[1]
           / "molbuilder/jobset/_cli.py").read_text()
    assert "NOT asked (past" in cli, "the skipped trials are not reported"


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


def test_asking_writes_NOTHING_to_the_tree(tmp_path, monkeypatch):
    """`--mode ask` is a question, and a question must not write.  Until
    2026-08-28 asking about a LAUNCHED hierarchical stage opened
    run-<n+1> and copied the warm files -- from an attempt that could
    still be running (a torn .DM/.XV copy) -- and the fresh empty attempt
    then hid the running one from `status`, which reports the latest.
    Found live during the full review: one ask, and a running relax
    vanished from the status table."""
    import json
    from molbuilder.jobset.materialize import attempts, write_run_launch
    from molbuilder.jobset.model import Job, JobSet, Resources
    from molbuilder.jobset.submit import submit_jobset

    js = JobSet(name="J", engine="siesta", kind="ladder", shared=[],
                jobs=[Job(name="coarse", script="J_01_coarse.fdf",
                          resources=Resources(mpi_np=2))])
    base = tmp_path
    d = base / "01_coarse"
    (d / "run-0").mkdir(parents=True)
    (d / "run-0" / "J_01_coarse.fdf").write_text("SystemLabel J\n")
    (base / "J_01_coarse.fdf").write_text("SystemLabel J\n")
    (d / "run-0" / "J.XV").write_text("warm state, mid-flight")
    write_run_launch(d / "run-0", mode="direct", command=["bash", "x"])

    before = attempts(d)
    try:
        submit_jobset(js, base, mode="ask", only="coarse", dry_run=False)
    except Exception:
        pass          # the refusal text is not this test's subject
    assert attempts(d) == before, (
        "asking opened a new attempt -- a question verb wrote to the tree")
    assert not (d / "run-1").exists()
