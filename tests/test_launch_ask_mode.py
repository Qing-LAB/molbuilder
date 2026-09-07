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


def test_the_number_of_QUERIES_is_bounded_and_says_what_it_skipped(tmp_path):
    """Politeness, not a rule about queues. And **no silent cap**: a partial
    answer that does not say it is partial reads as a complete one.

    CONVERTED 2026-09-06 (`plans/plan.md` § 5h).  This read `submit.py` for
    the strings `ASK_MAX_QUERIES` and `JobResult(job.name, [], "not asked")`.
    Both can be present while the cap counts the wrong thing, or while the
    skipped entries never reach a caller -- and neither would fail.  It now
    ASKS about more trials than the cap and reads the answer.

    MUTATION THIS MUST FAIL AGAINST: `continue` instead of appending the
    `"not asked"` result -- the cap still works, the strings are still there,
    and the answer silently covers 24 of 30.
    """
    from molbuilder.jobset.model import Job, JobSet, Resources
    from molbuilder.jobset.submit import ASK_MAX_QUERIES, submit_jobset

    n = ASK_MAX_QUERIES + 6
    js = JobSet(name="J", engine="siesta", kind="sweep", shared=[],
                jobs=[Job(name=f"p{i:02d}", script="J.fdf",
                          resources=Resources(mpi_np=1)) for i in range(n)])
    # A REAL deck, with the clean restart group written out: the submission
    # door verifies a trial's cold start against it since 2026-08-21, and a
    # deck without one is refused.  The first version of this fixture wrote
    # `SystemLabel J` alone and the test SKIPPED on that refusal -- which is
    # the shape `test_no_tests_read_the_projects_tree` calls out, a test that
    # reads green while never running.
    deck = ("SystemName test\nSystemLabel J\nNumberOfAtoms 2\n"
            "DM.UseSaveDM .false.\nMD.UseSaveXV .false.\n")
    (tmp_path / "J.fdf").write_text(deck)
    for i in range(n):
        d = tmp_path / "bench" / f"bench-p{i:02d}"
        d.mkdir(parents=True)
        (d / "J.fdf").write_text(deck)

    results = submit_jobset(js, tmp_path, mode="ask", dry_run=True)

    skipped = [r.name for r in results if r.status == "not asked"]
    assert skipped, (
        f"{n} trials, a cap of {ASK_MAX_QUERIES}, and nothing was reported as "
        "skipped -- a partial answer that does not say it is partial reads as "
        "a complete one")
    assert len(results) == n, (
        "trials past the cap were DROPPED rather than named: "
        f"{len(results)} results for {n} trials")
    assert skipped == [f"p{i:02d}" for i in range(ASK_MAX_QUERIES, n)], (
        "the skipped trials are not the ones past the cap, in order")


def test_ask_walks_the_SAME_path_as_submit():
    """The whole reason it is a mode and not a verb. If these ever became
    two code paths, the line asked about could stop being the line sent."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/jobset/submit.py").read_text()
    # Anchor the end AFTER the start: submit_transport_chain (P5b)
    # carries its own earlier `if mode == "direct":`, and a naive
    # first-occurrence slice inverted into an empty string.
    _start = src.index("    if mode in (\"submit\", \"ask\"):")
    disp = src[_start:src.index("    if mode == \"direct\":", _start)]
    assert "_submit_slurm" in disp
    assert disp.count("return") == 1, "ask branched away from submit"


def test_ask_adds_test_only_and_changes_nothing_else(tmp_path):
    """The flag goes into the REAL command, so what is asked about is the
    line that would be sent.

    CONVERTED 2026-09-06 (`plans/plan.md` § 5h).  This read `submit.py` for
    the exact expression `cmd = [cmd[0], "--test-only"] + cmd[1:]`.  Reformat
    it -- `cmd.insert(1, …)`, a different variable name -- and it fails while
    the behaviour is right; build a SECOND command for the question and it
    passes while the line asked about is not the line sent.

    No spying needed: a `JobResult` carries the command, so `ask` and a
    planned `submit` can simply be compared.  That is also the surface a
    person sees, which is the thing worth pinning.

    MUTATION THIS MUST FAIL AGAINST: append the flag instead of inserting it.
    `sbatch` takes the script last, so an appended flag lands after it and the
    question becomes a different command from the one that would be sent.
    """
    from molbuilder.jobset.model import Job, JobSet, Resources
    from molbuilder.jobset.submit import submit_jobset

    deck = ("SystemName test\nSystemLabel J\nNumberOfAtoms 2\n"
            "DM.UseSaveDM .false.\nMD.UseSaveXV .false.\n")
    js = JobSet(name="J", engine="siesta", kind="ladder", shared=[],
                jobs=[Job(name="coarse", script="J_01_coarse.fdf",
                          resources=Resources(mpi_np=1))])

    def _cmd(mode):
        # One tree per mode: sharing one lets the first call's attempt decide
        # what the second is allowed to do.
        base = tmp_path / mode
        base.mkdir()
        (base / "J_01_coarse.fdf").write_text(deck)
        d = base / "01_coarse"
        d.mkdir(parents=True)
        (d / "J_01_coarse.fdf").write_text(deck)
        (d / "J_01_coarse.sbatch").write_text("#!/bin/bash\n#SBATCH -J J\n")
        results = submit_jobset(js, base, mode=mode, dry_run=(mode != "ask"))
        assert len(results) == 1, results
        return list(results[0].command)

    asked, sent = _cmd("ask"), _cmd("submit")

    assert "--test-only" in asked, "ask did not add the flag"
    assert "--test-only" not in sent, "submit carried the question's flag"
    assert [a for a in asked if a != "--test-only"] == sent, (
        f"ask changed more than the flag:\n  asked {asked}\n  sent  {sent}")
    assert asked.index("--test-only") == 1, (
        "the flag must be inserted right after the program -- sbatch takes "
        "the script last, so appending it makes the question a different "
        "command from the one that would be sent")

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


def test_no_scheduler_previews_no_sbatch_either():
    """The same contradiction one line earlier (user, 2026-08-28): *"nothing
    to wait for"* followed by ``would send: sbatch ...`` reads as two
    answers.  On a machine with no scheduler nothing WOULD be sent, so
    nothing is previewed.  Guarded at the source like its sibling above,
    because the preview line lives in the CLI."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "molbuilder/jobset/_cli.py").read_text()
    i = src.index('click.echo("  would send: "')
    guard = src[max(0, i - 400):i]
    assert "not all(p.no_scheduler for p in preds)" in guard, (
        "the sbatch preview must be gated off when no scheduler exists")
