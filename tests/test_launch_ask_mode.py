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


# --------------------------------------------------------------------- #
#  what `launch --mode ask` says, on the road a person takes             #
# --------------------------------------------------------------------- #

def _jobset(*args):
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


def _a_prepped_stage(tree, monkeypatch):
    """One stage, described and prepped the way a person does it:
    `jobset init` on a structure in the projects tree, then `prep run
    --target this`.  NO engine runs -- asking needs a prepped attempt and
    nothing more."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "P" / "structure" / "h2.xyz").write_text(
        "2\nh2\nH 0 0 0\nH 0 0 0.74\n")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tree.parent)
    r = _jobset("init", "--structure", "P/structure/h2.xyz",
                "--bundle", "P/optimization/H2", "--engine", "pyscf",
                "--shape", "hierarchical", "--name", "H2")
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "optimization" / "H2"
    # How a shell enters the env on this machine -- no wrapper is written
    # without it (`running-a-job.md` § 5.2).
    (bundle / ".molbuilder.json").write_text(
        '{"script_generation": {"activation": "conda activate", '
        '"preamble": "true"}}')
    r = _jobset("prep", "run", "coarse", "--bundle", bundle,
                "--target", "this")
    assert r.exit_code == 0, r.output
    attempt = bundle / "01_coarse" / "run-0"
    assert attempt.is_dir(), sorted(p.name for p in bundle.iterdir())
    return bundle, attempt


def _a_scheduler_that_answers(bin_dir, calls):
    """An `sbatch` that answers `--test-only` the way Sol's did -- the
    prediction on STDERR, exit 0 -- and writes down every call.  Called
    WITHOUT the flag it would have queued a job, so it says so and the
    call is on record."""
    bin_dir.mkdir()
    f = bin_dir / "sbatch"
    f.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{calls}"\n'
        'case " $* " in\n'
        f'  *" --test-only "*) echo "{SOL_PREDICTION}" >&2; exit 0 ;;\n'
        "esac\n"
        'echo "Submitted batch job 4242"\n')
    f.chmod(0o755)


def _no_scheduler_on_path(monkeypatch):
    """PATH without any `sbatch` -- a workstation.  The suite's own
    refusing `sbatch` (conftest) is removed with the rest: it stands in
    for a scheduler, which is exactly what this machine does not have."""
    import os
    keep = [d for d in os.environ["PATH"].split(os.pathsep)
            if d and not os.path.exists(os.path.join(d, "sbatch"))]
    monkeypatch.setenv("PATH", os.pathsep.join(keep))


@pytest.mark.parametrize("machine", ["scheduler answers", "no scheduler"])
def test_ask_answers_on_the_road_and_launches_nothing(tmp_path, monkeypatch,
                                                      machine):
    """**What `launch run --mode ask` says, and that it leaves no launch.**

    The failures, each a contradiction a person acted on or could have:

    * the attempt RECORDED A LAUNCH -- `run.json` says a job exists, so
      `status` would report a job nobody submitted, and the next `launch`
      would refuse the attempt as already run;
    * on a machine with NO scheduler the closing line said *launch it with
      `--mode submit` when the answer suits you* right under *there is no
      scheduler here* -- a mode this machine cannot run (caught by running
      it, 2026-08-27);
    * and one line earlier it previewed ``would send: sbatch ...`` under
      *nothing to wait for* -- two answers (user, 2026-08-28).

    Contract: `execution/running-a-job.md` § 5.5 -- *"no job is created,
    nothing is recorded, and `status` sees nothing"*, and the line asked
    about is the line that would be sent.  Driven through `init` -> `prep`
    -> `launch`; the scheduler is a stub on PATH that answers the way Sol's
    `sbatch --test-only` did, and records every call.

    MUTATION THIS MUST FAIL AGAINST: `_submit_slurm` recording the launch
    (`_record_launch`) in its ask branch; the CLI's `would send:` preview
    printed without its no-scheduler guard; the closing line always naming
    `--mode submit`.
    """
    from molbuilder.jobset.materialize import read_run_launch, was_launched

    calls = tmp_path / "sbatch-calls.log"
    if machine == "scheduler answers":
        # A machine that HAS a queue: the probed record names one, so prep
        # writes the `.sbatch` the question is asked about.
        from conftest import write_machine_record
        from molbuilder.scheduler import Domain
        write_machine_record(scheduler="slurm", domains=[
            Domain(name="htc", partition="htc", qos="public",
                   max_time="0-04:00:00")])
        _a_scheduler_that_answers(tmp_path / "bin", calls)
        import os
        monkeypatch.setenv(
            "PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    bundle, attempt = _a_prepped_stage(tmp_path / "projects", monkeypatch)
    if machine == "no scheduler":
        _no_scheduler_on_path(monkeypatch)

    r = _jobset("launch", "run", "coarse", "--bundle", bundle,
                "--mode", "ask")
    assert r.exit_code == 0, r.output
    out = r.output

    assert read_run_launch(attempt) is None and not was_launched(attempt), (
        f"asking recorded a launch in {attempt.name}: `status` would now "
        f"report a job nobody submitted\n{out}")
    if machine == "scheduler answers":
        assert "2026-08-27T11:22:03" in out, (
            f"the scheduler's predicted start is not shown:\n{out}")
        assert "nothing was submitted" in out, out
        sent = calls.read_text().splitlines()
        assert len(sent) == 1 and "--test-only" in sent[0].split(), (
            f"the scheduler was not asked exactly once, with --test-only: "
            f"{sent}")
        assert "would send: sbatch --test-only" in out, out
        assert "--mode submit" in out, (
            f"the answer does not say how to act on it:\n{out}")
    else:
        assert "no scheduler on this machine" in out, out
        assert "--mode direct" in out, (
            f"no pointer at the mode that DOES work here:\n{out}")
        assert "would send" not in out, (
            f"previewed an sbatch line on a machine with no scheduler:\n{out}")
        assert "--mode submit" not in out, (
            f"pointed at a mode this machine cannot run:\n{out}")
