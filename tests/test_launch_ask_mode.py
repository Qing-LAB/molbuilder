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


# --------------------------------------------------------------------- #
#  what `launch --mode ask` says, on the road a person takes             #
# --------------------------------------------------------------------- #

def _jobset(*args):
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


def _a_prepped_stage(tree, monkeypatch, *stated):
    """One stage, described and prepped the way a person does it:
    `jobset init` on a structure in the projects tree, then `prep run
    --target this`, stating its launch values as flags (``stated``) -- a
    run states them or is refused (`architecture.md` § 5.2).  NO engine
    runs -- asking needs a prepped attempt and nothing more."""
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
    r = _jobset("prep", "run", "coarse", "--bundle", bundle,
                "--target", "this", *stated)
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
    from molbuilder.runrecord import launch_record

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
    # PySCF's threads, everywhere; where the machine has queues, the queue,
    # the wall and the memory too.
    bundle, attempt = _a_prepped_stage(
        tmp_path / "projects", monkeypatch, "--cpus-per-task", "1",
        *(("--domain", "htc", "--time", "1h", "--mem", "4G")
          if machine == "scheduler answers" else ()))
    if machine == "no scheduler":
        _no_scheduler_on_path(monkeypatch)

    # A queue is named where the machine has queues -- for `ask` as for
    # `submit`, so the line asked about is the one that would go (W52).
    r = _jobset("launch", "run", "coarse", "--bundle", bundle,
                "--mode", "ask",
                *(("--domain", "htc") if machine == "scheduler answers"
                  else ()))
    assert r.exit_code == 0, r.output
    out = r.output

    assert launch_record(attempt) is None, (
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
        # THE LINE TO SEND, without the question's flag (W52: it was shown
        # with `--test-only` in it, which is not the line that would go).
        would = next(ln for ln in out.splitlines() if "would send:" in ln)
        assert "sbatch -J" in would and "--test-only" not in would, would
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
