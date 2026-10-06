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

from molbuilder.jobset.ask import Prediction, prediction_table


# --------------------------------------------------------------------- #
#  reading what the scheduler said                                       #
# --------------------------------------------------------------------- #

# Retired 2026-10-06 (W57 T1, user: "there should be no assumption what so
# ever about the text returned by slurm"): the six tests that read SLURM's
# `--test-only` text -- two lines copied from Sol, the rest written by hand
# -- and the stand-in `sbatch` that answered with them.  Reading a real
# scheduler's answer is the field tier's
# (`tests/field/test_ask_the_target.py`).  The table below is molbuilder's
# own rendering of what was read, and stays.


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
                "--calculation", "optimization",
                "--shape", "hierarchical", "--name", "H2")
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "optimization" / "H2"
    r = _jobset("prep", "run", "coarse", "--bundle", bundle,
                "--target", "this", *stated)
    assert r.exit_code == 0, r.output
    attempt = bundle / "01_coarse" / "run-0"
    assert attempt.is_dir(), sorted(p.name for p in bundle.iterdir())
    return bundle, attempt


def _no_scheduler_on_path(monkeypatch):
    """PATH without any `sbatch` -- a workstation.  The suite's own
    refusing `sbatch` (conftest) is removed with the rest: it stands in
    for a scheduler, which is exactly what this machine does not have."""
    import os
    keep = [d for d in os.environ["PATH"].split(os.pathsep)
            if d and not os.path.exists(os.path.join(d, "sbatch"))]
    monkeypatch.setenv("PATH", os.pathsep.join(keep))


def test_ask_on_a_machine_with_no_scheduler_launches_nothing(tmp_path,
                                                            monkeypatch):
    """**What `launch run --mode ask` says where there is no scheduler, and
    that it leaves no launch.**

    The failures, each a contradiction a person acted on or could have:

    * the attempt RECORDED A LAUNCH -- `run.json` says a job exists, so
      `status` would report a job nobody submitted, and the next `launch`
      would refuse the attempt as already run;
    * the closing line said *launch it with `--mode submit`* right under
      *there is no scheduler here* -- a mode this machine cannot run (caught
      by running it, 2026-08-27);
    * and one line earlier it previewed ``would send: sbatch ...`` under
      *nothing to wait for* -- two answers (user, 2026-08-28).

    Contract: `execution/running-a-job.md` § 5.5 -- *"no job is created,
    nothing is recorded, and `status` sees nothing"*.  Driven through
    `init` -> `prep` -> `launch`.  *(Its other half -- a scheduler that
    answers, a stand-in replaying Sol's `--test-only` line -- retired
    2026-10-06, W57 T1; a real scheduler's answer is the field tier's,
    `tests/field/test_ask_the_target.py`.)*

    MUTATION THIS MUST FAIL AGAINST: the ask recording the launch
    (`_record_launch`); the CLI's `would send:` preview printed without its
    no-scheduler guard; the closing line always naming `--mode submit`.
    """
    from molbuilder.runrecord import launch_record

    bundle, attempt = _a_prepped_stage(
        tmp_path / "projects", monkeypatch, "--cpus-per-task", "1")
    _no_scheduler_on_path(monkeypatch)
    r = _jobset("launch", "run", "coarse", "--bundle", bundle,
                "--mode", "ask")
    assert r.exit_code == 0, r.output
    out = r.output

    assert launch_record(attempt) is None, (
        f"asking recorded a launch in {attempt.name}: `status` would now "
        f"report a job nobody submitted\n{out}")
    assert "no scheduler on this machine" in out, out
    assert "--mode direct" in out, (
        f"no pointer at the mode that DOES work here:\n{out}")
    assert "would send" not in out, (
        f"previewed an sbatch line on a machine with no scheduler:\n{out}")
    assert "--mode submit" not in out, (
        f"pointed at a mode this machine cannot run:\n{out}")
