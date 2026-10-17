"""What molbuilder does with runs that happened -- each made once in this pass
with the real SIESTA (`support/real_runs.py`, plan § 5y) and read here.

A stage builds on the newest run of the stage before it, and launched again
continues from its own latest run, warm, or takes nothing, cold
(`execution/job-system.md` § 5.4, *A stage launched again*); a launch shows
what it will run, asks, and writes down each decision as it is made, and a
dry run of it writes nothing (§ 6.0); a launch that names no stage sends the
one prepared and not launched; a benchmark run here walks its unlaunched
trials under one per-trial bound, passing over the ones launched before by
name; the run script's own dry run says where its ranks and threads came
from (`execution/architecture.md` § 5.2); and every name the Task setup card
gives a stage is a file on disk (`job-contracts.md` § 2.2).

Every expectation is the contract's words, read off what the commands
answered while the runs were made and off the folders they left -- never
a reader's number.
"""
from __future__ import annotations

import json

import pytest

from _road import conda_hook, env_available

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]


def _said(run, step, *words):
    """What ``step`` answered holds each of ``words``."""
    said = run.said[step]
    for w in words:
        assert w in said.output, f"{step}: {w!r} not in:\n{said.output}"


def _records(run, source: str) -> None:
    """The run in ``run`` records ``source`` as the run it continues from --
    its `.continued-from` and its launch record, read through their doors
    (`checkpointing.md` S3; `job-system.md` § 5.4)."""
    from molbuilder.runrecord import RunRef, launch_record, read_continued_from
    from molbuilder.runs import run_of
    names = run_of(run).names
    assert read_continued_from(run, names, 0) == RunRef.read(
        source, shared=False)
    assert launch_record(run, names)["continued_from"] == source


def _decided(bundle):
    """The decisions the calculation's ledger holds: each one's decision,
    and its verb's (``launch continues``)."""
    from molbuilder.jobset.ledger import LEDGER_FILE
    lines = [json.loads(x) for x in
             (bundle / LEDGER_FILE).read_text().splitlines() if x.strip()]
    return ({e["decision"] for e in lines}
            | {f"{e['verb']} {e['decision']}" for e in lines})


# --------------------------------------------------------------------- #
#  A launch, as it is shown, asked and written down (job-system.md § 6.0) #
# --------------------------------------------------------------------- #

def test_a_run_here_is_shown_and_asked_and_written_down(real_h2_layered):
    """A run on this machine is shown and asked as a submission is, then
    run -- waited on, under `--foreground`, saying where its output is as a
    run left in the background does -- and the ledger holds the question as
    asked, its answer and the launch (`job-system.md` § 6.0)."""
    from molbuilder.jobset.ledger import LEDGER_FILE
    _said(real_h2_layered, "launch coarse", "about to run here, waited on:",
          "bash H2_01_coarse.run.sh", "its output: 01_coarse/run-0/",
          "(--yes)", "[ran]")
    lines = [json.loads(x) for x in (real_h2_layered.bundle / LEDGER_FILE)
             .read_text().splitlines() if x.strip()]
    first = [e for e in lines if e.get("verb") == "launch"
             and e["decision"] in ("question", "launched")]
    assert (first[0].get("about"), first[0].get("answer")) == (
        "run here, in the foreground", "yes (--yes)"), first[:2]
    assert first[1]["decision"] == "launched" \
        and "background" not in first[1], first[:2]


def test_a_run_here_left_in_the_background_loses_nothing_it_prints(
        real_h2_layered):
    """A run here starts in the background unless said (`job-system.md`
    § 6.0): the launch shows it, takes `--yes` as the answer, starts it and
    returns, saying where its output is and naming its process in the
    ledger -- its own session, so closing the terminal does not end it, and
    not waited for: still this test's to wait for once the launch returned
    -- and nothing the run prints is lost: the run script's own lines are in
    its runwrap log, read by that log's reader through the engine's end
    (the engine's wall, which it prints once the engine has ended), the
    engine's own output is the run's `.out`, read by its parser to its end,
    and the run finished, by the one status door.  The silent failures: a
    launch that waits anyway, a run its terminal's closing ends, and a run
    whose terminal was gone with its account of how it ended printed to
    nowhere."""
    from molbuilder.jobset.ledger import LEDGER_FILE
    from molbuilder.parse import detect
    from molbuilder.runs import folder_answer, openable, run_answer
    _said(real_h2_layered, "launch naming no stage",
          "about to start here, in the background:", "(--yes)",
          "its output: 02_medium/run-0/", "[started in the background]")
    medium = real_h2_layered.bundle / "02_medium" / "run-0"
    said = run_answer(medium)
    assert (said["state"], said["detail"]) == ("finished",
                                               "job_completed"), said
    comp = folder_answer(medium)["record"]["computation"]
    assert comp["host"]["hostname"], comp
    assert comp["time"]["engine_elapsed_s"] > 0, comp
    out, trail = openable(medium)
    assert out is not None, trail
    traj = detect(out).parse(str(out))
    assert traj.run_state == "ended" and traj.frames, traj.run_state
    lines = [json.loads(x) for x in (real_h2_layered.bundle / LEDGER_FILE)
             .read_text().splitlines() if x.strip()]
    asked = [e for e in lines if e["decision"] == "question"
             and e.get("about") == "run here, in the background"]
    assert asked and asked[-1]["answer"] == "yes (--yes)", asked
    went = [e for e in lines if e["decision"] == "launched"
            and e.get("background")]
    assert len(went) == 1 and isinstance(went[0]["pid"], int), went
    left = real_h2_layered.left
    assert left["sid"] == left["pid"] and left["waited_here"], left


def test_a_run_here_that_ended_is_said_once_by_the_jobs_list(real_h2_layered):
    """Every launch here leaves its address, named on its ledger line
    (`job-system.md` § 6.0a): a run waited on is told by its own launch and
    leaves none behind; a run left in the background, once it has ended, is
    said by `molbuilder jobs status` once -- its process ended, its member's
    run in the status door's words -- and then dropped.  The silent
    failures: a list that holds every run that ever went, and a run that
    ended said as going."""
    from molbuilder.jobset.ledger import LEDGER_FILE
    from support.road import job_lines
    lines = [json.loads(x) for x in (real_h2_layered.bundle / LEDGER_FILE)
             .read_text().splitlines() if x.strip()]
    went = [e for e in lines if e["decision"] == "launched"]
    left = [e["job"] for e in went if e.get("background")]
    waited = [e["job"] for e in went if not e.get("background")]
    assert len(left) == 1 and waited, went
    after = real_h2_layered.jobs["after it ended"]
    assert after.exit_code == 0, after.output
    host, pid = left[0].rsplit("-", 1)
    mine = job_lines(after.output, left[0])
    assert mine and mine[0] == (f"{left[0]}  run here on {host}, in the "
                                f"background, pid {pid} -- its process has "
                                f"ended"), after.output
    assert any(ln.startswith("    medium  02_medium/run-0  finished")
               for ln in mine), mine
    assert mine[-1] == "  ended -- said once; its address is removed", mine
    assert not any(job_lines(after.output, job) for job in waited), \
        after.output
    again = real_h2_layered.jobs["again"]
    assert again.output.strip() == ("no job launched from this machine is "
                                    "on file."), again.output


def test_a_run_here_asked_at_a_terminal_goes_the_way_answered(
        real_h2_layered):
    """At a terminal a run here is asked *launch in the background?
    [Y/n/foreground(F)]*, and typed ``f`` it runs in the foreground: waited
    on, its line saying it ran, and the ledger holding the question as asked
    with the answer given and the launch with no background
    (`job-system.md` § 6.0).  The silent failure: a person typing ``f`` and
    the run left in the background, or the ledger saying what was not
    asked."""
    from molbuilder.jobset.ledger import LEDGER_FILE
    _said(real_h2_layered, "launch coarse again cold",
          "launch in the background? [Y/n/foreground(F)]", "[ran]")
    lines = [json.loads(x) for x in (real_h2_layered.bundle / LEDGER_FILE)
             .read_text().splitlines() if x.strip()]
    cold = [e for e in lines if e.get("verb") == "launch"
            and e["decision"] in ("question", "launched")
            and e.get("flags") == [["--cold", True]]]
    assert [(e["decision"], e.get("about"), e.get("answer"),
             e.get("background")) for e in cold] == [
        ("question", "run here, in the background", "foreground", None),
        ("launched", None, None, None)], cold


def test_a_dry_run_of_launching_again_writes_nothing(real_h2_layered):
    """It says what the launch would do -- the next run, warm from the
    stage's own latest -- and leaves every file as it was."""
    _said(real_h2_layered, "dry run of launching coarse again",
          "WOULD launch it again into run-1: it continues from "
          "01_coarse/run-0 (its own latest run; concluded rc=0",
          "would copy H2.XV")
    before, after = real_h2_layered.files_before, real_h2_layered.files_after
    assert before == after, sorted(set(before.items()) ^ set(after.items()))


def test_a_launch_naming_no_stage_sends_the_one_prepared_and_not_launched(
        real_h2_layered):
    """`medium` prepared and `coarse` launched: the bare launch sends
    `medium`, and its launch record is written as it starts."""
    assert (real_h2_layered.bundle / "02_medium" / "run-0"
            / "run.json").is_file()


# --------------------------------------------------------------------- #
#  A stage launched again (job-system.md § 5.4)                          #
# --------------------------------------------------------------------- #

def test_launched_again_warm_it_continues_from_its_own_latest_run(
        real_h2_layered):
    """Warm, the next run takes the restart file the stage's latest run
    left, recorded as prep's hand-over is -- beside the run and in the
    ledger."""
    b = real_h2_layered.bundle
    _said(real_h2_layered, "launch coarse again",
          "launched again into run-1: it continues from 01_coarse/run-0 "
          "(its own latest run; concluded rc=0", "copied H2.XV")
    for name in ("H2.XV", ".continued-from"):
        assert (b / "01_coarse" / "run-1" / name).is_file(), name
    _records(b / "01_coarse" / "run-1", "01_coarse/run-0")
    assert "launch continues" in _decided(b)


def test_launched_again_cold_it_takes_nothing_of_its_own(real_h2_layered):
    b = real_h2_layered.bundle
    _said(real_h2_layered, "launch coarse again cold",
          "launched again into run-2: it starts cold -- from its deck alone")
    assert (b / "01_coarse" / "run-2").is_dir()
    assert not (b / "01_coarse" / "run-2" / ".continued-from").exists()


# --------------------------------------------------------------------- #
#  What the next stage builds on (job-system.md § 5.4)                   #
# --------------------------------------------------------------------- #

def test_the_next_stage_builds_on_the_newest_run_of_the_stage_before_it(
        real_h2_layered):
    """`medium` is prepared taking nothing -- its prep and the plan's table
    say what its first launch will take, and from which stage -- and its
    first launch continues from `coarse`'s newest run, which finished (the
    cold one), saying so and what it copied (`job-system.md` § 5.0, § 6.0
    step 1)."""
    from molbuilder.jobset.plan import FILENAME as PLAN_FILE
    b = real_h2_layered.bundle
    _said(real_h2_layered, "prep medium",
          "at launch: takes", "from `coarse`'s newest run, which must have "
          "finished")
    plan = (b / PLAN_FILE).read_text()
    for words in ("restart files it declares",
                  "This prep: `medium` at launch: takes"):
        assert words in plan, (words, plan)
    _said(real_h2_layered, "launch naming no stage",
          "continues from 01_coarse/run-2 (the stage before it; concluded "
          "rc=0", ": copied H2.XV")
    for name in ("H2.XV", ".continued-from"):
        assert (b / "02_medium" / "run-0" / name).is_file(), name
    _records(b / "02_medium" / "run-0", "01_coarse/run-2")


def test_every_run_s_history_is_its_own(real_h2_layered):
    """A run that continues its stage's own run (`coarse`'s run-1, warm) and
    one that continues the stage before it (`medium`'s run-0) take no history
    from the run they continue: each holds its own steps, as the first run
    does (`job-contracts.md` § 2.2; `project-layout.md` § 2.3.4).

    MUTATION THIS MUST FAIL AGAINST: the run script laying no pointer --
    SIESTA writes ``H2.MD.nc`` in each run folder, and no run has a history
    of its own."""
    from support.road import history_is_the_runs_own
    b = real_h2_layered.bundle
    for out in ("01_coarse/run-0/H2_01_coarse-run0.out",
                "01_coarse/run-1/H2_01_coarse-run0.out",
                "02_medium/run-0/H2_02_medium-run0.out"):
        history_is_the_runs_own(b / out, "H2")


def test_a_run_that_ended_on_its_own_is_not_followed(real_h2_layered):
    """The server's one answer to a viewer about a stage's newest run
    (`runs.run_answer`, `web/results.md` § 4.1): finished, and not live."""
    from molbuilder.jobset.materialize import run_dir, stage_home
    from molbuilder.runfiles import stem
    from molbuilder.runs import run_answer
    from molbuilder.task import read_task
    b = real_h2_layered.bundle
    task = read_task(b / "task.json")
    home = stage_home(b, task, "coarse")
    got = run_answer(str(run_dir(home.dir) / (stem(task.label, home.token)
                                              + ".fdf")))
    assert (got.get("state"), got.get("live")) == ("finished", False), got


# --------------------------------------------------------------------- #
#  The names on the card are the files on disk (job-contracts.md § 2.2)  #
# --------------------------------------------------------------------- #

def test_every_name_the_card_gives_a_stage_is_on_disk(real_h2_layered):
    """A launched stage's names for its prep, its launch and its run, and
    the stage prepared after it -- in the layered folders (the flat ones:
    `test_siesta_flat_run_e2e.py`)."""
    from support.road import _road_card_written
    _road_card_written([{"stage": "coarse", "moments": ["prep", "launch", "run"]},
                        {"stage": "medium", "moments": ["prep"]}],
                       real_h2_layered.bundle)


# --------------------------------------------------------------------- #
#  The run script's own dry run (architecture.md § 5.2)                  #
# --------------------------------------------------------------------- #

def test_the_run_script_runs_the_stated_ranks_and_threads(real_h2_layered):
    said = real_h2_layered.said["script dry run"]
    assert said.exit_code == 0, said.output
    _said(real_h2_layered, "script dry run",
          "MPI ranks    : 1   (source: stated at prep)",
          "OMP threads  : 1   (source: stated at prep)")


def test_a_flag_on_the_run_script_still_overrides_and_says_so(
        real_h2_layered):
    said = real_h2_layered.said["script dry run -np 2"]
    assert said.exit_code == 0, said.output
    _said(real_h2_layered, "script dry run -np 2",
          "MPI ranks    : 2   (source: -np flag)")


# --------------------------------------------------------------------- #
#  A benchmark run here (job-system.md § 6.0)                            #
# --------------------------------------------------------------------- #

def test_a_benchmark_walks_its_unlaunched_trials_passing_the_launched_over(
        real_h2_bench):
    """One trial launched by name; the benchmark's launch then passes it
    over by name and walks the rest in one walk, as a shelf does on a
    queue -- its walk, its log and its folder mark written, the launch in
    the ledger."""
    b = real_h2_bench.bundle
    _said(real_h2_bench, "launch the benchmark", "G0K1C1",
          "skipped -- already launched", "bench-group", "rides the group",
          "[ran]")
    launch = b / "01_coarse" / "bench" / "launch"
    for name in ("bench-group.run.sh", "bench-group.log", "calcdir.json"):
        assert (launch / name).is_file(), name
    assert "launch launched" in _decided(b)


def test_a_benchmarks_per_trial_bound_is_the_walks(real_h2_bench):
    """Asked for five seconds a trial, the walk bounds each at the floor it
    states: five minutes, killed thirty seconds after."""
    walk = (real_h2_bench.bundle / "01_coarse" / "bench" / "launch"
            / "bench-group.run.sh").read_text()
    for words in ("per-trial-bound=300s", "timeout -k 30 300 bash"):
        assert words in walk, words
