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
a reader's number.  These rows ran on the suite's stand-in engine until
2026-10-08 (`tests/data/hand_overs.toml`, `launch_protocol.toml`,
`launch_values.toml`, `the_catalogue.toml`); a test that reads what a run
left is an end-to-end test, made with the engine.
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
    run, and the ledger holds the launch."""
    _said(real_h2_layered, "launch coarse", "about to run here:",
          "bash H2_01_coarse.run.sh", "(--yes)", "[ran]")
    assert "launched" in _decided(real_h2_layered.bundle)


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
    """`medium` continues from `coarse`'s newest run, which finished -- the
    cold one -- and says what it carried under the plan's table, whose
    column is each stage's declaration."""
    from molbuilder.jobset.plan import FILENAME as PLAN_FILE
    b = real_h2_layered.bundle
    _said(real_h2_layered, "prep medium", "continues from 01_coarse/run-2")
    plan = (b / PLAN_FILE).read_text()
    for words in ("restart files it declares",
                  "This prep: `medium` continues from 01_coarse/run-2 "
                  "(the stage before it; concluded rc=0",
                  ": copied H2.XV"):
        assert words in plan, (words, plan)
    for name in ("H2.XV", ".continued-from"):
        assert (b / "02_medium" / "run-0" / name).is_file(), name


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
