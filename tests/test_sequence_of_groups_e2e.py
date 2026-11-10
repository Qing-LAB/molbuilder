"""A sequence of groups -- one job launching each group in turn, each when
the one before has finished (`execution/job-system.md` § 6.0, *A sequence of
groups*) -- down the road with the real SIESTA.

``molbuilder describe`` (the shipped `publishable` ladder: two stages,
`coarse` and `medium`, the second building on the first, groups 1 2) ->
``prep task --group 1 --group 2 --one-job`` -> ``launch task --group 1
--group 2 --mode direct``: ONE walk, run here, launching each group as its
own launch would, each run recording the job it rode.

THE SILENT FAILURES IT GUARDS: a group launched before the one it builds on
finished (taking an older run, or nothing); a run inside the job recorded as
a run of its own (every reader then asks no job about it, and a job killed
leaves it reading running); a group the job still holds launched a second
time by hand.  Reading the code is not enough: the walk runs each group's
launch as a separate process, from molbuilder's own folder, on the
calculation named by its full path -- only a run shows those words reach.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from _road import conda_hook, env_available
from support.road import jobset, molbuilder

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]


def _groups(bundle: Path):
    """The job request's groups, each with its stages, in order."""
    from molbuilder.calculation import open as open_calculation
    request = open_calculation(bundle).request
    return [(n, request.members(n)) for n in request.numbers]


def _ledger(bundle: Path):
    """The calculation's ledger, line by line, as written."""
    from molbuilder.jobset import ledger
    return ledger.read(bundle)


@pytest.fixture(scope="module")
def road(tmp_path_factory):
    """Two calculations: one sent as a sequence run here and waited on;
    one sent in the background, looked at while it runs, its group launched
    alone refused, and then stopped."""
    from support.real_runs import h2_optimization, on_the_road
    said = {}
    # One rank a run: an H2 needs no more.
    one_rank = {"mpi_np": 1, "omp_threads": 1}
    with on_the_road(tmp_path_factory, "sequence") as tree:
        whole = h2_optimization(tree, "P/optimization/H2seq", execution=one_rank)
        groups = _groups(whole)
        words = [w for n, _ in groups for w in ("--group", str(n))]
        said["prep"] = jobset("prep", "task", *words, "--one-job",
                              "--bundle", whole, "--target", "this")
        assert said["prep"].exit_code == 0, said["prep"].output
        said["launch"] = jobset("launch", "task", *words, "--bundle", whole,
                                "--mode", "direct", "--foreground", "--yes")
        said["status"] = jobset("status", "--bundle", whole)

        left = h2_optimization(tree, "P/optimization/H2stop", execution=one_rank)
        r = jobset("prep", "task", *words, "--one-job", "--bundle", left,
                   "--target", "this")
        assert r.exit_code == 0, r.output
        said["background"] = jobset("launch", "task", *words, "--bundle",
                                    left, "--mode", "direct", "--yes")
        assert said["background"].exit_code == 0, said["background"].output
        said["held_status"] = jobset("status", "--bundle", left)
        said["alone"] = jobset("launch", "task", "--group", str(groups[-1][0]),
                               "--bundle", left, "--mode", "direct", "--yes")
        said["bare"] = jobset("launch", "task", "--bundle", left,
                              "--mode", "direct", "--yes")
        said["prep_held"] = jobset("prep", "task", "--group",
                                   str(groups[-1][0]), "--bundle", left,
                                   "--target", "this")
        said["jobs"] = molbuilder("jobs", "status")
        # THE READS ABOVE ARE OF A HELD GROUP only while the job has not
        # launched the last one -- asked of that group's own runs, after
        # the reads, so a job faster than them is said, not passed over.
        from molbuilder.calculation import open as open_calculation
        from molbuilder.jobset.materialize import ever_launched
        opened = open_calculation(left)
        said["reached_last"] = any(
            ever_launched(opened, opened.jobset.job(st))
            for st in groups[-1][1])
        from molbuilder.joblist import Job, every, going_here
        job = next(j for j in every()
                   if isinstance(j, Job) and j.sequence is not None
                   and Path(j.calculation) == left.resolve())
        # THE STOP LANDS ON A RUN THAT STARTED: group 1's first stage read
        # as begun (running, or already ended) before it is sent, as its
        # status door reads it -- a stop sent before the job's first run
        # began would leave it never started, not stopped.  Waited for,
        # bounded; what the stop then found is the tests' to judge
        # (`reached_last`, the stopped stage's state).
        from support.real_runs import stage_status
        first, deadline = groups[0][1][0], time.time() + 300
        while (stage_status(left, first).state
               not in ("running", "finished", "failed")
               and time.time() < deadline):
            time.sleep(0.5)
        said["stop"] = molbuilder("jobs", "stop", job.id, "--yes")
        # THE WALK ENDS ON ITS SIGNAL: waited for, bounded, never assumed.
        deadline = time.time() + 120
        while going_here(job) and time.time() < deadline:
            time.sleep(0.5)
        said["stopped_going"] = going_here(job)
        said["after_stop"] = jobset("status", "--bundle", left)
    return {"whole": whole, "left": left, "groups": groups, "said": said}


def test_the_sequence_runs_every_group_in_turn(road):
    """Each group launched when the one before had finished: every stage's
    first run finished, and each stage after the first continued from the
    run of the stage before it, which this very job had just made."""
    from molbuilder.runrecord import launch_record
    from molbuilder.jobset.runstatus import jobset_status
    from molbuilder.calculation import open as open_calculation
    r = road["said"]["launch"]
    assert r.exit_code == 0, r.output
    calc = open_calculation(road["whole"])
    rows = {s.name: s for s in jobset_status(calc).stages}
    stages = [st for _n, sts in road["groups"] for st in sts]
    assert [rows[s].state for s in stages] == ["finished"] * len(stages), \
        {s: (rows[s].state, rows[s].detail) for s in stages}
    from molbuilder.jobset.naming import stage_home
    for before, after in zip(stages, stages[1:]):
        home = stage_home(road["whole"], calc.task, after)
        rec = launch_record(home.dir / "run-0")
        assert rec["continued_from"].startswith(
            stage_home(road["whole"], calc.task, before).token + "/run-0"), \
            (after, rec)


def test_each_run_records_the_sequence_it_rode(road):
    """Every run the job sent records the sequence -- its name, and the
    job's line, the walk run here: ``mode: direct`` and the walk's line
    (`job-system.md` § 6.0, *The walk*: "run here, `mode: direct` and the
    walk's line") -- read back through the run record the Results tab reads,
    its folder answer (`runs.folder_answer`; `web/results.md` § 3a,
    `computation.launch.sequence`; `model/parse.md` § 5d.1b, the
    ``run-json`` row).

    THE SILENT FAILURE: a run inside the job recorded as a run of its own,
    so every reader asks no job about it and a job killed leaves it reading
    running."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.group import sequence_names
    from molbuilder.jobset.naming import stage_home
    from molbuilder.runs import folder_answer
    calc = open_calculation(road["whole"])
    numbers = [n for n, _ in road["groups"]]
    name = sequence_names(road["whole"], calc.task, calc.request,
                          numbers).stem
    for _n, stages in road["groups"]:
        for st in stages:
            home = stage_home(road["whole"], calc.task, st)
            launch = folder_answer(home.dir / "run-0")[
                "record"]["computation"]["launch"]
            assert launch["sequence"] == name, (st, launch)
            assert launch["mode"] == "direct" and launch["job_id"] is None, \
                launch
            assert launch["command"] == ["bash", f"launch/{name}.run.sh"], \
                launch


def test_the_sequence_record_and_the_ledger_say_it(road):
    """The sequence's launch record states its groups and how it went, and
    the ledger keeps the one launch that sent it and each group's launch
    inside it, naming the job (`job-system.md` § 6.0, *A sequence of
    groups*).

    And each group's first launch, inside the job, saved the folder saying
    so -- ``… · before launch task group 2 (medium), in job 4817 (sequence
    H2_sequence-01_coarse-02_medium)``, the job as `molbuilder jobs status`
    names it (a run here's walk, ``<host>-<pid>``: the address the ledger's
    ``launched`` line names) and the sequence by its name
    (`checkpointing.md` § 9, *The note an act writes*; 2.62.1), read back
    through the folder's history (`checkpoint.Repo.states`, § 10.2).  THE
    SILENT FAILURE: a state the job took read as one the person's own
    launch took -- or the job's words never reaching the note, only the
    walk carrying them to each group's launch on the job's node."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.checkpoint import Repo
    from molbuilder.jobset.group import sequence_names
    from molbuilder.jobset.sequence import read_record, record_path
    calc = open_calculation(road["whole"])
    numbers = [n for n, _ in road["groups"]]
    names = sequence_names(road["whole"], calc.task, calc.request, numbers)
    rec = read_record(record_path(road["whole"], names))
    assert [g["number"] for g in rec.groups] == numbers, rec
    assert rec.mode == "direct" and rec.scheduler is None, rec
    entries = _ledger(road["whole"])
    outer = [e for e in entries if e.get("decision") == "launched"
             and e.get("sequence") == numbers]
    inner = [e for e in entries if e.get("decision") == "launched"
             and e.get("in_job")]
    assert len(outer) == 1, entries
    assert len(inner) == len(numbers), inner
    assert all(e["in_sequence"] == names.stem for e in inner), inner
    notes = [s.note for s in Repo(str(road["whole"])).states()]
    for n, stages in road["groups"]:
        said = (f"· before launch task group {n} ({', '.join(stages)}),"
                f" in job {outer[0]['job']} (sequence {names.stem})")
        assert sum(note.endswith(said) for note in notes) == 1, (said, notes)


def test_a_group_the_job_holds_goes_in_it(road):
    """While the job runs, a group it has not launched yet is the job's:
    `status` says *goes in job N*, and launching it alone is refused,
    naming the stop (R-L45).  THE LAST GROUP IS NOT REACHED when these are
    read, by construction: it is launched only once the first group's
    SIESTA run has ended, while both reads follow the background launch at
    once, in this process."""
    assert not road["said"]["reached_last"], (
        "the job launched the last group before these reads were taken, so "
        "they read no held group")
    held = road["said"]["held_status"]
    assert held.exit_code == 0, held.output
    last = road["groups"][-1][0]
    assert f"group {last}" in held.output and "goes in job" in held.output, \
        held.output
    alone = road["said"]["alone"]
    assert alone.exit_code != 0, alone.output
    assert "goes in job" in alone.output and "jobs stop" in alone.output, \
        alone.output


@pytest.mark.parametrize("verb,says", [
    # a bare launch offers no group the job holds, and says where each goes
    ("bare", ["goes in job", "molbuilder jobs status"]),
    # R-P23: a group the job holds is not prepared again under it
    ("prep_held", ["goes in job", "jobs stop"]),
])
def test_what_the_job_holds_is_not_taken_from_it(road, verb, says):
    """While the job runs, its groups are its own: nothing else launches
    them, and no prep rewrites the files it is about to launch
    (`job-system.md` § 6.0, R-L45, R-P23)."""
    assert not road["said"]["reached_last"], (
        "the job launched the last group before these reads were taken")
    r = road["said"][verb]
    assert r.exit_code != 0, r.output
    assert all(w in r.output for w in says), r.output


def test_jobs_status_lists_the_sequence_and_a_stop_ends_it(road):
    """`molbuilder jobs status` names the job a sequence of its groups, with
    the groups not reached yet (`job-system.md` § 6.0a); `jobs stop` ends
    the walk -- SIGTERM to its process group, which reaches the engine --
    and "the run then reads *failed — stopped before its end* by its own
    files" (§ 6.0a, `jobs stop`; `running-a-job.md` § 4.2); the group it
    never reached is left as it was (§ 6.0, *A sequence of groups*: "a
    sequence that ended before a group leaves that group as it was") --
    prepared, never launched: *pending*, no launch record -- its own again,
    not the job's.

    THE SILENT FAILURES (2.61.0: a sequence stopped, untested): the run the
    job was running when stopped reading *running* for ever, or a group the
    job never reached reading as launched, or as still the job's."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.runstatus import jobset_status
    jobs = road["said"]["jobs"].output
    assert "a sequence of groups" in jobs, jobs
    assert "not launched yet" in jobs, jobs
    assert road["said"]["stop"].exit_code == 0, road["said"]["stop"].output
    assert road["said"]["stopped_going"] is False
    after = road["said"]["after_stop"].output
    assert "goes in job" not in after, after
    assert not road["said"]["reached_last"], (
        "the job launched the last group before it was stopped")
    rows = {s.name: s for s in
            jobset_status(open_calculation(road["left"])).stages}
    for st in road["groups"][0][1]:
        assert rows[st].state == "failed" and rows[st].detail.startswith(
            "stopped before its end"), (st, rows[st].state, rows[st].detail)
    for st in road["groups"][-1][1]:
        assert (rows[st].state, rows[st].launch) == ("pending", None), (
            st, rows[st].state, rows[st].detail)
