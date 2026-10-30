"""A sequence of groups -- one job launching each group in turn, each when
the one before has finished (`execution/job-system.md` § 6.0, *A sequence of
groups*) -- down the road with the real SIESTA.

``molbuilder describe`` (the shipped `publishable` ladder: three
stages, each building on the one before, groups 1 2 3) -> ``prep task
--group 1 --group 2 --group 3 --one-job`` -> ``launch task --group 1
--group 2 --group 3 --mode direct``: ONE walk, run here, launching each
group as its own launch would, each run recording the job it rode.

THE SILENT FAILURES IT GUARDS: a group launched before the one it builds on
finished (taking an older run, or nothing); a run inside the job recorded as
a run of its own (every reader then asks no job about it, and a job killed
leaves it reading running); a group the job still holds launched a second
time by hand.  Reading the code is not enough: the walk runs each group's
launch as a separate process, from molbuilder's own folder, on the
calculation named by its full path -- only a run shows those words reach.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from _road import conda_hook, env_available, live_siesta
from support.road import (H2, jobset, molbuilder, read_request,
                          road_structure, write_request)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]


def _calculation(tree: Path, name: str) -> Path:
    """A three-stage H2 relaxation ladder, one rank a run -- an H2 needs no
    more -- described as `molbuilder describe` describes it."""
    from molbuilder.workingcopy_structure import StructureCodec
    (tree / "P" / "structure").mkdir(parents=True, exist_ok=True)
    pair = tree / "P" / "structure" / "h2.xyz"
    if not pair.is_file():
        StructureCodec().write(road_structure(H2), pair)
    r = molbuilder("describe", "--structure", "P/structure/h2.xyz",
               "--bundle", f"P/opt/{name}", "--engine", "siesta",
               "--calculation", "optimization", "--name", "H2", "--psml-lib", "pseudopotential",
               "--stage-strategy", "publishable")
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "opt" / name
    request = read_request(bundle)
    request["execution"] = {**request.get("execution", {}), "mpi_np": 1,
                            "omp_threads": 1}
    write_request(bundle, request)
    return bundle


def _groups(bundle: Path):
    """The job request's groups, each with its stages, in order."""
    from molbuilder.calculation import open as open_calculation
    request = open_calculation(bundle).request
    return [(n, request.members(n)) for n in request.numbers]


def _ledger(bundle: Path):
    """The calculation's ledger, line by line, as written."""
    from molbuilder.jobset.ledger import LEDGER_FILE
    log = bundle / LEDGER_FILE
    return [json.loads(x) for x in log.read_text().splitlines() if x.strip()]


@pytest.fixture(scope="module")
def road(isolated_projects_root_module, tmp_path_factory):
    """Two calculations: one sent as a sequence run here and waited on;
    one sent in the background, looked at while it runs, its group launched
    alone refused, and then stopped."""
    tree = isolated_projects_root_module
    said = {}
    with live_siesta(tree, tmp_path_factory):
        whole = _calculation(tree, "H2seq")
        groups = _groups(whole)
        words = [w for n, _ in groups for w in ("--group", str(n))]
        said["prep"] = jobset("prep", "task", *words, "--one-job",
                              "--bundle", whole, "--target", "this")
        assert said["prep"].exit_code == 0, said["prep"].output
        said["launch"] = jobset("launch", "task", *words, "--bundle", whole,
                                "--mode", "direct", "--foreground", "--yes")
        said["status"] = jobset("status", "--bundle", whole)

        left = _calculation(tree, "H2stop")
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
        from molbuilder.jobset.materialize import newest_launch
        opened = open_calculation(left)
        said["reached_last"] = any(
            newest_launch(opened, opened.jobset.job(st)) is not None
            for st in groups[-1][1])
        from molbuilder.joblist import Job, every, going_here
        job = next(j for j in every()
                   if isinstance(j, Job) and j.sequence is not None
                   and Path(j.calculation) == left.resolve())
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
    job's line, the walk run here -- read back through the run record the
    Results tab reads (`computation.launch.sequence`)."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.group import sequence_names
    from molbuilder.jobset.naming import stage_home
    from molbuilder.runrecord import launch_record
    calc = open_calculation(road["whole"])
    numbers = [n for n, _ in road["groups"]]
    name = sequence_names(road["whole"], calc.task, calc.request,
                          numbers).stem
    for _n, stages in road["groups"]:
        for st in stages:
            home = stage_home(road["whole"], calc.task, st)
            rec = launch_record(home.dir / "run-0")
            assert rec["sequence"] == name, (st, rec)
            assert rec["mode"] == "direct" and rec["job_id"] is None, rec
            assert rec["command"] == ["bash", f"launch/{name}.run.sh"], rec


def test_the_sequence_record_and_the_ledger_say_it(road):
    """The sequence's launch record states its groups and how it went, and
    the ledger keeps the one launch that sent it and each group's launch
    inside it, naming the job."""
    from molbuilder.calculation import open as open_calculation
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
    ("prep_held", ["goes in job", "jobs stop", "R-P23"]),
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
    the groups not reached yet; `jobs stop` ends the walk, and the groups it
    never reached are back to their own state -- not the job's."""
    jobs = road["said"]["jobs"].output
    assert "a sequence of groups" in jobs, jobs
    assert "not launched yet" in jobs, jobs
    assert road["said"]["stop"].exit_code == 0, road["said"]["stop"].output
    assert road["said"]["stopped_going"] is False
    after = road["said"]["after_stop"].output
    assert "goes in job" not in after, after
