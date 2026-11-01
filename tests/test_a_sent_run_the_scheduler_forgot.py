"""A sent run the scheduler has forgotten is written down once and reads
`failed` from then on, wherever its folder goes (`running-a-job.md` § 4.2,
*Asked once, written once*; `project-layout.md` § 1.6.3) -- and so is a run
here whose process went with no marker written; a start with no output yet
reads *running* (*The order of evidence*).

The road to a queue is refused in the basic suite (the suite's own `sbatch`,
`support.road.a_machine_with_queues`: no scheduler text is ever fabricated),
so the launch record is written here by the product's own writer
(`runrecord.write_launch`) with what a sent job's holds, and the scheduler's
answer by its (`runrecord.record_ended`) -- the two records the door reads;
what a scheduler answers is the field tier's (`tests/field/`).

PREVENTS: a run that died with nothing written (Sol's job 66108241,
2026-10-19: the disk filled under the engine, the wrapper and the monitor)
reading *running* for ever on a machine that copied its folder.
"""
from __future__ import annotations

import os

import pytest

from support.road import (jobset, killed_outright, left_in_the_background,
                          prepared_here, until)


@pytest.fixture
def sent(tmp_path, monkeypatch):
    """An H2 stage prepared (`support.road.prepared_here`), its coarse run
    recorded as sent to a queue (job 4242, a five-second wall) and not yet
    written to: ``(the calculation, the run's folder, the stage's
    names)``."""
    from molbuilder.runrecord import write_launch
    bundle, run, names = prepared_here(tmp_path, monkeypatch,
                                       "conda activate", name="H2sent")
    write_launch(run, mode="submit",
                 command=["sbatch", "-J", "H2/coarse", "x.sbatch"],
                 job_id="4242",
                 placed_on={"domain": "public", "partition": "public",
                            "qos": "public", "time": "0-00:00:05",
                            "mem": None},
                 scheduler={"kind": "slurm", "stderr": None})
    return bundle, run, names


def test_the_schedulers_answer_is_written_once_and_read_everywhere(sent):
    """The launch record holds `ended` null at launch; `record_ended` fills
    it with the machine asked, when, and the scheduler's words; the run
    door reads that record before anything else, so the run reads `failed`
    with those words -- a forgotten job, and one listed ended, each in its
    own sentence.

    MUTATION THIS MUST FAIL AGAINST: the door not reading `ended` (the run
    stays *queued as job 4242*); the writer dropping the scheduler's words
    or the machine asked."""
    from molbuilder.parse.dirs.rundir import run_state_of
    from molbuilder.runrecord import launch_record, record_ended
    _bundle, run, names = sent
    assert launch_record(run)["ended"] is None
    st = run_state_of(run, names)
    assert (st.state, st.detail) == ("queued", "queued as job 4242"), st

    record_ended(run, asked_on="sol", said="")
    rec = launch_record(run)["ended"]
    assert rec["asked_on"] == "sol" and rec["listed"] is False, rec
    assert rec["said"] == "" and rec["asked_at"].endswith("Z"), rec
    st = run_state_of(run, names)
    assert st.state == "failed", st
    # ITS START NEVER BEGAN -- no session log -- so it is said so, never
    # *stopped before its end* (`running-a-job.md` § 4.2).
    assert st.detail == (
        f"never started: the scheduler no longer listed job 4242 "
        f"(asked on sol at {rec['asked_at']})"), st.detail

    record_ended(run, asked_on="sol", said="FAILED (NonZeroExitCode)")
    st = run_state_of(run, names)
    assert st.state == "failed" and st.detail.startswith(
        "the scheduler listed job 4242 ended: FAILED (NonZeroExitCode) "
        "(asked on sol at "), st.detail


def test_the_door_asks_nothing_where_the_scheduler_cannot_be_asked(sent):
    """`runs.concluded` on a machine with no scheduler (this one, the
    suite's workstation record) changes nothing: the record stays null and
    the run reads as its folder says.  The asking itself is the field
    tier's."""
    from molbuilder.runrecord import launch_record
    from molbuilder.runs import concluded
    bundle, run, names = sent
    st = concluded(run, names, base=bundle)
    assert (st.state, st.detail) == ("queued", "queued as job 4242"), st
    assert launch_record(run)["ended"] is None


def test_a_run_here_waited_on_whose_process_went_unrecorded(tmp_path,
                                                             monkeypatch):
    """A run here, waited on, killed outright before its engine -- no
    shell can catch SIGKILL, so no marker and no closing record are
    written: its launch reads its end through the run door while the job's
    address is still on file, the process gone is written once into the
    launch record (`ended`, asked on this machine, nothing said), the
    launch says *failed* in those words, and the address is removed once
    said (`job-system.md` § 6.0a).  THE SILENT FAILURE: the address removed
    before the read, and a run that died reading *running* for ever."""
    from molbuilder.joblist import every, this_host
    from molbuilder.runrecord import launch_record
    bundle, run, names = prepared_here(tmp_path, monkeypatch, "kill -KILL $$;")
    r = jobset("launch", "task", "--stage", "coarse", "--bundle", bundle,
               "--mode", "direct", "--foreground", "--yes")
    assert r.exit_code == 1, r.output
    said = (f"stopped before its end: its process on {this_host()} ended "
            f"without recording how")
    assert said in r.output, r.output
    ended = launch_record(run)["ended"]
    assert (ended["asked_on"], ended["said"]) == (this_host(), ""), ended
    assert not (run / names.name(".concluded")).exists()
    assert every() == []


def test_a_run_here_whose_address_cannot_be_kept_says_so(tmp_path,
                                                         monkeypatch):
    """A run here, waited on, whose address cannot be written -- this
    machine's jobs folder not writable, as a full disk leaves it -- runs
    all the same, and its launch's line says so beside how the run ended:
    the launch's own words kept when its run's are added (`job-system.md`
    § 6.0a).  THE SILENT FAILURE: a run `molbuilder jobs` will never list,
    and nothing said."""
    from molbuilder.config_dir import ensure_private_dir, jobs_dir
    bundle, _run, _names = prepared_here(tmp_path, monkeypatch, "true")
    folder = ensure_private_dir(jobs_dir())
    folder.chmod(0o500)
    try:
        r = jobset("launch", "task", "--stage", "coarse", "--bundle", bundle,
                   "--mode", "direct", "--foreground", "--yes")
    finally:
        folder.chmod(0o700)
    assert r.exit_code == 1, r.output
    (line,) = [ln for ln in r.output.splitlines() if "FAILED" in ln]
    assert "no env python" in line, line
    assert "its address could not be kept" in line, line


def test_a_run_here_stopped_at_its_terminal(tmp_path, monkeypatch):
    """A run here, waited on, stopped at its terminal -- Ctrl-C reaching
    the launch and its run script together, as a terminal sends it to its
    whole process group: the launch waits for the run script's own close,
    then reads the run once while its address is on file -- the process
    gone with no marker is written once into its launch record -- and only
    then removes the address (`job-system.md` § 6.0a).  THE SILENT FAILURE:
    a stopped run's end never written, so it reads *running* for ever; or
    its address kept, so `molbuilder jobs` lists a run long gone.  (The
    race the wait closes -- a run script still stopping its monitor as the
    launch reads it -- is not reached here: this run stops in its
    activation, before its monitor starts, and closes at once.)  The road:
    the launch typed in a shell of its own, its start begun -- its session
    log -- before the stop."""
    import signal
    import subprocess
    import sys
    from pathlib import Path
    from molbuilder.joblist import every, this_host
    from molbuilder.runrecord import launch_record
    from molbuilder.runs import concluded
    bundle, run, names = prepared_here(tmp_path, monkeypatch, "sleep 60;")
    shell = subprocess.Popen(
        [sys.executable, "-m", "molbuilder", "jobset", "launch", "task",
         "--stage", "coarse", "--bundle", str(bundle), "--mode", "direct",
         "--foreground", "--yes"],
        cwd=Path(__file__).resolve().parents[1], start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        until(lambda: (run / names.name(".runwrap.log")).is_file(),
              within=30, what="its start")
        os.killpg(shell.pid, signal.SIGINT)
        out, _ = shell.communicate(timeout=60)
    finally:
        if shell.poll() is None:
            os.killpg(shell.pid, signal.SIGKILL)
            shell.wait()
    assert shell.returncode != 0, out
    ended = launch_record(run)["ended"]
    assert ended is not None and ended["asked_on"] == this_host(), out
    st = concluded(run, names, base=bundle)
    assert st.state == "failed" and st.detail.startswith(
        f"stopped before its end: its process on {this_host()} ended "
        f"without recording how"), st
    assert every() == [], out


def test_a_run_here_left_in_the_background(tmp_path, monkeypatch):
    """A run here left in the background, its start begun -- its session
    log -- and nothing of its output yet, reads *running* (*started, no
    output yet*), never *queued*; killed outright, the next look writes its
    end once and reads *failed*, its address kept until `molbuilder jobs`
    says it.  THE SILENT FAILURE: a started run reading *queued*, and one
    killed reading *running* for ever."""
    from molbuilder.joblist import every, this_host
    from molbuilder.runrecord import launch_record
    from molbuilder.runs import concluded
    bundle, run, names = prepared_here(tmp_path, monkeypatch, "sleep 60;")
    with left_in_the_background("task", "--stage", "coarse",
                                "--bundle", bundle) as job:
        until(lambda: (run / names.name(".runwrap.log")).is_file(),
              within=15, what="its start")
        st = concluded(run, names, base=bundle)
        assert (st.state, st.detail) == ("running",
                                         "started, no output yet"), st
        killed_outright(job)
    st = concluded(run, names, base=bundle)
    assert st.state == "failed", st
    assert st.detail.startswith(
        f"stopped before its end: its process on {this_host()} ended "
        f"without recording how (asked at "), st.detail
    assert launch_record(run)["ended"]["asked_on"] == this_host()
    assert [j.id for j in every()] == [job.id]


def test_a_stage_its_groups_walk_never_started(tmp_path, monkeypatch):
    """A group of two stages run here, its walk killed outright while its
    first stage runs: the first, started, reads *failed -- stopped before
    its end*; the second, which the walk never started, reads *failed* in
    words that say it never started -- its job's end written once into its
    record -- never that it *stopped before its end*, never *queued* for
    ever (`running-a-job.md` § 4.2).  THE SILENT FAILURE: a run that never
    ran read as one that ran and was stopped.  The road: `medium` set to
    start from its own deck (its `restart` override, `clean`, through Task
    setup's Save), so the two may share a group."""
    import json
    from molbuilder.jobset.materialize import latest_attempt
    from molbuilder.jobset.naming import stage_home
    from molbuilder.joblist import this_host
    from molbuilder.runs import concluded
    from molbuilder.task import of as task_of
    from support.road import read_request, save_both
    bundle, run, names = prepared_here(tmp_path, monkeypatch, "sleep 60;")
    task = json.loads((bundle / "task.json").read_text())
    task["varies"] = sorted({*task.get("varies", []), "restart"})
    for st in task["stages"]:
        if st["name"] == "medium":
            st["overrides"]["restart"] = "clean"
    request = read_request(bundle)
    request["stages"]["medium"]["group"] = request["stages"]["coarse"][
        "group"]
    saved = save_both(bundle, task, request)
    assert saved.status_code == 200, saved.get_json()
    r = jobset("prep", "task", "--group", "1", "--bundle", bundle,
               "--target", "this")
    assert r.exit_code == 0, r.output
    with left_in_the_background("task", "--bundle", bundle) as job:
        run = latest_attempt(stage_home(bundle, task_of(bundle),
                                        "coarse").dir)
        until(lambda: (run / names.name(".runwrap.log")).is_file(),
              within=15, what="the first stage's start")
        killed_outright(job)
    first = concluded(run, names, base=bundle)
    assert first.detail.startswith(f"stopped before its end: its process "
                                   f"on {this_host()} ended without "
                                   f"recording how"), first
    home = stage_home(bundle, task_of(bundle), "medium")
    st = concluded(latest_attempt(home.dir), home.names, base=bundle)
    assert st.state == "failed" and not st.started, st
    assert st.detail.startswith(
        f"never started: its process on {this_host()} ended (asked at "), st
