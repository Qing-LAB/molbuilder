"""FIELD TEST -- the run door asks the scheduler about a sent run where this
machine sent its job, and writes down a job it no longer lists
(`docs/process/testing.md` § 0, tier 2).

Goal: a sent run whose job the scheduler has forgotten reads `failed` --
*the scheduler no longer listed job N (asked on <its cluster> at …)* -- the
first time its state is asked on the machine that sent it, by the one rule
`molbuilder jobs` asks by (`joblist.address_of`, `asked_here`), and the
answer is written into the run's launch record; a run no address on this
machine names is left as its folder says.  Contract: `execution/
running-a-job.md` § 4.2, *Asked once, written once*; `job-system.md`
§ 6.0a.  Nothing is submitted: the launch record and the address are
written by the product's own writers (`runrecord.write_launch`,
`joblist.queue_job`) for an id the scheduler has forgotten, and what the
scheduler answers is read here, where it answers (user, 2026-10-06).

PREVENTS: Sol's job 66232804, cancelled, reading *queued* on the Results
tab for ever (2026-10-22) -- the check asked whether this server's record
named the machine the calculation's copy names, and a probed record names
none.

Run ON the target, with its record (`molbuilder jobset probe --write`):

    MOLBUILDER_FIELD_RECORD=<the record> python tools/testrun.py run field
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from molbuilder.scheduler.record import read_environment

RECORD = Path(os.environ.get("MOLBUILDER_FIELD_RECORD", "")).expanduser()

#: An id no scheduler holds -- one it has forgotten, as
#: `test_the_queue_says_why.py` asks it.
_FORGOTTEN = "999999999"


@pytest.fixture
def sent(tmp_path, monkeypatch):
    """A flat H2 stage prepared on this machine, the target, its coarse run
    recorded as sent to the queue as job :data:`_FORGOTTEN`, and the job's
    address kept as the one sender keeps it."""
    env = read_environment(RECORD)
    assert env is not None, f"{RECORD} does not read as a machine record"
    if not env.kind.queue:
        pytest.skip("a machine with no scheduler holds no job")
    assert env.scheduler.query and Path(env.scheduler.query).is_file(), (
        f"the record's query command, {env.scheduler.query}, is not here: "
        f"run this on the machine {RECORD.name} describes")
    # THIS MACHINE IS THE TARGET: its record, as the probe wrote it.
    from molbuilder.scheduler import machine_scope_path, write_environment
    write_environment(env, machine_scope_path())
    from support.road import describe_calculation, group_words, jobset
    if not env.domains:
        pytest.skip("a record that names no queue sends no job")
    flat = describe_calculation(tmp_path, monkeypatch, engine="siesta",
                                calculation="optimization", shape="flat",
                                name="H2sent")
    # THE QUEUE, WALL AND MEMORY its header carries, stated in its request
    # (`running-a-job.md` § 5.3.1: stated nowhere, prep refuses).
    from support.road import read_request, write_request
    req = read_request(flat)
    req["allocation"] = {"domain": env.domains[0].name,
                         "time": "0-00:05:00", "mem": "1G"}
    write_request(flat, req)
    r = jobset("prep", "task", *group_words(flat, "coarse"),
               "--bundle", flat, "--target", "this")
    assert r.exit_code == 0, r.output
    from molbuilder import joblist
    from molbuilder.runfiles import RunNames
    from molbuilder.runrecord import RunRef, launch_record_path, write_launch
    from molbuilder.scheduler.queue import cluster
    names = RunNames.of("H2sent", "01_coarse", "flat")
    write_launch(flat, names=names, run=0, mode="submit",
                 command=[env.scheduler.submit or "sbatch", "x.sbatch"],
                 job_id=_FORGOTTEN,
                 placed_on={"domain": None, "partition": None, "qos": None,
                            "time": "0-00:05:00", "mem": None},
                 scheduler={"kind": env.scheduler.kind, "stderr": None})
    member = {"run": str(RunRef.named(names.run_name(0)) if names.shared
                         else RunRef.at(flat, flat)),
              "launch_record": str(launch_record_path(flat, names, 0)
                                   .relative_to(flat))}
    joblist.write(joblist.queue_job(
        job_id=_FORGOTTEN, cluster=cluster(env.scheduler),
        calculation=flat, members={"coarse": member}))
    return flat, names, env


def test_a_forgotten_job_is_written_down_where_this_machine_sent_it(sent):
    from molbuilder.runrecord import launch_record
    from molbuilder.runs import concluded
    from molbuilder.scheduler.queue import cluster
    flat, names, env = sent
    st = concluded(flat, names, base=flat, run=0)
    assert st.state == "failed", st
    assert (f"the scheduler no longer listed job {_FORGOTTEN}"
            in st.detail), st.detail
    rec = launch_record(flat, names, 0)["ended"]
    assert rec["listed"] is False and rec["said"] == "", rec
    assert rec["asked_on"] == cluster(env.scheduler), rec


def test_a_job_no_address_here_names_is_not_asked(sent):
    from molbuilder import joblist
    from molbuilder.runrecord import launch_record
    from molbuilder.runs import concluded
    flat, names, _env = sent
    joblist.forget(joblist.find(_FORGOTTEN))
    st = concluded(flat, names, base=flat, run=0)
    assert (st.state, st.detail) == ("queued", f"queued as job {_FORGOTTEN}")
    assert launch_record(flat, names, 0)["ended"] is None
