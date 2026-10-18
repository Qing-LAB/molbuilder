"""FIELD TEST -- `molbuilder jobs status` asks the target's own scheduler
about a job and says its state with the scheduler's reason for it
(`docs/process/testing.md` § 0, tier 2).

Goal: a job the scheduler took but holds says why, in the scheduler's own
words -- the scheduler decides, and its answer is shown (`execution/
scheduler.md` R6; `job-system.md` § 6.0a) -- and a job it has forgotten reads
as no longer listed.  Read here, where a scheduler prints it: what it prints
is never assumed (user, 2026-10-06).  Nothing is submitted: the job asked
about is one the scheduler already holds, whoever's.

Run ON the target, with its record (`molbuilder jobset probe --write`):

    MOLBUILDER_FIELD_RECORD=<the record> python tools/testrun.py run field
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from molbuilder.scheduler.queue import listed
from molbuilder.scheduler.record import read_environment

RECORD = Path(os.environ.get("MOLBUILDER_FIELD_RECORD", "")).expanduser()

#: Each kind's own words for *every job it lists in these states, with its
#: state and reason* -- the contract's (`scheduler.md` § 6a), never the
#: kinds file's, which is what this tests.
_LISTED = {"slurm": lambda states: ["--noheader", f"--states={states}",
                                    "--format=%i|%T|%r"]}
_WAITING = {"slurm": "PENDING"}
_ENDED = {"slurm": "COMPLETED,FAILED,TIMEOUT,CANCELLED,OUT_OF_MEMORY,"
                   "NODE_FAIL,PREEMPTED,BOOT_FAIL,DEADLINE"}


def _said(state, reason):
    """A state with the scheduler's reason for it, as `job-system.md`
    § 6.0a says it; a reason SLURM gives as ``None`` is none."""
    return state if reason in ("", "None") else f"{state} ({reason})"


def _rows(scheduler, states, job=None):
    """``[(id, state, reason)]`` of the jobs the scheduler lists in
    ``states`` -- whoever's, or ``job`` alone.  A REASON MAY HOLD SPACES
    ("ReqNodeNotAvail, Reserved for maintenance"): split on the bar only."""
    words = _LISTED[scheduler.kind](states) + ([f"--jobs={job}"] if job
                                               else [])
    out = subprocess.run([scheduler.query, *words], capture_output=True,
                         text=True, timeout=60).stdout.splitlines()
    return [tuple(x.strip() for x in r.split("|", 2))
            for r in out if r.strip()]


def _settled(scheduler, states, read):
    """``(the scheduler's own row, what ``read`` gave)`` for one job it
    lists in ``states`` -- its row asked just before and just after
    ``read`` and equal, so a job whose state or reason moved between is
    asked again (an array's range rows are not one job and are passed
    over); ``None`` when it lists none, or none held still."""
    for _try in range(3):
        rows = [r for r in _rows(scheduler, states)
                if "[" not in r[0] and "_" not in r[0]]
        if not rows:
            return None
        job = rows[0][0]
        before = _rows(scheduler, states, job)
        got = read(job)
        after = _rows(scheduler, states, job)
        if before and before == after:
            return before[0], got
    return None


@pytest.fixture(scope="module")
def scheduler():
    env = read_environment(RECORD)
    assert env is not None, f"{RECORD} does not read as a machine record"
    if not env.kind.queue:
        pytest.skip("a machine with no scheduler holds no job")
    assert env.scheduler.query and Path(env.scheduler.query).is_file(), (
        f"the record's query command, {env.scheduler.query}, is not here: "
        f"run this on the machine {RECORD.name} describes")
    return env.scheduler


def test_a_job_the_scheduler_holds_says_its_state_and_why(scheduler):
    seen = _settled(scheduler, _WAITING[scheduler.kind],
                    lambda job: listed(scheduler, job))
    if seen is None:
        pytest.skip("the scheduler holds no waiting job that held still")
    (job, state, reason), got = seen
    assert got == (True, _said(state, reason)), job


def test_a_job_the_scheduler_lists_ended_says_how_it_ended(scheduler):
    seen = _settled(scheduler, _ENDED[scheduler.kind],
                    lambda job: listed(scheduler, job))
    if seen is None:
        pytest.skip("the scheduler lists no ended job that held still")
    (job, state, reason), got = seen
    assert got == (False, _said(state, reason)), job


def test_a_job_the_scheduler_has_forgotten_is_no_longer_listed(scheduler):
    assert listed(scheduler, "999999999") == (False, "")
