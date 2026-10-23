"""The jobs list holds both clusters' jobs of one number
(`execution/job-system.md` § 6.0a: "a home folder two clusters share holds
both clusters' jobs of one number" -- the user, 2026-10-22, whose home Sol
shares with another cluster).

PREVENTS: the second cluster's address written over the first's, so a
sent run of the first reads *queued* for ever -- its scheduler never asked
again (`running-a-job.md` § 4.2) -- and `molbuilder jobs` loses the job.
The addresses are written by the one sender's own constructor and writer
(`joblist.queue_job`, `write`); nothing is asked of a scheduler.
"""
from __future__ import annotations

import pytest


def _sent(tmp_path, cluster: str, name: str):
    from molbuilder import joblist
    calc = tmp_path / name
    calc.mkdir()
    job = joblist.queue_job(
        job_id="4242", cluster=cluster, calculation=calc,
        members={"coarse": {"run": "01_coarse/run-0",
                            "launch_record": "01_coarse/run-0/run.json"}})
    joblist.write(job)
    return calc


def test_two_clusters_jobs_of_one_number_are_both_kept(tmp_path):
    """Each is on the list, each found by its calculation, and forgetting
    one leaves the other."""
    from molbuilder import joblist
    a = _sent(tmp_path, "sol", "A")
    b = _sent(tmp_path, "phx", "B")
    on_file = [j for j in joblist.every() if isinstance(j, joblist.Job)]
    assert sorted((j.cluster, j.job_id) for j in on_file) == [
        ("phx", "4242"), ("sol", "4242")]
    assert joblist.address_of("4242", a).cluster == "sol"
    assert joblist.address_of("4242", b).cluster == "phx"
    joblist.forget(joblist.address_of("4242", a))
    assert joblist.address_of("4242", a) is None
    assert joblist.address_of("4242", b).cluster == "phx"


def test_a_number_two_clusters_hold_is_refused_naming_both(tmp_path):
    """`jobs stop 4242` on a machine whose scheduler answers for neither
    cluster: refused, naming both -- never the first file found."""
    from molbuilder import joblist
    _sent(tmp_path, "sol", "A")
    _sent(tmp_path, "phx", "B")
    with pytest.raises(joblist.JobFileError, match="clusters phx, sol"):
        joblist.find("4242")
