"""A flat stage records its launch, as an attempt does (`project-layout.md`
§ 1.6.3).

Every stage of a flat calculation shares one directory, so there is no
attempt to hold a ``run.json``: the stage's own is ``<basename>.run.json``,
beside its deck.  Until 2026-09-27 a flat stage recorded nothing, so a stage
sent to the queue read ``pending -- prepped, not launched`` until its first
line of output -- in the ladder, and in the folder's own status.

Driven through ``jobset init --shape flat`` -> ``prep run`` -> ``launch run
--mode submit``, on a machine whose record names a queue; the scheduler is a
stub on PATH that queues nothing and answers with a job id.  Nothing runs,
so ``queued`` is what there is to see.
"""
from __future__ import annotations

import os

import pytest


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


@pytest.fixture
def submitted(tmp_path, monkeypatch):
    """A flat stage, prepped and submitted to a queue that answers."""
    from conftest import write_machine_record
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.scheduler import Domain

    write_machine_record(scheduler="slurm", domains=[
        Domain(name="htc", partition="htc", qos="public",
               max_time="0-04:00:00")])
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    calls = tmp_path / "sbatch-calls.log"
    (bin_ / "sbatch").write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{calls}"\n'
        'echo "Submitted batch job 4242"\n')
    (bin_ / "sbatch").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")

    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "P" / "structure" / "h2.xyz").write_text(
        "2\nh2\nH 0 0 0\nH 0 0 0.74\n")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tree.parent)
    r = _jobset("init", "--structure", "P/structure/h2.xyz",
                "--bundle", "P/optimization/H2", "--engine", "pyscf",
                "--shape", "flat", "--name", "H2")
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "optimization" / "H2"
    (bundle / ".molbuilder.json").write_text(
        '{"script_generation": {"activation": "conda activate", '
        '"preamble": "true"}}')
    r = _jobset("prep", "run", "coarse", "--bundle", bundle,
                "--target", "this")
    assert r.exit_code == 0, r.output
    r = _jobset("launch", "run", "coarse", "--bundle", bundle,
                "--mode", "submit", "--domain", "htc", "--yes")
    assert r.exit_code == 0, r.output
    assert calls.read_text().strip(), "nothing was submitted"
    return bundle


def test_a_submitted_flat_stage_records_its_launch_and_reads_queued(
        submitted):
    """The stage's own record, beside its deck -- and the ladder and the
    folder both read it: ``queued as job 4242``, not ``pending``.

    MUTATION THIS MUST FAIL AGAINST: submit writing a stage with no
    attempt anywhere but its own record (`submit._where_recorded`), or the
    ladder reading an attempt's record only -- the code before 2026-09-27.
    """
    from molbuilder.jobset.materialize import launch_record_path, read_run_launch
    from molbuilder.jobset.model import FILENAME, JobSet
    from molbuilder.jobset.runstatus import jobset_status
    from molbuilder.parse import parse_dir

    deck = next(submitted.glob("H2_01_coarse.*py"))
    record = launch_record_path(submitted, deck.stem)
    assert record.name == "H2_01_coarse.run.json" and record.is_file(), (
        sorted(p.name for p in submitted.iterdir()))
    assert read_run_launch(submitted, basename=deck.stem)["job_id"] == "4242"

    row = next(s for s in jobset_status(
        JobSet.load(submitted / FILENAME), submitted).stages
        if s.name == "coarse")
    assert (row.state, row.detail) == ("queued", "queued as job 4242"), row

    status = parse_dir(submitted).status
    assert (status["state"], status["detail"]) == ("queued",
                                                   "queued as job 4242"), status
