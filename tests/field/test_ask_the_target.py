"""FIELD TEST -- molbuilder asks the target's own scheduler about a line, and
reads its answer (`docs/process/testing.md` § 0, tier 2).

Goal: `jobset launch task --mode ask` on the machine a probed record describes:
the line asked about is the one `submit` would send, the real scheduler is
asked once with `--test-only`, its answer reads as a start time, and nothing is
submitted or recorded.  Contract: `execution/running-a-job.md` § 5.5 (*"no job
is created, nothing is recorded, and `status` sees nothing"*), `job-system.md`
§ 6.0 (the line asked about is the line that would be sent).

What a scheduler prints is read only here, where one prints it (user,
2026-10-06: *"there should be no assumption what so ever about the text
returned by slurm"*) -- the basic tier retired its stand-in answers.  Nothing
is submitted: the `sbatch` first on PATH passes a ``--test-only`` call to the
real one and refuses every other, so a defect that sent a job would fail here
instead of queueing it.

Run ON the target, with its record (`molbuilder jobset probe --write`):

    MOLBUILDER_FIELD_RECORD=<the record> python tools/testrun.py run field
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from molbuilder.scheduler.record import read_environment

RECORD = Path(os.environ.get("MOLBUILDER_FIELD_RECORD", "")).expanduser()


def _the_real_sbatch() -> str:
    """The target's own `sbatch`: found on PATH past the suite's refusing
    stand-ins (`conftest.the_suite_cannot_submit_a_job`, the road's)."""
    for d in os.environ.get("PATH", "").split(os.pathsep):
        f = Path(d) / "sbatch"
        if f.is_file() and os.access(f, os.X_OK) and \
                "the test suite must never run" not in f.read_text(
                    errors="replace")[:400]:
            return str(f)
    return ""


def test_ask_reads_the_targets_answer_and_submits_nothing(tmp_path,
                                                          monkeypatch):
    env = read_environment(RECORD)
    assert env is not None, f"{RECORD} does not read as a machine record"
    if env.scheduler != "slurm" or not env.domains:
        pytest.skip("a workstation has no scheduler to ask")
    real = _the_real_sbatch()
    assert real, ("no `sbatch` on PATH: run this on the machine "
                  f"{RECORD.name} describes")

    # THIS MACHINE IS THE TARGET: its record, as the probe wrote it.
    from molbuilder.scheduler import machine_scope_path
    shutil.copyfile(RECORD, machine_scope_path())

    # ONLY A QUESTION GETS THROUGH, and every call is written down.
    bin_dir = tmp_path / "ask-only"
    bin_dir.mkdir()
    calls = tmp_path / "sbatch-calls.log"
    gate = bin_dir / "sbatch"
    gate.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{calls}"\n'
        'case " $* " in\n'
        f'  *" --test-only "*) exec "{real}" "$@" ;;\n'
        "esac\n"
        'echo "the field test asks; it never submits" >&2\n'
        "exit 97\n")
    gate.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    # A held H2, its wall and memory stated, sent to the record's first queue.
    from support.road import describe_calculation, jobset
    queue = env.domains[0].name
    bundle = describe_calculation(tmp_path, monkeypatch)
    task = json.loads((bundle / "task.json").read_text())
    task["allocation"] = {"domain": queue, "time": "0-00:05:00", "mem": "1G"}
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
    r = jobset("prep", "task", "--stage", "coarse", "--bundle", bundle,
               "--target", "this")
    assert r.exit_code == 0, r.output

    r = jobset("launch", "task", "--stage", "coarse", "--bundle", bundle,
               "--mode", "ask", "--domain", queue)
    assert r.exit_code == 0, r.output
    out = r.output
    asked = calls.read_text().splitlines() if calls.is_file() else []
    assert len(asked) == 1 and "--test-only" in asked[0].split(), asked
    assert "nothing was submitted" in out, out
    assert not (bundle / "01_coarse" / "run-0" / "run.json").exists(), (
        "asking recorded a launch")
    # THE ANSWER READ AS A TIME -- or the scheduler's own words, shown whole
    # so the person judges: a queue that cannot take five minutes on two
    # cores, or an answer molbuilder no longer reads.
    assert "no prediction" not in out, (
        f"the scheduler's answer was not read as a start time:\n{out}")
