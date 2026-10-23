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
is submitted: the record's submit command is declared as a gate, as `jobset
probe --set scheduler.submit=PATH` declares a site's wrapper
(`configuration.md` M-2), which passes the kind's ask to the probed command
and refuses every other call, so a defect that sent a job would fail here
instead of queueing it.

Run ON the target, with its record (`molbuilder jobset probe --write`):

    MOLBUILDER_FIELD_RECORD=<the record> python tools/testrun.py run field
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from molbuilder.scheduler.record import read_environment

RECORD = Path(os.environ.get("MOLBUILDER_FIELD_RECORD", "")).expanduser()

#: Each kind's ask, as the contract states it (`scheduler.md` § 6a) -- the
#: one word the gate below lets reach the real scheduler; never the kinds
#: file's, which is what this tests.
_ASKS = {"slurm": "--test-only"}


def test_ask_reads_the_targets_answer_and_submits_nothing(tmp_path,
                                                          monkeypatch):
    env = read_environment(RECORD)
    assert env is not None, f"{RECORD} does not read as a machine record"
    if not env.kind.queue or not env.domains:
        pytest.skip("a machine with no scheduler has none to ask")
    ask = _ASKS[env.scheduler.kind]
    real = env.scheduler.submit
    assert real and Path(real).is_file(), (
        f"the record's submit command, {real}, is not here: run this on the "
        f"machine {RECORD.name} describes")

    # ONLY A QUESTION GETS THROUGH, and every call is written down.
    bin_dir = tmp_path / "ask-only"
    bin_dir.mkdir()
    calls = tmp_path / "submit-calls.log"
    gate = bin_dir / "submit-gate"
    gate.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{calls}"\n'
        'case " $* " in\n'
        f'  *" {ask} "*) exec "{real}" "$@" ;;\n'
        "esac\n"
        'echo "the field test asks; it never submits" >&2\n'
        "exit 97\n")
    gate.chmod(0o755)

    # THIS MACHINE IS THE TARGET: its record, as the probe wrote it -- its
    # submit command declared as the gate.
    import dataclasses
    from molbuilder.scheduler import machine_scope_path, write_environment
    write_environment(dataclasses.replace(
        env, scheduler=dataclasses.replace(env.scheduler, submit=str(gate))),
        machine_scope_path())

    # A held H2, its wall and memory stated, sent to the record's first queue.
    from support.road import describe_calculation, jobset
    queue = env.domains[0].name
    bundle = describe_calculation(tmp_path, monkeypatch)
    from support.road import read_request, write_request
    req = read_request(bundle)
    req["allocation"] = {"domain": queue, "time": "0-00:05:00", "mem": "1G"}
    write_request(bundle, req)
    r = jobset("prep", "task", "--group", "1", "--bundle", bundle,
               "--target", "this")
    assert r.exit_code == 0, r.output

    r = jobset("launch", "task", "--stage", "coarse", "--bundle", bundle,
               "--mode", "ask")
    assert r.exit_code == 0, r.output
    out = r.output
    asked = calls.read_text().splitlines() if calls.is_file() else []
    assert len(asked) == 1 and ask in asked[0].split(), asked
    assert "nothing was submitted" in out, out
    assert not (bundle / "01_coarse" / "run-0" / "run.json").exists(), (
        "asking recorded a launch")
    # THE ANSWER READ AS A TIME -- or the scheduler's own words, shown whole
    # so the person judges: a queue that cannot take five minutes on two
    # cores, or an answer molbuilder no longer reads.
    assert "no prediction" not in out, (
        f"the scheduler's answer was not read as a start time:\n{out}")
