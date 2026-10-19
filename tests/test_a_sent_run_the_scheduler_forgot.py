"""A sent run the scheduler has forgotten is written down once and reads
`failed` from then on, wherever its folder goes (`running-a-job.md` § 4.2,
*Asked once, written once*; `project-layout.md` § 1.6.3).

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

import pytest

from support.road import describe_calculation, group_words, jobset


@pytest.fixture
def sent(tmp_path, monkeypatch):
    """A flat H2 stage prepared (`support.road.describe_calculation`), its
    coarse run recorded as sent to a queue (job 4242, a five-second wall)
    and not yet written to."""
    from molbuilder.runfiles import RunNames
    from molbuilder.runrecord import write_launch
    flat = describe_calculation(tmp_path, monkeypatch, engine="siesta",
                                calculation="optimization", shape="flat",
                                name="H2sent")
    r = jobset("prep", "task", *group_words(flat, "coarse"),
               "--bundle", flat, "--target", "this")
    assert r.exit_code == 0, r.output
    names = RunNames.of("H2sent", "01_coarse", "flat")
    write_launch(flat, names=names, run=0, mode="submit",
                 command=["sbatch", "-J", "H2/coarse", "x.sbatch"],
                 job_id="4242",
                 placed_on={"domain": "public", "partition": "public",
                            "qos": "public", "time": "0-00:00:05",
                            "mem": None},
                 scheduler={"kind": "slurm", "stderr": None})
    return flat, names


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
    flat, names = sent
    assert launch_record(flat, names, 0)["ended"] is None
    st = run_state_of(flat, names, 0)
    assert (st.state, st.detail) == ("queued", "queued as job 4242"), st

    record_ended(flat, names, 0, asked_on="sol", said="")
    rec = launch_record(flat, names, 0)["ended"]
    assert rec["asked_on"] == "sol" and rec["listed"] is False, rec
    assert rec["said"] == "" and rec["asked_at"].endswith("Z"), rec
    st = run_state_of(flat, names, 0)
    assert st.state == "failed", st
    assert st.detail == (
        f"stopped before its end: the scheduler no longer listed job 4242 "
        f"(asked on sol at {rec['asked_at']}); no exit recorded"), st.detail

    record_ended(flat, names, 0, asked_on="sol",
                 said="FAILED (NonZeroExitCode)")
    st = run_state_of(flat, names, 0)
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
    flat, names = sent
    st = concluded(flat, names, base=flat, run=0)
    assert (st.state, st.detail) == ("queued", "queued as job 4242"), st
    assert launch_record(flat, names, 0)["ended"] is None
