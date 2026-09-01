"""`task.json` carries what this calculation asks the scheduler for.

**Why the key exists.** Five Sol jobs (62039301-05, 2026-08-23) died against
limits nobody chose: a per-GPU memory default, and a wall the framework
invented.  The two numbers had no home that travelled with the calculation,
so the Task-setup tab could not set them and `prep` could not learn them --
every launch had to be told again, and a launch that forgot was a launch
that guessed.

`allocation` is that home (`task.Allocation`): the queue, the wall, the
memory, spelled as a person types them.
"""
from __future__ import annotations

import json

import pytest

from molbuilder.task import SCHEMA, Allocation, Task

_BASE = {
    "schema": SCHEMA,
    "engine": {"name": "siesta"},
    "shape": "hierarchical",
    "run": {"name": "JOB", "id": "JOB_H2"},
    "structure": {"source": "h2.xyz", "formula": "H2", "atoms": 2},
    "varies": [],
    "stages": [{"name": "coarse", "enabled": True, "overrides": {}}],
}


def _task(**extra) -> Task:
    return Task.from_dict({**_BASE, **extra})


# --------------------------------------------------------------------- #
#  the shape on disk                                                     #
# --------------------------------------------------------------------- #

def test_the_three_asks_round_trip():
    t = _task(allocation={"domain": "htc", "time": "4h", "mem": "128G"})
    # "4h" is what a person types; the record holds SLURM's spelling,
    # normalised on the way in (task.py::Allocation, 2026-08-24).
    assert t.allocation == Allocation(
        domain="htc", time="0-04:00:00", mem="128G")
    assert Task.from_dict(t.to_dict()).allocation == t.allocation


def test_absent_is_a_state_and_writes_no_key():
    """A description that asks for nothing must round-trip BYTE-identical --
    otherwise every file written before 2026-08-24 changes on first save,
    and "unstated" acquires a second spelling on disk (submission.md S1)."""
    t = _task()
    assert not t.allocation
    assert "allocation" not in t.to_dict()


def test_a_partial_ask_writes_only_what_was_said():
    """Stating the queue is not stating the memory.  Each field is absent on
    its own, so an unstated one cannot be read back as a chosen empty."""
    t = _task(allocation={"domain": "htc"})
    assert t.to_dict()["allocation"] == {"domain": "htc"}


def test_the_record_holds_ONE_SPELLING_AND_IT_IS_SLURMS():
    """Replaces `test_the_values_stay_AS_A_PERSON_TYPES_THEM` (2026-08-24,
    user: *"your record should set unified time format while it is the UI
    that can do some translation for human readability/input"*).

    The old rule let the file hold whichever spelling arrived.  Nothing
    could then read it correctly, because the two vocabularies DISAGREE --
    `04:30` is four minutes thirty to SLURM and four and a half hours to a
    person -- and `prep` copied the browser's `"4h"` straight into
    `Resources.time`, which `sbatch` refused as `-t 4h`.

    A person still types `"0.5T"`; it is normalised at the door."""
    t = _task(allocation={"time": "7-00:00:00", "mem": "0.5T"})
    assert t.to_dict()["allocation"] == {"time": "7-00:00:00", "mem": "512G"}


def test_a_human_spelling_is_accepted_at_the_door_and_normalised():
    """A hand-edited file is a human edge too, so the reader takes what a
    person would type -- and returns the record's spelling."""
    t = _task(allocation={"time": "4h", "mem": "80GB"})
    assert t.to_dict()["allocation"] == {"time": "0-04:00:00", "mem": "80G"}


def test_a_machine_answered_value_is_read_as_one():
    """§ 6.8a, extended 2026-09-01: the block carries what the RUN should
    use, beside what it asks the scheduler for.  One value each."""
    t = _task(allocation={"domain": "htc", "mpi_np": 8, "use_gpu": True})
    assert t.allocation.values == {"mpi_np": 8, "use_gpu": True}
    assert t.to_dict()["allocation"] == {
        "domain": "htc", "mpi_np": 8, "use_gpu": True}


def test_a_list_is_refused_with_the_difference_named():
    """Several points is a measurement and one value is a run.  A list here
    is `bench`'s shape arriving in the wrong block, and the refusal says
    which -- "invalid" would send a person to the wrong file."""
    with pytest.raises(Exception) as e:
        _task(allocation={"mpi_np": [4, 8]})
    assert "bench" in str(e.value) and "ONE value" in str(e.value)


def test_an_unknown_key_is_refused_by_VALIDATION_not_the_reader():
    """**The refusal moved; it did not go away.**

    The reader rejected any key outside `domain`/`time`/`mem` until
    2026-09-01. The machine-answered values made that impossible -- an
    unknown key is now indistinguishable, at L1, from a value the catalogue
    declares. So the check moved to where the vocabulary is in hand, which
    is the same split `bench` has always made: shape in `task.py`,
    membership in `validation/task.py`.

    A typo carried silently would be the worst outcome: a number nobody
    applies, in a file that looks configured.
    """
    from molbuilder.validation.task import preflight
    t = _task(allocation={"domain": "htc", "cores": 48})
    assert t.allocation.values == {"cores": 48}          # shape: fine
    issues = [i for i in preflight(t) if i.severity == "error"]
    assert any("cores" in i.message for i in issues), (
        f"an unknown allocation key passed validation: {issues}")


def test_a_number_is_refused_with_the_spelling_it_wants():
    """`{"mem": 128}` is ambiguous -- 128 what? -- so it is refused rather
    than guessed at."""
    with pytest.raises(Exception) as e:
        _task(allocation={"mem": 128})
    assert "mem" in str(e.value)


def test_an_empty_object_is_refused_as_a_second_spelling_of_absent():
    with pytest.raises(Exception) as e:
        _task(allocation={})
    assert "allocation" in str(e.value)


def test_a_queue_this_machine_never_heard_of_is_ACCEPTED():
    """Shape only.  A description written for Sol is opened on a laptop, and
    refusing it there for naming `htc` would refuse a file that is perfectly
    correct where it is going."""
    assert _task(allocation={"domain": "no-such-queue"}).allocation.domain \
        == "no-such-queue"


# --------------------------------------------------------------------- #
#  what prep does with it -- FIELD by field                              #
# --------------------------------------------------------------------- #

def test_the_description_supplies_what_no_flag_states():
    from molbuilder.jobset.prep import _under_description
    got = _under_description(None, Allocation(domain="htc", time="2:00:00",
                                              mem="200GB"))
    assert (got.domain, got.time, got.mem) == (
        "htc", "0-02:00:00", "200G")


def test_a_stated_flag_wins():
    """A person typing `--mem` now is answering about now."""
    from molbuilder.jobset.model import Resources
    from molbuilder.jobset.prep import _under_description
    got = _under_description(Resources(mem="64G"),
                             Allocation(time="2:00:00", mem="200GB"))
    assert got.mem == "64G" and got.time == "0-02:00:00"


def test_an_UNRELATED_flag_erases_nothing():
    """**The reason precedence is per-field.**  Whole-object precedence would
    make `--np 8` silently drop a memory ask nobody mentioned -- the exact
    class of loss this whole round of work was about."""
    from molbuilder.jobset.model import Resources
    from molbuilder.jobset.prep import _under_description
    got = _under_description(Resources(mpi_np=8),
                             Allocation(domain="htc", time="2:00:00",
                                        mem="200GB"))
    assert got.mpi_np == 8
    assert (got.domain, got.time, got.mem) == (
        "htc", "0-02:00:00", "200G")


def test_no_declaration_leaves_the_flags_alone():
    from molbuilder.jobset.model import Resources
    from molbuilder.jobset.prep import _under_description
    got = _under_description(Resources(mpi_np=4), Allocation())
    assert got.mpi_np == 4 and got.mem is None and got.time is None


# --------------------------------------------------------------------- #
#  Per stage, and the bench's own — §§ 6.8b, 6.8c                        #
# --------------------------------------------------------------------- #

def test_a_rung_overrides_field_by_field_over_the_flat_block():
    """§ 6.8b: a stage states only what it differs in.  Whole-object
    precedence would let a rung that wanted a longer wall silently drop the
    queue nobody restated -- the loss § 6.8a's rule already ends between a
    flag and the block."""
    t = _task(allocation={"domain": "htc", "time": "0-04:00:00",
                          "mem": "128G", "mpi_np": 8},
              stage_allocation={"tight": {"time": "2-00:00:00",
                                          "mpi_np": 16}})
    merged = t.stage_allocation["tight"].merged_over(t.allocation)
    assert merged.domain == "htc"                     # inherited
    assert merged.mem == "128G"                       # inherited
    assert merged.time == "2-00:00:00"                # overridden
    assert merged.values == {"mpi_np": 16}            # overridden


def test_the_bench_may_ask_for_a_different_wall():
    """§ 6.8c: a benchmark is short by construction and a run is not.  One
    wall serving both queues a thirty-second job behind a two-day
    reservation, or kills the calculation."""
    t = _task(allocation={"domain": "htc", "time": "2-00:00:00"},
              bench_allocation={"domain": "general", "time": "0-00:30:00"})
    assert t.allocation.time == "2-00:00:00"
    assert t.bench_allocation.time == "0-00:30:00"
    assert t.bench_allocation.domain == "general"
    assert Task.from_dict(t.to_dict()).bench_allocation == t.bench_allocation


def test_absent_means_use_the_runs_and_writes_no_key():
    """Absent-is-a-state, everywhere here.  A description written before
    2026-09-01 says exactly what it always said, byte for byte."""
    t = _task(allocation={"domain": "htc"})
    assert not t.bench_allocation and not t.stage_allocation
    d = t.to_dict()
    assert "bench_allocation" not in d and "stage_allocation" not in d


def test_the_three_blocks_round_trip_together():
    t = _task(allocation={"domain": "htc", "time": "1-00:00:00", "mpi_np": 8},
              bench_allocation={"time": "0-00:30:00"},
              stage_allocation={"tight": {"mpi_np": 16}})
    back = Task.from_dict(t.to_dict())
    assert back.allocation == t.allocation
    assert back.bench_allocation == t.bench_allocation
    assert back.stage_allocation == t.stage_allocation
