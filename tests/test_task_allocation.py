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
    assert t.allocation == Allocation(domain="htc", time="4h", mem="128G")
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


def test_the_values_stay_AS_A_PERSON_TYPES_THEM():
    """`"7-00:00:00"` is SLURM's own spelling and `"0.5T"` is the CLI's --
    the file holds what you would type, so one parser serves both surfaces
    and nothing is converted twice."""
    t = _task(allocation={"time": "7-00:00:00", "mem": "0.5T"})
    assert t.to_dict()["allocation"] == {"time": "7-00:00:00", "mem": "0.5T"}


def test_an_unknown_key_is_refused_not_ignored():
    with pytest.raises(Exception) as e:
        _task(allocation={"domain": "htc", "cores": 48})
    assert "cores" in str(e.value)


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
    assert (got.domain, got.time, got.mem) == ("htc", "2:00:00", "200GB")


def test_a_stated_flag_wins():
    """A person typing `--mem` now is answering about now."""
    from molbuilder.jobset.model import Resources
    from molbuilder.jobset.prep import _under_description
    got = _under_description(Resources(mem="64G"),
                             Allocation(time="2:00:00", mem="200GB"))
    assert got.mem == "64G" and got.time == "2:00:00"


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
    assert (got.domain, got.time, got.mem) == ("htc", "2:00:00", "200GB")


def test_no_declaration_leaves_the_flags_alone():
    from molbuilder.jobset.model import Resources
    from molbuilder.jobset.prep import _under_description
    got = _under_description(Resources(mpi_np=4), Allocation())
    assert got.mpi_np == 4 and got.mem is None and got.time is None
