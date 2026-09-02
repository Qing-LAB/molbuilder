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


def test_a_launch_shape_here_is_sent_to_the_block_that_owns_it():
    """`generator.md` § 4.3a: ranks, threads and devices are a `bench` entry
    -- several points to measure, one point to use -- so the knob you might
    instead measure has ONE home.  This block is the SCHEDULER's ask.

    The refusal has to say that, because it is not a typo: `mpi_np` is a real
    setting in the wrong block, and "unknown key" alone sends a person
    looking for a spelling mistake they did not make.  (It IS what this block
    briefly held, on 2026-09-01.)"""
    for value in ([4, 8], 8):
        with pytest.raises(Exception) as e:
            _task(allocation={"mpi_np": value})
        assert "bench" in str(e.value) and "ONE point" in str(e.value), e.value

    # and a real typo still gets the near-miss, not the lecture
    with pytest.raises(Exception) as e:
        _task(allocation={"tyme": "4h"})
    assert "did you mean 'time'" in str(e.value) and "bench" not in str(e.value)


class TestTheRunsConditionIsOneValueEach:
    """`stages.md` § 6.8d.  `bench` and `execution` are one vocabulary at two
    arities, so the ARITY is what tells them apart -- and a list in the wrong
    one has to say which block it belongs in, because "invalid" would send a
    person hunting for a typo instead of to the other lane."""

    def test_a_list_is_refused_and_names_the_block_it_belongs_in(self):
        with pytest.raises(Exception) as e:
            _task(execution={"mpi_np": [4, 8]})
        assert "ONE value" in str(e.value) and "bench" in str(e.value), e.value

    def test_a_single_point_list_is_refused_too(self):
        """A one-item list is still `bench`'s shape.  Accepting it would make
        two spellings of one thing, and the next reader would have to check
        both."""
        with pytest.raises(Exception) as e:
            _task(execution={"mpi_np": [8]})
        assert "ONE value" in str(e.value)

    def test_a_scalar_is_what_it_takes(self):
        t = _task(execution={"mpi_np": 8, "diag_algorithm": "ELPA-2STAGE"})
        assert t.execution == {"mpi_np": 8, "diag_algorithm": "ELPA-2STAGE"}
        assert t.to_dict()["execution"] == t.execution

    def test_a_name_that_is_not_an_execution_setting_is_refused(self):
        """Membership is the catalogue's, checked where the vocabulary is
        (`validation/task.py`) -- the same split `bench` makes.  A physics
        value here would otherwise ride to prep and land in a deck as
        something nobody can read back."""
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation.task import preflight
        found = [i for i in preflight(_task(execution={"mesh_cutoff": 400}),
                                      SiestaConfig)
                 if "execution" in (i.where or "")]
        assert found, "a physics value passed as a run condition"
        assert "not an execution setting" in found[0].message
        # and it says where it DOES belong, rather than only that it is wrong
        assert "template" in found[0].message

    def test_the_two_LANE_ASKS_are_admitted_and_mem_is_not(self):
        """`stages.md` § 6.8e -- the membership door, both directions.

        `time` and `domain` are not catalogue items (no engine has an opinion
        about a wall clock or a queue) and are admitted by name, because a
        BENCH and a RUN want different ones: a trial's steps are cut so it
        wants minutes; the run wants days.

        `mem` is refused, and that is the load-bearing half.  A trial and a
        run compute the same system with the same basis and hold about the
        same amount, so a second home for it would be a second place to look
        and no new answer -- and two places to write one value is how a run
        ends up asking for something nobody typed."""
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation.task import preflight

        def _bad(block):
            return [i for i in preflight(_task(execution=block), SiestaConfig)
                    if "execution" in (i.where or "")]

        assert not _bad({"time": "2-00:00:00"}), "a run may state its wall"
        assert not _bad({"domain": "public"}), "a run may state its queue"

        found = _bad({"mem": "256G"})
        assert found, "`mem` was admitted to `execution` -- it has one home"
        assert "not an execution setting" in found[0].message

    def test_absent_writes_no_key(self):
        """Absent-is-a-state: a description that runs at the wrapper's own
        policy round-trips without gaining a block."""
        assert "execution" not in _task().to_dict()


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
