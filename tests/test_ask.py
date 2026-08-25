"""One question, one answer, one interface, one output — `jobset/ask.py`.

**The design this replaced was the mistake.**  The numbers used to arrive by
themselves: 128 GB because SLURM grants 2 GB a core and the job had 64 of
them; 38 minutes because a per-trial default nobody set was multiplied by a
trial count nobody saw.  Both were correct arithmetic on inputs the person had
never been offered.

The first fix was a provenance system — five categories, an announcement rule,
a display — machinery whose entire purpose was to *cope with* numbers nobody
chose.  **Asking removes the problem instead of labelling it** (user,
2026-08-23), and most of that machinery went with it.

Four things, and the CLI and the browser call the same four.  Two surfaces
asking one question two ways is how they come to disagree about what was
asked.
"""
from __future__ import annotations

import pytest

from molbuilder.jobset.ask import Ask, confirm
from molbuilder.scheduler.quantities import parse_duration, parse_memory
from molbuilder.scheduler import Domain


# --------------------------------------------------------------------- #
#  the answer                                                           #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("text,secs", [
    ("4h", 14400), ("90m", 5400), ("45", 2700), ("30s", 30), ("1.5h", 5400)])
def test_a_duration_is_written_the_way_a_person_says_it(text, secs):
    assert parse_duration(text) == secs


@pytest.mark.parametrize("text,gb", [
    ("128G", 128.0), ("0.5T", 512.0), ("128", 128.0), ("512M", 0.5)])
def test_memory_uses_slurms_own_spelling(text, gb):
    """What a person types here is what they would have typed into --mem."""
    assert parse_memory(text) == pytest.approx(gb)


@pytest.mark.parametrize("bad", ["soon", "-5", "0", "lots"])
def test_an_answer_that_is_not_a_number_is_refused_with_the_shape(bad):
    """A refusal that does not show the accepted forms is one you have to
    guess past."""
    for fn, word in ((parse_duration, "4h"), (parse_memory, "128G")):
        with pytest.raises(ValueError) as e:
            fn(bad)
        assert word in str(e.value) or "positive" in str(e.value)


def test_unanswered_is_None_and_never_zero():
    """``None`` is *not answered*.  Zero is an answer, and a wrong one."""
    for blank in (None, "", "   "):
        assert parse_duration(blank) is None
        assert parse_memory(blank) is None
    assert not Ask()
    assert Ask(time_s=60)


# The benchmark-arithmetic and render() tests that lived here are DELETED
# with the code they pinned (user dictation, 2026-08-24): time is never
# derived -- the user states it, or the target queue's own ceiling stands.
# The one output is now the launch door's plan (the exact sbatch command of
# every job); its tests live with the launch tests.


# --------------------------------------------------------------------- #
#  the one interface                                                     #
# --------------------------------------------------------------------- #

def test_without_yes_the_answer_is_the_persons():
    said = []
    assert confirm("x", echo=said.append, prompt=lambda: False) is False
    assert confirm("x", echo=said.append, prompt=lambda: True) is True
    assert said[0] == "x", "the request must be shown before it is answered"


def test_yes_is_a_decision_to_trust_not_an_absence_of_one():
    """`--yes` says *I have decided*; its absence is not permission."""
    said = []
    def _never():
        raise AssertionError("asked despite --yes")
    assert confirm("x", auto_yes=True, echo=said.append, prompt=_never)
    assert "(--yes)" in said[1], "the skip is recorded, not silent"


def test_the_request_is_shown_even_when_it_is_accepted_unasked():
    """--yes skips the question, never the output: a person scrolling back
    must be able to see what was sent."""
    said = []
    confirm("about to request:\n  time 4h", auto_yes=True, echo=said.append,
            prompt=lambda: True)
    assert "about to request" in said[0]


# --------------------------------------------------------------------- #
#  the queue is the person's to name                                     #
# --------------------------------------------------------------------- #

def _menu():
    return [
        Domain(name="debug", partition="htc", qos="debug",
               max_time="0-00:15:00", max_cores=128, max_mem_gb=251.0),
        Domain(name="htc", partition="htc", qos="public",
               max_time="0-04:00:00", max_cores=128, max_mem_gb=251.0),
        Domain(name="general", partition="general", qos="public",
               max_time="7-00:00:00", max_cores=48, max_mem_gb=502.9,
               gpu={"a100": 4}),
    ]


def _table(**kw):
    from molbuilder.jobset.ask import queue_table
    return queue_table(_menu(), Ask(time_s=4 * 3600, mem_gb=128), **kw)


def test_every_queue_is_listed_with_what_it_offers():
    """The framework does not choose.  Which queue to spend a day of
    wall-clock in is a judgement about priority, contention and what else is
    running — none of it on this machine's record, all of it the person's."""
    t = _table(cores=64)
    for name in ("debug", "htc", "general"):
        assert name in t
    assert "251 GB" in t and "a100 x4" in t


def test_a_queue_that_cannot_take_the_job_is_listed_WITH_THE_REASON():
    """Hiding it would answer *"why is my queue not an option?"* with
    silence, and that question has a real answer worth reading."""
    t = _table(cores=64)
    assert "needs 240 min but debug allows" in t
    assert "needs 64 cores but general allows 48" in t


def test_the_listing_and_the_submission_cannot_disagree():
    """The table reuses the scheduler's own admission, so a row it marks as
    fitting is one the submission will accept.  A table that says yes where
    the check says no is worse than no table."""
    from molbuilder.jobset.ask import _why_not
    from molbuilder.scheduler.admit import Request, admits
    ask, cores = Ask(time_s=4 * 3600, mem_gb=128), 64
    for row in _menu():
        assert _why_not(row, ask, cores=cores) == list(
            admits(row, Request(ranks=cores, walltime_s=ask.time_s,
                                mem_gb=ask.mem_gb)))


def test_it_says_how_to_choose_and_that_nothing_has_happened_yet():
    t = _table(cores=64)
    assert "--domain" in t
    assert "Nothing is submitted until you do" in t


def test_a_machine_with_no_queues_says_the_job_runs_directly():
    from molbuilder.jobset.ask import queue_table
    assert "runs directly" in queue_table([], Ask(time_s=60))
