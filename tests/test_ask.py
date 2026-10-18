"""One question, one answer, one interface, one output — `jobset/ask.py`.

**Asking removes the problem instead of labelling it** (user, 2026-08-23): a
number that arrives by itself -- 128 GB because SLURM grants 2 GB a core, a
time from a per-trial default nobody set -- is correct arithmetic on inputs
the person was never offered.

Four things, and the CLI and the browser call the same four.  Two surfaces
asking one question two ways is how they come to disagree about what was
asked.
"""
from __future__ import annotations

import pytest

from molbuilder.jobset.ask import confirm
from molbuilder.scheduler.quantities import parse_duration, parse_memory
from molbuilder.scheduler import Domain, Request


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


# --------------------------------------------------------------------- #
#  the one interface                                                     #
# --------------------------------------------------------------------- #

def test_without_yes_the_answer_is_the_persons():
    said = []
    assert confirm("x", echo=said.append, prompt=lambda: False).yes is False
    assert confirm("x", echo=said.append, prompt=lambda: True).yes is True
    assert said[0] == "x", "the request must be shown before it is answered"


@pytest.mark.parametrize("typed, said", [
    ("y", (True, "yes", False)),                # Enter gives the default, y
    ("n", (False, "no", False)),
    ("f", (True, "foreground", True)),          # the other way, its letter
    ("foreground", (True, "foreground", True)),  # or its word
])
def test_a_run_here_is_asked_which_way_too(typed, said):
    """A run here is asked with the other way of running it as a third
    answer (`job-system.md` § 6.0): ``y`` -- Enter's default -- goes as shown, ``n``
    sends nothing, the other way's letter or word goes that way -- its word
    what the ledger keeps.  The silent failure: a person typing ``f`` for
    the foreground, and the run left in the background."""
    got = confirm("x", echo=lambda _s: None, prompt=lambda: typed,
                  question="launch in the background?",
                  other=("f", "foreground"))
    assert (got.yes, got.words, got.other) == said, got


def test_yes_is_a_decision_to_trust_not_an_absence_of_one():
    """`--yes` says *I have decided*; its absence is not permission."""
    said = []
    def _never():
        raise AssertionError("asked despite --yes")
    assert confirm("x", auto_yes=True, echo=said.append, prompt=_never)
    assert "(--yes)" in said[1], "the skip is recorded, not silent"


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


def _table():
    from molbuilder.jobset.ask import queue_table
    return queue_table(_menu(), Request(ranks=64, walltime_s=4 * 3600,
                                        mem_gb=128))


def test_every_queue_is_listed_with_what_it_offers():
    """The framework does not choose.  Which queue to spend a day of
    wall-clock in is a judgement about priority, contention and what else is
    running — none of it on this machine's record, all of it the person's."""
    t = _table()
    for name in ("debug", "htc", "general"):
        assert name in t
    assert "251 GB" in t and "a100 x4" in t


def test_every_queue_says_the_jobs_fit_on_it_WITH_THE_NUMBERS():
    """`scheduler.md` R4, R6: each queue is listed with how the job sits
    on it -- what it is over, with the numbers -- and none is hidden or
    struck: the person names one, and the scheduler decides.  A record that
    lists no machines is read by its largest machine's figures (R3)."""
    t = _table()
    assert "4h is over debug's wall (0-00:15:00)" in t
    assert "64 cores is over general's largest machine (48 cores)" in t
    assert "64 cores · 128 GB · 4h fits htc" in t
