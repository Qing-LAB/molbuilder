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

from molbuilder.jobset.ask import (GROUP_SLACK, GROUP_STARTUP_S, Ask,
                                   bench_bound, bench_total, confirm, fits,
                                   parse_duration, parse_memory, render)
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


# --------------------------------------------------------------------- #
#  the benchmark's arithmetic — stated as a total, derived per trial     #
# --------------------------------------------------------------------- #

def test_the_reported_case_reproduces_exactly():
    """2 trials, a 15-minute bound: 38 minutes.  Pinned so the formula and
    the number it produced stay tied together."""
    assert bench_total(15 * 60, 2) == 38 * 60


def test_a_total_becomes_a_per_trial_bound_that_fits_inside_it():
    for text, n in (("4h", 36), ("15m", 2), ("90m", 6)):
        total_s = parse_duration(text)
        assert bench_total(bench_bound(total_s, n), n) <= total_s


def test_a_bound_under_a_minute_measures_nothing_so_the_total_gives():
    """Honouring a budget into uselessness is the worse answer — and § 2's
    *timing out is a result* only holds if a trial had time to produce one."""
    assert bench_bound(parse_duration("15m"), 36) >= 60


def test_the_launcher_shares_this_formula():
    """The CLI prints a total and the launcher computes one.  Two spellings of
    one formula is how a displayed number comes to differ from the one that
    reaches the scheduler."""
    import inspect
    from molbuilder.jobset import submit
    src = inspect.getsource(submit._submit_side_group)
    assert f"* {GROUP_SLACK}" in src and f"+ {GROUP_STARTUP_S}" in src


# --------------------------------------------------------------------- #
#  fits — answered before anything is submitted                          #
# --------------------------------------------------------------------- #

def _sol():
    return [Domain(name="htc", partition="htc", qos="public",
                   max_time="0-04:00:00", max_cores=128, max_mem_gb=251.0),
            Domain(name="general", partition="general", qos="public",
                   max_time="7-00:00:00", max_cores=128, max_mem_gb=502.9)]


def test_an_ask_this_machine_can_hold_passes():
    ok, why = fits(Ask(time_s=2280, mem_gb=128), _sol())
    assert ok and not why


def test_an_impossible_ask_is_caught_while_changing_it_is_free():
    """The whole point: a queue rejects this after you have waited for it."""
    ok, why = fits(Ask(time_s=2280, mem_gb=900), _sol())
    assert not ok
    assert "900" in why[0] and "502.9" in why[0], (
        "the refusal must name both what was asked and what is available")


def test_a_machine_with_no_queues_contradicts_nothing():
    assert fits(Ask(time_s=2280, mem_gb=900), [])[0] is True


def test_an_unstated_ceiling_never_bars():
    """R3.  A row that does not say how much memory it has is not claiming to
    have none."""
    silent = [Domain(name="s", partition="p", qos="q")]
    assert fits(Ask(time_s=999999, mem_gb=9999), silent)[0] is True


# --------------------------------------------------------------------- #
#  the one output                                                        #
# --------------------------------------------------------------------- #

def test_every_number_that_reaches_the_scheduler_is_shown():
    out = render(Ask(time_s=2280, mem_gb=128))
    assert "0h 38m" in out and "128 GB" in out


def test_the_benchmark_shows_its_arithmetic_not_just_the_total():
    """How 00:38:00 became a number nobody could question."""
    out = render(Ask(time_s=None), n_trials=2, bound_s=15 * 60)
    assert "2 trial(s), 15 min each" in out and "38 min total" in out


def test_a_total_that_cannot_hold_the_trials_says_so_with_both_ways_out():
    out = render(Ask(time_s=15 * 60), n_trials=36,
                 bound_s=bench_bound(15 * 60, 36))
    assert "do not fit" in out
    assert "Fewer trials" in out and "more time" in out


def test_the_queue_is_named_when_one_was_chosen():
    class _P:
        partition, qos = "htc", "public"
    assert "htc / public" in render(Ask(time_s=60), placement=_P())


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
