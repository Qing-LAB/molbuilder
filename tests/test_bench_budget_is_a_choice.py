"""A benchmark is BOUNDED, never estimated — and the bound is chosen.

`execution/submission.md` § 2.  A benchmark exists to measure the per-cycle
cost, so feeding it an estimate of that cost is circular and the estimate has
nowhere to come from.  The grouped launcher already bounded rather than
predicted:

    total = trials × per-trial bound × 1.1 + 5 min startup

and 2 × 15 min × 1.1 + 5 = **38:00** — the number that prompted all of this.
The formula was right.  **Nobody chose the 15 minutes and nobody saw the
arithmetic**: it was a flag default, and the total arrived as an opaque
`00:38:00` in a scheduler record.

Two things change here, and neither is a new estimate:

  * **the person states the TOTAL**, which is the decision they can actually
    make — how long to wait, how big a slot to queue for — and the per-trial
    bound is arithmetic on top of it (user, 2026-08-23);
  * **the arithmetic is printed**, with its provenance, so a default announces
    itself as one (S1).
"""
from __future__ import annotations

import pytest

from molbuilder.jobset._cli import (GROUP_SLACK, GROUP_STARTUP_S,
                                    _announce_bench_bound, _bound_from_budget,
                                    _group_total_s, _parse_budget)


# --------------------------------------------------------------------- #
#  the arithmetic, and that it inverts                                  #
# --------------------------------------------------------------------- #

def test_the_reported_case_reproduces_exactly():
    """2 trials, a 15-minute bound: 38 minutes.  Pinned so the formula and
    the number it produced stay tied together."""
    assert _group_total_s(15 * 60, 2) == 38 * 60


def test_a_budget_becomes_a_per_trial_bound_that_fits_inside_it():
    for text, n in (("4h", 36), ("15m", 2), ("90m", 6)):
        budget = _parse_budget(text)
        total = _group_total_s(_bound_from_budget(budget, n), n)
        assert total <= budget, (
            f"{text} over {n} trials produced {total}s, which exceeds the "
            f"budget the person set")


@pytest.mark.parametrize("text,secs", [
    ("4h", 14400), ("90m", 5400), ("45", 2700), ("30s", 30), ("1.5h", 5400)])
def test_a_budget_is_written_the_way_a_person_says_it(text, secs):
    """Bare numbers are minutes, because that is the unit the flag this
    replaces already used."""
    assert _parse_budget(text) == secs


@pytest.mark.parametrize("bad", ["soon", "", "-5", "0"])
def test_a_budget_that_is_not_a_duration_is_refused(bad):
    import click
    with pytest.raises(click.ClickException):
        _parse_budget(bad)


def test_the_floor_stops_a_bound_that_would_measure_nothing():
    """A per-trial bound under a minute measures nothing, so the budget is
    exceeded rather than honoured into uselessness — and § 2's *timing out is
    a result* only holds if a trial had time to produce one."""
    assert _bound_from_budget(_parse_budget("15m"), 36) >= 60


# --------------------------------------------------------------------- #
#  and it is SAID                                                       #
# --------------------------------------------------------------------- #

def _said(capsys, *a, **kw):
    _announce_bench_bound(*a, **kw)
    return capsys.readouterr().out


def test_the_arithmetic_is_printed_not_just_the_total(capsys):
    out = _said(capsys, 2, 15 * 60, chosen=False)
    assert "2 trial(s) x 15 min" in out and "38 min total" in out, (
        "the total appeared without the sum that produced it, which is how "
        "00:38:00 became a number nobody could question")


def test_a_default_announces_itself_as_one(capsys):
    """**S1.**  Not buried in a comment, not recorded in a file read later:
    said at the moment of choosing, with what it would take to replace it."""
    out = _said(capsys, 2, 15 * 60, chosen=False)
    assert "A DEFAULT YOU DID NOT SET" in out
    assert "--budget" in out, "a default owes the way to decide it"


def test_a_chosen_bound_is_not_scolded(capsys):
    """Announcing a decision the person made as though it were a guess is
    noise, and noise is what makes real warnings unreadable."""
    out = _said(capsys, 2, 4 * 60, budget_s=15 * 60, chosen=True)
    assert "DEFAULT" not in out


def test_a_budget_that_cannot_hold_the_trials_says_so(capsys):
    """The floor above exceeds the budget rather than measuring nothing; that
    is a decision the person is owed sight of, with both ways out."""
    b = _parse_budget("15m")
    out = _said(capsys, 36, _bound_from_budget(b, 36), budget_s=b, chosen=True)
    assert "cannot fit" in out
    assert "fewer trials" in out and "larger budget" in out


def test_the_bound_is_described_as_a_bound(capsys):
    """**S2.**  A bound cannot be wrong -- it can only be reached, and
    reaching it is a result.  The trials that finished still yield the
    per-cycle cost, so hitting it must not read as a failure."""
    out = _said(capsys, 2, 15 * 60, chosen=False)
    assert "killed and reads incomplete" in out
    assert "walk continues" in out


def test_the_cli_and_the_launcher_share_one_formula():
    """Two spellings of one formula is how a displayed total comes to differ
    from the one actually requested."""
    import inspect
    from molbuilder.jobset import submit
    src = inspect.getsource(submit._submit_side_group)
    assert f"* {GROUP_SLACK}" in src, (
        f"the launcher no longer multiplies by {GROUP_SLACK}, so the CLI is "
        f"showing a total the allocation does not ask for")
    assert f"+ {GROUP_STARTUP_S}" in src, (
        f"the launcher's startup margin no longer matches the "
        f"{GROUP_STARTUP_S}s the CLI shows")
