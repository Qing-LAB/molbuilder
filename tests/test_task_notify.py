"""`task.json` carries WHEN a calculation should say something.

**Why the key exists.** The monitor has carried a notifier hook since it was
written, and `job-contracts.md` calls it *"the deliberate customization
point… why nobody has to be at the cluster. A run that ends at 3am can say
so."*  What it never had was a cadence anyone would want or a place to set
one: the hook fired on every changed sample, so a webhook configured against
it would have sent a message every ten seconds for the length of a run.

`notify` is where the answer lives (`task.Notify`): fire when an SCF cycle
converges, fire every N hours, or neither.  Finish is not settable — a run
ending always reports, which is the reason the hook exists at all.

**What is deliberately NOT here: the destination and its credential.**  This
file travels — to a cluster, into a handoff bundle, to whoever is handed the
calculation.  A policy is safe to carry; a token is not.  The URL and its
secret live in `~/.molbuilder/notify` on the machine that runs the job, mode
0600, and the split is what keeps the rest of the record shareable
(`plans/bench-and-junction-plan.md` § 2.9).
"""
from __future__ import annotations

import pytest

from molbuilder.task import SCHEMA, Notify, Task

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

def test_both_triggers_round_trip():
    t = _task(notify={"on_scf_converged": True, "every_hours": 6})
    assert t.notify == Notify(on_scf_converged=True, every_hours=6.0)
    assert Task.from_dict(t.to_dict()).notify == t.notify


def test_the_triggers_are_independent_not_a_choice():
    """They combine with OR — checkboxes, not a picker.  Either alone is a
    valid policy, and so is both."""
    scf_only = _task(notify={"on_scf_converged": True})
    assert scf_only.notify.on_scf_converged and not scf_only.notify.every_hours

    clock_only = _task(notify={"every_hours": 2})
    assert clock_only.notify.every_hours == 2.0
    assert not clock_only.notify.on_scf_converged

    for t in (scf_only, clock_only):
        assert Task.from_dict(t.to_dict()).notify == t.notify


def test_absent_is_a_state_and_writes_no_key():
    """A description that reports on nothing must round-trip BYTE-identical,
    or every file written before 2026-08-26 changes on first save and
    "off" acquires a second spelling on disk."""
    t = _task()
    assert not t.notify
    assert "notify" not in t.to_dict()


def test_a_policy_that_says_nothing_writes_no_key():
    """Explicit falsey values are the same state as absent, and must not
    survive as a block full of zeros -- that is the second spelling again,
    arriving by a different door."""
    t = _task(notify={"on_scf_converged": False, "every_hours": 0})
    assert not t.notify
    assert "notify" not in t.to_dict()


# --------------------------------------------------------------------- #
#  what it refuses, and why each refusal is worth its line                #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("bad,expect", [
    ({"on_scf_converged": "yes"}, "true or false"),
    ({"on_scf_converged": 1},     "true or false"),
    ({"every_hours": "6h"},       "number of HOURS"),
    ({"every_hours": "6"},        "number of HOURS"),
    ({"every_hours": True},       "number of HOURS"),
])
def test_a_value_of_the_wrong_type_is_refused_by_name(bad, expect):
    """Not coerced.  A record exists so a person can read it and know what
    their job will do; `"true"` accepted as a boolean makes a file that
    reads one way and behaves another.

    ``True`` is refused for ``every_hours`` specifically because Python
    would otherwise take it as 1 -- notify every hour, from a value that
    was meant as a checkbox.
    """
    with pytest.raises(ValueError, match=expect):
        _task(notify=bad)


def test_a_negative_period_is_refused():
    """There is no reading of "every minus two hours", and a timer armed
    with one fires on every pass -- which is the noise this block exists
    to stop."""
    with pytest.raises(ValueError, match="cannot be negative"):
        _task(notify={"every_hours": -2})


def test_an_empty_block_is_refused_rather_than_read_as_off():
    """Absent already means off.  Accepting `{}` as a second way to say it
    is how one state grows two spellings."""
    with pytest.raises(ValueError, match="non-empty object"):
        _task(notify={})


def test_a_misspelled_trigger_is_refused_with_the_near_miss():
    """§ 6.1 rule 1 -- an unknown key is refused, not ignored.  Ignored, a
    typo is a calculation that silently reports nothing while its file
    appears to ask for reports."""
    with pytest.raises(ValueError, match="on_scf_converged"):
        _task(notify={"on_scf_convrged": True})


def test_the_destination_is_not_a_field_here():
    """The credential must have no home in this record.  A URL or a token
    key would be accepted-and-ignored without the allowlist, which is the
    worst outcome: it looks configured and sends nothing, and the secret
    travels anyway."""
    for secret_ish in ("url", "webhook", "token", "notify_url"):
        with pytest.raises(ValueError, match="unknown key"):
            _task(notify={"on_scf_converged": True, secret_ish: "x"})


def test_finish_is_not_settable():
    """A run ending always reports.  Offering a switch for it would let
    someone turn off the one message the hook exists to deliver."""
    with pytest.raises(ValueError, match="unknown key"):
        _task(notify={"on_finish": False})
