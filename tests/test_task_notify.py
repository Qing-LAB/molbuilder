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
secret live in **`config_dir()/notify`**, mode `0600`, on the machine that
runs the job, and the split is what keeps the rest of the record shareable
(`web/this-machine.md`, which owns that file; `run-reports.md` § 3 owns its
format).

*(This said `~/.molbuilder/notify` and cited the 2026-09-01 plan until
2026-09-02.  The path was never right — it is the config directory, which
`MOLBUILDER_CONFIG_DIR` may move — and the plan records how the split was
decided rather than what is true now.)*
"""
from __future__ import annotations

import pytest

from molbuilder.task import SCHEMA, Notify, Task

_BASE = {
    "schema": SCHEMA,
    "engine": {"name": "siesta"},
    "calculation": "optimization",
    "shape": "hierarchical",
    "run": {"name": "JOB", "id": "JOB_H2"},
    "structure": {"source": "h2.xyz", "formula": "H2", "atoms": 2},
    "varies": [],
    "stages": [{"name": "coarse", "enabled": True, "overrides": {}}],
}


def _task(**extra) -> Task:
    return Task.from_dict({**_BASE, **extra})


def _block(**stated) -> dict:
    """A notify block WHOLE, as every description writes one since
    2026-10-06 (`run-reports.md` § 3.0): its four values, ``null`` for
    never, ``["*"]`` for every channel and every field, with ``stated``
    over them."""
    return {"on_scf_converged": False, "every_hours": None,
            "channels": ["*"], "report": ["*"], **stated}


# --------------------------------------------------------------------- #
#  the shape on disk                                                     #
# --------------------------------------------------------------------- #

def test_both_triggers_round_trip():
    t = _task(notify=_block(on_scf_converged=True, every_hours=6))
    assert t.notify == Notify(on_scf_converged=True, every_hours=6.0)
    assert Task.from_dict(t.to_dict()).notify == t.notify


def test_the_triggers_are_independent_not_a_choice():
    """They combine with OR — checkboxes, not a picker.  Either alone is a
    valid policy, and so is both."""
    scf_only = _task(notify=_block(on_scf_converged=True))
    assert scf_only.notify.on_scf_converged
    assert scf_only.notify.every_hours is None

    clock_only = _task(notify=_block(every_hours=2))
    assert clock_only.notify.every_hours == 2.0
    assert not clock_only.notify.on_scf_converged

    for t in (scf_only, clock_only):
        assert Task.from_dict(t.to_dict()).notify == t.notify


def test_absent_is_a_state_and_writes_no_key():
    """A description that reports on nothing must round-trip BYTE-identical,
    or every file written before 2026-08-26 changes on first save and
    "off" acquires a second spelling on disk."""
    t = _task()
    assert t.notify is None
    assert "notify" not in t.to_dict()


def test_a_block_with_every_trigger_off_is_still_a_block():
    """A block reports the start and the end at least (`run-reports.md`
    § 2) -- so one whose triggers are all off is a block, kept and written
    whole, never read as no block.  *(It was read as none until 2026-10-06,
    `test_a_policy_that_says_nothing_writes_no_key`: `{"every_hours": 0}`
    meant off, against § 2.)*"""
    t = _task(notify=_block())
    assert t.notify is not None
    assert t.to_dict()["notify"] == _block()


def test_a_block_that_leaves_a_key_out_is_refused_naming_it():
    """A block states all four (2026-10-06) -- an explicit choice and a key
    left out were the same bytes until then."""
    with pytest.raises(ValueError, match="'every_hours', 'channels', "
                                         "'report'"):
        _task(notify={"on_scf_converged": True})


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
        _task(notify=_block(**bad))


@pytest.mark.parametrize("hours", [-2, 0])
def test_a_period_that_is_not_positive_is_refused(hours):
    """There is no reading of "every minus two hours" or "every 0 hours",
    and a timer armed with either fires on every pass -- the noise this
    block exists to stop.  Never is ``null`` (W57 R10: ``0`` was never until
    2026-10-06)."""
    with pytest.raises(ValueError, match="positive number of hours"):
        _task(notify=_block(every_hours=hours))


def test_an_empty_block_is_refused_rather_than_read_as_off():
    """No block already means no notification.  Accepting `{}` as a second
    way to say it is how one state grows two spellings."""
    with pytest.raises(ValueError, match="non-empty object"):
        _task(notify={})


def test_a_misspelled_trigger_is_refused_with_the_near_miss():
    """§ 6.1 rule 1 -- an unknown key is refused, not ignored.  Ignored, a
    typo is a calculation that silently reports nothing while its file
    appears to ask for reports."""
    with pytest.raises(ValueError, match="on_scf_converged"):
        _task(notify={**_block(), "on_scf_convrged": True})


# --------------------------------------------------------------------- #
#  which channels -- every, none, or these                               #
# --------------------------------------------------------------------- #

def test_names_round_trip_and_are_deduplicated():
    """A name is a label the person chose on the machine that runs the job.
    It grants nothing, so it may travel where an address may not."""
    t = _task(notify=_block(channels=["slack", "lab", "slack"]))
    assert t.notify.channels == ("slack", "lab")
    assert Task.from_dict(t.to_dict()).notify == t.notify


def test_every_channel_is_written_as_a_star():
    """*Every channel the running machine has* is written ``["*"]`` and read
    back as every channel (``None``) -- it was the key left out until
    2026-10-06 (`test_naming_no_channels_writes_no_key`), the same bytes as
    a block that never said."""
    t = _task(notify=_block(on_scf_converged=True))
    assert t.notify.channels is None
    assert t.to_dict()["notify"]["channels"] == ["*"]
    assert Task.from_dict(t.to_dict()).notify == t.notify


def test_an_EMPTY_list_survives_the_round_trip():
    """`[]` and `["*"]` are two INTENTIONS: send this calculation nowhere,
    versus send it everywhere the running machine can reach
    (`run-reports.md` § 3.0).  Read as the other, it would send reports to
    every channel the person had just unticked, from a description that
    says so in as many words.
    """
    t = _task(notify=_block(channels=[]))
    assert t.notify.channels == ()
    assert t.to_dict()["notify"]["channels"] == []
    assert Task.from_dict(t.to_dict()).notify.channels == ()


# `test_channels_alone_is_a_policy_worth_writing` and
# `test_a_selection_alone_is_a_policy_worth_writing` retired 2026-10-06: they
# pinned `Notify.__bool__`, which decided whether a block was there by its
# values.  A block is there or not (`Task.notify` is None without one), and
# is written whole.


@pytest.mark.parametrize("bad,why", [
    ("slack",          "a bare string, not a list"),
    ([3],              "not a name"),
    (["*", "slack"],   "every channel stands alone, never beside names"),
    (["has space"],    "would not survive the monitor's command line"),
    (["a/b"],          "a slash means a different thing in a path"),
    ({"slack": True},  "an object, not a list"),
])
def test_a_channel_list_that_could_not_travel_is_refused(bad, why):
    """A name is written into this file, read back out of it, and rendered
    into the monitor's command line.  Anything outside the rule would mean
    one thing here and another in the shell."""
    with pytest.raises(ValueError):
        _task(notify=_block(channels=bad))


def test_the_destination_is_not_a_field_here():
    """The credential must have no home in this record.  A URL or a token
    key would be accepted-and-ignored without the allowlist, which is the
    worst outcome: it looks configured and sends nothing, and the secret
    travels anyway."""
    for secret_ish in ("url", "webhook", "token", "notify_url"):
        with pytest.raises(ValueError, match="unknown key"):
            _task(notify={**_block(on_scf_converged=True), secret_ish: "x"})


def test_finish_is_not_settable():
    """A run ending always reports.  Offering a switch for it would let
    someone turn off the one message the hook exists to deliver."""
    with pytest.raises(ValueError, match="unknown key"):
        _task(notify={"on_finish": False})


# --------------------------------------------------------------------- #
#  WHAT each report carries (`stages.md` § 6.9)                          #
#                                                                        #
#  `notify` says WHEN and TO WHOM; `report` says WHAT IS IN IT.  Added   #
#  2026-09-02 and unpinned until now -- which is how the serializer came #
#  to parse the key and write nothing.                                   #
# --------------------------------------------------------------------- #

def test_a_field_selection_survives_the_round_trip():
    """**The defect this closes.** `read_task` parsed `notify.report` and
    `to_dict` did not write it, so a description carrying a selection lost
    it the first time anything read and wrote the file -- which the task-setup
    tab does on every save.  No error anywhere; the ticks simply were not
    there when you came back.
    """
    t = _task(notify=_block(every_hours=6, channels=["lab"],
                            report=["energy", "n_iters"]))
    assert t.notify.report == ("energy", "n_iters")
    assert t.to_dict()["notify"]["report"] == ["energy", "n_iters"], (
        "the selection was parsed and then not written: "
        + repr(t.to_dict()["notify"]))
    assert Task.from_dict(t.to_dict()).notify == t.notify


def test_EVERY_field_is_a_star_and_an_EMPTY_LIST_is_the_summary_alone():
    """Two states, both written (§ 6.9): ``["*"]`` is *everything the
    monitor could determine*, ``[]`` the name, the state and the summary
    line with no field grid.  *(Every field was the key left out until
    2026-10-06.)*
    """
    every = _task(notify=_block(every_hours=1))
    assert every.notify.report is None
    assert every.to_dict()["notify"]["report"] == ["*"]

    empty = _task(notify=_block(every_hours=1, report=[]))
    assert empty.notify.report == ()
    assert empty.to_dict()["notify"]["report"] == [], (
        "the empty list was dropped, so 'the summary line alone' came back "
        "meaning 'every field'")
    assert Task.from_dict(empty.to_dict()).notify.report == ()


@pytest.mark.parametrize("bad,why", [
    ("energy",              "a bare string, not a list"),
    ({"energy": True},      "an object, not a list"),
    (["wall_time"],         "a plausible name that is not a report field"),
    (["name"],              "the name is always sent and is not settable"),
    (["job_id"],            "nor is the job id"),
])
def test_a_field_that_is_not_a_report_field_is_refused_by_name(bad, why):
    """One vocabulary (§ 6.9): what you tick, what travels, and what a
    listener parses are the same words, so a name outside them is refused
    at save rather than silently producing a report without it.

    `name` and `job_id` are in this list on purpose: they ARE in every
    report, and that is precisely why they are not selectable -- a report you
    cannot attribute to a job is a notification you have to go and look up.
    """
    with pytest.raises(ValueError):
        _task(notify=_block(report=bad))


def test_the_refusal_names_the_fields_that_do_exist():
    """A refusal that does not say what IS allowed makes the person guess."""
    from molbuilder.report_fields import NAMES
    with pytest.raises(ValueError) as exc:
        _task(notify=_block(report=["nonsense"]))
    said = str(exc.value)
    for item in NAMES:
        assert item in said, (
            f"the refusal does not name {item!r}, so the reader cannot tell "
            f"what to write instead: {said}")


def test_a_repeated_field_is_kept_once_and_in_order():
    """The same rule the channel names follow -- a list is a set with an
    order, and a duplicate is a slip rather than a second request."""
    t = _task(notify=_block(report=["energy", "n_iters", "energy"]))
    assert t.notify.report == ("energy", "n_iters")
