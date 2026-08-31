"""R14 — a QoS also caps how many jobs you may have submitted at once.

2026-08-30, on Sol.  A six-shelf bench sweep went to the `debug` domain,
which allows **two** submitted jobs per user.  Two were accepted; four came
back `QOSMaxSubmitJobPerUserLimit`.  The preview had listed all six and asked
*submit this?*, and every check it could run had passed — because the limit
that decided the outcome **was not among the facts the record holds**.

`Domain` had no job-count field at all.  The number was in `sacctmgr show qos`,
in the very table the probe already fetched, and the format list never asked
for it: R13's rule (*a field you did not request is not an absence the record
may report as silence*) landing a third time, one column over.

Two things are pinned here, because the incident had two halves:

* the arithmetic — said BEFORE the prompt, while saying no is still free; and
* the DOMAIN NAME — the preview showed `-p htc -q debug` and nothing else, and
  on Sol `debug` *is* (htc, debug) while `htc` is (htc, public).  Two domains,
  one partition, so the flags read as *htc* and the run was believed to have
  gone to the wrong queue.  It had not.
"""
from __future__ import annotations

import pytest

from molbuilder.jobset.submit import submitted_cap_notes, _domain_name
from molbuilder.scheduler import Domain


class _Placement:
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain


class _Plan:
    def __init__(self, placement):
        self.placement = placement


def _shelf(name="debug", cap=2):
    row = {"name": name, "partition": "htc", "qos": name,
           "max_time": "00:15:00"}
    if cap is not None:
        row["max_submit_jobs"] = cap
    return _Plan(_Placement(name, Domain.from_row(row)))


class TestTheArithmeticIsSaidBeforeTheAsk:

    def test_the_incident_reproduced(self):
        """Six shelves at Sol's debug: the note must name the cap, the ask,
        and how many the scheduler will refuse."""
        note = submitted_cap_notes([_shelf() for _ in range(6)])
        assert len(note) == 1, note
        said = note[0]
        assert "debug" in said
        assert "2 submitted job(s)" in said
        assert "this sweep is 6" in said
        assert "refuse 4" in said
        # ...and what it costs, so the reader knows this is survivable.
        assert "stay pending" in said and "picks up exactly them" in said

    def test_a_sweep_that_fits_says_nothing(self):
        assert submitted_cap_notes([_shelf() for _ in range(2)]) == []

    def test_each_domain_is_judged_on_its_own(self):
        """A split sweep sends its CPU and GPU sides to different queues, so
        the count that matters is per domain, not per sweep."""
        plans = [_shelf("debug", 2)] * 3 + [_shelf("htc", 100)] * 3
        notes = submitted_cap_notes(plans)
        assert len(notes) == 1 and notes[0].startswith("debug takes")


class TestWhatIsNotALimit:

    def test_a_qos_that_states_no_cap(self):
        """R3: an unstated limit never bars.  `None` is *asked, and this QoS
        states none* — it is a measurement, not a ceiling."""
        row = {"name": "public", "partition": "public", "qos": "public",
               "max_time": "7-00:00:00", "max_submit_jobs": None}
        plans = [_Plan(_Placement("public", Domain.from_row(row)))] * 50
        assert submitted_cap_notes(plans) == []

    def test_a_record_that_never_asked(self):
        """The key absent means the probe predates the question.  R3 forbids
        reading an unstated limit as a bar; this asserts the other direction
        too — a never-asked one is not permission to warn about."""
        plans = [_shelf(cap=None)] * 50
        assert submitted_cap_notes(plans) == []
        assert plans[0].placement.domain.max_submit_jobs is not None, (
            "absent must be UNSET, distinguishable from a stated null")

    def test_no_menu_no_domain_no_note(self):
        """R6: with no record of this machine's queues nothing was promised,
        so there is no cap to check and no domain to name."""
        class _Bare:
            placement = None
        assert _domain_name(_Bare()) is None
        assert submitted_cap_notes([_Bare()] * 9) == []


class TestThePreviewNamesTheDomain:

    def test_the_shelf_reports_which_domain_it_was_placed_on(self):
        assert _domain_name(_shelf("debug")) == "debug"

    def test_two_domains_on_one_partition_are_told_apart(self):
        """The whole reason the name is carried.  Both render `-p htc`; only
        the name says which queue the sweep is actually going to."""
        debug = Domain.from_row({"name": "debug", "partition": "htc",
                                 "qos": "debug", "max_time": "00:15:00"})
        htc = Domain.from_row({"name": "htc", "partition": "htc",
                               "qos": "public", "max_time": "4:00:00"})
        assert debug.partition == htc.partition == "htc"
        assert _domain_name(_Plan(_Placement("debug", debug))) == "debug"
        assert _domain_name(_Plan(_Placement("htc", htc))) == "htc"
