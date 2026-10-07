"""P3 unit 3 — the id is editable once, before anything has run.

Contract: ``docs/execution/run-identity.md`` § 1 (*"a label edited between
runs — the warm files no longer match, and a run that should have resumed
starts cold instead"*), § 3 rule 1, § 5 (reported, not prevented) and
``docs/execution/job-contracts.md`` § 4.2 (the warm-file inventory).

**These tests write real files.** The unit's whole claim is about what is on
disk beside a deck, and a mocked ``is_file`` would pass while the real
predicate read the wrong names — which is exactly the seam that matters here.
"""
from __future__ import annotations

import pytest

from molbuilder.validation.identity import check_id_change, warm_files_present


#: A **label**, which is what a file stem is (§ 2.0a, decision 26).
ID = "BDT_Au_relax"


@pytest.fixture
def calc(tmp_path):
    """A calculation directory with a deck and nothing else run yet."""
    (tmp_path / f"{ID}_coarse.fdf").write_text("SystemLabel " + ID + "\n")
    return tmp_path


# --------------------------------------------------------------------- #
#  "before anything has run" — the common case, and it must be silent   #
# --------------------------------------------------------------------- #

def test_before_anything_has_run_the_id_is_freely_editable(calc):
    """Naming a calculation and thinking better of it is ordinary. A warning
    here would be noise on the one path everybody takes."""
    assert check_id_change(calc, ID, "BDT_Au_relax_v2", "siesta") == []


def test_a_deck_is_not_state(calc):
    """A written deck is an input, not something a run produced — it carries
    no geometry to orphan. § 1's failure is about restart files."""
    assert warm_files_present(calc, ID, "siesta") == []


# --------------------------------------------------------------------- #
#  P3 unit 5 — § 5: reported rather than pinned                         #
# --------------------------------------------------------------------- #
#
# § 5's table has four rows and a "who says it" column; the fourth names the
# READER at preflight and is deliberately not this function's.

from molbuilder.validation.identity import check_prior_state       # noqa: E402


def test_a_clean_directory_reports_nothing(calc):
    """Every row of § 5 is about state that exists. A first run in a fresh
    folder is the common path and must be silent, or the report becomes
    something users learn to dismiss."""
    assert check_prior_state(calc, ID, "siesta") == []


# -- § 5's first row: a changed cell is a NO-OP, not a mismatch --------- #

_CELL = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]


def test_the_cell_row_needs_state_to_be_about(calc):
    """No .XV, nothing was written under any cell -- so there is nothing to
    report, however different the deck's cell is."""
    assert check_prior_state(calc, ID, "siesta", cell=_CELL) == []
