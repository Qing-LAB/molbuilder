"""P3 unit 3 — the id is editable once, before anything has run.

Contract: ``docs/execution/run-identity.md`` § 1 (*"a label edited between
runs — the warm files no longer match, and a run that should have resumed
starts cold instead"*), § 3 rule 1, § 5 (reported, not prevented) and
``docs/execution/job-contracts.md`` § 4.2 (the warm-file inventory).

**These tests write real files.** The unit's whole claim is about what is on
disk beside a deck, and a mocked ``is_file`` would pass while the real
predicate read the wrong names — which is exactly the seam that matters here,
since four disagreeing inventories exist in this tree.
"""
from __future__ import annotations

import pytest

from molbuilder.runwrap import _SIESTA_WARM_SUFFIXES
from molbuilder.validation.identity import check_id_change, warm_files_present


ID = "BDT_Au_relax_C6H4S2Au38"


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


def test_an_unchanged_id_says_nothing(calc):
    (calc / f"{ID}.XV").write_text("coords\n")
    assert check_id_change(calc, ID, ID, "siesta") == []


# --------------------------------------------------------------------- #
#  "after that, it is a different calculation"                          #
# --------------------------------------------------------------------- #

def test_once_state_exists_the_edit_is_reported(calc):
    """§ 1. The run does not fail -- it silently starts over, which is why
    something has to say so before the user commits."""
    (calc / f"{ID}.XV").write_text("relaxed coords\n")
    (calc / f"{ID}.DM").write_text("density\n")
    issues = check_id_change(calc, ID, "BDT_Au_relax_v2", "siesta")
    assert len(issues) == 1
    assert issues[0].severity == "warn"
    assert issues[0].where == "run.id"


def test_the_finding_names_the_files_not_a_count(calc):
    """*"3 warm files"* tells a user nothing; the name of the file holding
    the relaxed geometry tells them what they are about to walk away from."""
    (calc / f"{ID}.XV").write_text("relaxed coords\n")
    msg = check_id_change(calc, ID, "other", "siesta")[0].message
    assert f"{ID}.XV" in msg
    assert "starts cold" in msg


def test_it_is_a_warning_and_never_a_refusal(calc):
    """`job-system.md § 2` decision 5: molbuilder informs, the user decides.
    Renaming after a run is legitimate; doing it unknowingly is not."""
    (calc / f"{ID}.XV").write_text("x\n")
    assert all(i.severity == "warn"
               for i in check_id_change(calc, ID, "other", "siesta"))


# --------------------------------------------------------------------- #
#  The inventory — the reason this unit could have been built wrong      #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("suffix", _SIESTA_WARM_SUFFIXES)
def test_every_suffix_the_cold_glob_covers_is_detected(calc, suffix):
    """`job-contracts.md § 4.2`: *"Every file below is in that engine's
    --cold move-aside glob"* — so the glob's list is the authority on what an
    id keys, and all thirteen count.

    Parametrised over ``runwrap``'s own tuple rather than a retyped list, so a
    fourteenth warm hook is detected here the day it is added. Built against
    this list and not ``runstatus._WARM_FILES``, which names three: a
    directory holding only a `.TSHS` has state keyed by the id, and reading it
    as 'nothing has run' would declare the id editable when it is not."""
    (calc / f"{ID}{suffix}").write_text("state\n")
    assert warm_files_present(calc, ID, "siesta") == [f"{ID}{suffix}"]


def test_state_belonging_to_a_different_id_is_not_this_id_s(calc):
    """§ 5's third row -- prior state from another calculation. The engine
    will not load it, so renaming orphans nothing and this stays quiet."""
    (calc / "someone_else.XV").write_text("not ours\n")
    assert warm_files_present(calc, ID, "siesta") == []
    assert check_id_change(calc, ID, "other", "siesta") == []


def test_a_dangling_carry_symlink_is_not_state(calc):
    """A carried restart file that no longer resolves cannot be continued
    from. ``runstatus`` reads them the same way; disagreeing would mean two
    surfaces describing one directory differently."""
    (calc / f"{ID}.XV").symlink_to(calc / "gone.XV")
    assert warm_files_present(calc, ID, "siesta") == []


def test_pyscf_has_its_own_inventory_and_it_is_used(calc):
    """§ 4.2 gives PySCF five files, not SIESTA's thirteen. The per-engine
    split is the point -- `run-identity.md § 4` is an entire section on the
    two engines meaning the same thing by different mechanisms."""
    (calc / f"{ID}.chk").write_text("scf guess\n")
    assert warm_files_present(calc, ID, "pyscf") == [f"{ID}.chk"]
    assert warm_files_present(calc, ID, "siesta") == []


def test_an_unknown_engine_is_not_refused_a_second_time(calc):
    """Whether the engine is supported is `stages.md § 6.6`'s first check and
    belongs to the preflight. Refusing again here would put one rule in two
    places and let them drift apart."""
    (calc / f"{ID}.XV").write_text("x\n")
    assert warm_files_present(calc, ID, "vasp") == []
    assert check_id_change(calc, ID, "other", "vasp") == []
