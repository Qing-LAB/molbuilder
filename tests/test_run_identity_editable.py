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


def test_the_question_is_no_longer_per_engine(calc):
    """**Retired and replaced 2026-08-08.** This used to assert that a `.chk`
    counted for PySCF and not for SIESTA — a real property of the per-engine
    lists, and one that stopped being true when the lists went away.

    *"Has anything run here"* is now answered by name, so a file named after
    the id counts no matter which engine is named. The engine argument is
    vestigial and the assertion says so, rather than being deleted quietly."""
    (calc / f"{ID}.chk").write_text("scf guess\n")
    assert warm_files_present(calc, ID, "pyscf") == [f"{ID}.chk"]
    assert warm_files_present(calc, ID, "siesta") == [f"{ID}.chk"]
    assert warm_files_present(calc, ID) == [f"{ID}.chk"]


def test_a_file_no_list_ever_named_is_still_detected(calc):
    """**The whole reason for the change** (`job-contracts.md § 4.2`,
    2026-08-08). SIESTA's output set depends on its version and its options,
    so any enumeration is a snapshot of one build. Here is a suffix that
    appears in none of this project's four inventories: under a list it was
    invisible, and 'nothing has run here' would have been the answer while a
    finished calculation sat in the folder."""
    (calc / f"{ID}.SOMETHING_NEW_IN_SIESTA_6").write_text("state\n")
    assert warm_files_present(calc, ID) == [
        f"{ID}.SOMETHING_NEW_IN_SIESTA_6"]


@pytest.mark.parametrize("name", [
    f"{ID}.fdf", f"{ID}_coarse.fdf", f"{ID}.fdf.template",
    f"{ID}.run.sh", f"{ID}.sbatch", f"{ID}.molwatch.log",
    f"{ID}-run0.out", f"{ID}_tight-run2.out", f"{ID}.log",
])
def test_what_molbuilder_wrote_is_never_mistaken_for_engine_state(calc, name):
    """The other half of the inversion, and the half that has to be exact.

    We cannot enumerate what an engine writes, but we always know what **we**
    wrote — so the subtraction is only as good as this list. Two kinds of
    mistake live here: counting our own deck as leftover state (which would
    say a fresh folder has run), and counting the run-indexed `.out` files,
    which are **history a user goes back to read**, not leftovers."""
    (calc / name).write_text("ours\n")
    assert warm_files_present(calc, ID) == []


def test_an_unknown_engine_is_not_refused_a_second_time(calc):
    """Whether the engine is supported is `stages.md § 6.6`'s first check and
    belongs to the preflight. Refusing again here would put one rule in two
    places and let them drift apart -- and now that the answer is by name, an
    unknown engine's files are found like any other."""
    (calc / f"{ID}.XV").write_text("x\n")
    assert warm_files_present(calc, ID, "vasp") == [f"{ID}.XV"]
    assert len(check_id_change(calc, ID, "other", "vasp")) == 1


# --------------------------------------------------------------------- #
#  P3 unit 5 — § 5: reported rather than pinned                         #
# --------------------------------------------------------------------- #
#
# § 5's table has four rows and a "who says it" column.  Three name the
# surface at check time and are asserted here; the fourth names the READER at
# preflight and is deliberately not this function's -- see the last test.

from molbuilder.validation.identity import check_prior_state       # noqa: E402


def _by_where(issues):
    return {i.where: i for i in issues}


def test_a_clean_directory_reports_nothing(calc):
    """Every row of § 5 is about state that exists. A first run in a fresh
    folder is the common path and must be silent, or the report becomes
    something users learn to dismiss."""
    assert check_prior_state(calc, ID, "siesta") == []


def test_matching_state_is_reported_as_info_because_nothing_is_wrong(calc):
    """§ 5's fourth row: *nothing is wrong; the user should still know before
    they start.* ``info`` is what says both halves at once."""
    (calc / f"{ID}.XV").write_text("coords\n")
    found = _by_where(check_prior_state(calc, ID, "siesta"))
    assert found["identity.prior_state"].severity == "info"
    assert f"{ID}.XV" in found["identity.prior_state"].message


def test_state_from_another_calculation_is_reported_and_warns(calc):
    """§ 5's third row, and the one the wrapper banner **cannot** say:
    `WARM-RESTART (silent)` has nothing beside it recording which run made
    the files. Warned rather than noted because one directory is meant to
    hold one job (`job-contracts.md § 2.1` Rule 1)."""
    (calc / "someone_else.XV").write_text("not ours\n")
    (calc / "someone_else.DM").write_text("not ours\n")
    found = _by_where(check_prior_state(calc, ID, "siesta"))
    assert found["identity.foreign_state"].severity == "warn"
    assert "someone_else.XV" in found["identity.foreign_state"].message
    # ...and it is NOT reported as this calculation's own state.
    assert "identity.prior_state" not in found


def test_the_two_kinds_of_state_are_told_apart_in_one_directory(calc):
    """Both rows at once -- ours to resume from, and a stranger's beside it.
    Reporting either alone would describe half the folder."""
    (calc / f"{ID}.XV").write_text("ours\n")
    (calc / "someone_else.XV").write_text("theirs\n")
    found = _by_where(check_prior_state(calc, ID, "siesta"))
    assert set(found) == {"identity.prior_state", "identity.foreign_state"}


def test_pyscf_foreign_state_is_found_by_its_own_suffixes(calc):
    """The suffixes are subtracted from the shipped inventory's filenames
    rather than listed again, which is what keeps this right for PySCF --
    where a 'suffix' is ``_optimized.xyz``, not an extension."""
    (calc / "other_job_optimized.xyz").write_text("coords\n")
    found = _by_where(check_prior_state(calc, ID, "pyscf"))
    assert "other_job_optimized.xyz" in found["identity.foreign_state"].message


# -- § 5's first row: a changed cell is a NO-OP, not a mismatch --------- #

_CELL = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]


def _write_xv(path, cell_ang):
    """A minimal .XV: three cell rows in BOHR, then an atom line."""
    bohr = 1.8897261254578281
    rows = "".join(
        f"  {v[0]*bohr:.9f} {v[1]*bohr:.9f} {v[2]*bohr:.9f}  0.0 0.0 0.0\n"
        for v in cell_ang)
    path.write_text(rows + "     1\n  1  1   0.0 0.0 0.0  0.0 0.0 0.0\n")


def test_a_cell_that_matches_says_nothing_extra(calc):
    _write_xv(calc / f"{ID}.XV", _CELL)
    found = _by_where(check_prior_state(calc, ID, "siesta", cell=_CELL))
    assert "identity.saved_cell" not in found


def test_a_changed_cell_is_reported_as_the_no_op_it_is(calc):
    """§ 5: *a `.XV` carries its own cell and **wins*** -- so widening the
    vacuum changes the deck and changes nothing about the run. The message
    has to say that, not 'mismatch', or the user reads it as an error and
    goes looking for the wrong thing."""
    _write_xv(calc / f"{ID}.XV", _CELL)
    wider = [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]
    msg = _by_where(check_prior_state(calc, ID, "siesta",
                                      cell=wider))["identity.saved_cell"]
    assert msg.severity == "warn"
    assert "the saved one wins" in msg.message
    assert "start this stage clean" in msg.message


def test_an_unreadable_xv_is_not_turned_into_a_cell_finding(calc):
    """This row exists to explain a silent no-op. Inventing a second failure
    out of a parse problem would be worse than staying quiet, and the bundle
    parsers already report unreadable files where that is their job."""
    (calc / f"{ID}.XV").write_text("not a real XV\n")
    found = _by_where(check_prior_state(calc, ID, "siesta", cell=_CELL))
    assert "identity.saved_cell" not in found
    assert "identity.prior_state" in found       # the file still IS state


def test_the_cell_row_needs_state_to_be_about(calc):
    """No .XV, nothing was written under any cell -- so there is nothing to
    report, however different the deck's cell is."""
    assert check_prior_state(calc, ID, "siesta", cell=_CELL) == []


def test_the_structure_witness_row_is_deliberately_not_here():
    """§ 5's second row -- *the structure moved under a saved description* --
    names **the reader, at preflight** in its 'who says it' column, not the
    surface. It is answered from the description's own witness
    (`stages.md § 6.3`: a reference plus formula + atom count), never from
    what happens to be lying in a directory.

    Pinned behaviourally rather than by grepping the source -- the first
    version of this test inspected the text and failed on the docstring that
    *explains* the absence, which is the brittleness such a test is made of.
    What actually holds is a namespace rule: everything this reports is an
    ``identity.*`` finding, so it cannot quietly grow a rule whose author is
    someone else."""
    import tempfile
    from pathlib import Path as _P
    with tempfile.TemporaryDirectory() as d:
        for name in (f"{ID}.XV", f"{ID}.DM", "someone_else.XV"):
            (_P(d) / name).write_text("state\n")
        wheres = {i.where for i in check_prior_state(_P(d), ID, "siesta")}
    assert wheres, "the fixture should have produced findings"
    assert all(w.startswith("identity.") for w in wheres), wheres
