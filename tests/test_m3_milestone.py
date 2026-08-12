"""M3 — P3's milestone, asserted as the plan states it.

``docs/execution/staged-runs-implementation-plan.md`` P3:

  > A two-stage description whose second stage continues renders **every**
  > bound parameter set, and a stage set to `clean` renders **none** —
  > asserted together, because the failure mode is that they disagree. A
  > second produce into a folder that already holds warm files **refuses
  > unless told to overwrite, and never renames**.

Both halves live here rather than in the unit files, so "did M3 pass" is one
file and not an archaeology exercise.
"""
from __future__ import annotations

import pytest

from molbuilder.config.siesta import SIESTA_RESTART_GROUP, SiestaConfig
from molbuilder.issues import ValidationError
from molbuilder.siesta.input import render_siesta_stage_fdfs
from molbuilder.structure import Structure
from molbuilder.task import Stage
from molbuilder.validation.identity import check_overwrite
from molbuilder.validation.task import refuse_on_error


ID = "BDT_Au_relax_C6H4S2Au38"


@pytest.fixture
def h2o():
    return Structure(elements=["O", "H", "H"],
                     positions=[[0.0, 0.0, 0.0],
                                [0.76, 0.59, 0.0],
                                [-0.76, 0.59, 0.0]])


def _keys_in(deck: str):
    """The group's members that this deck actually EMITS, as keys.

    Read as keys and not as substrings: the explanatory comment beside the
    block names the very keys being looked for, so a substring test passes
    for the wrong reason.  (That trap caught the first version of unit 4's
    test, which is why it is spelled out here too.)
    """
    return sorted(ln.split()[0] for ln in deck.splitlines()
                  if ln.split() and ln.split()[0] in SIESTA_RESTART_GROUP.keys)


# --------------------------------------------------------------------- #
#  M3, first half — a real two-stage description                        #
# --------------------------------------------------------------------- #

def test_a_two_stage_ladder_renders_all_of_the_group_and_none_of_it(h2o):
    """**One test, both halves**, exactly as the plan words it: the failure
    mode is that they disagree, so asserting them apart would let precisely
    that through.

    A ladder, not two single configs -- the two decks come out of ONE
    ``render_siesta_stage_fdfs`` call against one template, which is the
    path a user actually takes."""
    template = SiestaConfig(system_label=ID, relax_type="CG")
    decks = render_siesta_stage_fdfs(h2o, template, [
        Stage(name="coarse", overrides={"restart": "clean",
                                        "mesh_cutoff": 150.0}),
        Stage(name="tight",  overrides={"restart": "continue",
                                        "mesh_cutoff": 300.0}),
    ])

    # Keyed by FILENAME (`<label>_<stage>.fdf`), not by stage name -- which
    # is itself the § 3.2 rule that anything a stage produced carries
    # `_<stage>` while anything the engine resumes from carries the bare id.
    assert _keys_in(decks[f"{ID}_01_coarse.fdf"]) == []
    assert _keys_in(decks[f"{ID}_02_tight.fdf"]) == \
        sorted(SIESTA_RESTART_GROUP.keys)


def test_the_two_stages_are_otherwise_the_same_calculation(h2o):
    """The ladder is one calculation, so both decks carry ONE SystemLabel --
    which is what lets `tight` find what `coarse` left (`run-identity.md`
    § 1). If this drifted, the group above would be set correctly and still
    resume from nothing."""
    template = SiestaConfig(system_label=ID, relax_type="CG")
    decks = render_siesta_stage_fdfs(h2o, template, [
        Stage(name="coarse", overrides={"restart": "clean"}),
        Stage(name="tight",  overrides={"restart": "continue"}),
    ])
    # Read as a VALUE, not pinned as text: the emitter pads the key, and this
    # test is about the label being one label, not about its column width.
    labels = {ln.split()[1] for deck in decks.values()
              for ln in deck.splitlines()
              if ln.split()[:1] == ["SystemLabel"]}
    assert labels == {ID}


# --------------------------------------------------------------------- #
#  M3, second half — producing twice into one directory                 #
# --------------------------------------------------------------------- #

def test_producing_into_a_fresh_directory_is_not_refused(tmp_path):
    """The common path stays open. § 6 is about a SECOND produce."""
    assert check_overwrite(tmp_path, ID, "siesta") == []


def test_a_second_produce_over_warm_files_refuses(tmp_path):
    """§ 6: *refuse unless the user says overwrite*."""
    (tmp_path / f"{ID}.XV").write_text("relaxed coords\n")
    issues = check_overwrite(tmp_path, ID, "siesta")
    assert [i.severity for i in issues] == ["error"]
    with pytest.raises(ValidationError):
        refuse_on_error(issues)


def test_told_to_overwrite_it_proceeds(tmp_path):
    (tmp_path / f"{ID}.XV").write_text("relaxed coords\n")
    assert check_overwrite(tmp_path, ID, "siesta", overwrite=True) == []


def test_it_never_renames_and_never_touches_the_warm_files(tmp_path):
    """*"and never renames"* -- the other half of § 6's sentence, and the
    reason it is there: the warm files are what the next run continues from,
    so "make the name unique" would throw them away. Asserted on disk, not
    on the message: this function must not move anything, with or without
    permission."""
    xv = tmp_path / f"{ID}.XV"
    xv.write_text("relaxed coords\n")
    before = sorted(p.name for p in tmp_path.iterdir())
    check_overwrite(tmp_path, ID, "siesta")
    check_overwrite(tmp_path, ID, "siesta", overwrite=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert xv.read_text() == "relaxed coords\n"
    assert not list(tmp_path.glob("*-restart-aside-*"))


def test_the_refusal_says_the_files_are_safe(tmp_path):
    """A refusal that reads as "you are about to lose your geometry" would
    make users pass overwrite in a panic. The decks are what gets replaced;
    § 6 is explicit that rewriting them does not touch the warm files."""
    (tmp_path / f"{ID}.DM").write_text("density\n")
    msg = check_overwrite(tmp_path, ID, "siesta")[0].message
    assert "NOT touched" in msg
    assert "nothing is renamed" in msg


# ``test_the_cli_produce_path_refuses_a_second_time`` was RETIRED 2026-08-11.
#
# It was a strict xfail recording a real defect: `molbuilder fdf` called no
# producer-side overwrite check, so a second produce wrote over an existing deck
# beside a relaxed ``.XV`` silently.  **The verb was deleted, so the defect went
# with it** -- and a strict xfail that starts passing is itself a failure, which
# is the mechanism working: it refused to be quietly right.
#
# The CHECK it guarded (``check_overwrite``) is still asserted directly above.
# Whether `prep` re-rendering a deck in place needs the same guard is a
# DIFFERENT question -- a deck is derived from the description now, and
# `project-layout.md` § 1.5 makes each ATTEMPT immutable rather than each deck.
# Reinstating this against the new path would assert an answer nobody has made.
