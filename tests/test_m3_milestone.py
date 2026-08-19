"""M3 — P3's milestone, asserted as the plan states it.

``docs/archive/2026-08-19-staged-runs-implementation-plan.md`` P3:

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
from molbuilder.structure import Structure
from molbuilder.task import Stage
from molbuilder.validation.identity import warm_files_present
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


def _answers_in(deck: str):
    """``{key: ".true."/".false."}`` -- what the deck actually INSTRUCTS.

    Added 2026-08-18.  Every member is now written for both answers, so which
    keys appear no longer distinguishes a continuing deck from a clean one;
    the answer beside each key does.
    """
    return {ln.split()[0]: ln.split()[1] for ln in deck.splitlines()
            if ln.split() and ln.split()[0] in SIESTA_RESTART_GROUP.keys}


# --------------------------------------------------------------------- #
#  M3, first half — a real two-stage description                        #
# --------------------------------------------------------------------- #

def test_a_two_stage_ladder_answers_the_group_both_ways(h2o):
    """**One test, both halves**, exactly as the plan words it: the failure
    mode is that they disagree, so asserting them apart would let precisely
    that through.

    A ladder, not two single configs -- the two decks come out of ONE
    ladder walk against one template, which is the
    path a user actually takes.

    **The clean half reads `.false.`, not absence** (2026-08-18).  This was
    ``..._renders_all_of_the_group_and_none_of_it`` and asserted the coarse
    deck carried NO member, which pinned the premise that a key left out is a
    key not honoured.  SIESTA 5.4.2 reads ``<SystemLabel>.DM`` whenever the
    file is there regardless, so *none of it* instructed nothing and the first
    rung continued from whatever the directory held."""
    template = SiestaConfig(system_label=ID, relax_type="CG")
    decks = _live_ladder_decks(h2o, template, [
        Stage(name="coarse", overrides={"restart": "clean",
                                        "mesh_cutoff": 150.0}),
        Stage(name="tight",  overrides={"restart": "continue",
                                        "mesh_cutoff": 300.0}),
    ])

    # Keyed by FILENAME (`<label>_<stage>.fdf`), not by stage name -- which
    # is itself the § 3.2 rule that anything a stage produced carries
    # `_<stage>` while anything the engine resumes from carries the bare id.
    coarse = _answers_in(decks[f"{ID}_01_coarse.fdf"])
    tight = _answers_in(decks[f"{ID}_02_tight.fdf"])
    assert sorted(coarse) == sorted(tight) == sorted(SIESTA_RESTART_GROUP.keys)
    assert set(coarse.values()) == {".false."}     # the rung that starts fresh
    assert set(tight.values()) == {".true."}       # the rung that continues


def test_the_two_stages_are_otherwise_the_same_calculation(h2o):
    """The ladder is one calculation, so both decks carry ONE SystemLabel --
    which is what lets `tight` find what `coarse` left (`run-identity.md`
    § 1). If this drifted, the group above would be set correctly and still
    resume from nothing."""
    template = SiestaConfig(system_label=ID, relax_type="CG")
    decks = _live_ladder_decks(h2o, template, [
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
#                                                                       #
#  ``check_overwrite`` was RETIRED at U14 (2026-08-12): it implemented  #
#  "refuse unless the user says overwrite", the rule run-identity.md    #
#  § 6 softened away on 2026-08-08 ("Warn, do not refuse") — which is   #
#  why it had zero callers.  What § 6 still requires is pinned below on #
#  its survivors: ``warm_files_present`` says what is there, the        #
#  surface asks (test_prep_bench_fold pins the ask), and NOTHING is     #
#  ever renamed or touched to make room.                                #
# --------------------------------------------------------------------- #


def test_a_fresh_directory_reports_nothing_underway(tmp_path):
    """The common path stays quiet. § 6 is about a SECOND produce."""
    assert warm_files_present(tmp_path, ID, "siesta") == []


def test_warm_files_are_reported_by_name(tmp_path):
    """§ 6's surviving rule: before writing, say WHAT is in the folder."""
    (tmp_path / f"{ID}.XV").write_text("relaxed coords\n")
    assert warm_files_present(tmp_path, ID, "siesta") == [f"{ID}.XV"]


def test_saying_never_renames_and_never_touches_the_warm_files(tmp_path):
    """*"and never renames"* -- the non-negotiable half of § 6's sentence:
    the warm files are what the next run continues from, so "make the
    name unique" would throw away the geometry the user is keeping.
    Asserted on disk, not on a message."""
    xv = tmp_path / f"{ID}.XV"
    xv.write_text("relaxed coords\n")
    before = sorted(p.name for p in tmp_path.iterdir())
    warm_files_present(tmp_path, ID, "siesta")
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert xv.read_text() == "relaxed coords\n"
    assert not list(tmp_path.glob("*-restart-aside-*"))


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


from _ladder_helpers import _live_ladder_decks  # noqa: E402  (U20: the one renderer)
