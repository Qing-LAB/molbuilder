"""A reserved label is an ordinary label, plus one designated read.

The contract (`web/molview.md` § 6.6, `model/structure-annotations.md` § 2):

> A reserved meaning costs a **name** and **one accessor** — nothing else. The
> alternative, which this design rejects, is to give each special meaning its own
> field on the structure, its own kind of thing to filter by, its own key in the
> saved file, its own control in the panel, and a translation between the name
> the user sees and the name the field has.

`frozen_atoms` is the first reserved label: the atoms carrying it are held still by a
later calculation, which is why something downstream needs to collect them. That
is the ONLY thing it gets of its own — :attr:`Structure.frozen_atoms`, one place
that answers "which atoms carry this name" so no caller spells the name itself.

Until 2026-07-31 the backend had all five of the rejected things, and they had
already drifted into a live defect: `_expose_frozen_as_region` copied the frozen
set into the label store on `/api/selection/eval` and deliberately NOT on
`/api/selection/atoms`, because doing both would render an atom's frozen state
twice in the panel. Same structure, two answers, depending which route you asked.

These tests pin the properties that make that impossible to reintroduce, so they
are written against BEHAVIOUR — what a caller can observe — rather than against
the storage that currently produces it.
"""
from __future__ import annotations

import json

import pytest

from molbuilder.sidecars import molstruct
from molbuilder.structure import FROZEN_LABEL, Structure
from molbuilder.workingcopy_structure import StructureCodec


def _labelled() -> Structure:
    """Three atoms, one ordinary label and the reserved one, written the way
    every label is written."""
    s = Structure.from_xyz("3\n\nC 0 0 0\nO 1 0 0\nH 0 1 0\n")
    s.regions = {"L-electrode": [0]}
    s.frozen_atoms = [2, 1]
    return s


# --------------------------------------------------------------------- #
#  One store                                                            #
# --------------------------------------------------------------------- #

def test_the_reserved_label_lives_in_the_label_store_with_every_other_label():
    """"It is just another label, treated equally in format, storage and UI."

    Not "reachable from" the store — IN it, indistinguishable from a label a
    user typed. Anything that walks labels therefore sees it with no case.
    """
    s = _labelled()

    assert s.regions == {"L-electrode": [0], FROZEN_LABEL: [1, 2]}
    assert sorted(s.channels()) == ["L-electrode", FROZEN_LABEL]
    kinds = {name: ch.kind for name, ch in s.channels().items()}
    assert kinds[FROZEN_LABEL] == kinds["L-electrode"] == "tag", (
        f"the reserved label is a different KIND of thing: {kinds}"
    )


def test_a_walk_over_the_labels_needs_no_case_for_the_reserved_one():
    """The payoff, stated as the thing a caller does: ask each atom what it
    carries, and the reserved label is in the answer like any other name."""
    s = _labelled()

    assert s.atom_annotations(0) == {"L-electrode": True}
    assert s.atom_annotations(1) == {FROZEN_LABEL: True}
    assert s.atom_annotations(2) == {FROZEN_LABEL: True}


def test_there_is_no_second_place_the_fact_is_kept():
    """The whole defect in one assertion: if a second store existed, the two
    could disagree — and the only way to be sure they cannot is for the second
    one not to exist. Checked on the serialised form, where a second store would
    have to surface as a second key."""
    meta = _labelled().metadata_to_dict()

    assert "frozen_atoms" not in meta, (
        f"a second home for the fact the label store already holds: {sorted(meta)}"
    )
    assert meta["regions"][FROZEN_LABEL] == [1, 2]


# --------------------------------------------------------------------- #
#  One designated read                                                  #
# --------------------------------------------------------------------- #

def test_the_accessor_is_a_cut_of_the_store_and_never_a_copy_of_it():
    """"A cut of the label store, not a second home." So it cannot be stale:
    edit the label through the ordinary door and the accessor already agrees,
    because there is nothing to keep in step.
    """
    s = _labelled()
    assert s.frozen_atoms == [1, 2]

    s.regions[FROZEN_LABEL] = [0]           # an ordinary label write
    assert s.frozen_atoms == [0], "the accessor answered from a copy"

    del s.regions[FROZEN_LABEL]
    assert s.frozen_atoms == []


def test_the_accessor_hands_back_something_a_caller_cannot_edit_the_store_with():
    """A read is a read. Handing back the live list would make every caller a
    writer by accident — the same class of defect as a second store, since the
    edit would land without going through the door that owns the name."""
    s = _labelled()

    got = s.frozen_atoms
    got.append(0)

    assert s.frozen_atoms == [1, 2], "a read mutated the structure"


def test_writing_through_the_accessor_writes_an_ordinary_label():
    """The write side of the same door: the one place the name is spelled. What
    it produces is indistinguishable from a label written any other way."""
    s = Structure.from_xyz("3\n\nC 0 0 0\nO 1 0 0\nH 0 1 0\n")

    s.frozen_atoms = [2, 1, 1]              # unsorted, duplicated

    assert s.regions == {FROZEN_LABEL: [1, 2]}, "not normalised like other labels"


def test_clearing_the_label_removes_it_rather_than_storing_an_empty_one():
    """"Carries no label" and "carries an empty label" must not both exist, or
    every reader needs to know which one means what."""
    s = _labelled()

    s.frozen_atoms = []

    assert FROZEN_LABEL not in s.regions
    assert s.frozen_atoms == []


def test_the_name_has_exactly_one_spelling():
    """A reserved meaning costs a name — ONE. The label kept the name it always
    had; what the second storage added was a SECOND spelling for the same fact,
    a `frozen` flag channel synthesised beside the label, and an alias between
    them at every boundary that touched both."""
    s = _labelled()

    assert FROZEN_LABEL in s.regions
    assert FROZEN_LABEL in s.metadata_to_dict()["regions"]
    assert FROZEN_LABEL in s.channels()


# --------------------------------------------------------------------- #
#  Every label operation applies to it identically                      #
# --------------------------------------------------------------------- #

def test_deleting_atoms_remaps_the_reserved_label_by_the_same_rule():
    """§ 2.1: a channel that is not remapped silently points at the wrong
    atoms — and for THIS label that means the wrong atoms are held still in
    someone's relaxation. It used to need its own line in the remap, in lockstep
    with the labels' line."""
    from molbuilder.modify import delete_atoms
    s = Structure.from_xyz("4\n\nC 0 0 0\nO 1 0 0\nH 0 1 0\nN 0 0 1\n")
    s.regions = {"keep": [0, 3]}
    s.frozen_atoms = [1, 3]

    after = delete_atoms(s, [0])            # everything shifts down by one

    assert after.regions["keep"] == [2]
    assert after.frozen_atoms == [0, 2], (
        "the reserved label did not follow the atoms it was on"
    )


def test_concatenating_offsets_the_reserved_label_by_the_same_rule():
    a = Structure.from_xyz("2\n\nC 0 0 0\nO 1 0 0\n")
    a.frozen_atoms = [1]
    b = Structure.from_xyz("2\n\nH 0 1 0\nN 0 0 1\n")
    b.regions = {"second": [0]}
    b.frozen_atoms = [0]

    merged = Structure.concat([a, b])

    assert merged.frozen_atoms == [1, 2]
    assert merged.regions["second"] == [2]


def test_copying_carries_it_without_being_told_about_it():
    """`copy()` names no field for it — it carries the label store, and the
    reserved label is in there."""
    s = _labelled()
    assert s.copy().frozen_atoms == [1, 2]
    assert s.translated([1.0, 0.0, 0.0]).frozen_atoms == [1, 2]


def test_it_is_validated_by_the_same_validator_as_every_other_label():
    """One validator, because one store. An out-of-range index is refused with
    the same message a bad `L-electrode` index gets."""
    s = Structure.from_xyz("2\n\nC 0 0 0\nO 1 0 0\n")

    with pytest.raises(ValueError, match=r"out of range"):
        s.regions = {FROZEN_LABEL: [5]}
        s.__post_init__()


def test_an_extra_channel_may_not_shadow_it_any_more_than_another_label():
    """`annotations` holds channels BESIDE the labels; a name already taken by a
    label is refused — and the reserved label is a label, so it is covered by
    that one rule rather than by a clause of its own."""
    s = _labelled()

    with pytest.raises(ValueError, match=r"already a label"):
        s.set_channel(FROZEN_LABEL, __import__(
            "molbuilder.structure", fromlist=["AtomChannel"]).AtomChannel("tag", [0]))


# --------------------------------------------------------------------- #
#  The saved file                                                       #
# --------------------------------------------------------------------- #

def test_the_saved_file_keeps_one_key_for_labels(tmp_path):
    """"Its own key in the saved file" is one of the five things the design
    rejects. A saved structure has a label store and nothing beside it."""
    target = tmp_path / "labelled.xyz"
    StructureCodec().write(_labelled(), target)

    saved = json.loads(molstruct.sidecar_path_for(target).read_text("utf-8"))

    assert "frozen_atoms" not in saved, f"a second key: {sorted(saved)}"
    assert saved["regions"][FROZEN_LABEL] == [1, 2]
    assert saved["schema_version"] == molstruct.SCHEMA_VERSION


def test_a_saved_structure_comes_back_the_same(tmp_path):
    target = tmp_path / "round.xyz"
    StructureCodec().write(_labelled(), target)

    back = StructureCodec().read(target)

    assert back.regions == {"L-electrode": [0], FROZEN_LABEL: [1, 2]}
    assert back.frozen_atoms == [1, 2]


def test_a_file_written_before_the_fold_still_opens(tmp_path):
    """Schema 6 kept the reserved label in a top-level key. Those files are on
    disk in real projects; they open, and what they held lands in the label
    store where the rest of the application now looks for it."""
    xyz = tmp_path / "old.xyz"
    xyz.write_text("3\n\nC 0 0 0\nO 1 0 0\nH 0 1 0\n", encoding="utf-8")
    molstruct.sidecar_path_for(xyz).write_text(json.dumps({
        "schema_version":  6,
        "n_atoms_total":   3,
        "structure_hash":  "0" * 64,
        "regions":         {"L-electrode": [0]},
        "frozen_atoms":    [1, 2],          # where schema 6 put it
        "created_by":      "molbuilder",
        "created_at":      "2026-01-01T00:00:00Z",
    }), encoding="utf-8")

    back = StructureCodec().read(xyz)

    assert back.frozen_atoms == [1, 2], "a schema-6 file lost its frozen atoms"
    assert back.regions == {"L-electrode": [0], FROZEN_LABEL: [1, 2]}


def test_the_designated_read_works_on_a_saved_payload_of_either_schema():
    """The same "one place spells the name" rule, one level down: code holding a
    sidecar dict rather than a Structure asks `molstruct.frozen_atoms` and does
    not need to know which schema wrote the file."""
    assert molstruct.frozen_atoms({"regions": {FROZEN_LABEL: [1, 2]}}) == [1, 2]
    assert molstruct.frozen_atoms({"frozen_atoms": [1, 2]}) == [1, 2]
    assert molstruct.frozen_atoms({"regions": {"L-electrode": [0]}}) == []
    assert molstruct.frozen_atoms(None) == []


# --------------------------------------------------------------------- #
#  Every route gives the same answer                                    #
# --------------------------------------------------------------------- #

@pytest.fixture
def served(tmp_path):
    """A labelled structure inside the file-picker root, and a client. The root
    patching follows `test_selection_blueprint.py` -- these routes resolve a
    path against the configured roots, so a bare tmp path is refused."""
    pytest.importorskip("flask")
    from molbuilder import diagnostics
    from molbuilder.web.app import create_app

    xyz = tmp_path / "labelled.xyz"
    StructureCodec().write(_labelled(), xyz)

    caps = diagnostics.Capabilities(runtime_config={}, conda_binary=None)
    cls = type(caps)
    old = cls.file_picker_roots
    cls.file_picker_roots = lambda self: ((tmp_path.resolve(), "projects"),)
    diagnostics.set_capabilities(caps)
    try:
        yield create_app(config={}).test_client(), str(xyz.resolve())
    finally:
        cls.file_picker_roots = old
        diagnostics.reset_capabilities()


def test_an_atom_carries_the_reserved_label_once_and_only_as_a_label(served):
    """THE regression this closes. The atom list used to carry the fact twice --
    a `frozen_atoms` tag AND an `is_frozen` flag -- which double-rendered in the
    panel, and the workaround was to withhold the tag on this route only.

    One representation: the label, in the list of labels.
    """
    client, path = served
    answer = client.post("/api/selection/atoms",
                         json={"structure_path": path}).get_json()
    assert answer.get("atoms"), answer
    rows = answer["atoms"]

    assert "is_frozen" not in rows[1], (
        f"the fact is on the atom twice: {sorted(rows[1])}"
    )
    assert rows[1]["regions"] == [FROZEN_LABEL]
    assert rows[0]["regions"] == ["L-electrode"]


def test_filtering_by_the_reserved_label_needs_no_case_of_its_own(served):
    """"Filtering for frozen atoms needs no case of its own" (§ 9.5). The
    by-label rule that finds `L-electrode` finds this with the same rule and the
    same name -- nothing synthetic injected onto the structure first."""
    client, path = served

    def by_label(name):
        answer = client.post("/api/selection/eval", json={
            "structure_path": path,
            "rule": {"op": "by_region", "name": name},
        }).get_json()
        assert answer.get("ok"), answer
        return answer["selected_indices"]

    assert by_label(FROZEN_LABEL) == [1, 2]
    assert by_label("L-electrode") == [0]


def test_the_two_routes_cannot_disagree(served):
    """The property the conditional broke, asserted directly: what the atom list
    says an atom carries, and what filtering by that name selects, are the same
    set. No route sees a structure another route does not."""
    client, path = served

    rows = client.post("/api/selection/atoms",
                       json={"structure_path": path}).get_json()["atoms"]
    from_rows = [r["index"] for r in rows if FROZEN_LABEL in r["regions"]]

    from_rule = client.post("/api/selection/eval", json={
        "structure_path": path,
        "rule": {"op": "by_region", "name": FROZEN_LABEL},
    }).get_json()["selected_indices"]

    assert from_rows == from_rule == [1, 2]
