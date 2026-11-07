"""What an edit does to the record a real run left on its structure
(`web/molview.md` § 8.4a, `model/structure.md` § 2.2a; `process/testing.md`
§ 0, tier 3, and § 6).

A structure saved from a finished run carries that run's record,
``info.calculation``.  An edit VOIDS it, and the way it says so is a flag ON the
record -- `structure_modified` for a geometry or cell op -- which
`electronic_state._recorded` reads as *no longer the structure that run came out
of*.  A rebuild that drops ``info`` deletes the thing that flag marks.  Voiding
is a MARK on the record; a record that is gone cannot carry one.

The record here is real: a formate ion a real SIESTA relaxed, saved as a pair
by `molbuilder xv2xyz --from-run` (`support.real_runs.charged_formate_relaxed`),
a junction's for the sort -- and every mark is an op's, never one set by hand.
"""
from __future__ import annotations

import copy as _copy

import numpy as np
import pytest

from _road import conda_hook, env_available
from molbuilder.modify import (add_atom, append_structure, delete_atoms,
                               orient_along_axis, rotate_around_axis,
                               translate)
from molbuilder.periodicity_gate import apply_edit
from molbuilder.structure import Structure
from molbuilder.workingcopy_structure import StructureCodec

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]

_CELL = [[9., 0, 0], [0, 9., 0], [0, 0, 9.]]


@pytest.fixture
def recorded(real_charged_formate):
    """The pair a real relaxation was saved as, read as every reader reads
    it -- carrying the record, or none of this shows anything."""
    s = StructureCodec().load(real_charged_formate.pair)
    assert s.info["calculation"]["contract"], s.info
    return s


def _fragment():
    return Structure(elements=["C"], positions=np.array([[4., 4, 4]]))


#: Each atom edit, what it makes of a structure -- and whether it is an edit: a
#: copy changes nothing, so it marks nothing.
_ATOM_EDITS = {
    "copy":             (lambda x: x.copy(), False),
    "delete_atoms":     (lambda x: delete_atoms(x, [2]), True),
    "add_atom":         (lambda x: add_atom(x, "C", 0, [2., 2., 2.]), True),
    "translate":        (lambda x: translate(x, [1., 0, 0]), True),
    "translate_subset": (lambda x: translate(x, [1., 0, 0], indices=[0]), True),
    "rotate":           (lambda x: rotate_around_axis(x, "z", 30.0), True),
    "orient":           (lambda x: orient_along_axis(x, [0, 1], "z"), True),
    "append":           (lambda x: append_structure(x, _fragment())[0], True),
}


@pytest.mark.parametrize("name", list(_ATOM_EDITS))
def test_an_atom_edit_keeps_the_record_and_marks_it_outdated(recorded, name):
    """`molbuilder modify IN OUT --delete N` wrote an edited pair still
    carrying the relaxation's mesh cutoff and k-mesh with no staleness mark,
    until the backend marked what MolView marks.  So every edit: the record
    whole, the mark on the copy it returns, and its source left unmarked.

    THE SILENT FAILURES: an op rebuilding the structure without ``info``, the
    flag then having nothing to mark; an op that does not mark, so a later
    calculation starts from settings converged for a structure that is no
    longer there; the mark reaching the source."""
    was = _copy.deepcopy(recorded.info)
    edit, marks = _ATOM_EDITS[name]
    out = edit(recorded)
    assert out.info["calculation"]["contract"] == was["calculation"]["contract"]
    assert bool(out.info["calculation"].get("structure_modified")) is marks
    assert recorded.info == was, "the edit marked its SOURCE too"


def test_the_mark_rides_what_is_derived_after_it_and_voids_the_record(
        recorded, tmp_path):
    """The flag is the point: a real op sets it, it rides a copy no op marks
    and the pair written to disk, and its reader then takes nothing from the
    record -- the charge of -1 the same structure's record gave before the
    edit.  The silent failure: a calculation of an edited structure starting
    from what the run recorded of another."""
    from molbuilder.electronic_state import _recorded
    assert _recorded(recorded)["net_charge"].value == -1
    derived = delete_atoms(recorded, [3]).replace(title="derived")
    assert derived.info["calculation"]["structure_modified"] is True
    StructureCodec().write(derived, tmp_path / "edited.xyz")
    back = StructureCodec().load(tmp_path / "edited.xyz")
    assert back.info["calculation"]["structure_modified"] is True
    assert back.info["calculation"]["contract"]["net_charge"] == -1
    assert _recorded(back) is None


@pytest.mark.parametrize("op, payload", [
    ("vacuum",      [4.0, 4.0, 4.0]),
    ("axis_kind",   ["periodic", "periodic", "isolated"]),
    ("cell",        _CELL),
    ("box_corner",  [-1.0, -1.0, -1.0]),
    ("block",       {"cell": _CELL, "box_corner": None,
                     "axis_kind": ["periodic", "periodic", "periodic"],
                     "vacuum": None}),
])
def test_a_box_edit_marks_the_record_outdated(recorded, op, payload):
    """Every op of the periodicity door, not just the geometry ones: mesh
    cutoff is a grid density over the CELL and the k-mesh samples the
    reciprocal cell, so changing the box invalidates the settings as much as
    moving an atom does.  The door marks and the answer carries it."""
    was = _copy.deepcopy(recorded.info)
    out, _notices = apply_edit(recorded, op, payload)
    assert out.info["calculation"]["structure_modified"] is True
    assert out.info["calculation"]["contract"] == was["calculation"]["contract"]
    assert out.info["calculation"].get("labels_modified") is None, \
        "a box edit claimed the labels moved"
    assert recorded.info == was, "the edit marked its SOURCE too"


def test_a_refused_box_edit_marks_nothing(recorded):
    """The mark rides the returned copy, and a refusal returns none --
    `apply_edit` marks before its per-op branches, safe only because every
    refusal raises.  Answering with the copy on a rejected edit would leave
    the pair a staleness flag for an edit that never happened."""
    was = _copy.deepcopy(recorded.info)
    with pytest.raises(ValueError):
        apply_edit(recorded, "axis_kind", ["periodic", "sideways", "isolated"])
    assert recorded.info == was


def test_a_reorder_is_not_an_edit(real_bridged_chain):
    """The transport sort derives through the same door but changes no
    geometry, so marking there would warn about every composed junction: a
    real junction's record, the atoms listed last to first as a person's file
    may list them, sorted into their order."""
    from molbuilder.transport.sort import SORT_TRANSPORT, sort_by
    s = StructureCodec().load(real_bridged_chain.pair)
    backwards = s.take(list(range(s.n_atoms))[::-1])
    out = sort_by(backwards, SORT_TRANSPORT).structure
    assert out.atom_order is not None, "nothing moved: this shows no reorder"
    assert "structure_modified" not in out.info["calculation"]
    assert out.info["calculation"]["contract"] == \
        s.info["calculation"]["contract"]


def test_append_takes_the_record_from_the_structure_appended_to(recorded):
    """Not from whichever structure carries the cell -- `concat` picks the
    lattice that way and `info` rode along, so a base with no box lost its
    record to the incoming one: the cell cleared by its own op, a fragment
    with a box of its own appended."""
    base, _notices = apply_edit(recorded, "cell", None)
    assert base.cell is None
    other = _fragment()
    other.cell = np.diag([9., 9., 9.])
    out, _notes = append_structure(base, other)
    assert out.info["calculation"]["contract"] == \
        recorded.info["calculation"]["contract"]


# --------------------------------------------------------------------- #
#  A derived copy is a copy (`model/structure.md` § 2.2a)                #
# --------------------------------------------------------------------- #

def test_writing_to_a_derived_copy_does_not_reach_the_record(recorded):
    """``dataclasses.replace`` re-passes the mutable fields by reference, so a
    copy shared the ``info`` dict with its source: writing to one wrote to the
    other."""
    was = _copy.deepcopy(recorded.info)
    out = recorded.replace(title="derived")
    out.info["calculation"]["contract"]["mesh_cutoff"] = 1.0
    assert recorded.info == was, "the nested info dict was shared, not copied"


def test_the_record_travels_through_a_change_nobody_named_it_in(recorded):
    """§ 2.2a: it travels; a strip is explicit, never a field a rebuild
    forgot."""
    was = _copy.deepcopy(recorded.info)
    assert recorded.replace(positions=recorded.positions + 1.0).info == was
    assert recorded.replace(info={}).info == {}
    assert recorded.info == was, "stripping the copy emptied the source"
