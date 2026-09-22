"""`Structure.replace` — deriving a copy without the frozen label coming back.

The trap this closes
====================

``frozen_atoms`` is deliberately two things at once: an ``__init__`` field, so
``Structure(..., frozen_atoms=[...])`` reaches the one place that spells the
reserved label, and a derived READ of ``regions[FROZEN_LABEL]``, so "which
atoms are held still" is answered in one place.

``dataclasses.replace`` re-passes every field by reading it off the instance.
Reading ``frozen_atoms`` goes through the property, which returns a list and
never ``None`` — so the setter's documented "``None`` says nothing about it"
can never be expressed, and an explicit new ``regions`` silently gets the OLD
frozen set stamped into it.  Measured 2026-08-22:

    dataclasses.replace(s, regions={"electrode_L": [1]})
    -> {"electrode_L": [1], "frozen_atoms": [0]}

Nothing in production passed ``regions=`` to ``replace`` at the time, which is
why this was scheduled rather than urgent — and written down rather than left
for the next person to rediscover from a structure that would not unfreeze.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from molbuilder.structure import FROZEN_LABEL, Structure


@pytest.fixture
def held():
    """Five atoms: one frozen, two in an electrode region."""
    return Structure(elements=["C", "H", "H", "H", "H"],
                     positions=np.zeros((5, 3)),
                     regions={FROZEN_LABEL: [0], "electrode_L": [1, 2]})


class TestStatingRegionsStatesTheWholeStore:

    def test_the_old_frozen_set_does_not_come_back(self, held):
        got = held.replace(regions={"electrode_L": [1]})
        assert got.regions == {"electrode_L": [1]}
        assert got.frozen_atoms == []

    def test_unfreezing_by_rewriting_regions_actually_unfreezes(self, held):
        """The failure the trap produces, stated as the user would meet it."""
        assert held.frozen_atoms == [0]
        assert held.replace(regions={}).frozen_atoms == []

    def test_frozen_atoms_still_wins_when_it_is_stated(self, held):
        """Not "regions always wins" — "the caller gets what they asked for"."""
        got = held.replace(regions={"electrode_L": [1]}, frozen_atoms=[3])
        assert got.regions == {"electrode_L": [1], FROZEN_LABEL: [3]}

    def test_replacing_something_else_leaves_regions_alone(self, held):
        got = held.replace(positions=np.ones((5, 3)))
        assert got.regions == held.regions
        assert got.frozen_atoms == [0]

    def test_the_other_fields_survive_the_rebuild(self, held):
        """The regions path rebuilds from fields rather than delegating, so
        prove nothing is dropped on the way."""
        s = dataclasses.replace(held, title="a name")
        got = s.replace(regions={"electrode_L": [1]})
        assert got.title == "a name"
        assert list(got.elements) == ["C", "H", "H", "H", "H"]
        assert got.positions.shape == (5, 3)


class TestTheInterpreterHook:
    """WHICH stdlib helper dispatches through ``__replace__`` — and which
    does not, which this class asserted backwards until 2026-09-22.

    It claimed `dataclasses.replace` would route through the hook on 3.13+,
    "so the door becomes automatic on a newer interpreter".  Checked against
    the stdlib source: `dataclasses.replace` ends
    `return obj.__class__(**changes)` and never mentions `__replace__`, on
    any version.  The 3.13 addition is `copy.replace`, a different helper.

    The consequence was not academic.  The trap test below was
    `skipif(>=3.13)` on that belief, so a 3.13 upgrade would have SILENCED
    the only guard on the aliasing while the aliasing was still there.
    """

    def test_the_replace_hook_is_installed(self):
        """For `copy.replace` (3.13+), which is the helper that honours it."""
        assert Structure.__replace__ is Structure.replace

    # `test_the_stdlib_dataclasses_helper_never_routes_through_the_hook`
    # was RETIRED the day it was written (2026-09-22).  It asserted that the
    # string "__replace__" does not appear in
    # `inspect.getsource(dataclasses.replace)` -- a grep over SOURCE TEXT,
    # which `process/testing.md` disqualifies unless the goal has no other
    # check.  It has one: the test below demonstrates the behaviour itself,
    # on a real Structure, and would fail the moment the stdlib started
    # dispatching.  The source check only restated it less directly.

    def test_the_stdlib_helper_carries_the_trap_on_every_version(self, held):
        """NOT skipped on 3.13 — there is no version where this stops being
        true, so `Structure.replace` is the only correct door, permanently.

        Kept as a live assertion rather than a comment because the whole
        reason `replace()` exists is that this alternative looks equivalent
        and is not."""
        got = dataclasses.replace(held, regions={"electrode_L": [1]})
        assert got.regions == {"electrode_L": [1], FROZEN_LABEL: [0]}


class TestTheDerivedCopyIsACopy:
    """The door is `copy()` plus the changes, so nothing it carried can be
    written through.

    ``dataclasses.replace`` re-passes the mutable fields BY REFERENCE, so
    the structure it returned shared ``positions``, ``cell``,
    ``cell_origin`` and the ``info`` dict with its source — writing to one
    wrote to the other.  That is what every hand-listed rebuild in the tree
    was working around with its own ``.copy()`` calls, and enumerating
    fields in order to copy them is how ``cell_origin`` and ``info`` came
    to be left out of four of those lists (`model/structure.md` § 2.2a).
    """

    @pytest.fixture
    def carrying(self):
        return Structure(
            elements=["C", "H"], positions=np.array([[0.0, 0.0, 0.0],
                                                     [0.0, 0.0, 1.1]]),
            cell=np.diag([8.0, 8.0, 8.0]),
            cell_origin=np.array([-1.0, -1.0, -1.0]),
            info={"calculation": {"contract": {"mesh_cutoff_ry": 400}}})

    def test_writing_to_the_derived_copy_does_not_reach_the_source(
            self, carrying):
        out = carrying.replace(title="derived")
        out.positions[0, 0] = 7.0
        out.cell[0, 0] = 99.0
        out.cell_origin[0] = 5.0
        out.info["calculation"]["contract"]["mesh_cutoff_ry"] = 1
        assert carrying.positions[0, 0] == 0.0
        assert carrying.cell[0, 0] == 8.0
        assert carrying.cell_origin[0] == -1.0
        assert (carrying.info["calculation"]["contract"]["mesh_cutoff_ry"]
                == 400), "the nested info dict was shared, not copied"

    def test_info_travels_through_a_change_nobody_named_it_in(self, carrying):
        """§ 2.2a: it travels; a strip is explicit, never a field a
        rebuild forgot."""
        out = carrying.replace(positions=carrying.positions + 1.0)
        assert out.info["calculation"]["contract"]["mesh_cutoff_ry"] == 400
        assert out.cell_origin is not None, "the corner went with it"

    def test_a_strip_is_something_a_caller_says(self, carrying):
        assert carrying.replace(info={}).info == {}
        assert carrying.info, "stripping the copy emptied the source"

    # `test_a_stated_pbc_is_not_overruled_by_the_carried_kinds` and
    # `test_kinds_that_agree_with_the_stated_pbc_are_kept` were RETIRED
    # 2026-09-22, with the special case they pinned.
    #
    # They guarded a patch inside `replace()`: because the method seeded BOTH
    # `pbc` and `axis_kind` from the source, and `__post_init__` let the kind
    # win, a caller who stated only the boolean had it silently discarded --
    # so `replace()` compared the two and dropped the carried kind when they
    # contradicted.  A precedence rule in two places, which is what the door
    # exists to prevent.
    #
    # `pbc` is no longer a field (user: "why the fuck need pbc when axis_kind
    # fully contains this information and more").  There is one periodicity
    # field to carry, nothing to reconcile, and `replace(pbc=...)` is a
    # TypeError rather than a case to handle -- pinned by
    # `test_structure_periodicity.py::TestAxisKindIsTheOnePeriodicityField`.

