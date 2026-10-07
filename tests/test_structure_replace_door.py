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
    does not.

    Checked against the stdlib source: `dataclasses.replace` ends
    `return obj.__class__(**changes)` and never mentions `__replace__`, on
    any version.  The 3.13 addition is `copy.replace`, a different helper.
    """

    def test_the_replace_hook_is_installed(self):
        """For `copy.replace` (3.13+), which is the helper that honours it."""
        assert Structure.__replace__ is Structure.replace


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
    the structure it returns shares ``positions``, ``cell`` and the ``info``
    dict with its source — writing to one writes to the other; enumerating
    fields in order to copy them is how fields come to be forgotten
    (`model/structure.md` § 2.2a).
    """

    @pytest.fixture
    def carrying(self):
        return Structure(
            elements=["C", "H"], positions=np.array([[0.0, 0.0, 0.0],
                                                     [0.0, 0.0, 1.1]]),
            cell=np.diag([8.0, 8.0, 8.0]),
            engine_offset=np.array([1.0, 1.0, 1.0]),
            info={"calculation": {"contract": {"mesh_cutoff_ry": 400}}})

    def test_writing_to_the_derived_copy_does_not_reach_the_source(
            self, carrying):
        out = carrying.replace(title="derived")
        out.positions[0, 0] = 7.0
        out.cell[0, 0] = 99.0
        out.engine_offset[0] = 5.0
        out.info["calculation"]["contract"]["mesh_cutoff_ry"] = 1
        assert carrying.positions[0, 0] == 0.0
        assert carrying.cell[0, 0] == 8.0
        assert carrying.engine_offset[0] == 1.0
        assert (carrying.info["calculation"]["contract"]["mesh_cutoff_ry"]
                == 400), "the nested info dict was shared, not copied"

    def test_info_travels_through_a_change_nobody_named_it_in(self, carrying):
        """§ 2.2a: it travels; a strip is explicit, never a field a
        rebuild forgot."""
        out = carrying.replace(positions=carrying.positions + 1.0)
        assert out.info["calculation"]["contract"]["mesh_cutoff_ry"] == 400
        assert out.engine_offset is not None and np.allclose(
            out.engine_offset, [1.0, 1.0, 1.0]), "the stated offset went with it"

    def test_a_strip_is_something_a_caller_says(self, carrying):
        assert carrying.replace(info={}).info == {}
        assert carrying.info, "stripping the copy emptied the source"


def test_replace_carries_every_field_the_dataclass_declares():
    """COMPLETE BY CONSTRUCTION, not by memory.

    `replace()` names the fields it carries: nine in a literal dict plus
    the five `_carry_nonatom()` supplies. That union is complete today —
    and only because someone remembered. Add a sixteenth field, forget it
    in both places, and every derived copy silently resets it to its
    default -- the failure this door exists to prevent.

    So the check iterates the LIVE field list rather than a copy of it. A
    field added tomorrow is covered the moment it is declared, and the
    test cannot go stale the way a hand-listed set would.

    `frozen_atoms` is the one exclusion and it is deliberate: it has no
    storage of its own — it reads and writes `regions[FROZEN_LABEL]` — so
    a copied `regions` already carries it (see the module docstring).
    """
    import dataclasses as _dc

    s = Structure(
        elements=["C", "O"],
        positions=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        atom_names=["C1", "O1"], residue_ids=[7, 7],
        residue_names=["LIG", "LIG"], chain_ids=["B", "B"],
        title="every field non-default",
        cell=np.diag([11.0, 12.0, 13.0]), engine_offset=[-0.5, -1.5, -2.5],
        axis_kind=("periodic", "isolated", "transport"),
        vacuum=(1.0, 2.0, 0.0),
        regions={"lead": [0], FROZEN_LABEL: [1]},
        info={"calculation": {"engine": "siesta"}},
    )
    out = s.replace(title="derived")          # one field stated, no more

    for f in _dc.fields(Structure):
        if f.name in ("frozen_atoms", "title"):
            continue
        got, want = getattr(out, f.name), getattr(s, f.name)
        if isinstance(want, np.ndarray):
            assert np.allclose(got, want), f"{f.name} did not survive replace()"
        else:
            assert got == want, (
                f"{f.name} did not survive replace() -- it is declared on the "
                f"dataclass but carried by neither the explicit list nor "
                f"_carry_nonatom(), so every derived copy resets it")
    # And the reserved label rode along inside `regions`, as designed.
    assert out.frozen_atoms == [1]
