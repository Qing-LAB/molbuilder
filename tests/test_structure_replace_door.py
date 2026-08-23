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
import sys

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

    def test_the_replace_hook_is_installed(self):
        """``__replace__`` is what Python 3.13+ dispatches
        ``dataclasses.replace`` through, so the door becomes automatic on a
        newer interpreter without any call site changing."""
        assert Structure.__replace__ is Structure.replace

    @pytest.mark.skipif(sys.version_info >= (3, 13),
                        reason="3.13+ routes dataclasses.replace through "
                               "__replace__, so the trap is gone")
    def test_on_312_the_stdlib_helper_still_carries_the_trap(self, held):
        """Stated, not hidden.  This interpreter has no ``__replace__`` hook,
        so ``dataclasses.replace`` keeps the old behaviour and
        ``Structure.replace`` is the only correct door.  When this repo moves
        to 3.13 this test starts being skipped and the door becomes belt AND
        braces."""
        got = dataclasses.replace(held, regions={"electrode_L": [1]})
        assert got.regions == {"electrode_L": [1], FROZEN_LABEL: [0]}
