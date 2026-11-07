"""The species ORDER — one rule, every engine (`model/chemistry.md` § 3a).

Why this file exists: the order fixes each element's index in
`%block ChemicalSpeciesLabel`, and that index fixes the orbital ordering
inside SIESTA's `.DM` and `.TSHS`.  Two emitters carried two rules until
2026-09-23 -- `siesta/input.py` sorted by atomic number and
`transport/transiesta.py` alphabetically -- so one structure was declared
`H, C, S, Au` by one and `Au, C, H, S` by the other.

**THIS FILE OWNS THE LITERALS, AND IT IS THE ONLY ONE THAT MAY.**  Every
other test asks `chemistry.species_order` instead of spelling an order out
(user, 2026-09-23).  The split is not style:

* here the order IS the subject, so deriving the expectation from the
  function would assert the code equals itself and pin nothing;
* everywhere else the order is incidental -- a deck renders, a message names
  the elements, a sort put the lead first -- so a spelled-out order pins a
  contract that test has no opinion about, and goes red when the rule
  changes.
"""
from __future__ import annotations

import pytest

from molbuilder.chemistry import species_order


@pytest.mark.parametrize("elements,expected,why", [
    (["Au", "C", "H", "S"], ["C", "H", "S", "Au"],
     "carbon wins over the sulfur -- the Au-BDT-Au junction"),
    (["C", "H", "O"], ["C", "H", "O"],
     "carbon wins over the oxygen -- methanol is CH4O, never HCO"),
    (["H", "O"], ["H", "O"], "group VI puts hydrogen first -- H2O"),
    (["H", "F"], ["H", "F"], "group VII -- HF"),
    (["H", "Cl"], ["H", "Cl"], "group VII -- HCl"),
    (["H", "S"], ["H", "S"], "group VI -- H2S"),
    (["H", "N", "O"], ["H", "N", "O"],
     "the oxygen decides, not the nitrogen -- HNO3"),
    (["H", "N"], ["N", "H"], "no carbon, no VI/VII -- NH3, not H3N"),
    (["B", "H"], ["B", "H"], "anchor is the lightest non-H -- B2H6"),
    (["Si", "H"], ["Si", "H"], "SiH4"),
    (["Au"], ["Au"], "a bulk lead: one species, nothing to order"),
])
def test_the_order_is_the_one_a_chemist_would_write(elements, expected, why):
    assert species_order(elements) == expected, why


def test_numbered_labels_stay_distinct_and_hydrogen_lands_after_the_group():
    """`Au1`/`Au2` are two species a person wrote on purpose, and hydrogen
    belongs after the carbons as a GROUP -- not between `C1` and `C2`."""
    assert species_order(["C1", "H", "C2", "S"]) == \
        ["C1", "C2", "H", "S"]
    assert species_order(["H1", "C", "H2"]) == ["C", "H1", "H2"]
    assert species_order(["Au2", "Au1"]) == ["Au2", "Au1"], \
        "a tie keeps first-seen order, so the file's own order survives"


